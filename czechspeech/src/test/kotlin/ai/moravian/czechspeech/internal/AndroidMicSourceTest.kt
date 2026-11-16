package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The microphone source against a fake platform (android.jar has no AudioRecord on the JVM): every platform failure
 * reaches the app as an exception it can act on.
 */
class AndroidMicSourceTest {
    private class FakeRecorder(
        var initialized: Boolean = true,
        var recordingAfterStart: Boolean = true,
        var startFailure: RuntimeException? = null,
        /** Results of successive read() calls; when empty, every read fills the rest of the frame. */
        val reads: ArrayDeque<Int> = ArrayDeque(),
    ) : AndroidMicSource.Recorder {
        var started = false
        var stopped = false
        var releases = 0

        override val isInitialized: Boolean get() = initialized
        override val isRecording: Boolean get() = started && recordingAfterStart

        override fun start() {
            startFailure?.let { throw it }
            started = true
        }

        override fun read(frame: FloatArray, offset: Int, count: Int): Int {
            val n = reads.removeFirstOrNull() ?: count
            if (n > 0) frame.fill(0.25f, offset, offset + n)
            return n
        }

        override fun stop() {
            check(started) { "stop() called on an AudioRecord that is not recording" }
            stopped = true
        }

        override fun release() {
            releases++
        }
    }

    private class FakePlatform(
        val recorder: FakeRecorder = FakeRecorder(),
        val minBytes: Int = 2_048,
        val recorderFailure: RuntimeException? = null,
        val priorityFailure: RuntimeException? = null,
    ) : AndroidMicSource.Platform {
        var bufferBytes = 0

        override fun raiseThreadPriority() {
            priorityFailure?.let { throw it }
        }

        override fun minBufferBytes(): Int = minBytes

        override fun newRecorder(bufferBytes: Int): AndroidMicSource.Recorder {
            recorderFailure?.let { throw it }
            this.bufferBytes = bufferBytes
            return recorder
        }
    }

    private fun openFails(platform: FakePlatform): CzechSpeechException =
        failure { AndroidMicSource(platform).open() }

    @Test
    fun opensWithHalfASecondOfBufferReadsWholeFramesAndCloses() {
        val recorder = FakeRecorder(reads = ArrayDeque(listOf(200, 0, 312)))
        val platform = FakePlatform(recorder)
        val mic = AndroidMicSource(platform)
        mic.open()
        assertEquals("at least 0.5 s of 16 kHz float audio", SAMPLE_RATE * 4 / 2, platform.bufferBytes)
        val frame = FloatArray(FRAME_SAMPLES)
        assertEquals("partial reads are joined into one frame", FRAME_SAMPLES, mic.read(frame))
        assertTrue(frame.all { it == 0.25f })
        mic.close()
        assertTrue(recorder.stopped)
        assertEquals(1, recorder.releases)
        mic.close()
        assertEquals("close() twice releases once", 1, recorder.releases)
    }

    @Test
    fun aDeviceThatCannotRecord16kHzFloatIsReportedAsSuch() {
        val e = openFails(FakePlatform(minBytes = -2))
        assertTrue(e.message, e.message!!.contains("cannot record 16 kHz mono float audio"))
        for (cause in listOf(IllegalArgumentException("bad format"), UnsupportedOperationException("no float"))) {
            val wrapped = openFails(FakePlatform(recorderFailure = cause))
            assertTrue(wrapped.message, wrapped.message!!.contains("cannot record 16 kHz mono float audio"))
            assertSame("the platform exception is the cause", cause, wrapped.cause)
        }
    }

    @Test
    fun aRecorderThatDoesNotInitialiseIsReleasedAndReported() {
        val recorder = FakeRecorder(initialized = false)
        val e = openFails(FakePlatform(recorder))
        assertTrue(e.message, e.message!!.contains("could not be opened") && e.message!!.contains("another app"))
        assertEquals(1, recorder.releases)
    }

    @Test
    fun startRecordingFailingIsReportedAndTheRecorderReleased() {
        val cause = IllegalStateException("startRecording() called on an uninitialized AudioRecord")
        val recorder = FakeRecorder(startFailure = cause)
        val e = openFails(FakePlatform(recorder))
        assertTrue(e.message, e.message!!.contains("could not be started") && e.message!!.contains("another app"))
        assertSame(cause, e.cause)
        assertEquals(1, recorder.releases)
    }

    @Test
    fun aMicrophoneHeldByAnotherAppIsReportedAndTheRecorderReleased() {
        val recorder = FakeRecorder(recordingAfterStart = false)
        val e = openFails(FakePlatform(recorder))
        assertTrue(e.message, e.message!!.contains("in use by another app"))
        assertEquals(1, recorder.releases)
    }

    @Test
    fun aPermissionRevokedAfterTheCheckStaysASecurityException() {
        val cause = SecurityException("Permission Denial: RECORD_AUDIO")
        val e = failure<SecurityException> { AndroidMicSource(FakePlatform(recorderFailure = cause)).open() }
        assertTrue(e.message, e.message!!.contains("RECORD_AUDIO"))
        assertSame(cause, e.cause)
    }

    @Test
    fun theCaptureThreadPriorityIsBestEffort() {
        // Some builds refuse THREAD_PRIORITY_URGENT_AUDIO; that must not fail the session, nor read as a missing
        // RECORD_AUDIO permission.
        for (refusal in listOf(SecurityException("priority"), IllegalArgumentException("priority"))) {
            val recorder = FakeRecorder()
            val mic = AndroidMicSource(FakePlatform(recorder, priorityFailure = refusal))
            mic.open()
            assertTrue(recorder.started)
            mic.close()
        }
    }

    @Test
    fun readErrorsSayWhatHappened() {
        fun readFails(code: Int): CzechSpeechException {
            val mic = AndroidMicSource(FakePlatform(FakeRecorder(reads = ArrayDeque(listOf(code)))))
            mic.open()
            return failure { mic.read(FloatArray(FRAME_SAMPLES)) }
        }
        val dead = readFails(-6)                    // AudioRecord.ERROR_DEAD_OBJECT
        assertTrue(dead.message, dead.message!!.contains("start listening again"))
        val other = readFails(-3)                   // AudioRecord.ERROR_INVALID_OPERATION
        assertTrue(other.message, other.message!!.contains("-3"))

        val silent = AndroidMicSource(FakePlatform(FakeRecorder(reads = ArrayDeque(List(101) { 0 }))))
        silent.open()
        val stalled = failure<CzechSpeechException> { silent.read(FloatArray(FRAME_SAMPLES)) }
        assertTrue(stalled.message, stalled.message!!.contains("stopped delivering audio"))
    }

    @Test
    fun closeBeforeOpenDoesNothingAndReadBeforeOpenFails() {
        val recorder = FakeRecorder()
        val mic = AndroidMicSource(FakePlatform(recorder))
        mic.close()
        assertEquals(0, recorder.releases)
        failure<IllegalStateException> { mic.read(FloatArray(FRAME_SAMPLES)) }
        assertFalse(recorder.started)
    }
}
