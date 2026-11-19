package ai.moravian.czechspeech.internal

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

/**
 * An abort stops whisper.cpp inside the encoder, between two nodes of its graph
 * (native/patches/whisper.cpp-encoder-abort.patch). Unpatched, whisper.cpp looks at the abort callback only after the
 * encoder pass, so an abort that came during it waited for its end: on a phone the final text waited 2 to 4 s behind
 * a preview that had been dropped.
 *
 * The measurement needs the host library and CZECHSPEECH_MODEL (no recording: any audio keeps the encoder busy); the
 * source check runs always.
 */
class NativeEncoderAbortTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(300)

    private var handle = 0L

    @After
    fun free() {
        if (handle != 0L) JniNativeApi.whisperFree(handle)
    }

    @Test
    fun thePatchPassesTheAbortCallbackToEveryGraphOfTheEncoder() {
        val patch = File("../native/patches/whisper.cpp-encoder-abort.patch").takeIf { it.exists() }?.readText()
            ?: error("native/patches/whisper.cpp-encoder-abort.patch not found from ${File(".").absolutePath}")
        val passed = Regex("""^\+\s*if \(!ggml_graph_compute_helper\(sched, gf, n_threads, true, abort_callback, abort_callback_data\)\)""", RegexOption.MULTILINE)
        assertTrue("the conv, encoder and cross-attention graphs", passed.findAll(patch).count() == 3)
        assertTrue("taken off again after the graph", patch.contains("czechspeech_set_abort_callback(sched, nullptr, nullptr)"))
    }

    @Test
    fun anAbortDuringTheEncoderReturnsWithinAFractionOfItsTimeAndTheContextStillWorks() {
        TestEnv.assumeNativeLibrary()
        TestEnv.assumeFile(TestEnv.model, "CzechSpeech model (CZECHSPEECH_MODEL)")
        handle = JniNativeApi.whisperInit(TestEnv.model.absolutePath)
        assertTrue("model loaded", handle != 0L)
        val threads = TestEnv.threads
        // 25 s: an encoder window of 1296 frames, seconds of work on any machine
        val audio = Signals.concat(Signals.tone(12.5), Signals.whiteNoise(12.5, amplitude = 0.05f))

        var t0 = System.nanoTime()
        val first = JniNativeApi.whisperTranscribe(handle, 1, audio, audio.size, threads, "cs", null)
        val fullMs = (System.nanoTime() - t0) / 1_000_000
        assertNotNull("an unaborted recognition", first)

        // Abort a tenth of that time in: the mel spectrogram is long done, the encoder is at work.
        val abortAfterMs = (fullMs / 10).coerceIn(50, 1_000)
        var result: ByteArray? = ByteArray(0)
        var endedNs = 0L
        val started = CountDownLatch(1)
        val worker = thread(name = "abort-test") {
            started.countDown()
            result = JniNativeApi.whisperTranscribe(handle, 2, audio, audio.size, threads, "cs", null)
            endedNs = System.nanoTime()
        }
        assertTrue(started.await(10, TimeUnit.SECONDS))
        Thread.sleep(abortAfterMs)
        t0 = System.nanoTime()
        JniNativeApi.whisperAbort(handle, 2)
        worker.join(120_000)
        assertTrue("the aborted call returned", !worker.isAlive)
        val abortMs = (endedNs - t0) / 1_000_000
        println(
            "encoder abort: full recognition of ${audio.size / SAMPLE_RATE} s $fullMs ms on $threads threads; aborted " +
                "$abortAfterMs ms in, returned $abortMs ms after the abort",
        )
        assertNull("aborted: no result", result)
        assertTrue("returned $abortMs ms after the abort; a full run takes $fullMs ms", abortMs * 4 < fullMs)

        val again = JniNativeApi.whisperTranscribe(handle, 3, audio, audio.size, threads, "cs", null)
        assertArrayEquals("the same result after an aborted run", first, again)
    }
}
