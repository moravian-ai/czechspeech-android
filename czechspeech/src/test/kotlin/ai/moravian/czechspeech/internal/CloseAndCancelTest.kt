package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * close() in each window of the start of a session or a call (held in a test hook, so the interleaving is exact),
 * awaitClosed(), and cancellation that never waits for the whisper thread.
 */
class CloseAndCancelTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))

    /** A hook that holds the calling thread at one point until [proceed] is released. */
    private class Hold {
        val reached = CountDownLatch(1)
        val proceed = CountDownLatch(1)
        fun here() {
            reached.countDown()
            check(proceed.await(10, TimeUnit.SECONDS)) { "never released" }
        }
    }

    // close() in each window

    @Test
    fun closeJustAfterASessionRegisteredCancelsItBeforeItOpensTheMicrophone() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val source = ArrayAudioSource(utterance)
        val engine = fakeEngine(native, hooks = object : SpeechEngine.Hooks {
            override fun sessionRegistered() = hold.here()
        }) { source }
        val collected = async(Dispatchers.Default) { runCatching { engine.listen().toList() } }
        assertTrue(hold.reached.await(10, TimeUnit.SECONDS))
        engine.close()                       // the session is registered, nothing of it has started
        hold.proceed.countDown()
        val outcome = collected.awaitWithin(5_000, "the session registered before close()")
        assertTrue("the session is cancelled; got $outcome", outcome.exceptionOrNull() is CancellationException)
        engine.awaitClosed()
        assertFalse("the microphone was never opened", source.opened)
        assertEquals("nothing loaded", 0, native.whisperInits.get())
        assertFalse(engine.isListening)
    }

    @Test
    fun closeJustAfterATranscribeRegisteredFailsItAndNothingRuns() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = fakeEngine(native, hooks = object : SpeechEngine.Hooks {
            override fun callRegistered() = hold.here()
        }) { error("no microphone in this test") }
        val call = async(Dispatchers.Default) { runCatching { engine.transcribe(Signals.tone(1.0)) } }
        assertTrue(hold.reached.await(10, TimeUnit.SECONDS))
        engine.close()                       // registered, not queued on the whisper thread yet
        hold.proceed.countDown()
        assertClosed(call.awaitWithin(5_000, "the transcribe() registered before close()"))
        engine.awaitClosed()
        assertEquals("the gate never ran", 0, native.gateCalls.get())
        assertEquals("nothing loaded", 0, native.whisperInits.get())
    }

    @Test
    fun closeBetweenTheCheckAndTheRegistrationOfATranscribeFailsIt() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = fakeEngine(native, hooks = object : SpeechEngine.Hooks {
            override fun callStarting() = hold.here()
        }) { error("no microphone in this test") }
        val call = async(Dispatchers.Default) { runCatching { engine.transcribe(Signals.tone(1.0)) } }
        assertTrue(hold.reached.await(10, TimeUnit.SECONDS))
        engine.close()                       // the call passed its first check, it is not registered
        hold.proceed.countDown()
        assertClosed(call.awaitWithin(5_000, "the transcribe() starting at close()"))
        engine.awaitClosed()
        assertEquals("the gate never ran", 0, native.gateCalls.get())
        assertEquals("nothing loaded", 0, native.whisperInits.get())
    }

    // awaitClosed()

    @Test
    fun awaitClosedBeforeCloseIsAnError() = runBlocking {
        val engine = fakeEngine(FakeNative()) { error("no microphone in this test") }
        val e = failure<IllegalStateException> { engine.awaitClosed() }
        assertTrue(e.message, e.message!!.contains("close()"))
    }

    @Test
    fun awaitClosedWithNothingRunningReturnsAtOnce() = runBlocking {
        val engine = fakeEngine(FakeNative()) { error("no microphone in this test") }
        engine.close()
        assertTrue(withTimeoutOrNull(2_000) { engine.awaitClosed(); true } == true)
    }

    @Test
    fun awaitClosedReturnsOnceTheModelLoadRunningAtCloseHasEndedAndTheModelIsFreed() = runBlocking {
        val load = CountDownLatch(1)
        val native = FakeNative().apply { holdInit = load }
        val source = ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1)
        val engine = fakeEngine(native) { source }
        val job = launch(Dispatchers.Default) { runCatching { engine.listen().collect { } } }
        try {
            assertTrue("the model is loading", native.initStarted.await(10, TimeUnit.SECONDS))
            engine.close()                   // returns at once; the load cannot be interrupted
            job.joinWithin(2_000, "the session cancelled by close()")
            val waiting = async(Dispatchers.Default) { engine.awaitClosed() }
            assertNull("awaitClosed() waits for the load", withTimeoutOrNull(300) { waiting.await() })
            load.countDown()
            waiting.awaitWithin(5_000, "awaitClosed() after the load ended")
            assertEquals("the model is freed when awaitClosed() returns", 1, native.whisperFrees.get())
            assertFalse(engine.isModelLoaded)
            assertTrue("the microphone is closed", source.closed)
            assertFalse(engine.isListening)
        } finally {
            load.countDown()
        }
    }

    @Test
    fun awaitClosedReturnsOnceTheRecognitionRunningAtCloseHasEndedAndTheModelIsFreed() = runBlocking {
        val recognition = CountDownLatch(1)
        val native = FakeNative().apply { holdRecognition = recognition }
        val engine = fakeEngine(native) { error("no microphone in this test") }
        val call = async(Dispatchers.Default) { runCatching { engine.transcribe(Signals.tone(1.0)) } }
        try {
            assertTrue("the recogniser is running", native.transcribeStarted.await(10, TimeUnit.SECONDS))
            engine.close()
            assertClosed(call.awaitWithin(2_000, "the transcribe() failed by close()"))
            val waiting = async(Dispatchers.Default) { engine.awaitClosed() }
            assertNull("awaitClosed() waits for the recognition", withTimeoutOrNull(300) { waiting.await() })
            recognition.countDown()
            waiting.awaitWithin(5_000, "awaitClosed() after the recognition ended")
            assertEquals("the model is freed when awaitClosed() returns", 1, native.whisperFrees.get())
            assertFalse(engine.isModelLoaded)
        } finally {
            recognition.countDown()
        }
    }

    // cancellation never waits for the whisper thread

    @Test
    fun cancellingASessionDoesNotWaitForAnotherCallersTranscribe() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative()
        // Silence only: the session itself never runs the speech gate.
        val source = ArrayAudioSource(FloatArray(0), tail = { 0f }, sleepPerFrameMs = 1)
        val engine = fakeEngine(native) { source }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        awaitCondition("the session's model") { engine.isModelLoaded }
        native.holdGate = gate
        val call = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        try {
            assertTrue("another caller's transcribe() is in the gate", native.gateStarted.await(10, TimeUnit.SECONDS))
            job.cancel()
            job.joinWithin(2_000, "the cancelled session, with a transcribe() of another caller running")
            assertFalse(engine.isListening)
            assertTrue("the microphone closed", source.closed)
        } finally {
            gate.countDown()
        }
        assertEquals("dobrý den", call.awaitWithin(10_000, "the other caller's transcribe()"))
        assertTrue("the model is freed once both are done", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun aSessionThatEndsNormallyHasFreedTheModelWhenTheFlowCompletes() = runBlocking {
        // stop() and the end of the audio are not cancellation: the release is still awaited.
        val native = FakeNative()
        val engine = fakeEngine(native) { ArrayAudioSource(utterance) }
        val events = engine.listen().toList()
        assertEquals(1, events.count { it is SpeechEvent.Transcript })
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }
}
