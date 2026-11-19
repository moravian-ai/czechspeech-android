package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections

/**
 * Dictation must never crash the host app: whatever the work of one utterance throws on the recognition thread, an
 * exception or an Error (a class that failed to initialise, which is what the crash on the phone was), reaches the app
 * as one Error event for that utterance, never as an exception in the collector's scope; the session goes on.
 */
class UtteranceFailureContainmentTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val logs = Collections.synchronizedList(mutableListOf<String>())
    private val sink = CsLog.sink

    @Before
    fun captureLogs() {
        CsLog.sink = { _, message -> logs += message }
    }

    @After
    fun restoreLogs() {
        CsLog.sink = sink
    }

    private val config384 = CzechSpeechConfig(minEncoderWindow = 384)

    /** Two utterances of 2 s of speech (clear speech), a pause between. */
    private val twoUtterances = Signals.concat(
        Signals.zeros(0.5), Signals.tone(2.0), Signals.zeros(1.5), Signals.tone(2.0), Signals.zeros(1.5),
    )

    private fun listen(native: FakeNative, config: CzechSpeechConfig = config384, patternCheck: () -> Int = SdkPatterns::verify): List<SpeechEvent> =
        runBlocking {
            val engine = fakeEngine(native, config = config, patternCheck = patternCheck) { ArrayAudioSource(twoUtterances) }
            // an exception here is what the collector's scope would have received: the test fails with it
            withContext(Dispatchers.Default) { engine.listen().toList() }
        }

    private fun texts(events: List<SpeechEvent>) = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.text }

    private fun failsFirstCallWith(failure: Throwable) {
        val native = FakeNative().apply {
            text = "dobrý den"
            onTranscribe = { if (it == 0) throw failure }
        }
        val events = listen(native)
        val errors = events.filterIsInstance<SpeechEvent.Error>()
        assertEquals("one error event: $events", 1, errors.size)
        // the coroutines machinery may hand over a copy of the exception (stack trace recovery): same type and message
        assertEquals(failure.javaClass, errors.single().cause.javaClass)
        assertEquals(failure.message, errors.single().cause.message)
        assertEquals("the next utterance is recognised normally", listOf("dobrý den"), texts(events))
        assertTrue("after the error", events.indexOf(errors.single()) < events.indexOfFirst { it is SpeechEvent.Transcript })
        val first = events.filterIsInstance<SpeechEvent.SpeechStarted>().first().atMs
        assertEquals("clear speech that is missing is told once", listOf(first), events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().map { it.startMs })
        assertTrue("every array handed to the recogniser is zeroed", native.passedToWhisper.isNotEmpty() && native.passedToWhisper.all { a -> a.all { it == 0f } })
        assertEquals("the session ended normally and freed the model", native.whisperInits.get(), native.whisperFrees.get())
    }

    @Test
    fun aRuntimeExceptionOfTheRecogniserIsOneErrorEventAndTheSessionGoesOn() = failsFirstCallWith(RuntimeException("boom"))

    @Test
    fun anErrorOfTheRecogniserIsOneErrorEventToo() = failsFirstCallWith(ExceptionInInitializerError("static initialiser failed"))

    @Test
    fun anOutOfMemoryErrorEndsTheSessionAsItAlwaysDidAndTheAudioIsZeroed() {
        // the engine's precedent (StreamingSessionTest): the flow throws it, the app ends dictation (its collector catches Throwable)
        val native = FakeNative().apply { onTranscribe = { if (it == 0) throw OutOfMemoryError("fake") } }
        val outcome = runCatching { listen(native) }
        assertTrue("", outcome.exceptionOrNull() is OutOfMemoryError)
        assertTrue(native.passedToWhisper.isNotEmpty() && native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    @Test
    fun aFailureOnTheRetryPutsTheEncoderWindowFloorBack() {
        val failure = ExceptionInInitializerError("on the retry")
        val native = FakeNative().apply {
            texts += listOf("...")                  // the first run of the first utterance comes out empty: a retry on 512 follows
            onTranscribe = { if (it == 1) throw failure }
        }
        val events = listen(native)
        assertEquals(listOf(failure.javaClass), events.filterIsInstance<SpeechEvent.Error>().map { it.cause.javaClass })
        assertEquals(listOf("dobrý den"), texts(events))
        val floors = native.minAudioCtxSet.toList()
        assertEquals("the floor raised for the retry is lowered again: $floors", 384, floors[floors.indexOf(512) + 1])
        assertEquals(384, native.minAudioCtxSet.last())
    }

    // ---- the start-up check of the regular expressions ----

    @Test
    fun aPatternThatThisRuntimeRefusesIsAnErrorEventAtTheStartOfTheSessionAndAFailureOfTranscribe() {
        val defect = CzechSpeechException("Regular expression X is not valid on this runtime's regex engine")
        val native = FakeNative().apply { text = "dobrý den" }
        val events = listen(native) { throw defect }
        assertSame("the first thing the app hears of the session", defect, (events.first { it !is SpeechEvent.SpeechStarted } as SpeechEvent.Error).cause)
        assertEquals("the defect is reported once per session", 1, events.count { it is SpeechEvent.Error })
        assertEquals("dictation otherwise goes on", listOf("dobrý den", "dobrý den"), texts(events))
        assertTrue(logs.any { it.contains("start-up check of the regular expressions failed") })

        val engine = fakeEngine(FakeNative(), patternCheck = { throw defect }) { ArrayAudioSource(twoUtterances) }
        try {
            runBlocking { engine.transcribe(Signals.tone(1.0)) }
            fail("transcribe should fail")
        } catch (e: CzechSpeechException) {
            assertSame(defect, e)
        }
    }

    @Test
    fun aPatternCheckThatThrowsAnErrorDoesNotStopTheEngineFromBeingCreated() {
        val engine = fakeEngine(FakeNative(), patternCheck = { throw ExceptionInInitializerError("static") }) { ArrayAudioSource(twoUtterances) }
        engine.close()
    }

    @Test
    fun aGoodEngineReportsNoDefect() {
        val events = listen(FakeNative().apply { text = "dobrý den" })
        assertTrue(events.none { it is SpeechEvent.Error })
    }
}
