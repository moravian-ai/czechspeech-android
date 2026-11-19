package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger

/**
 * A final recognition that comes out without a letter or a digit is repeated with a longer encoder window when the
 * endpointer saw clear speech in it (the fake recogniser answers from [FakeNative.texts], one text per call; the
 * floors the engine set show which window each call ran on).
 */
class EmptyFinalRetrySessionTest {
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

    /** 2 s of speech: clear speech (at least 1 s), 2.9 s of audio: on the lowest window of the floor. */
    private fun speech(seconds: Double = 2.0) = Signals.concat(Signals.zeros(0.5), Signals.tone(seconds), Signals.zeros(1.5))

    private fun session(
        native: FakeNative,
        config: CzechSpeechConfig = config384,
        audio: FloatArray = speech(),
        before: (SpeechEngine) -> Unit = {},
    ): List<SpeechEvent> = runBlocking {
        val engine = fakeEngine(native, config = config) { ArrayAudioSource(audio) }
        before(engine)
        withContext(Dispatchers.Default) { engine.listen().toList() }
    }

    private fun native(vararg answers: String) = FakeNative().apply {
        texts += answers.toList()
        keepCopies = true
    }

    private val retryLines get() = logs.filter { it.contains("empty result, retry with window") }
    private val droppedLines get() = logs.filter { it.contains("no letters or digits, dropped") }
    private fun List<SpeechEvent>.texts() = filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.text }

    @Test
    fun anEmptyResultIsRescuedByTheWindowOf512() {
        val native = native("...", "dobrý den")
        val events = session(native)
        assertEquals(listOf("dobrý den"), events.texts())
        assertEquals("two calls, the second on the window of 512, the floor then put back", listOf(384, 512, 384), native.minAudioCtxSet.toList())
        assertEquals(listOf("utterance: empty result, retry with window 512: rescued"), retryLines)
        assertTrue("no dropped line: $logs", droppedLines.isEmpty())
        assertTrue(events.none { it is SpeechEvent.UtteranceNotWritten })
        val started = events.filterIsInstance<SpeechEvent.SpeechStarted>().single().atMs
        assertEquals("the same startMs as a normal final", started, events.filterIsInstance<SpeechEvent.Transcript>().single().transcript.startMs)
    }

    @Test
    fun anEmptyResultIsRescuedByTheFullWindowAfterTheWindowOf512() {
        val native = native("...", ",", "dobrý den")
        val events = session(native)
        assertEquals(listOf("dobrý den"), events.texts())
        assertEquals(listOf(384, 512, 384, 1488, 384), native.minAudioCtxSet.toList())
        assertEquals(
            listOf(
                "utterance: empty result, retry with window 512: still empty",
                "utterance: empty result, retry with window 1488: rescued",
            ),
            retryLines,
        )
        assertTrue(droppedLines.isEmpty())
    }

    @Test
    fun aResultThatStaysEmptyAfterBothRetriesIsDroppedOnceAndTheAppIsToldOnceBeforeUtteranceDone() {
        val native = native("...", "[ ... ]", "♪")
        val events = session(native, config = config384.copy(streaming = StreamingConfig())) { it.streamingGuard!!.recorded(6_000, 3_000) }
        assertEquals(3, native.transcribeCalls.get())
        assertTrue(events.texts().isEmpty())
        assertEquals(2, retryLines.size)
        assertTrue(retryLines.all { it.endsWith(": still empty") })
        val dropped = droppedLines
        assertEquals("the dropped line comes once, after the last attempt: $logs", 1, dropped.size)
        assertTrue(logs.indexOf(dropped.single()) > logs.indexOf(retryLines.last()))
        assertTrue("the last attempt's kind of nothing: ${dropped.single()}", dropped.single().contains("model returned: symbols"))
        val unwritten = events.filterIsInstance<SpeechEvent.UtteranceNotWritten>()
        assertEquals(1, unwritten.size)
        val done = events.filterIsInstance<SpeechEvent.UtteranceDone>().single()
        assertEquals(done.startMs, unwritten.single().startMs)
        assertEquals(events.filterIsInstance<SpeechEvent.SpeechStarted>().single().atMs, unwritten.single().startMs)
        assertTrue("before UtteranceDone", events.indexOf(unwritten.single()) < events.indexOf(done))
    }

    @Test
    fun withoutClearSpeechThereIsNoRetryAndNoEvent() {
        // 0.6 s of speech: more than minSpeechMs, less than a second
        val native = native("...", "dobrý den")
        val events = session(native, audio = speech(0.6))
        assertEquals(1, native.transcribeCalls.get())
        assertTrue(retryLines.isEmpty())
        assertEquals(1, droppedLines.size)
        assertTrue(events.none { it is SpeechEvent.UtteranceNotWritten })
        assertTrue(events.texts().isEmpty())
    }

    @Test
    fun aRecognitionOnTheFullWindowAlreadyIsNotRepeatedButTheAppIsTold() {
        val native = native("...", "dobrý den")
        val events = session(native, config = CzechSpeechConfig(minEncoderWindow = EmptyFinalRetry.FULL_WINDOW_FLOOR))
        assertEquals(1, native.transcribeCalls.get())
        assertTrue(retryLines.isEmpty())
        assertEquals(1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
    }

    @Test
    fun fromTheFloorOf512OnlyTheFullWindowIsTried() {
        val native = native("...", "dobrý den")
        val events = session(native, config = CzechSpeechConfig(minEncoderWindow = 512))
        assertEquals(listOf("dobrý den"), events.texts())
        assertEquals(listOf(512, 1488, 512), native.minAudioCtxSet.toList())
        assertEquals(listOf("utterance: empty result, retry with window 1488: rescued"), retryLines)
    }

    @Test
    fun theSwitchTurnsTheRetryOffButTheEventStays() {
        val native = native("...", "dobrý den")
        val events = session(native, config = config384.copy(retryEmptyFinals = false))
        assertEquals(1, native.transcribeCalls.get())
        assertEquals(listOf(384), native.minAudioCtxSet.toList())
        assertTrue(retryLines.isEmpty())
        assertEquals(1, droppedLines.size)
        assertEquals(1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
    }

    @Test
    fun theSameAudioIsRecognisedEveryTimeAndIsZeroedAfterTheLastAttempt() {
        val native = native("...", ",", "dobrý den")
        session(native)
        assertEquals(3, native.copiesPassed.size)
        assertTrue("the audio is not silent", native.copiesPassed[0].any { it != 0f })
        assertArrayEquals(native.copiesPassed[0], native.copiesPassed[1], 0f)
        assertArrayEquals(native.copiesPassed[0], native.copiesPassed[2], 0f)
        assertTrue("zeroed afterwards", native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    @Test
    fun noWordsAtAllIsNotAnEmptyResultForTheEventWhenTheGateFoundNoSpeech() {
        // silence: the gate runs nothing, no retry and no event
        val native = native("...")
        val events = session(native, audio = Signals.zeros(3.0))
        assertEquals(0, native.transcribeCalls.get())
        assertTrue(events.none { it is SpeechEvent.UtteranceNotWritten })
    }

    // ---- streaming ----

    @Test
    fun aRescuedFinalOfAStreamedPieceIsAChunkAndTheRetryDoesNotPoisonTheFlatCost() {
        val native = native("...", "dobrý den").apply {
            recognitionMs = 50
            onTranscribe = { if (it > 0) Thread.sleep(250) }            // the retry is slow, as a longer window is
        }
        lateinit var guard: StreamingGuard
        val events = session(native, config = config384.copy(streaming = StreamingConfig())) {
            guard = it.streamingGuard!!
            guard.recorded(6_000, 3_000)                                // passes paused: the final is the only recognition
        }
        assertEquals(listOf("dobrý den"), events.texts())
        assertEquals(listOf(384, 512, 384), native.minAudioCtxSet.toList())
        val done = events.filterIsInstance<SpeechEvent.UtteranceDone>().single()
        assertTrue("the Transcript comes before UtteranceDone", events.indexOfFirst { it is SpeechEvent.Transcript } < events.indexOf(done))
        assertTrue(events.none { it is SpeechEvent.UtteranceNotWritten })
        assertEquals(1, guard.retryCount)
        assertTrue("the retry time is booked apart: ${guard.retryRecogniserMs}", guard.retryRecogniserMs >= 250)
        assertTrue("the flat cost is the first run's, not the retry's: ${guard.closingEstimateMs(5_000)}", guard.closingEstimateMs(5_000) < 200)
        assertFalse(guard.closingEstimateMs(5_000) == 0L)
    }

    // ---- the full-window retry waits for an idle recogniser; stop(), close() and cancellation; pieces that wrote words ----

    /** Capture goes on only once each pass has been absorbed (as `Paced` in the streaming tests), so the passes are deterministic. */
    private class PacedPasses : SpeechEngine.Hooks {
        val requested = AtomicInteger()
        val finished = AtomicInteger()
        override fun streamingPassRequested(samples: FloatArray) {
            val n = requested.incrementAndGet()
            val deadline = System.nanoTime() + 10_000_000_000L
            while (finished.get() < n) {
                check(System.nanoTime() < deadline) { "pass $n never finished" }
                Thread.sleep(1)
            }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }
    }

    private fun waitUntil(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (!condition()) {
            check(System.nanoTime() < deadline) { "timed out: $what" }
            Thread.sleep(1)
        }
    }

    /** Two utterances of clear speech 0.9 s apart; the fake recogniser takes 150 ms per call, so the second is queued by the time the first is recognised. */
    private fun twoUtterances() = Signals.concat(Signals.zeros(0.5), Signals.tone(1.5), Signals.zeros(0.9), Signals.tone(1.5), Signals.zeros(1.5))

    @Test
    fun theFullWindowRetryIsSkippedWhileAnotherUtteranceWaitsInTheQueue() {
        val native = native("...", "...", "dobrý den").apply { recognitionMs = 150 }
        val events = session(native, audio = twoUtterances())
        assertEquals("the second utterance is recognised on its own turn", listOf("dobrý den"), events.texts())
        assertEquals("no 1488 in it: the first utterance got its 512 retry only", listOf(384, 512, 384), native.minAudioCtxSet.toList())
        assertEquals(listOf("utterance: empty result, retry with window 512: still empty"), retryLines)
        assertEquals(1, droppedLines.size)
        assertEquals("the first utterance is told as not written (the retries the rule allows were made)", 1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
    }

    @Test
    fun inStreamingTheFullWindowRetryIsSkippedWhileAnotherFinalIsPending() {
        val native = native("...", "...", "dobrý den").apply { recognitionMs = 150 }
        val events = session(native, config = config384.copy(streaming = StreamingConfig()), audio = twoUtterances()) { it.streamingGuard!!.recorded(6_000, 3_000) }
        assertEquals(listOf("dobrý den"), events.texts())
        assertEquals(listOf(384, 512, 384), native.minAudioCtxSet.toList())
        assertEquals(listOf("utterance: empty result, retry with window 512: still empty"), retryLines)
        assertEquals(2, events.filterIsInstance<SpeechEvent.UtteranceDone>().size)
    }

    @Test
    fun afterStopNoRetryStartsButTheStretchIsStillReportedAsUnwritten() {
        val native = native("...", "dobrý den")
        lateinit var engine: SpeechEngine
        native.onTranscribe = { if (it == 0) engine.stop() }      // during the first run
        val events = session(native) { engine = it }
        assertEquals("the first run happened, no retry", 1, native.transcribeCalls.get())
        assertEquals(listOf(384), native.minAudioCtxSet.toList())
        assertTrue(retryLines.isEmpty())
        assertEquals("the dropped line, once", 1, droppedLines.size)
        assertEquals("speech never disappears silently: the stretch is told even though stop() kept the retry from running", 1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
        assertTrue(events.texts().isEmpty())
        assertTrue(native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    @Test
    fun stopDuringTheRetryLetsItFinishButNoFurtherRetryStartsAndTheStretchIsStillTold() {
        val native = native("...", "...", "dobrý den")
        lateinit var engine: SpeechEngine
        native.onTranscribe = { if (it == 1) engine.stop() }      // during the retry on 512
        val events = session(native) { engine = it }
        assertEquals(2, native.transcribeCalls.get())
        assertEquals("the floor is put back after the retry that was running", listOf(384, 512, 384), native.minAudioCtxSet.toList())
        assertEquals(listOf("utterance: empty result, retry with window 512: still empty"), retryLines)
        assertEquals(1, droppedLines.size)
        assertEquals(1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
        assertTrue(events.texts().isEmpty())
        assertTrue("zeroed", native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    private fun endedWhileTheRetryRuns(end: suspend (SpeechEngine, kotlinx.coroutines.Job) -> Unit) = runBlocking {
        val native = native("...", "...", "dobrý den").apply { abortableCalls = setOf(1) }       // the retry blocks until aborted
        val engine = fakeEngine(native, config = config384) { ArrayAudioSource(speech()) }
        val events = Collections.synchronizedList(mutableListOf<SpeechEvent>())
        val job = launch(Dispatchers.Default) { runCatching { engine.listen().collect { events += it } } }
        waitUntil("the retry started") { native.transcribeCalls.get() >= 2 }
        end(engine, job)
        engine.close()
        engine.awaitClosed()
        assertEquals("the retry was aborted, no more", 2, native.transcribeCalls.get())
        assertEquals("the floor is put back", listOf(384, 512, 384), native.minAudioCtxSet.toList())
        assertTrue("nothing is sent after the end: $events", events.none { it is SpeechEvent.Transcript || it is SpeechEvent.UtteranceNotWritten })
        assertTrue("zeroed", native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    @Test
    fun cancellingTheCollectionDuringTheRetryAbortsItAndPutsTheFloorBack() = endedWhileTheRetryRuns { _, job -> job.cancelAndJoin() }

    @Test
    fun closeDuringTheRetryAbortsItAndPutsTheFloorBack() = endedWhileTheRetryRuns { engine, job ->
        engine.close()
        job.join()
    }

    @Test
    fun aPieceThatWroteWordsGetsNoRetryAndNoEventWhenNoWordsWerePendingButOneWhenTheyWere() {
        // passes: "a b c", "a b c d" (a b c committed, d pending), then nothing: the tail of the piece is only breath
        val breath = native("slovoa slovob slovoc", "slovoa slovob slovoc slovod", "").apply { text = "" }
        val hooks = PacedPasses()
        val first = runBlocking {
            val engine = fakeEngine(breath, hooks, config = config384.copy(streaming = StreamingConfig())) { ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(3.5), Signals.zeros(1.5))) }
            withContext(Dispatchers.Default) { engine.listen().toList() }
        }
        assertEquals("only the chunk of the passes: ${first.texts()}", listOf("slovoa slovob slovoc"), first.texts().take(1))
        assertEquals("no retry: the floor was set once, for the model", listOf(384), breath.minAudioCtxSet.toList())
        assertEquals(breath.previewCalls.get() + 1, breath.transcribeCalls.get())
        assertTrue(first.none { it is SpeechEvent.UtteranceNotWritten })
        assertTrue(retryLines.isEmpty())

        // the last pass still had a word pending: what the final finds in the rest may be speech, so it is repeated
        logs.clear()
        val pending = native("slovoa slovob slovoc", "slovoa slovob slovoc slovod", "", "slovod").apply { text = "" }
        val second = runBlocking {
            val engine = fakeEngine(pending, PacedPasses(), config = config384.copy(streaming = StreamingConfig())) { ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(2.2), Signals.zeros(1.5))) }
            withContext(Dispatchers.Default) { engine.listen().toList() }
        }
        assertEquals(listOf("utterance: empty result, retry with window 512: rescued"), retryLines)
        assertEquals(listOf(384, 512, 384), pending.minAudioCtxSet.toList())
        assertTrue(second.texts().contains("slovod"))
        assertTrue(second.none { it is SpeechEvent.UtteranceNotWritten })
    }
}
