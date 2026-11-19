package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.EarlyStart
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.Transcript
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Early start of the final recognition ([CzechSpeechConfig.earlyStart]) in a session, on the fake native layer.
 *
 * The audio, in 32 ms frames counted from 0: a tone from 0.5 s for 1 s is speech in frames 15..46 (pre-roll from frame
 * 5); the pause after it is handed out early with frame 54 (8 frames of silence) and ends the utterance with frame 65
 * (19 frames). The source's onFrame(n) runs while frame n-1 is being read, when the frames up to n-2 have been
 * processed: from onFrame(56) on, the early start of frame 54 has happened. Where a test needs the recogniser at a
 * certain point, the capture thread waits for it there, so nothing depends on timing.
 */
class EarlyStartSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val on = CzechSpeechConfig(earlyStart = EarlyStart())
    private val startMs = 480L
    private val endMs = 47 * FRAME_MS
    private val earlyAt = 56                 // onFrame(n) from here on: the first pause has been handed out early
    private val endAt = 67                   // ... the first utterance has ended
    private val utteranceFrames = 61         // frames 5..65
    private val audibleFrames = 50           // frames 5..54: up to the early start

    /** Not digital silence: a microphone's noise floor, far under the fake VAD's speech level. */
    private fun quiet(seconds: Double, seed: Long) = Signals.whiteNoise(seconds, amplitude = 0.01f, seed = seed)

    /** One utterance: speech for 1 s, then quiet. */
    private fun one() = Signals.concat(quiet(0.5, 1), Signals.tone(1.0), quiet(1.5, 2))

    /**
     * One utterance with a pause of 0.4 s inside: speech in frames 15..46 and 59..90. The pause is handed out early
     * with frame 54 and retracted with frame 59; the pause after it with frame 98, and the utterance ends with 109.
     */
    private fun withPause() =
        Signals.concat(quiet(0.5, 1), Signals.tone(1.0), quiet(0.4, 2), Signals.tone(1.0), quiet(1.5, 3))

    private val pausedEndMs = 91 * FRAME_MS
    private val pausedFrames = 105           // frames 5..109

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

    /** Whether an event like this has been sent; safe on any thread while the session runs. */
    private fun seen(predicate: (ListenEvent) -> Boolean): Boolean = synchronized(events) { events.any(predicate) }

    /** The log of the session run by [withLogs], while it runs and afterwards. */
    private val logs: MutableList<String> = Collections.synchronizedList(mutableListOf())

    private fun logged(predicate: (String) -> Boolean): Boolean = synchronized(logs) { logs.any(predicate) }

    private suspend fun collect(flow: Flow<ListenEvent>) =
        withContext(Dispatchers.Default) { flow.collect { events += it } }

    /** Blocks the capture thread until [condition] holds; fails the session after 10 s. */
    private fun waitOnCapture(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (!condition()) {
            if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for $what")
            Thread.sleep(1)
        }
    }

    private fun withLogs(block: suspend () -> Unit): List<String> {
        logs.clear()
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            runBlocking { block() }
        } finally {
            CsLog.sink = sink
        }
        return synchronized(logs) { logs.toList() }
    }

    private fun assertAllZero(arrays: List<FloatArray>) {
        assertTrue("whisper was called", arrays.isNotEmpty())
        for (a in arrays) assertTrue("audio buffer zeroed after use", a.all { it == 0f })
    }

    private class Counting : SpeechEngine.Hooks {
        val offered = AtomicInteger()
        val queued = AtomicInteger()
        val discardedAudio: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
        val earlyDiscarded = AtomicInteger()
        val awaited = AtomicInteger()
        override fun earlyStartAwaited() {
            awaited.incrementAndGet()
        }

        override fun earlyStartOffered(queued: Boolean) {
            offered.incrementAndGet()
            if (queued) this.queued.incrementAndGet()
        }

        override fun utteranceDiscarded(samples: FloatArray) {
            discardedAudio += samples
        }

        override fun earlyStartDiscarded() {
            earlyDiscarded.incrementAndGet()
        }
    }

    @Test
    fun offByDefaultAndThenTheRecogniserGetsTheMicrophonesAudioToTheEnd() = runBlocking {
        assertNull(CzechSpeechConfig().earlyStart)
        val native = FakeNative().apply { keepCopies = true }
        val hooks = Counting()
        val engine = fakeEngine(native, hooks) { ArrayAudioSource(one()) }
        collect(engine.listen())
        assertEquals(1, native.transcribeCalls.get())
        assertEquals(0, hooks.offered.get())
        val audio = native.copiesPassed.single()
        assertEquals(utteranceFrames * FRAME_SAMPLES, audio.size)
        assertTrue("no zeros where the pause was recorded", audio.drop(audibleFrames * FRAME_SAMPLES).none { it == 0f })
        assertEquals(0, engine.earlyStartsUsed)
    }

    @Test
    fun theRecognitionStartsInThePauseAndGivesTheTranscriptWhenTheUtteranceEndsThere() {
        val native = FakeNative().apply { keepCopies = true }
        val hooks = Counting()
        // 5 ms a frame, so the pause takes time; the recognition is held until the utterance has ended.
        val ended = CountDownLatch(1)
        native.onTranscribe = { check(ended.await(10, TimeUnit.SECONDS)) { "the utterance never ended" } }
        val source = ArrayAudioSource(one(), sleepPerFrameMs = 5, onFrame = { n ->
            if (n == earlyAt + 1) {
                waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
                check(!seen { it is SpeechEvent.SpeechEnded }) { "the utterance has ended already" }
            }
            if (n == endAt) ended.countDown()
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        val logs = withLogs { collect(engine.listen()) }

        val transcript = transcripts.single()
        assertEquals(
            listOf(SpeechEvent.SpeechStarted(startMs), SpeechEvent.SpeechEnded(endMs), SpeechEvent.Transcript(transcript)),
            events.toList(),
        )
        assertEquals(Transcript("dobrý den", startMs, endMs, transcript.processingMs, "cs"), transcript)
        assertEquals("one recognition: the early one is the final one", 1, native.transcribeCalls.get())
        assertEquals(1, native.gateCalls.get())
        assertEquals("the call of the final text, with the temperature fallback", 1, native.greedyCalls.get())
        assertEquals(0, native.previewCalls.get())
        assertTrue("never aborted", native.aborts.isEmpty())
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertEquals(1, hooks.queued.get())
        assertEquals(1, engine.earlyStartsUsed)
        assertEquals(0, engine.earlyStartsDiscarded)
        // The pause went on for 9 more frames of 5 ms after the recognition had started.
        assertTrue("saved ${engine.earlyStartSavedMs} ms", engine.earlyStartSavedMs in 30..5_000)
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)

        val line = logs.single { it.startsWith("utterance: ") }
        assertTrue(line, Regex(""", transcript kept, early start used, \d+ ms saved$""").containsMatchIn(line))
        assertTrue(
            logs.joinToString("\n"),
            logs.any { Regex("""^early start: 1 used, \d+ ms saved, 0 too late; .* 0 discarded, .* 0 ms of recogniser time$""").matches(it) },
        )
        assertTrue("never the text", logs.none { "dobrý" in it })
    }

    @Test
    fun theRecogniserGetsTheAudioUpToTheEarlyStartAndZerosWhateverTheTiming() {
        // The early recognition is used (a), or cannot start before the utterance has ended because the model is
        // still loading (b): the recogniser gets the same samples. Without early start (c) it gets the microphone's
        // own pause instead of the zeros: the one difference, and the same length.
        fun run(config: CzechSpeechConfig, loadHeldUntilTheEnd: Boolean): FloatArray {
            events.clear()
            val native = FakeNative().apply { keepCopies = true }
            val release = CountDownLatch(1)
            if (loadHeldUntilTheEnd) native.holdInit = release
            val source = ArrayAudioSource(one(), onFrame = { n ->
                if (n == earlyAt + 1 && !loadHeldUntilTheEnd && config.earlyStart != null) {
                    waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
                }
                if (n == endAt + 2) release.countDown()
            })
            val engine = fakeEngine(native, config = config) { source }
            runBlocking { collect(engine.listen()) }
            assertEquals("dobrý den", transcripts.single().text)
            assertEquals(1, native.transcribeCalls.get())
            assertAllZero(native.passedToWhisper)
            return native.copiesPassed.single()
        }

        val used = run(on, loadHeldUntilTheEnd = false)
        val late = run(on, loadHeldUntilTheEnd = true)
        val without = run(CzechSpeechConfig(), loadHeldUntilTheEnd = false)
        assertTrue("the same samples whether the early recognition ran early or not", used.contentEquals(late))
        assertEquals(utteranceFrames * FRAME_SAMPLES, used.size)
        assertEquals("as long as without early start", without.size, used.size)
        val audible = audibleFrames * FRAME_SAMPLES
        assertTrue("the same audio up to the early start", used.copyOf(audible).contentEquals(without.copyOf(audible)))
        assertTrue("then zeros", used.drop(audible).all { it == 0f })
        assertTrue("where the microphone's pause is without early start", without.drop(audible).none { it == 0f })
    }

    @Test
    fun whenTheSpeechGoesOnTheRecognitionIsAbortedAndNothingOfItReachesTheApp() {
        val native = FakeNative().apply {
            keepCopies = true
            abortableCalls = setOf(0)           // the early recognition runs until it is aborted
        }
        val hooks = Counting()
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        val logs = withLogs { collect(engine.listen()) }

        val transcript = transcripts.single()
        assertEquals(
            listOf(SpeechEvent.SpeechStarted(startMs), SpeechEvent.SpeechEnded(pausedEndMs), SpeechEvent.Transcript(transcript)),
            events.toList(),
        )
        assertEquals("dobrý den", transcript.text)
        assertEquals("the dropped recognition, then the utterance's", 2, native.transcribeCalls.get())
        assertEquals("one abort, of the dropped one", 1, native.aborts.toSet().size)
        assertEquals(1, engine.abortedRecognitions)
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertEquals(listOf(utteranceFrames, pausedFrames).map { it * FRAME_SAMPLES }, native.copiesPassed.map { it.size })
        assertEquals(1, engine.earlyStartsDiscarded)
        assertEquals(1, hooks.earlyDiscarded.get())
        assertEquals("the pause that ended the utterance was started early too", 1, engine.earlyStartsUsed + engine.earlyStartsTooLate)
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)

        assertTrue(
            logs.joinToString("\n"),
            logs.any { Regex("""^utterance: \d+ ms of audio, recogniser cancelled after \d+ ms, .*, early start discarded(, stopped \d+ ms after the abort)?$""").matches(it) },
        )
        assertTrue(logs.any { Regex("""^early start: \d used, \d+ ms saved, \d too late; .* 1 discarded, .* \d+ ms of recogniser time$""").matches(it) })
        assertTrue("no error for a dropped recognition", logs.none { it.startsWith("utterance failed") })
    }

    @Test
    fun aRecognitionThatFinishedBeforeTheSpeechWentOnIsDroppedWithItsText() {
        // The early recognition returns text at once; then the speech goes on. That text never leaves the engine.
        val native = FakeNative().apply { texts += "nikdy" }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            // its own log line has been written, without a decision in it
            if (n == earlyAt + 1) waitOnCapture("the early recognition's end") { logged { "not decided yet" in it } }
        })
        val engine = fakeEngine(native, config = on) { source }
        val logs = withLogs { collect(engine.listen()) }

        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(3, events.size)
        assertTrue(events.none { it.toString().contains("nikdy") })
        assertTrue(logs.none { "nikdy" in it || "dobrý" in it })
        assertEquals(2, native.transcribeCalls.get())
        assertEquals(1, engine.earlyStartsDiscarded)
        assertTrue("the text it returned is zeroed", native.returned.all { bytes -> bytes.all { it == 0.toByte() } })
        assertTrue(
            logs.joinToString("\n"),
            logs.any { Regex("""^utterance: early start discarded, \d+ ms of recogniser time$""").matches(it) },
        )
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aFailedEarlyRecognitionIsAnErrorOnlyWhenTheUtteranceEndsThere() {
        // dropped: no error, the utterance is recognised at its end
        val dropped = FakeNative().apply { failNext = 1 }
        var source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the failed recognition") { dropped.transcribeCalls.get() == 1 && dropped.failNext == 0 }
        })
        runBlocking { collect(fakeEngine(dropped, config = on) { source }.listen()) }
        assertTrue(events.none { it is SpeechEvent.Error })
        assertEquals("dobrý den", transcripts.single().text)

        // used: the error of that utterance, as without early start
        events.clear()
        val used = FakeNative().apply { failNext = 1 }
        source = ArrayAudioSource(one())
        runBlocking { collect(fakeEngine(used, config = on) { source }.listen()) }
        assertEquals(1, events.count { it is SpeechEvent.Error })
        assertTrue(transcripts.isEmpty())
    }

    @Test
    fun retractedBeforeTheRecogniserWasFreeItNeverRunsAndIsZeroedAtOnce() {
        // The model is still loading when the pause is handed out and when the speech goes on.
        val native = FakeNative().apply { keepCopies = true }
        val release = CountDownLatch(1)
        native.holdInit = release
        val hooks = Counting()
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 62) {          // the speech went on with frame 59
                check(hooks.discardedAudio.size == 1) { "the retracted utterance was not dropped at once" }
                check(hooks.discardedAudio[0].all { it == 0f }) { "not zeroed" }
                release.countDown()
            }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        runBlocking { collect(engine.listen()) }
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals("only the utterance itself was recognised", 1, native.transcribeCalls.get())
        assertEquals(pausedFrames * FRAME_SAMPLES, native.copiesPassed.single().size)
        assertTrue(native.aborts.isEmpty())
        assertEquals("nothing was computed for nothing", 0, engine.earlyStartsDiscarded)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun anEarlyRecognitionWithoutWordsGivesNoTranscript() = runBlocking {
        val native = FakeNative().apply { texts += " ... " }
        val engine = fakeEngine(native, config = on) { ArrayAudioSource(one()) }
        collect(engine.listen())
        assertTrue(transcripts.isEmpty())
        assertEquals(2, events.count { it !is SpeechEvent.UtteranceNotWritten })
        assertEquals("clear speech without a word is told to the app (an early recognition is not repeated)", 1, events.count { it is SpeechEvent.UtteranceNotWritten })
        assertEquals(1, native.transcribeCalls.get())
        assertEquals(1, engine.earlyStartsUsed + engine.earlyStartsTooLate)
    }

    @Test
    fun stopInThePauseUsesTheEarlyRecognition() {
        for (stopFrame in listOf(40, earlyAt + 2)) {       // during the speech; in the pause already handed out
            events.clear()
            val native = FakeNative().apply { keepCopies = true }
            lateinit var engine: SpeechEngine
            val source = ArrayAudioSource(
                Signals.concat(quiet(0.5, 1), Signals.tone(1.0)),
                tail = { 0.001f },                          // a pause that never ends by itself
                onFrame = { n -> if (n == stopFrame) engine.stop() },
            )
            engine = fakeEngine(native, config = on) { source }
            runBlocking { collect(engine.listen()) }
            assertEquals("stop() at frame $stopFrame", "dobrý den", transcripts.single().text)
            assertEquals(1, native.transcribeCalls.get())
            assertTrue(native.aborts.isEmpty())
            assertEquals(1, engine.earlyStartsUsed + engine.earlyStartsTooLate)
            // as if silence had followed: the utterance handed out early, zeros included
            val audio = native.copiesPassed.single()
            assertEquals(utteranceFrames * FRAME_SAMPLES, audio.size)
            assertTrue(audio.drop(audibleFrames * FRAME_SAMPLES).all { it == 0f })
            assertAllZero(native.passedToWhisper)
            assertFalse(engine.isModelLoaded)
        }
    }

    @Test
    fun cancellingDuringAnEarlyRecognitionAbortsItAndZeroesTheAudio() = runBlocking {
        val native = FakeNative().apply { blockUntilAborted = true }
        val running = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val source = ArrayAudioSource(one(), onFrame = { n ->
            if (n == earlyAt + 1) {
                waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
                running.countDown()
                check(cancelled.await(10, TimeUnit.SECONDS)) { "never cancelled" }
            }
        })
        val engine = fakeEngine(native, config = on) { source }
        val job = launch(Dispatchers.Default) { engine.listen().collect { events += it } }
        assertTrue(running.await(10, TimeUnit.SECONDS))
        job.cancel()
        cancelled.countDown()
        job.joinWithin(10_000, "the cancelled session")
        engine.awaitWhisperThread()
        assertTrue(transcripts.isEmpty())
        assertEquals(1, native.aborts.toSet().size)
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun closeDuringAnEarlyRecognitionEndsTheSessionAndFreesTheModel() = runBlocking {
        val native = FakeNative().apply { blockUntilAborted = true }
        val running = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val source = ArrayAudioSource(one(), onFrame = { n ->
            if (n == earlyAt + 1) {
                waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
                running.countDown()
                check(closed.await(10, TimeUnit.SECONDS)) { "never closed" }
            }
        })
        val engine = fakeEngine(native, config = on) { source }
        val job = launch(Dispatchers.Default) { runCatching { engine.listen().collect { events += it } } }
        assertTrue(running.await(10, TimeUnit.SECONDS))
        engine.close()
        closed.countDown()
        job.joinWithin(10_000, "the closed session")
        engine.awaitClosed()
        assertTrue(transcripts.isEmpty())
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)
    }

    // Sentence joining: the texts and the language model's answer of SentenceJoiningEngineTest.
    private val a = "Jasně, ale můžeš nějak."
    private val b = "s tebou spojit."
    private val joined = SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, b, "S tebou spojit.")
    private val tokenizer by lazy { Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm)) }

    private fun FakeNative.joinsWithSmallLetter() {
        val texts = SentenceBoundary.languageModelTexts(a, b)
        val score = mapOf(
            SentenceBoundary.Choice.JOIN to -40.0, SentenceBoundary.Choice.JOIN_KEEP to -50.0,
            SentenceBoundary.Choice.SPLIT to -52.6, SentenceBoundary.Choice.COMMA to -47.0,
        )
        val bySequence = texts.entries.associate { (choice, text) ->
            (intArrayOf(tokenizer.startToken) + tokenizer.encode(text)).toList() to score.getValue(choice)
        }
        lmScoreOf = { bySequence[it.toList()] ?: -100.0 }
    }

    @Test
    fun sentenceJoiningDecidesOnlyAboutUtterancesAndADroppedRecognitionNeverReachesIt() {
        // Two utterances; the second has a pause of 0.4 s inside: speech in frames 78..109 and 121..153, the pause
        // handed out early with frame 117 and retracted with 121, the last pause handed out with frame 161, the end
        // with frame 172. Recognitions: 0 the first utterance, 1 dropped, 2 the second utterance.
        val audio = Signals.concat(
            quiet(0.5, 1), Signals.tone(1.0), quiet(1.0, 2), Signals.tone(1.0), quiet(0.4, 3), Signals.tone(1.0), quiet(1.5, 4),
        )
        val joining = SentenceJoining(TestEnv.tinyLm)

        events.clear()
        val plain = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        runBlocking {
            collect(fakeEngine(plain, config = CzechSpeechConfig(sentenceJoining = joining)) { ArrayAudioSource(audio) }.listen())
        }
        val expected = transcripts.map { it.copy(processingMs = 0) }
        val expectedSpeech = events.filter { it !is SpeechEvent.Transcript }
        assertEquals(joined, transcripts[1].sentenceJoin)

        // The second utterance's recognition ends before the capture thread confirms it (the sentence is decided in a
        // whisper task of its own), or runs until after (decided in the same task).
        for (endsBeforeTheConfirmation in listOf(true, false)) {
            events.clear()
            val firstConfirmed = CountDownLatch(1)
            val confirmed = CountDownLatch(1)
            val native = FakeNative().apply {
                texts += listOf(a, b)
                abortableCalls = setOf(1)
                joinsWithSmallLetter()
                if (!endsBeforeTheConfirmation) {
                    onTranscribe = {
                        if (it == 0) check(firstConfirmed.await(10, TimeUnit.SECONDS)) { "never confirmed" }
                        if (it == 2) check(confirmed.await(10, TimeUnit.SECONDS)) { "never confirmed" }
                    }
                }
            }
            val hooks = Counting()
            val source = ArrayAudioSource(audio, onFrame = { n ->
                if (n == endAt) firstConfirmed.countDown()
                // the first utterance is done with, so the queue is free for the next early start
                if (n == endAt + 1) waitOnCapture("the first transcript") { seen { it is SpeechEvent.Transcript } }
                if (n == 120) waitOnCapture("the recognition to drop") { native.transcribeCalls.get() == 2 }
                if (n == 164 && endsBeforeTheConfirmation) {
                    // the first utterance, the dropped one, and now the second: its recognition has ended undecided
                    waitOnCapture("the second utterance's recognition") { hooks.awaited.get() == 3 }
                }
                if (n == 174) confirmed.countDown()
            })
            val engine = fakeEngine(native, hooks, config = on.copy(sentenceJoining = joining)) { source }
            val logs = withLogs { collect(engine.listen()) }
            val what = "recognition ended before the confirmation: $endsBeforeTheConfirmation"
            assertEquals(what, expected, transcripts.map { it.copy(processingMs = 0) })
            assertEquals(what, expectedSpeech, events.filter { it !is SpeechEvent.Transcript })
            assertEquals(what, joined, transcripts[1].sentenceJoin)
            assertEquals(
                "$what: the decision has a log line of its own",
                endsBeforeTheConfirmation,
                logs.any { Regex("""^utterance: early start used, \d+ ms saved$""").matches(it) },
            )
            assertEquals(3, native.transcribeCalls.get())
            assertEquals("the boundary was scored once", 1, native.lmScoreCalls.size)
            assertEquals(2, engine.earlyStartsUsed + engine.earlyStartsTooLate)
            assertEquals(1, engine.earlyStartsDiscarded)
            assertEquals(0, native.liveLanguageModels)
        }
    }

    @Test
    fun aPreviewRunningWhenThePauseIsHandedOutIsAbortedSoThatTheFinalRecognitionStarts() = runBlocking {
        // 6 s of speech (frames 15..203): its preview starts after frame 172 and runs until aborted. The pause is
        // handed out early with frame 211 and ends the utterance with frame 222: the final recognition has started
        // before that, so the preview was aborted at the early start, not at the end.
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val audio = Signals.concat(quiet(0.5, 1), Signals.tone(6.0), quiet(1.5, 2))
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 176) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
            if (n == 215) {
                waitOnCapture("the final recognition, started early") { native.transcribeCalls.get() == 2 }
                check(!seen { it is SpeechEvent.SpeechEnded }) { "the utterance has ended already" }
            }
        })
        val engine = fakeEngine(native, config = on) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue(events.none { it is SpeechPreview })
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals("the preview, then the final recognition", 2, native.transcribeCalls.get())
        assertEquals(1, native.previewCalls.get())
        assertEquals(1, native.greedyCalls.get())
        assertEquals(1, engine.abortedPreviews)
        assertEquals(1, engine.earlyStartsUsed)
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun previewsGoOnAfterADroppedEarlyRecognition() = runBlocking {
        // 4 s of speech, a pause of 0.4 s, 3 s of speech: the pause is handed out and retracted (without
        // PreviewConfig.fromPauses its recognition is dropped); the timed preview of the utterance still comes, 5 s
        // after that pause began, and the transcript replaces it.
        val native = FakeNative().apply {
            abortableCalls = setOf(0)
            texts += listOf("dobrý", "dobrý den, jak se máte")
        }
        val audio = Signals.concat(quiet(0.5, 1), Signals.tone(4.0), quiet(0.4, 2), Signals.tone(3.0), quiet(1.5, 3))
        val requested = AtomicInteger()
        val hooks = object : SpeechEngine.Hooks {
            override fun previewRequested(samples: FloatArray) {
                requested.incrementAndGet()
                waitOnCapture("the preview") { seen { it is SpeechPreview } }
            }
        }
        // speech to frame 140, the pause handed out with frame 148; the speech goes on with frame 153
        lateinit var engine: SpeechEngine
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 151) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            // the recogniser is free again before the preview is due (a microphone gives it 0.6 s for that)
            if (n == 157) waitOnCapture("the dropped recognition's end") { engine.earlyStartsDiscarded == 1 }
        })
        engine = fakeEngine(native, hooks, config = on) { source }
        collect(engine.listenWithPreviews(PreviewConfig(fromPauses = false)))
        assertEquals(listOf(SpeechPreview(startMs, "dobrý")), events.filterIsInstance<SpeechPreview>())
        assertEquals("dobrý den, jak se máte", transcripts.single().text)
        assertEquals(1, requested.get())
        assertEquals("dropped, preview, final", 3, native.transcribeCalls.get())
        assertEquals(1, engine.earlyStartsDiscarded)
        assertEquals(1, native.maxConcurrentRecognitions.get())
    }

    /** Pixel 9a (Tensor G4) under the floating button: cpus 0-6 allowed, three threads pinned to the big cores. */
    private val tensorG4 = CpuTopology(
        mapOf(0 to 191L, 1 to 191L, 2 to 191L, 3 to 191L, 4 to 741L, 5 to 741L, 6 to 741L, 7 to 1024L),
        "cpu_capacity",
    )

    @Test
    fun aDroppedRecognitionLeavesTheThreadUnpinnedAndNeverPausesPinning() {
        val native = FakeNative().apply {
            cpuset = 0x7f
            abortableCalls = setOf(0)
            // cores that look idle when the dropped recognition ends must not be taken for cores that cannot run
            isolated = 0x40
        }
        val guard = PinGuard()
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 62) native.isolated = 0        // the speech went on with frame 59
        })
        val engine = fakeEngine(native, config = on, threads = 4, cpus = tensorG4, pinGuard = guard) { source }
        runBlocking { collect(engine.listen()) }
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(listOf("greedy/3/0x70", "greedy/3/0x70"), native.computes.map { it.toString() })
        assertEquals(listOf("set 0x70", "unpin", "set 0x70", "unpin"), native.cpuCallNames)
        assertEquals("unpinned afterwards", FakeNative.ALL_CPUS, native.request)
        assertFalse("an aborted recognition says nothing about the cores", guard.paused)
    }

    @Test
    fun whilePinningIsPausedTheEarlyRecognitionRunsAsAnyOther() {
        var now = 0L
        val guard = PinGuard { now }.apply { pause() }
        val native = FakeNative().apply { cpuset = 0x7f }
        val engine = fakeEngine(native, config = on, threads = 4, cpus = tensorG4, pinGuard = guard) { ArrayAudioSource(one()) }
        runBlocking { collect(engine.listen()) }
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(listOf("greedy/4/0x7f"), native.computes.map { it.toString() })
        assertTrue("nothing pinned", native.cpuCalls.isEmpty())
        assertEquals(1, engine.earlyStartsUsed + engine.earlyStartsTooLate)
        now += PinGuard.PAUSE_MS
        assertFalse(guard.paused)
    }

    @Test
    fun aBacklogGetsNoEarlyStartAndLosesNothing() {
        // The first recognition is held while a second utterance comes and waits; the third utterance's pause finds
        // the queue occupied, so it is not started early, and all three are transcribed in order.
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) { "never released" } }
        }
        val hooks = Counting()
        val utterance = Signals.concat(Signals.tone(1.0), quiet(1.0, 2))
        val audio = Signals.concat(quiet(0.5, 1), utterance, utterance, utterance, quiet(1.0, 3))
        // the first utterance is in its (held) recognition, so the queue is empty for the second one's early start
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the first recognition") { native.transcribeCalls.get() == 1 }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        runBlocking {
            val job = launch(Dispatchers.Default) { engine.listen().collect { events += it } }
            awaitCondition("the end of the audio") { source.closed }
            release.countDown()
            job.joinWithin(10_000, "the session")
        }
        assertEquals(3, transcripts.size)
        assertEquals(transcripts.sortedBy { it.startMs }, transcripts)
        assertEquals(3, native.transcribeCalls.get())
        assertEquals("every pause was handed out", 3, hooks.offered.get())
        assertEquals("the queue took the first two; the third waited behind the second", 2, hooks.queued.get())
        // Only the first one's recognition ran ahead of its end: the other two began after theirs.
        assertEquals(1, engine.earlyStartsUsed)
        assertEquals("not counted as used: nothing was gained", 2, engine.earlyStartsTooLate)
        assertTrue(events.none { it is SpeechEvent.Error })
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun withVariantSelectionADroppedRecognitionStopsBeforeTheLanguageModelsNextWindow() {
        // Beam search has returned and the language model scores the first of two windows when the speech goes on.
        // That one run cannot be interrupted; the second window is not scored any more, and nothing native is aborted.
        val window = listOf(
            Hyp(" Viděl jsem velký psy.", -2.0, 9, best = true, score = -0.22),
            Hyp(" Viděl jsem velké psy.", -2.3, 9, score = -0.25),
        )
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            hypotheses = listOf(window, window)
            holdLmScore = release
        }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == earlyAt + 1) check(native.lmScoreStarted.await(10, TimeUnit.SECONDS)) { "the language model never ran" }
            if (n == 62) {          // the speech went on with frame 59
                check(native.lmScoreCalls.size == 1) { "the language model went on" }
                release.countDown()
            }
        })
        val engine = fakeEngine(native, config = on.copy(variantSelection = VariantSelection(TestEnv.tinyLm))) { source }
        val logs = withLogs { collect(engine.listen()) }
        assertEquals(1, transcripts.size)
        assertTrue(transcripts.single().text.startsWith("Viděl jsem"))
        assertEquals(3, events.size)
        assertEquals(listOf(5, 5), native.beamSizes)
        assertEquals("one window of the dropped recognition, two of the utterance's", 3, native.lmScoreCalls.size)
        assertTrue("whisper.cpp had returned: nothing to abort", native.aborts.isEmpty())
        assertEquals(0, engine.aborts)
        assertEquals(1, engine.earlyStartsDiscarded)
        assertTrue(logs.joinToString("\n"), logs.any { it.startsWith("utterance: ") && "language model stopped by a cancellation" in it })
        assertTrue("never the text", logs.none { "Viděl" in it })
        assertEquals(0, native.liveLanguageModels)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun whenCaptureFailsAnUndecidedRecognitionIsAbortedAndTheSessionFailsCleanly() {
        // The VAD fails four frames after the pause was handed out, while its recognition runs.
        val native = FakeNative().apply {
            blockUntilAborted = true
            failVadProbsAt = earlyAt + 2
        }
        val hooks = Counting()
        val source = ArrayAudioSource(one(), onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        val outcome = runCatching { runBlocking { collect(engine.listen()) } }
        val error = outcome.exceptionOrNull()
        assertTrue("$error", error is CzechSpeechException && error.message == "Voice activity detection failed")
        runBlocking { engine.awaitWhisperThread() }
        assertTrue(transcripts.isEmpty())
        assertEquals("the recognition was aborted", 1, native.aborts.toSet().size)
        assertEquals(1, native.transcribeCalls.get())
        assertAllZero(native.passedToWhisper)
        assertTrue(source.closed)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
        assertEquals(native.vadInits.get(), native.vadFrees.get())
    }

    @Test
    fun releaseModelDuringAnUndecidedRecognitionWaitsForTheSession() {
        // keepModelLoaded: the release comes while the early recognition runs and its utterance has not ended. The
        // model stays until the session has ended (the fake fails any use of a freed context), then it is freed.
        val ended = CountDownLatch(1)
        val native = FakeNative().apply {
            onTranscribe = { check(ended.await(10, TimeUnit.SECONDS)) { "the utterance never ended" } }
        }
        lateinit var engine: SpeechEngine
        val source = ArrayAudioSource(one(), onFrame = { n ->
            if (n == earlyAt + 1) {
                waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
                engine.releaseModel()
                check(engine.isModelLoaded) { "freed under a running recognition" }
            }
            if (n == endAt) ended.countDown()
        })
        engine = fakeEngine(native, config = on, keepModelLoaded = true) { source }
        runBlocking { collect(engine.listen()) }
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(1, engine.earlyStartsUsed)
        assertFalse("released once the session had ended", engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
        assertEquals(1, native.whisperInits.get())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aKeptModelStaysLoadedAfterASessionWithADroppedRecognition() {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == earlyAt + 1) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
        })
        val engine = fakeEngine(native, config = on, keepModelLoaded = true) { source }
        runBlocking { collect(engine.listen()) }
        assertEquals("dobrý den", transcripts.single().text)
        assertTrue(engine.isModelLoaded)
        assertEquals(0, native.whisperFrees.get())
        engine.close()
        runBlocking { engine.awaitClosed() }
        assertFalse(engine.isModelLoaded)
    }
}
