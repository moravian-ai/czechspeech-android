package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.Transcript
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.EmptyCoroutineContext

/**
 * Previews in a [SpeechEngine.listenWithPreviews] session, with the fake native layer. Speech is a tone from 0.5 s:
 * its utterance starts at 480 ms (frame 15), with pre-roll from frame 5. With the default config its previews are
 * due after the frames in [dueFrames] (5, 10, 15, 20 and 25 s of speech). Where a test needs every preview to reach
 * the app, the capture thread waits for it at each request ([Paced]), so nothing depends on timing.
 */
class PreviewSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val startMs = 480L
    private val dueFrames = listOf(172, 329, 486, 643, 800)
    private val preRollFrom = 5

    private fun speech(seconds: Double) = Signals.concat(Signals.zeros(0.5), Signals.tone(seconds), Signals.zeros(1.5))

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val previews: List<SpeechPreview> get() = synchronized(events) { events.filterIsInstance<SpeechPreview>() }

    /** Blocks the capture thread until [condition] holds; fails the session after 10 s. */
    private fun waitOnCapture(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (!condition()) {
            if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for $what")
            Thread.sleep(1)
        }
    }

    /**
     * Counts preview requests and final recognitions; [waitForEach]: capture goes on only once each preview has
     * reached the app.
     */
    private inner class Paced(private val waitForEach: Boolean = true) : SpeechEngine.Hooks {
        val requests = AtomicInteger()
        val recognised = AtomicInteger()
        override fun previewRequested(samples: FloatArray) {
            val n = requests.incrementAndGet()
            if (waitForEach) waitOnCapture("preview $n") { previews.size >= n }
        }

        override fun utteranceRecognised() {
            recognised.incrementAndGet()
        }
    }

    private suspend fun collect(flow: Flow<ListenEvent>) =
        withContext(Dispatchers.Default) { flow.collect { events += it } }

    @Test
    fun listenWithoutPreviewsIsUnchangedAndRunsNoPreview() = runBlocking {
        // 26.5 s of speech, then 40 s that the cap cuts in two: with listen() every utterance is recognised once.
        for (seconds in listOf(26.5, 40.0)) {
            val native = FakeNative()
            val hooks = Paced(waitForEach = false)
            val engine = fakeEngine(native, hooks) { ArrayAudioSource(speech(seconds)) }
            val heard = engine.listen().toList()
            val utterances = if (seconds < 28) 1 else 2
            assertEquals(1, heard.count { it is SpeechEvent.SpeechStarted })
            assertEquals(utterances, heard.count { it is SpeechEvent.Transcript })
            assertEquals(utterances + 2, heard.size)
            assertEquals("one recognition per utterance", utterances, native.transcribeCalls.get())
            assertEquals(0, hooks.requests.get())
            assertEquals(0, engine.previewRecognitions)
        }
    }

    @Test
    fun listenWithPreviewsGivesTheSpeechEventsOfListenWithPreviewsBetweenThem() = runBlocking {
        val withoutPreviews = fakeEngine(FakeNative()) { ArrayAudioSource(speech(12.0)) }.listen().toList()
        val engine = fakeEngine(FakeNative(), Paced()) { ArrayAudioSource(speech(12.0)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        fun SpeechEvent.withoutTiming() =
            if (this is SpeechEvent.Transcript) SpeechEvent.Transcript(transcript.copy(processingMs = 0)) else this
        assertEquals(
            withoutPreviews.map { it.withoutTiming() },
            events.filterIsInstance<SpeechEvent>().map { it.withoutTiming() },
        )
        assertEquals(2, previews.size)
    }

    @Test
    fun previewsComeWhileSpeakingAndTheTranscriptOfTheSameUtteranceReplacesThem() = runBlocking {
        val native = FakeNative().apply { texts += listOf("dobrý", "dobrý den", "dobrý den, jak se máte") }
        val engine = fakeEngine(native, Paced()) { ArrayAudioSource(speech(12.0)) }
        val logs = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            collect(engine.listenWithPreviews(PreviewConfig()))
        } finally {
            CsLog.sink = sink
        }
        val transcript = (events.last() as SpeechEvent.Transcript).transcript
        assertEquals(
            listOf(
                SpeechEvent.SpeechStarted(startMs),
                SpeechPreview(startMs, "dobrý"),
                SpeechPreview(startMs, "dobrý den", stableLength = 5),     // "dobrý": both previews start with it
                SpeechEvent.SpeechEnded(transcript.endMs),
                SpeechEvent.Transcript(transcript),
            ),
            events.toList(),
        )
        assertEquals(Transcript("dobrý den, jak se máte", startMs, 12_512, transcript.processingMs, "cs"), transcript)
        // Each preview is recognised from the audio captured so far: pre-roll up to the frame that made it due.
        assertEquals(
            dueFrames.take(2).map { (it - preRollFrom) * FRAME_SAMPLES },
            native.passedToWhisper.take(2).map { it.size },
        )
        assertEquals(3, native.passedToWhisper.size)
        assertEquals("the speech check runs for the final text only", 1, native.gateCalls.get())
        assertEquals(2, engine.previewRecognitions)
        assertEquals("previews through the call without temperature fallback", 2, native.previewCalls.get())
        assertAllZero(native.passedToWhisper)
        // Never the text, in a log line or in toString().
        assertEquals(2, logs.count { it.startsWith("preview: ") })
        assertTrue(logs.joinToString("\n"), logs.none { Regex("dobrý|máte").containsMatchIn(it) })
        assertTrue(events.none { it.toString().contains("dobrý") })
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aPreviewRunningWhenItsUtteranceEndsIsAbortedAndNeverReachesTheApp() = runBlocking {
        // 6 s of speech: the preview starts at 5 s and waits for its abort, which the end of the speech brings.
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val source = ArrayAudioSource(speech(6.0), onFrame = { n ->
            if (n == dueFrames[0] + 2) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
        })
        val engine = fakeEngine(native) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue(events.none { it is SpeechPreview })
        assertEquals("dobrý den", events.filterIsInstance<SpeechEvent.Transcript>().single().transcript.text)
        assertEquals("the preview, then the final recognition", 2, native.transcribeCalls.get())
        assertEquals(1, native.aborts.toSet().size)
        assertEquals(1, engine.abortedPreviews)
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aPreviewFinishedAfterItsUtteranceEndedIsDropped() = runBlocking {
        // The preview runs past the end of the speech and returns text (whisper.cpp may finish before its next
        // check): its utterance has ended, so nothing reaches the app, and its text is zeroed.
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            texts += "pozdě"
            onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) }
        }
        val source = ArrayAudioSource(speech(6.0), onFrame = { n ->
            if (n == dueFrames[0] + 2) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
            if (n == 240) release.countDown()           // the speech ended at frame 203, its utterance at 222
        })
        val engine = fakeEngine(native) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue(events.none { it is SpeechPreview })
        assertEquals("dobrý den", events.filterIsInstance<SpeechEvent.Transcript>().single().transcript.text)
        assertTrue("the abort was signalled", native.aborts.isNotEmpty())
        assertEquals(1, engine.previewRecognitions)
        assertEquals("it ran to its end", 0, engine.abortedPreviews)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aPreviewThatFailsOrHasNoWordsShowsNothingAndTheSessionGoesOn() = runBlocking {
        for (fails in listOf(true, false)) {
            events.clear()
            val logs = Collections.synchronizedList(mutableListOf<String>())
            val native = FakeNative().apply {
                if (fails) failNext = 1 else texts += " ... "
            }
            val expected = if (fails) "preview failed: CzechSpeechException: Preview recognition" else "preview: "
            // Capture waits until the preview has run to its end, before the speech ends.
            val source = ArrayAudioSource(speech(6.0), onFrame = { n ->
                if (n == dueFrames[0] + 2) {
                    waitOnCapture("the preview") { synchronized(logs) { logs.any { it.startsWith(expected) } } }
                }
            })
            val engine = fakeEngine(native) { source }
            val sink = CsLog.sink
            CsLog.sink = { _, message -> logs += message }
            try {
                collect(engine.listenWithPreviews(PreviewConfig()))
            } finally {
                CsLog.sink = sink
            }
            assertTrue(events.none { it is SpeechPreview })
            assertEquals("dobrý den", events.filterIsInstance<SpeechEvent.Transcript>().single().transcript.text)
            assertTrue("no error event for a preview", events.none { it is SpeechEvent.Error })
            assertEquals(1, engine.previewRecognitions)
            assertAllZero(native.passedToWhisper)
        }
    }

    @Test
    fun noPreviewStartsWhileAFinalRecognitionWaitsOrRuns() = runBlocking {
        val release = CountDownLatch(1)
        val native = FakeNative().apply { onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) } }
        val hooks = Paced()
        // A short utterance (its final recognition is held), then 12 s of speech from frame 78 (2496 ms, pre-roll
        // from frame 68): its preview would be due at frame 235; at frame 330 nothing has run but the held
        // recognition. Capture then waits for that recognition to end, so the preview starts at the frame read next.
        val first = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))
        val audio = Signals.concat(first, Signals.tone(12.0), Signals.zeros(1.5))
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 330) {
                // The held recognition may not have reached whisper yet on a busy machine: wait for it, not for time.
                waitOnCapture("the first final recognition held") { native.transcribeCalls.get() >= 1 }
                check(native.transcribeCalls.get() == 1 && hooks.requests.get() == 0) { "a preview before the final" }
                release.countDown()
            }
            if (n == 331) waitOnCapture("the final recognition") { hooks.recognised.get() == 1 }
        })
        val engine = fakeEngine(native, hooks) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>()
        assertEquals(2, transcripts.size)
        val preview = previews.first()
        assertEquals(transcripts[1].transcript.startMs, preview.startMs)
        assertTrue("the earlier transcript first", events.indexOf(transcripts[0]) < events.indexOf(preview))
        assertEquals("the preview of frames 68 to 330", (331 - 68) * FRAME_SAMPLES, native.passedToWhisper[1].size)
    }

    @Test
    fun aPreviewIsWithdrawnWhenItsUtteranceGivesNoTranscript() = runBlocking {
        for (finalFails in listOf(false, true)) {
            events.clear()
            val native = FakeNative().apply {
                texts += listOf("něco", "")
                if (finalFails) onTranscribe = { if (it == 1) failNext = 1 }
            }
            val engine = fakeEngine(native, Paced(), config = CzechSpeechConfig(retryEmptyFinals = false)) { ArrayAudioSource(speech(6.0)) }
            collect(engine.listenWithPreviews(PreviewConfig()))
            assertTrue(events.none { it is SpeechEvent.Transcript })
            assertEquals(listOf(SpeechPreview(startMs, "něco"), SpeechPreview(startMs, "")), previews)
            assertEquals("withdrawn last", SpeechPreview(startMs, ""), events.last())
            if (finalFails) assertTrue(events.dropLast(1).last { it !is SpeechEvent.UtteranceNotWritten } is SpeechEvent.Error)    // clear speech that failed is told too
            assertAllZero(native.passedToWhisper)
        }
    }

    @Test
    fun stopEndsThePreviewsWithTheUtteranceInProgress() = runBlocking {
        lateinit var engine: SpeechEngine
        val native = FakeNative().apply { texts += "náhled" }
        // Speech that never pauses; stop() once the first preview has reached the app.
        val hooks = object : SpeechEngine.Hooks {
            override fun previewRequested(samples: FloatArray) {
                waitOnCapture("the first preview") { previews.isNotEmpty() }
                engine.stop()
            }
        }
        engine = fakeEngine(native, hooks) { ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0))) }
        collect(engine.listenWithPreviews(PreviewConfig()))         // completes normally
        assertEquals(
            listOf("SpeechStarted", "SpeechPreview", "SpeechEnded", "Transcript"),
            events.map { it::class.java.simpleName },
        )
        assertEquals(1, engine.previewRecognitions)
        assertFalse(engine.isListening)
        assertFalse(engine.isModelLoaded)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun noPreviewStartsAfterStopEvenWhenOneFallsDueWhileTheLastWordEnds() = runBlocking {
        // stop() two frames before the first preview is due, in speech that never pauses: capture goes on for 1 s
        // to finish the word, and a preview started then would only hold up the final text.
        lateinit var engine: SpeechEngine
        val native = FakeNative()
        val hooks = Paced(waitForEach = false)
        val source = ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0)), onFrame = { n ->
            if (n == dueFrames[0] - 2) engine.stop()
        })
        engine = fakeEngine(native, hooks) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue("capture went past the frame where it was due", source.framesRead > dueFrames[0])
        assertEquals("no preview after stop()", 0, hooks.requests.get())
        assertEquals("the final recognition only", 1, native.transcribeCalls.get())
        assertEquals(
            listOf("SpeechStarted", "SpeechEnded", "Transcript"),
            events.map { it::class.java.simpleName },
        )
    }

    @Test
    fun stopAbortsARunningPreviewAtOnceNotWhenTheUtteranceEnds() = runBlocking {
        lateinit var engine: SpeechEngine
        val native = FakeNative().apply { abortableCalls = setOf(0) }     // the preview waits for its abort
        val source = ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0)), onFrame = { n ->
            if (n == dueFrames[0] + 2) {
                check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
                engine.stop()
            }
            // The utterance ends up to 1 s after stop(); the abort must not wait for that.
            if (n == dueFrames[0] + 4) waitOnCapture("the abort after stop()") { native.aborts.isNotEmpty() }
        })
        engine = fakeEngine(native) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue(events.none { it is SpeechPreview })
        assertEquals("dobrý den", events.filterIsInstance<SpeechEvent.Transcript>().single().transcript.text)
        assertEquals(1, engine.abortedPreviews)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aPreviewNeverLoadsTheModelEvenWhenTheSessionsLoadFailed() = runBlocking {
        // The load is held until a preview waits behind it on the whisper thread, then fails: the session fails,
        // and the preview, which runs next, must not try to load the model a second time.
        val load = CountDownLatch(1)
        val native = FakeNative().apply {
            holdInit = load
            initFails = true
        }
        lateinit var engine: SpeechEngine
        val hooks = object : SpeechEngine.Hooks {
            override fun previewRequested(samples: FloatArray) {
                waitOnCapture("the preview queued behind the load") { engine.whisperTasksWaiting >= 1 }
                load.countDown()
            }
        }
        engine = fakeEngine(native, hooks) { ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0))) }
        val outcome = runCatching { collect(engine.listenWithPreviews(PreviewConfig())) }
        val error = outcome.exceptionOrNull()
        assertTrue("$outcome", error is CzechSpeechException && error.message!!.contains("Could not load the model"))
        engine.awaitWhisperThread()                                 // the preview behind the load has run
        assertEquals("loaded once, never by the preview", 1, native.whisperInits.get())
        assertEquals(0, engine.previewRecognitions)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun cancellingTheCollectionAbortsARunningPreviewAndZeroesItsAudio() = runBlocking {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val cancelling = CountDownLatch(1)
        val source = ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0)), onFrame = { n ->
            if (n == dueFrames[0] + 2) {
                check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
                check(cancelling.await(10, TimeUnit.SECONDS)) { "not cancelled" }
            }
        })
        val engine = fakeEngine(native) { source }
        val job = launch(Dispatchers.Default) { engine.listenWithPreviews(PreviewConfig()).collect { events += it } }
        assertTrue("the preview is running", native.transcribeStarted.await(10, TimeUnit.SECONDS))
        job.cancel()
        cancelling.countDown()
        job.joinWithin(2_000, "the cancelled collection")
        awaitCondition("the preview aborted") { native.aborts.isNotEmpty() }
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
        assertTrue(events.none { it is SpeechPreview || it is SpeechEvent.Transcript })
        assertEquals("only the preview reached the recogniser", 1, native.transcribeCalls.get())
        assertTrue(source.closed)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aPreviewRequestNobodyTookIsZeroedWhenTheSessionIsCancelled() = runBlocking {
        // The collector's only thread is blocked from the start of the session, so the preview runner never starts:
        // the request waits in its channel when the session is cancelled.
        val executor = Executors.newSingleThreadExecutor()
        val dispatcher = executor.asCoroutineDispatcher()
        val unblock = CountDownLatch(1)
        val requested = CountDownLatch(1)
        var audio = FloatArray(0)
        val native = FakeNative()
        val hooks = object : SpeechEngine.Hooks {
            override fun sessionRegistered() {
                dispatcher.dispatch(EmptyCoroutineContext, Runnable { check(unblock.await(10, TimeUnit.SECONDS)) })
            }

            override fun previewRequested(samples: FloatArray) {
                audio = samples
                requested.countDown()
            }
        }
        val endless = Signals.concat(Signals.zeros(0.5), Signals.tone(20.0))
        val engine = fakeEngine(native, hooks) { ArrayAudioSource(endless) }
        try {
            val job = launch(dispatcher) { engine.listenWithPreviews(PreviewConfig()).collect { } }
            try {
                assertTrue("a preview requested", requested.await(10, TimeUnit.SECONDS))
                assertTrue("not taken yet", audio.isNotEmpty() && audio.any { it != 0f })
                job.cancel()
            } finally {
                unblock.countDown()
            }
            job.joinWithin(5_000, "the cancelled collection")
            assertTrue("its audio is zeroed", audio.all { it == 0f })
            assertEquals("never recognised", 0, native.transcribeCalls.get())
            assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        } finally {
            dispatcher.close()
        }
    }

    @Test
    fun closeAbortsARunningPreviewAndAwaitClosedWaitsForIt() = runBlocking {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val closing = CountDownLatch(1)
        // Capture waits, with the preview running, until close() has been called.
        val source = ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0)), onFrame = { n ->
            if (n == dueFrames[0] + 2) {
                check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no preview" }
                check(closing.await(10, TimeUnit.SECONDS)) { "not closed" }
            }
        })
        val engine = fakeEngine(native) { source }
        val collected = async(Dispatchers.Default) {
            runCatching { engine.listenWithPreviews(PreviewConfig()).collect { events += it } }
        }
        assertTrue("the preview is running", native.transcribeStarted.await(10, TimeUnit.SECONDS))
        engine.close()
        closing.countDown()
        val outcome = collected.awaitWithin(2_000, "the collection closed during a preview")
        assertTrue("$outcome", outcome.exceptionOrNull() is CancellationException)
        engine.awaitClosed()
        assertTrue("the preview was aborted", native.aborts.isNotEmpty())
        assertEquals(1, engine.abortedPreviews)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
        assertTrue(source.closed)
        assertTrue(events.none { it is SpeechPreview })
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun previewsKeepAKeptModelAndReleaseModelWaitsForTheSessionWithItsPreview() = runBlocking {
        // Session 1 (preview, final text): the model stays loaded. Session 2: releaseModel() while its preview runs
        // frees nothing until the session ends; the fake fails any use of a freed context.
        val secondPreview = CountDownLatch(1)
        val goOn = CountDownLatch(1)
        val native = FakeNative().apply {
            onTranscribe = {
                if (it == 2) {
                    secondPreview.countDown()
                    check(goOn.await(10, TimeUnit.SECONDS))
                }
            }
        }
        val engine = fakeEngine(native, Paced(), keepModelLoaded = true) { ArrayAudioSource(speech(6.0)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue("kept", engine.isModelLoaded)
        assertEquals(1, engine.previewRecognitions)

        val second = async(Dispatchers.Default) { collect(engine.listenWithPreviews(PreviewConfig())) }
        assertTrue("the second session's preview runs", secondPreview.await(10, TimeUnit.SECONDS))
        engine.releaseModel()
        assertTrue("not freed under a running preview", engine.isModelLoaded)
        goOn.countDown()
        second.awaitWithin(10_000, "the second session")
        assertFalse("freed when the session ended", engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
        assertEquals(2, previews.size)
        assertEquals(2, events.count { it is SpeechEvent.Transcript })
    }

    @Test
    fun withVariantSelectionAPreviewIsStillOneGreedyPassWithoutBeamsOrTheLanguageModel() = runBlocking {
        // 12 s of speech: two previews, then the final text. The final text takes the beams and the language model
        // (the second variant wins by it); each preview is the greedy pass of whisperPreview and nothing else.
        val native = FakeNative().apply {
            texts += listOf("viděl jsem", "viděl jsem velký")
            hypotheses = listOf(
                listOf(
                    Hyp(" Viděl jsem velký psy.", -2.0, 9, best = true, score = -0.22),
                    Hyp(" Viděl jsem velké psy.", -2.3, 9, score = -0.25),
                ),
            )
        }
        val tokenizer = Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm))
        val good = intArrayOf(tokenizer.startToken) + tokenizer.encode("Viděl jsem velké psy.")
        native.lmScoreOf = { if (it.contentEquals(good)) -40.0 else -50.0 }
        val atEachRecognition = Collections.synchronizedList(mutableListOf<String>())
        native.onTranscribe = { index ->
            atEachRecognition += "$index: previews ${native.previewCalls.get()}, beams ${native.beamSizes.size}, " +
                "LM runs ${native.lmScoreCalls.size}, greedy ${native.greedyCalls.get()}"
        }
        val config = CzechSpeechConfig(variantSelection = VariantSelection(TestEnv.tinyLm))
        val engine = fakeEngine(native, Paced(), config = config) { ArrayAudioSource(speech(12.0)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertEquals(listOf(SpeechPreview(startMs, "viděl jsem"), SpeechPreview(startMs, "viděl jsem velký", 10)), previews)
        val transcript = (events.last() as SpeechEvent.Transcript).transcript
        assertEquals("Viděl jsem velké psy.", transcript.text)
        assertEquals("two previews through whisperPreview", 2, native.previewCalls.get())
        assertEquals("one beam search, the final text's", listOf(5), native.beamSizes)
        assertEquals("the language model ran once, for the final text", 1, native.lmScoreCalls.size)
        assertEquals("no greedy final text", 0, native.greedyCalls.get())
        assertEquals(
            listOf(
                "0: previews 1, beams 0, LM runs 0, greedy 0",
                "1: previews 2, beams 0, LM runs 0, greedy 0",
                "2: previews 2, beams 1, LM runs 0, greedy 0",
            ),
            atEachRecognition.toList(),
        )
        assertEquals("loaded once, with the model", 1, native.lmInits.get())
        assertEquals("and freed with it", 0, native.liveLanguageModels)
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
    }

    @Test
    fun theLanguageModelIsNeverFreedUnderARunningPreview() = runBlocking {
        // keepModelLoaded with variant selection: session 1 loads the model and the language model, which stay.
        // Session 2: releaseModel() while its preview runs frees neither until the session ends.
        val secondPreview = CountDownLatch(1)
        val goOn = CountDownLatch(1)
        val native = FakeNative().apply {
            onTranscribe = {
                if (it == 2) {
                    secondPreview.countDown()
                    check(goOn.await(10, TimeUnit.SECONDS))
                }
            }
        }
        val config = CzechSpeechConfig(variantSelection = VariantSelection(TestEnv.tinyLm))
        val engine = fakeEngine(native, Paced(), keepModelLoaded = true, config = config) {
            ArrayAudioSource(speech(6.0))
        }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertEquals("kept", 1, native.liveLanguageModels)

        val second = async(Dispatchers.Default) { collect(engine.listenWithPreviews(PreviewConfig())) }
        assertTrue("the second session's preview runs", secondPreview.await(10, TimeUnit.SECONDS))
        engine.releaseModel()
        assertEquals("not freed under a running preview", 1, native.liveLanguageModels)
        assertEquals(0, native.lmFrees.get())
        goOn.countDown()
        second.awaitWithin(10_000, "the second session")
        assertEquals("freed when the session ended", 0, native.liveLanguageModels)
        assertEquals(1, native.lmInits.get())
        assertEquals(1, native.lmFrees.get())
        assertEquals(listOf("whisperFree", "lmFree"), native.events.toList())
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
        assertEquals(2, native.previewCalls.get())
        assertEquals("only the final texts searched beams", listOf(5, 5), native.beamSizes)
    }

    @Test
    fun aPreviewNeverLoadsTheLanguageModelWhenTheSessionsLoadFailed() = runBlocking {
        // The language model loads with the model, before it, in the session's load. That load is held until a
        // preview waits behind it on the whisper thread, then fails: the session fails, and the preview, which runs
        // next, must not try to load either model.
        val load = CountDownLatch(1)
        val native = FakeNative().apply {
            holdLmInit = load
            failLmInit = true
        }
        lateinit var engine: SpeechEngine
        val hooks = object : SpeechEngine.Hooks {
            override fun previewRequested(samples: FloatArray) {
                waitOnCapture("the preview queued behind the load") { engine.whisperTasksWaiting >= 1 }
                load.countDown()
            }
        }
        val config = CzechSpeechConfig(variantSelection = VariantSelection(TestEnv.tinyLm))
        engine = fakeEngine(native, hooks, config = config) {
            ArrayAudioSource(Signals.concat(Signals.zeros(0.5), Signals.tone(20.0)))
        }
        val outcome = runCatching { collect(engine.listenWithPreviews(PreviewConfig())) }
        val error = outcome.exceptionOrNull()
        assertTrue("$outcome", error is CzechSpeechException && error.message!!.contains("language model"))
        engine.awaitWhisperThread()                                 // the preview behind the load has run
        assertEquals("one load attempt, the session's", 1, native.lmThreads.size)
        assertEquals(0, native.whisperInits.get())
        assertEquals(0, engine.previewRecognitions)
        assertEquals(0, native.previewCalls.get())
    }

    @Test
    fun previewsAreNeverJoined() = runBlocking {
        // A short utterance, then 6 s of speech after a pause of 1 s: one preview of the second (at 5 s of it), then
        // its final text. With sentence joining on, only the final text's boundary runs the language model, after
        // the preview; the preview shows the recogniser's text as it is.
        val first = "Jasně, ale můžeš nějak."
        val native = FakeNative().apply { texts += listOf(first, "s tebou", "s tebou spojit.") }
        val lmCallsAtPreview = AtomicInteger(-1)
        native.onTranscribe = { if (it == 1) lmCallsAtPreview.set(native.lmScoreCalls.size) }
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0), Signals.tone(6.0), Signals.zeros(1.5))
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm))
        val hooks = Paced()
        // Capture waits for the first final text before the second utterance's preview falls due (frame ~230).
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 150) waitOnCapture("the first final recognition") { hooks.recognised.get() == 1 }
        })
        val engine = fakeEngine(native, hooks, config = config) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
        assertEquals(listOf(first, "s tebou spojit."), transcripts.map { it.text })
        assertEquals(listOf(SpeechPreview(transcripts[1].startMs, "s tebou")), previews)
        assertEquals("no language model for the preview", 0, lmCallsAtPreview.get())
        assertEquals("one boundary, the final text's", 1, native.lmScoreCalls.size)
        assertEquals(1, native.previewCalls.get())
    }

    @Test
    fun withAFullQueueOnlyUtterancesThatGetAFinalRecognitionArePreviewed() = runBlocking {
        // Four short utterances while the first one's final recognition is held: the second waits in the queue (3 s
        // bound), the third and fourth are skipped. Then 8 s of speech from frame 328 (10496 ms). A preview needs
        // every final recognition done, so a skipped utterance was never previewed and has nothing to withdraw; the
        // long one is previewed once the two queued ones are recognised (capture waits for that at frame 485).
        val release = CountDownLatch(1)
        val native = FakeNative().apply { onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) } }
        val hooks = Paced()
        val short = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))
        val audio = Signals.concat(short, short, short, short, speech(8.0))
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 78) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "the first not in recognition" }
            if (n == 400) release.countDown()
            if (n == 485) waitOnCapture("the queued final recognitions") { hooks.recognised.get() == 2 }
        })
        val engine = fakeEngine(native, hooks, maxQueuedAudioMs = 3_000) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertEquals(2, engine.skippedUtterances)
        val skipped = events.filterIsInstance<SpeechEvent.Error>()
        assertEquals(2, skipped.size)
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
        assertEquals(3, transcripts.size)
        assertEquals(10_496L, transcripts.last().startMs)
        assertEquals("the long utterance's only, none withdrawn", listOf(SpeechPreview(10_496, "dobrý den")), previews)
        assertTrue(events.indexOf(previews.single()) < events.indexOfLast { it is SpeechEvent.Transcript })
    }

    @Test
    fun speechLongerThanTheCapGetsPreviewsPerUtteranceInOrder() = runBlocking {
        // 40 s of speech: cut at the cap (frame 880) into an utterance from 480 ms and one from 26176 ms (frame 818),
        // where the cut fell. The second one's previews wait for the final recognition of the first; capture waits
        // for it just before the second one's first preview is due (frame 975).
        val native = FakeNative()
        val hooks = Paced()
        val source = ArrayAudioSource(speech(40.0), onFrame = { n ->
            if (n == 975) waitOnCapture("the final recognition of the first") { hooks.recognised.get() == 1 }
        })
        val engine = fakeEngine(native, hooks) { source }
        collect(engine.listenWithPreviews(PreviewConfig()))
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
        assertEquals(2, transcripts.size)
        assertEquals(startMs, transcripts[0].startMs)
        assertEquals(transcripts[0].endMs, transcripts[1].startMs)
        val positions = events.indices
        val firstTranscript = positions.first { events[it] is SpeechEvent.Transcript }
        val firstOfSecond = positions.filter { (events[it] as? SpeechPreview)?.startMs == transcripts[1].startMs }
        val ofFirst = positions.filter { (events[it] as? SpeechPreview)?.startMs == startMs }
        assertEquals("previews at 5, 10, 15, 20 and 25 s of the first utterance", 5, ofFirst.size)
        assertEquals("at 5 and 10 s of the second", 2, firstOfSecond.size)
        assertEquals(26_176L, transcripts[1].startMs)
        assertEquals(
            "the second one's previews from its start (frame 818) up to frames 975 and 1132",
            listOf(975, 1132).map { (it - 818) * FRAME_SAMPLES },
            native.passedToWhisper.slice(6..7).map { it.size },     // after 5 previews and the first final text
        )
        assertEquals(ofFirst.size + firstOfSecond.size, previews.size)
        assertTrue("those of the first before its transcript", ofFirst.all { it < firstTranscript })
        assertTrue("those of the second after the first transcript", firstOfSecond.all { it > firstTranscript })
        val ended = events.indexOfFirst { it is SpeechEvent.SpeechEnded }
        assertTrue("and before its SpeechEnded", firstOfSecond.all { it < ended })
    }

    @Test
    fun theWorkOfPreviewsForALongUtterance() = runBlocking {
        // 26.5 s of speech, the longest that still ends by a pause before the 28 s cap (pre-roll and 600 ms of
        // silence count): every preview due reaches the app before capture goes on, as on a device that keeps up.
        val native = FakeNative()
        val engine = fakeEngine(native, Paced()) { ArrayAudioSource(speech(26.5)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        val sizes = native.passedToWhisper.map { it.size }
        val previewSizes = sizes.dropLast(1)
        assertEquals(dueFrames.map { (it - preRollFrom) * FRAME_SAMPLES }, previewSizes)
        val previewMs = previewSizes.sum() * 1000L / SAMPLE_RATE
        val finalMs = sizes.last() * 1000L / SAMPLE_RATE
        assertEquals(5, engine.previewRecognitions)
        assertEquals(previewMs, engine.previewAudioMs)
        println("previews of an utterance of $finalMs ms: ${previewSizes.size} runs over $previewMs ms of audio")
        assertTrue("about 2.8 times the audio of the final recognition", previewMs in 2 * finalMs..3 * finalMs)
    }

    @Test
    fun aPreviewThatWouldStillRunAtTheCapIsNotStartedJudgedByTheSpeedOfTheLastOne() = runBlocking {
        // A 6 s cap and a preview every second of 5.5 s of speech; each preview takes 700 ms, as all fall under the
        // 10.24 s encoder window. Previews at 1, 2, 3 and 4 s of speech (positions 1504 to 4576 ms); the one due at
        // 5 s (5600 ms) would end after the cap (at 6144 ms here), so it does not start, nor any after it.
        val native = FakeNative().apply { onTranscribe = { Thread.sleep(700) } }
        val config = CzechSpeechConfig(maxUtteranceMs = 6_000)
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(5.5), Signals.zeros(1.5))
        val engine = fakeEngine(native, Paced(), config = config) { ArrayAudioSource(audio) }
        collect(engine.listenWithPreviews(PreviewConfig(firstAfterMs = 1_000, intervalMs = 1_000)))
        assertEquals(4, engine.previewRecognitions)
        assertEquals(4, previews.size)
        assertEquals(
            "previews from the audio up to 1504, 2528, 3552 and 4576 ms",
            listOf(47, 79, 111, 143).map { (it - preRollFrom) * FRAME_SAMPLES },
            native.passedToWhisper.take(4).map { it.size },
        )
    }

    private fun assertAllZero(arrays: List<FloatArray>) {
        assertTrue("whisper was called", arrays.isNotEmpty())
        for (a in arrays) assertTrue("audio buffer zeroed after use", a.all { it == 0f })
    }
}
