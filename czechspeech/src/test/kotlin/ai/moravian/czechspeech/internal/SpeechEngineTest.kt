package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.io.File
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.sin

/**
 * Pipeline behaviour with the fake native layer ([FakeNative]): no model needed, so these always run. "Speech" is any
 * frame with a sample above 0.05 in magnitude.
 */
class SpeechEngineTest {
    /** Backstop: a test that blocks (a latch, a join without a timeout) fails instead of hanging the build. */
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private fun engine(
        native: FakeNative,
        hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE,
        maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS,
        source: () -> AudioSource,
    ) = fakeEngine(native, hooks, maxQueuedAudioMs, source = source)

    /** Records the utterances the engine drops without recognition. */
    private class Discarded : SpeechEngine.Hooks {
        val utterances: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
        override fun utteranceDiscarded(samples: FloatArray) {
            utterances += samples
        }
    }

    // 2.5 s, 78 frames: the endpointer emits the utterance (pre-roll, 1 s of speech, 600 ms of silence) about 66
    // frames in, so after utteranceFrames * k + 20 frames read, k utterances have been emitted.
    private val utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))
    private val utteranceFrames = utterance.size / FRAME_SAMPLES

    @Test
    fun listenEmitsStartedEndedTranscriptAndFreesTheModel() = runBlocking {
        val native = FakeNative()
        val source = ArrayAudioSource(utterance)
        val engine = engine(native) { source }
        val events = engine.listen().toList()

        assertEquals(3, events.size)
        assertTrue(events[0] is SpeechEvent.SpeechStarted)
        assertTrue(events[1] is SpeechEvent.SpeechEnded)
        val t = (events[2] as SpeechEvent.Transcript).transcript
        assertEquals("dobrý den", t.text)
        assertEquals(480L, t.startMs)              // the tone starts at 0.5 s, inside frame 15 (480..512 ms)
        assertTrue(t.endMs in 1_500L..1_540L)
        assertFalse("toString must not print the text", t.toString().contains("dobrý"))

        assertTrue(source.opened && source.closed)
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertFalse(engine.isListening)
        assertEquals("streaming VAD + gate VAD", 2, native.vadInits.get())
        assertEquals(native.vadInits.get(), native.vadFrees.get())
        assertEquals("streaming VAD reset before first use", 0, native.unresetStreamingCalls.get())
        assertEquals("reset at start and after the utterance", 2, native.vadResets.get())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun czechIsForcedAndTheTranscriptSaysSo() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native) { ArrayAudioSource(utterance) }
        val t = (engine.listen().toList().last() as SpeechEvent.Transcript).transcript
        assertEquals("cs", t.language)
        assertEquals(listOf("cs"), native.languagesPassed.toList())
        assertEquals("a forced language asks for no detection result", 0, native.whisperLanguageCalls.get())
        assertTrue("no detection pass for a single language", t.toString().contains("language=cs"))
    }

    @Test
    fun listenForcesTheRequestedLanguageOfTheModel() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native, languages = listOf("pl", "en")) { ArrayAudioSource(utterance) }
        val t = (engine.listen("en").toList().last() as SpeechEvent.Transcript).transcript
        assertEquals(listOf("en"), native.languagesPassed.toList())
        assertEquals("en", t.language)
        assertEquals("no extra native call for a forced language", 0, native.whisperLanguageCalls.get())
    }

    @Test
    fun aMissingOrUnlistedDetectedLanguageFailsTheRecognitionInsteadOfMakingOneUp() = runBlocking {
        for (detected in listOf(null, "", "de")) {
            val native = FakeNative().also { it.detectedLanguage = detected }
            val engine = fakeEngine(native, languages = listOf("pl", "en")) { ArrayAudioSource(utterance) }
            val events = engine.listen().toList()
            assertTrue("detected=$detected", events.none { it is SpeechEvent.Transcript })
            val error = events.filterIsInstance<SpeechEvent.Error>().single()
            assertTrue("detected=$detected: ${error.cause}", error.cause is CzechSpeechException)
            assertEquals("Recognition failed", error.cause.message)
            assertEquals(1, native.whisperLanguageCalls.get())
        }
    }

    @Test
    fun duplicateLanguagesCountAsOneAndAreForcedWithoutDetection() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native, languages = listOf("cs", "cs")) { ArrayAudioSource(utterance) }
        val t = (engine.listen().toList().last() as SpeechEvent.Transcript).transcript
        assertEquals(listOf("cs"), native.languagesPassed.toList())
        assertEquals("cs", t.language)
        assertEquals("no detection pass", 0, native.whisperLanguageCalls.get())
        val several = FakeNative().also { it.detectedLanguage = "en" }
        val engine2 = fakeEngine(several, languages = listOf("pl", "en", "pl")) { ArrayAudioSource(utterance) }
        engine2.listen().toList()
        assertEquals(listOf("pl,en"), several.languagesPassed.toList())
    }

    @Test
    fun languageCodesMustBeTwoOrThreeLowercaseLetters() = runBlocking {
        for (bad in listOf("czech", "CS", "c", "cs ")) {
            val e = failure<IllegalArgumentException> { fakeEngine(FakeNative(), languages = listOf(bad)) { error("unused") } }
            assertTrue(e.message!!, e.message!!.contains("\"$bad\""))
            val engine = fakeEngine(FakeNative(), languages = listOf("pl", "en")) { error("unused") }
            failure<IllegalArgumentException> { engine.listen(bad) }
            failure<IllegalArgumentException> { engine.transcribe(Signals.tone(1.0), bad) }
        }
        val engine = fakeEngine(FakeNative(), languages = listOf("cs", "yue")) { error("unused") }
        engine.listen("yue")
        engine.listen("cs")
        Unit
    }

    @Test
    fun blankLanguageCodesAreRejectedAndTheListIsCopied() {
        failure<IllegalArgumentException> { fakeEngine(FakeNative(), languages = listOf("pl", " ")) { error("unused") } }
        failure<IllegalArgumentException> { fakeEngine(FakeNative(), languages = listOf("")) { error("unused") } }
        val mutable = mutableListOf("pl", "en")
        val engine = fakeEngine(FakeNative(), languages = mutable) { error("unused") }
        mutable.clear()
        mutable += "de"
        failure<IllegalArgumentException> { engine.listen("de") }   // still ["pl", "en"]
        engine.listen("en")
        Unit
    }

    @Test
    fun severalLanguagesWithoutAChoiceAreDetectedAmongThemAndReported() = runBlocking {
        val native = FakeNative().also { it.detectedLanguage = "en" }
        val engine = fakeEngine(native, languages = listOf("pl", "en")) { ArrayAudioSource(utterance) }
        val t = (engine.listen().toList().last() as SpeechEvent.Transcript).transcript
        assertEquals(listOf("pl,en"), native.languagesPassed.toList())
        assertEquals("en", t.language)
    }

    @Test
    fun listenRejectsALanguageOutsideTheModelsList() {
        val native = FakeNative()
        val engine = fakeEngine(native, languages = listOf("pl", "en")) { ArrayAudioSource(utterance) }
        val e = failure<IllegalArgumentException> { runBlocking { engine.listen("de").toList() } }
        assertTrue(e.message!!, e.message!!.contains("de"))
        // eager: listen() itself throws, nothing is collected
        failure<IllegalArgumentException> { engine.listen("de") }
        assertEquals(0, native.whisperInits.get())
        assertFalse(engine.isListening)
    }

    @Test
    fun transcribeTakesALanguageAndRejectsOneOutsideTheList() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native, languages = listOf("pl", "en")) { error("no microphone in this test") }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0), "en"))
        assertEquals(listOf("en"), native.languagesPassed.toList())
        failure<IllegalArgumentException> { engine.transcribe(Signals.tone(1.0), "de") }
        assertEquals(1, native.whisperInits.get())
    }

    @Test
    fun anEmptyLanguageListIsRejected() {
        failure<IllegalArgumentException> { fakeEngine(FakeNative(), languages = emptyList()) { error("unused") } }
        Unit
    }

    @Test
    fun noSpeechMeansNoTranscriptAndTheRecogniserNeverRuns() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { ArrayAudioSource(Signals.zeros(10.0)) }
        val events = engine.listen().toList()
        assertTrue(events.isEmpty())
        assertTrue(native.passedToWhisper.isEmpty())
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun transcribeGatesBeforeLoadingTheModel() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { error("no microphone in this test") }
        assertEquals("", engine.transcribe(Signals.zeros(3.0)))
        assertEquals("model not loaded for silence", 0, native.whisperInits.get())
        assertEquals(1, native.gateCalls.get())
        assertEquals("", engine.transcribe(FloatArray(0)))
    }

    @Test
    fun transcribeLoadsAndFreesAndNeverTouchesTheCallersArray() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { error("no microphone in this test") }
        val samples = Signals.tone(1.0)
        val copy = samples.copyOf()
        assertEquals("dobrý den", engine.transcribe(samples))
        assertArrayEquals(copy, samples, 0f)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
        engine.transcribe(samples)
        assertEquals("loaded again for the second call", 2, native.whisperInits.get())
    }

    @Test
    fun transcribeRejectsNaNAndInfinityNamingTheSample() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { error("no microphone in this test") }
        for (bad in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)) {
            val samples = Signals.tone(1.0).also { it[1234] = bad }
            val e = failure<IllegalArgumentException> { engine.transcribe(samples) }
            assertTrue(e.message, e.message!!.contains("samples[1234] is $bad") && e.message!!.contains("finite"))
        }
        assertEquals("rejected before the speech gate", 0, native.gateCalls.get())
        assertEquals(0, native.whisperInits.get())
    }

    @Test
    fun transcribeRejectsUnscaled16BitPcmInsteadOfReturningNothing() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { error("no microphone in this test") }
        // int16 values passed as floats, the common mistake: the real VAD sees no speech in them and "" comes back.
        val pcm16 = FloatArray(SAMPLE_RATE) { Math.round(8_000 * sin(2 * PI * 220 * it / SAMPLE_RATE)).toFloat() }
        val e = failure<IllegalArgumentException> { engine.transcribe(pcm16) }
        assertTrue(e.message, e.message!!.contains("[-1, 1]") && e.message!!.contains("32768"))
        assertEquals("rejected before the speech gate", 0, native.gateCalls.get())

        // Float audio that runs a little hot is accepted, up to a magnitude of exactly 2.
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0, amplitude = 1.5f)))
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0).also { it[0] = -MAX_SAMPLE_MAGNITUDE }))
        val justOver = Math.nextUp(MAX_SAMPLE_MAGNITUDE)
        val over = failure<IllegalArgumentException> { engine.transcribe(Signals.tone(1.0).also { it[0] = justOver }) }
        assertTrue(over.message, over.message!!.contains("samples[0] is $justOver"))
    }

    @Test
    fun transcribeTakesAtMostTenMinutesOfAudio() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { error("no microphone in this test") }
        assertEquals(9_600_000, MAX_TRANSCRIBE_SAMPLES)
        val e = failure<IllegalArgumentException> { engine.transcribe(FloatArray(MAX_TRANSCRIBE_SAMPLES + 1)) }
        assertTrue(e.message, e.message!!.contains("at most 600 s"))
        assertEquals("rejected before the speech gate", 0, native.gateCalls.get())
        assertEquals("exactly the maximum is taken (silence: no speech)", "", engine.transcribe(FloatArray(MAX_TRANSCRIBE_SAMPLES)))
        assertEquals(1, native.gateCalls.get())
    }

    @Test
    fun resultsWithoutLettersOrDigitsAreDropped() = runBlocking {
        val native = FakeNative().apply { text = " ... , ..." }
        val engine = engine(native) { ArrayAudioSource(utterance) }
        assertEquals("", engine.transcribe(Signals.tone(1.0)))
        val events = engine.listen().toList()
        assertTrue(events.none { it is SpeechEvent.Transcript })
        assertEquals(2, events.size)                // started + ended only
        native.text = "91"
        assertEquals("91", engine.transcribe(Signals.tone(1.0)))
    }

    @Test
    fun anEmptyResultIsDroppedSilentlyAndTheUtteranceIsNotDecodedAgain() = runBlocking {
        // A model that learned to answer "no speech" with an empty transcript (suppress_blank is off): no Transcript
        // event, no error, and one pass of the recogniser per utterance (the SDK never retries on its own).
        for (empty in listOf("", " ", "\n")) {
            val native = FakeNative().apply { text = empty }
            val engine = engine(native) { ArrayAudioSource(utterance) }
            assertEquals("transcribe()", "", engine.transcribe(Signals.tone(1.0)))
            assertEquals(1, native.passedToWhisper.size)
            val events = engine.listen().toList()
            assertTrue("no Transcript or Error for '$empty': $events", events.none { it is SpeechEvent.Transcript || it is SpeechEvent.Error })
            assertEquals(2, events.size)                // started + ended only
            assertEquals("one pass per utterance", 2, native.passedToWhisper.size)
        }
    }

    @Test
    fun aRepetitionLoopIsRemovedBeforeTheTranscriptReachesTheAppAndTheLogHasCountsOnly() = runBlocking {
        val loop = "pojišťování vývozu se státní podporou v roce 2012 až se státní podporou v roce 2012 až se státní " +
            "podporou v roce 2012 až se státní podporou v roce 2012 až se st"
        val cleaned = "pojišťování vývozu se státní podporou v roce 2012 až"
        val native = FakeNative().apply { text = loop }
        val engine = engine(native) { ArrayAudioSource(utterance) }
        val logs = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            val events = engine.listen().toList()
            assertEquals(3, events.size)
            assertEquals(cleaned, (events[2] as SpeechEvent.Transcript).transcript.text)
            assertEquals(cleaned, engine.transcribe(Signals.tone(1.0)))
        } finally {
            CsLog.sink = sink
        }
        assertEquals(
            listOf(
                "utterance: repetition loop removed (7 words, 3 repeats, partial copy at the end)",
                "transcribe: repetition loop removed (7 words, 3 repeats, partial copy at the end)",
            ),
            logs.filter { it.contains("repetition loop removed") },
        )
        assertTrue(logs.joinToString("\n"), logs.none { Regex("pojišťování|státní|podporou").containsMatchIn(it) })
    }

    @Test
    fun transcribeDuringListenUsesTheSessionModel() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val firstTranscript = CompletableDeferred<Unit>()
        val job = launch(Dispatchers.Default) {
            engine.listen().collect { if (it is SpeechEvent.Transcript) firstTranscript.complete(Unit) }
        }
        withTimeout(10_000) { firstTranscript.await() }
        assertTrue(engine.isModelLoaded)
        assertEquals("dobrý den", engine.transcribe(Signals.tone(0.5)))
        assertTrue("the session still holds the model", engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
        job.cancel()
        job.join()
        // A cancelled session does not wait for the whisper thread: the model is freed right after, in the background.
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun cancellingAbortsARunningTranscriptionAndFreesTheModel() = runBlocking {
        val native = FakeNative().apply { blockUntilAborted = true }
        val source = ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1)
        val engine = engine(native) { source }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        assertTrue(native.transcribeStarted.await(10, TimeUnit.SECONDS))
        val t0 = System.nanoTime()
        job.cancel()
        withTimeout(5_000) { job.join() }
        val stopMs = (System.nanoTime() - t0) / 1_000_000
        assertEquals(1, native.aborts.size)
        assertTrue(source.closed)
        assertFalse(engine.isListening)
        // A cancelled session does not wait for the whisper thread: the model is freed right after, in the background.
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
        assertEquals(1, native.whisperFrees.get())
        assertAllZero(native.passedToWhisper)
        println("stopped ${stopMs} ms after cancel")
    }

    @Test
    fun cancellingDuringTheModelLoadCompletesAtOnceAndTheModelIsFreedWhenTheLoadEnds() = runBlocking {
        val load = CountDownLatch(1)
        val native = FakeNative().apply { holdInit = load }
        val engine = engine(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        try {
            assertTrue("the model is loading", native.initStarted.await(10, TimeUnit.SECONDS))
            job.cancel()
            job.joinWithin(2_000, "the collection cancelled during the model load")
            assertFalse(engine.isListening)
            assertEquals("a load cannot be interrupted, so nothing is freed yet", 0, native.whisperFrees.get())
        } finally {
            load.countDown()
        }
        assertTrue("the model is freed as soon as its load ends", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
        assertEquals(native.vadInits.get(), native.vadFrees.get())
    }

    @Test
    fun aSessionCancelledBeforeItsModelLoadStartedNeverLoadsTheModel() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative().apply { holdGate = gate }
        val source = ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1)
        val engine = engine(native) { source }
        // The whisper thread is busy: a transcribe() of silence is held in the speech gate.
        val busy = async(Dispatchers.Default) { engine.transcribe(Signals.zeros(1.0)) }
        try {
            assertTrue(native.gateStarted.await(10, TimeUnit.SECONDS))
            val job = launch(Dispatchers.Default) { engine.listen().collect { } }
            // The load is queued as the session starts, before capture: once a frame is read, it waits behind the gate.
            awaitCondition("the session capturing") { source.framesRead > 0 }
            job.cancel()
            job.joinWithin(2_000, "the collection cancelled before its model load started")
            assertFalse(engine.isListening)
        } finally {
            gate.countDown()
        }
        assertEquals("", busy.awaitWithin(10_000, "the transcribe() held in the gate"))
        // The whisper thread runs its tasks in order: when this call returns, the session's load task has run.
        assertEquals("", engine.transcribe(Signals.zeros(0.1)))
        assertEquals("a cancelled session never loads the model", 0, native.whisperInits.get())
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aTranscribeCancelledDuringTheSpeechGateNeverLoadsTheModel() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative().apply { holdGate = gate }
        val engine = engine(native) { error("no microphone in this test") }
        val call = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        assertTrue(native.gateStarted.await(10, TimeUnit.SECONDS))
        call.cancel()
        gate.countDown()                                         // the gate then finds speech
        call.joinWithin(2_000, "the cancelled transcribe()")
        assertEquals("", engine.transcribe(Signals.zeros(0.1)))  // the cancelled call has left the whisper thread
        assertEquals("cancelled before the recogniser: the model is never loaded", 0, native.whisperInits.get())
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun stopFinishesTheUtteranceInProgressAndCompletesTheFlow() = runBlocking {
        val native = FakeNative()
        // Speech that never pauses: without stop() the utterance would only end at the length cap.
        val source = ArrayAudioSource(Signals.zeros(0.5), tail = { i -> 0.3f * kotlin.math.sin(i * 0.0864f) }, sleepPerFrameMs = 1)
        val engine = engine(native) { source }
        val speechStarted = CompletableDeferred<Unit>()
        val events = Collections.synchronizedList(mutableListOf<SpeechEvent>())
        val collected = async(Dispatchers.Default) {
            try {
                engine.listen().collect { event ->
                    events += event
                    if (event is SpeechEvent.SpeechStarted) speechStarted.complete(Unit)
                }
                "completed"
            } catch (e: CancellationException) {
                "cancelled"
            }
        }
        withTimeout(10_000) { speechStarted.await() }   // speech confirmed and still going on
        engine.stop()
        assertEquals("the flow completes normally", "completed", withTimeout(10_000) { collected.await() })
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>()
        assertEquals("the utterance in progress is transcribed", 1, transcripts.size)
        assertEquals("dobrý den", transcripts[0].transcript.text)
        assertTrue(source.closed)
        assertFalse(engine.isListening)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
        assertTrue("nothing aborted", native.aborts.isEmpty())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun stopTranscribesUtterancesStillWaitingInTheQueue() = runBlocking {
        val hold = CountDownLatch(1)
        val native = FakeNative().apply { holdRecognition = hold }
        val tailFrames = AtomicInteger()
        // Two utterances, then silence forever: the second one waits in the queue while the first is held.
        val source = ArrayAudioSource(
            Signals.concat(utterance, utterance),
            tail = { i -> if (i % FRAME_SAMPLES == 0L) tailFrames.incrementAndGet(); 0f },
            sleepPerFrameMs = 1,
        )
        val engine = engine(native) { source }
        val collected = async(Dispatchers.Default) { engine.listen().toList() }
        assertTrue(native.transcribeStarted.await(10, TimeUnit.SECONDS))
        awaitCondition("the end of the second utterance") { tailFrames.get() >= 40 }
        engine.stop()
        hold.countDown()
        val events = withTimeout(10_000) { collected.await() }
        assertEquals(2, events.filterIsInstance<SpeechEvent.Transcript>().size)
        assertTrue(source.closed)
        assertFalse(engine.isListening)
        assertFalse(engine.isModelLoaded)
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun cancellingZeroesTheUtterancesStillWaitingInTheQueue() = runBlocking {
        val hold = CountDownLatch(1)
        val native = FakeNative().apply { holdRecognition = hold }
        val discarded = Discarded()
        val allQueued = CountDownLatch(1)
        // Three utterances, then silence: the first is held in recognition, the other two wait in the queue.
        val source = ArrayAudioSource(
            Signals.concat(utterance, utterance, utterance),
            tail = { 0f },
            sleepPerFrameMs = 1,
            onFrame = { if (it == 3 * utteranceFrames + 20) allQueued.countDown() },
        )
        val engine = engine(native, discarded) { source }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        assertTrue("the first utterance is being recognised", native.transcribeStarted.await(10, TimeUnit.SECONDS))
        assertTrue("the third utterance has been queued", allQueued.await(10, TimeUnit.SECONDS))
        job.cancel()
        hold.countDown()
        withTimeout(5_000) { job.join() }

        assertEquals("the two utterances waiting in the queue are discarded", 2, discarded.utterances.size)
        for (samples in discarded.utterances) {
            assertTrue("a whole utterance", samples.size > utteranceFrames / 2 * FRAME_SAMPLES)
            assertTrue("a queued utterance is zeroed when the session is cancelled", samples.all { it == 0f })
        }
        assertEquals("only the first utterance reached the recogniser", 1, native.passedToWhisper.size)
        assertAllZero(native.passedToWhisper)
        // A cancelled session does not wait for the whisper thread: the model is freed right after, in the background.
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
    }

    @Test
    fun whenRecognitionFallsBehindTheQueueIsBoundedAndSkippedSpeechIsReported() = runBlocking {
        val hold = CountDownLatch(1)
        val native = FakeNative().apply { holdRecognition = hold }
        val discarded = Discarded()
        // Four utterances of 61 frames (1952 ms) each; the queue may hold 3 s of audio. The first is in recognition
        // (held) before the others end, the second fits in the queue, the third and fourth do not.
        val source = ArrayAudioSource(
            Signals.concat(utterance, utterance, utterance, utterance),
            onFrame = { if (it == utteranceFrames) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) },
        )
        val engine = engine(native, discarded, maxQueuedAudioMs = 3_000) { source }
        val collected = async(Dispatchers.Default) { engine.listen().toList() }
        try {
            awaitCondition("two utterances skipped") { engine.skippedUtterances == 2 }
        } finally {
            hold.countDown()
        }
        val events = collected.awaitWithin(10_000, "the session")

        assertEquals(4, events.count { it is SpeechEvent.SpeechEnded })
        assertEquals("the utterance in recognition and the one queued", 2, events.count { it is SpeechEvent.Transcript })
        val skipped = events.filterIsInstance<SpeechEvent.Error>()
        assertEquals("each skipped utterance is reported", 2, skipped.size)
        for (error in skipped) {
            assertTrue(error.cause is CzechSpeechException)
            assertTrue(error.cause.message, error.cause.message!!.contains("falling behind") && error.cause.message!!.contains("skipped"))
        }
        assertTrue("the queue never held more than the bound", engine.peakQueuedAudioMs in 1..3_000)
        assertEquals("skipped audio is dropped, and zeroed", 2, discarded.utterances.size)
        for (samples in discarded.utterances) assertTrue(samples.isNotEmpty() && samples.all { it == 0f })
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aBacklogOverTheBoundIsSkippedExactlyAndEveryUtteranceIsAccounted() = runBlocking {
        // Paced, not timed: the first recognition is held until capture has ended, so the queue sees all 12
        // utterances (61 or 62 frames each, depending on how the 2.5 s period falls on 32 ms frames) with that one
        // in recognition. A 4 s bound takes two (under 4 s together, three would be over 5.8 s); nine are skipped.
        val hold = CountDownLatch(1)
        val native = FakeNative().apply { holdRecognition = hold }
        val source = ArrayAudioSource(
            Signals.concat(*Array(12) { utterance }),
            onFrame = { if (it == utteranceFrames) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) },
        )
        val engine = engine(native, maxQueuedAudioMs = 4_000) { source }
        val collected = async(Dispatchers.Default) { engine.listen().toList() }
        try {
            awaitCondition("the end of capture") { source.closed }
        } finally {
            hold.countDown()
        }
        val events = collected.awaitWithin(10_000, "the session")

        assertEquals(12, events.count { it is SpeechEvent.SpeechEnded })
        assertEquals("the one in recognition and the two that fit", 3, events.count { it is SpeechEvent.Transcript })
        assertEquals(9, events.filterIsInstance<SpeechEvent.Error>().count { it.cause.message!!.contains("skipped") })
        assertEquals(9, engine.skippedUtterances)
        val twoQueued = (native.passedToWhisper[1].size + native.passedToWhisper[2].size) * 1000L / SAMPLE_RATE
        assertEquals("the two queued utterances were the most that waited", twoQueued, engine.peakQueuedAudioMs)
        assertTrue("within the bound", engine.peakQueuedAudioMs <= 4_000)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aRecogniserSlowerThanTheAudioKeepsTheInvariantsWhateverTheTiming() = runBlocking {
        // A real backlog with a sleeping recogniser: how many utterances are skipped depends on the machine, so only
        // what holds for any timing is checked. The exact counts are in the paced test above.
        val native = FakeNative().apply { recognitionMs = 300 }
        val source = ArrayAudioSource(Signals.concat(*Array(12) { utterance }), sleepPerFrameMs = 1)
        val engine = engine(native, maxQueuedAudioMs = 4_000) { source }
        val events = engine.listen().toList()

        val ended = events.count { it is SpeechEvent.SpeechEnded }
        val transcripts = events.count { it is SpeechEvent.Transcript }
        val skipped = events.filterIsInstance<SpeechEvent.Error>().count { it.cause.message!!.contains("skipped") }
        assertEquals(12, ended)
        assertEquals("every utterance is transcribed or reported as skipped", ended, transcripts + skipped)
        assertEquals(skipped, engine.skippedUtterances)
        assertTrue("never more than the bound queued: ${engine.peakQueuedAudioMs} ms", engine.peakQueuedAudioMs <= 4_000)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun stopWithoutASessionDoesNotStopTheNextOne() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { ArrayAudioSource(utterance) }
        engine.stop()
        val events = engine.listen().toList()
        assertEquals(1, events.filterIsInstance<SpeechEvent.Transcript>().size)
    }

    // stop() during speech waits for the end of the word: a pause of 256 ms (8 frames), at most 1 s. In these tests
    // the source calls stop() while the capture thread reads frame number stopFrame, so the capture loop sees it at
    // exactly stopFrame * 32 = 1504 ms of audio. Speech (a tone) starts at frame 16; the pre-roll from frame 6.
    private val stopFrame = 47
    private val stopMs = stopFrame * FRAME_MS
    private val speechFrom = 16
    private val preRollFrom = 6

    @Test
    fun stopDuringSpeechWaitsForAPauseAndTheUtteranceKeepsTheSpeechAfterStop() = runBlocking {
        val native = FakeNative()
        // the speech goes on for 400 ms after stop(), then silence
        val audio = Signals.concat(
            FloatArray(speechFrom * FRAME_SAMPLES),
            Signals.toneSamples((stopFrame - speechFrom) * FRAME_SAMPLES + 400 * SAMPLE_RATE / 1000),
            FloatArray(100 * FRAME_SAMPLES),
        )
        lateinit var engine: SpeechEngine
        val source = ArrayAudioSource(audio, onFrame = { if (it == stopFrame) engine.stop() })
        engine = engine(native) { source }
        val t = engine.listen().toList().filterIsInstance<SpeechEvent.Transcript>().single().transcript
        // the speech ends 400 ms after stop() (1904 ms, inside frame 59); endMs is the end of that frame
        assertEquals(stopMs + 416, t.endMs)
        // the utterance: pre-roll from frame 6, the speech to frame 59, then the 8 frames of the pause (60..67)
        assertEquals((68 - preRollFrom) * FRAME_SAMPLES, native.passedToWhisper.single().size)
        assertEquals("reading ended with the pause, not at stop() nor after minSilenceMs", 68, source.framesRead)
        assertTrue(source.closed)
        assertFalse(engine.isListening)
    }

    @Test
    fun stopDuringSpeechThatNeverPausesEndsTheUtteranceOneSecondAfterStop() = runBlocking {
        val native = FakeNative()
        lateinit var engine: SpeechEngine
        // a second stop() during the wait changes nothing
        val source = ArrayAudioSource(
            FloatArray(speechFrom * FRAME_SAMPLES),
            tail = { Signals.toneAt(it) },
            onFrame = { if (it == stopFrame || it == stopFrame + 10) engine.stop() },
        )
        engine = engine(native) { source }
        val t = engine.listen().toList().filterIsInstance<SpeechEvent.Transcript>().single().transcript
        val waited = t.endMs - stopMs
        assertTrue("ended $waited ms after stop()", waited in (1000 - FRAME_MS)..(1000 + FRAME_MS))
        assertEquals("the last frame read is the last one in the utterance", t.endMs / FRAME_MS, source.framesRead.toLong())
        assertFalse(engine.isListening)
    }

    @Test
    fun stopWithoutSpeechInProgressFinishesAtOnce() = runBlocking {
        // after an utterance has ended (more than minSilenceMs of silence) no word is in progress
        val a = FakeNative()
        lateinit var ea: SpeechEngine
        val sa = ArrayAudioSource(utterance, tail = { 0f }, onFrame = { if (it == 100) ea.stop() })
        ea = engine(a) { sa }
        assertEquals(1, ea.listen().toList().filterIsInstance<SpeechEvent.Transcript>().size)
        assertEquals("no frame read after the one in flight", 100, sa.framesRead)
        // nor in pure silence
        val b = FakeNative()
        lateinit var eb: SpeechEngine
        val sb = ArrayAudioSource(FloatArray(0), tail = { 0f }, onFrame = { if (it == 40) eb.stop() })
        eb = engine(b) { sb }
        assertTrue(eb.listen().toList().isEmpty())
        assertEquals("no frame read after the one in flight", 40, sb.framesRead)
    }

    @Test
    fun aPauseUnderWayCountsSoStopWaitsOnlyForTheRestOfIt() = runBlocking {
        // speech up to frame 46, silence from frame 47
        val audio = Signals.concat(
            FloatArray(speechFrom * FRAME_SAMPLES),
            Signals.toneSamples((stopFrame - speechFrom) * FRAME_SAMPLES),
            FloatArray(100 * FRAME_SAMPLES),
        )
        // stop() 96 ms into the pause waits for the other 160 ms
        val a = FakeNative()
        lateinit var ea: SpeechEngine
        val sa = ArrayAudioSource(audio, onFrame = { if (it == stopFrame + 3) ea.stop() })
        ea = engine(a) { sa }
        assertEquals(stopMs, ea.listen().toList().filterIsInstance<SpeechEvent.Transcript>().single().transcript.endMs)
        assertEquals(stopFrame + 8, sa.framesRead)
        // stop() 320 ms into the pause finishes at once
        val b = FakeNative()
        lateinit var eb: SpeechEngine
        val sb = ArrayAudioSource(audio, onFrame = { if (it == stopFrame + 10) eb.stop() })
        eb = engine(b) { sb }
        assertEquals(1, eb.listen().toList().filterIsInstance<SpeechEvent.Transcript>().size)
        assertEquals(stopFrame + 10, sb.framesRead)
    }

    @Test
    fun cancellingWhileStopWaitsForAPauseStillDiscards() = runBlocking {
        val native = FakeNative()
        val waiting = CountDownLatch(1)
        lateinit var engine: SpeechEngine
        val source = ArrayAudioSource(
            FloatArray(speechFrom * FRAME_SAMPLES),
            tail = { Signals.toneAt(it) },
            sleepPerFrameMs = 5,
            onFrame = {
                if (it == stopFrame) engine.stop()
                if (it == stopFrame + 5) waiting.countDown()
            },
        )
        engine = engine(native) { source }
        val events = Collections.synchronizedList(mutableListOf<SpeechEvent>())
        val job = launch(Dispatchers.Default) { engine.listen().collect { events += it } }
        assertTrue("stop() is waiting for a pause", waiting.await(10, TimeUnit.SECONDS))
        job.cancel()
        withTimeout(5_000) { job.join() }
        assertTrue("discarded", events.none { it is SpeechEvent.Transcript })
        assertTrue(native.passedToWhisper.isEmpty())
        assertTrue("cancelled within the wait", source.framesRead < stopFrame + 31)
        assertTrue(source.closed)
        assertFalse(engine.isListening)
        // A cancelled session does not wait for the whisper thread: the model is freed right after, in the background.
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
    }

    @Test
    fun oneSessionAtATime() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { ArrayAudioSource(Signals.zeros(1.0), tail = { 0f }, sleepPerFrameMs = 1) }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        awaitCondition("the first session") { engine.isListening }
        try {
            engine.listen().first()
            fail("a second session must fail")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("already"))
        }
        job.cancel()
        job.join()
        assertFalse(engine.isListening)
    }

    @Test
    fun closeCancelsTheSessionAndRejectsNewWork() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val collected = async(Dispatchers.Default) {
            try {
                engine.listen().collect { }
                "completed"
            } catch (e: CancellationException) {
                "cancelled"
            }
        }
        awaitCondition("the session's model") { engine.isModelLoaded }
        engine.close()
        assertEquals("cancelled", withTimeout(5_000) { collected.await() })
        engine.awaitClosed()                                    // the model is freed in the background; wait for it
        assertFalse(engine.isModelLoaded)
        try {
            engine.transcribe(Signals.tone(1.0))
            fail("closed")
        } catch (e: IllegalStateException) {
            // expected
        }
    }

    @Test
    fun closeWhileListenIsStartingMeansTheSessionNeverRuns() = runBlocking {
        val starting = CountDownLatch(1)
        val proceed = CountDownLatch(1)
        val hooks = object : SpeechEngine.Hooks {
            override fun sessionStarting() {
                starting.countDown()
                check(proceed.await(10, TimeUnit.SECONDS)) { "never released" }
            }
        }
        val native = FakeNative()
        val source = ArrayAudioSource(utterance)
        val engine = engine(native, hooks) { source }
        val collected = async(Dispatchers.Default) { runCatching { engine.listen().toList() } }
        assertTrue(starting.await(10, TimeUnit.SECONDS))
        engine.close()                       // listen() has started, its session is not registered yet
        proceed.countDown()
        val outcome = collected.awaitWithin(5_000, "the collection started before close()")
        val error = outcome.exceptionOrNull()
        assertTrue(
            "no session may run after close() returned; got $outcome",
            error is IllegalStateException && error.message == "CzechSpeech is closed",
        )
        assertEquals("the microphone was never opened", 0, source.framesRead)
        assertFalse(source.opened)
        assertEquals(0, native.whisperInits.get())
        assertFalse(engine.isListening)
    }

    @Test
    fun listenAfterCloseFailsWithAClearErrorAndCloseAndStopAreIdempotent() = runBlocking {
        val native = FakeNative()
        val source = ArrayAudioSource(utterance)
        val engine = engine(native) { source }
        engine.close()
        engine.close()
        engine.stop()
        try {
            engine.listen().first()
            fail("listen() after close() must fail")
        } catch (e: IllegalStateException) {
            assertEquals("CzechSpeech is closed", e.message)
        }
        assertFalse(source.opened)
        assertFalse(engine.isListening)
    }

    @Test
    fun closeRacingTheStartOfListenNeverLeavesASessionRunning() = runBlocking {
        // No hook: close() and the start of a collection race freely. Whichever wins, once close() has returned
        // the collection ends (rejected or cancelled); a session left running would never end (endless silence).
        repeat(100) { run ->
            val engine = engine(FakeNative()) { ArrayAudioSource(FloatArray(0), tail = { 0f }, sleepPerFrameMs = 1) }
            val go = CountDownLatch(1)
            val collected = async(Dispatchers.Default) {
                go.await()
                runCatching { engine.listen().collect { } }
            }
            val closer = async(Dispatchers.Default) {
                go.await()
                engine.close()
            }
            go.countDown()
            closer.await()
            val outcome = collected.awaitWithin(5_000, "run $run: the collection after close()")
            val error = outcome.exceptionOrNull()
            assertTrue(
                "run $run: $outcome",
                error is CancellationException || (error is IllegalStateException && error.message == "CzechSpeech is closed"),
            )
            assertFalse(engine.isListening)
        }
    }

    @Test
    fun closeCancelsARunningTranscribeAndAbortsTheRecogniser() = runBlocking {
        val native = FakeNative().apply { blockUntilAborted = true }
        val engine = engine(native) { error("no microphone in this test") }
        val call = async(Dispatchers.Default) {
            val outcome = runCatching { engine.transcribe(Signals.tone(1.0)) }
            outcome to currentCoroutineContext().isActive
        }
        assertTrue("the recogniser is running", native.transcribeStarted.await(10, TimeUnit.SECONDS))
        engine.close()
        val (outcome, callerActive) = call.awaitWithin(2_000, "the transcribe() running at close()")
        assertClosed(outcome)
        assertTrue("close() fails the call; it does not cancel the caller's coroutine", callerActive)
        assertEquals("the abort reached the running recognition", 1, native.aborts.size)
        assertTrue("the model is freed", native.modelFreed.await(10, TimeUnit.SECONDS))
        awaitCondition("the model flag cleared") { !engine.isModelLoaded }
    }

    @Test
    fun closeCancelsTranscribeCallsWaitingForTheWhisperThreadAndNothingLoads() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative().apply { holdGate = gate }
        val engine = engine(native) { error("no microphone in this test") }
        val first = async(Dispatchers.Default) { runCatching { engine.transcribe(Signals.tone(1.0)) } }
        try {
            assertTrue("the first call is in the speech gate", native.gateStarted.await(10, TimeUnit.SECONDS))
            // Queued behind the first, or (if close() wins) rejected at once: closed either way.
            val second = async(Dispatchers.Default) { runCatching { engine.transcribe(Signals.tone(1.0)) } }
            engine.close()
            assertClosed(first.awaitWithin(2_000, "the transcribe() in the gate at close()"))
            assertClosed(second.awaitWithin(2_000, "the transcribe() queued at close()"))
        } finally {
            gate.countDown()                                    // the gate then finds speech in the first call
        }
        awaitCondition("the first call leaving the whisper thread (gate VAD freed)") { native.vadFrees.get() == 1 }
        assertEquals("the queued call never ran the gate", 1, native.gateCalls.get())
        assertEquals("a call cancelled by close() never loads the model", 0, native.whisperInits.get())
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aFailedUtteranceIsAnErrorEventAndTheSessionGoesOn() = runBlocking {
        val native = FakeNative().apply { failNext = 1 }
        val audio = Signals.concat(utterance, utterance)
        val engine = engine(native) { ArrayAudioSource(audio) }
        val events = engine.listen().toList()
        val errors = events.filterIsInstance<SpeechEvent.Error>()
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>()
        assertEquals(1, errors.size)
        assertTrue(errors[0].cause is CzechSpeechException)
        assertEquals(1, transcripts.size)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aMissingModelEndsTheSessionWithAnException() = runBlocking {
        val native = FakeNative()
        val source = ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1)
        val engine = SpeechEngine(File("/nonexistent/model.bin"), CzechSpeechConfig(), native, { ByteArray(16) }, { source }, { 2 })
        try {
            engine.listen().toList()
            fail("expected an exception")
        } catch (e: CzechSpeechException) {
            assertTrue(e.message!!.contains("not found"))
        }
        assertTrue("microphone closed", source.closed)
        assertFalse(engine.isListening)
        assertEquals(0, native.whisperInits.get())
    }

    @Test
    fun microphoneErrorsEndTheSession() = runBlocking {
        val native = FakeNative()
        val engine = engine(native) { throw SecurityException("RECORD_AUDIO") }
        try {
            engine.listen().toList()
            fail("expected SecurityException")
        } catch (e: SecurityException) {
            // expected
        }
        assertFalse(engine.isListening)
        assertFalse(engine.isModelLoaded)
    }

    private fun assertAllZero(arrays: List<FloatArray>) {
        assertTrue("whisper was called", arrays.isNotEmpty())
        for (a in arrays) assertTrue("utterance buffer zeroed after use", a.all { it == 0f })
    }
}
