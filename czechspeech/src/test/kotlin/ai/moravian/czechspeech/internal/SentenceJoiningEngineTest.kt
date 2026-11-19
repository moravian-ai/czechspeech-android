package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.Transcript
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.io.File
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Sentence joining in a session, on the fake native layer with the tiny language model's tokenizer: off by default
 * (then nothing changes), and on, independently of variant selection, with the language model the two share.
 */
class SentenceJoiningEngineTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val a = "Jasně, ale můžeš nějak."
    private val b = "s tebou spojit."
    private val joined = SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, b, "S tebou spojit.")
    private val joining = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm))
    private val tokenizer by lazy { Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm)) }

    /** Two utterances, the second [pauseS] after the first (speech to speech, about). */
    private fun two(pauseS: Double = 1.0) =
        Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(pauseS), Signals.tone(1.0), Signals.zeros(1.5))

    /**
     * The fake language model's answer for the boundary of [first] and [second]: the small-letter join wins by 12.6
     * over the end of the sentence, by 10 over the capital; anything else scores -100.
     */
    private fun FakeNative.joinsWithSmallLetter(first: String = a, second: String = b) {
        val texts = SentenceBoundary.languageModelTexts(first, second)
        val score = mapOf(
            SentenceBoundary.Choice.JOIN to -40.0, SentenceBoundary.Choice.JOIN_KEEP to -50.0,
            SentenceBoundary.Choice.SPLIT to -52.6, SentenceBoundary.Choice.COMMA to -47.0,
        )
        val bySequence = texts.entries.associate { (choice, text) -> seq(text) to score.getValue(choice) }
        lmScoreOf = { bySequence[it.toList()] ?: -100.0 }
    }

    private fun seq(text: String) = (intArrayOf(tokenizer.startToken) + tokenizer.encode(text)).toList()

    private fun engine(native: FakeNative, config: CzechSpeechConfig, audio: FloatArray, hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE, maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS) =
        fakeEngine(native, hooks, maxQueuedAudioMs = maxQueuedAudioMs, config = config) { ArrayAudioSource(audio) }

    /** The session's events. */
    private class Collected(val events: List<SpeechEvent>) {
        val transcripts: List<Transcript> get() = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
    }

    private fun collect(engine: SpeechEngine): Collected = runBlocking {
        val events = Collections.synchronizedList(mutableListOf<SpeechEvent>())
        withContext(Dispatchers.Default) { engine.listen().collect { events += it } }
        Collected(events.toList())
    }

    /** Language model calls run when each utterance's events had been sent (on the recognition loop). */
    private class LmCallsAtEachUtterance(private val native: FakeNative) : SpeechEngine.Hooks {
        val calls: MutableList<Int> = Collections.synchronizedList(mutableListOf())
        override fun utteranceRecognised() {
            calls += native.lmScoreCalls.size
        }
    }

    @Test
    fun offByDefaultAndThenNothingChanges() {
        assertNull(CzechSpeechConfig().sentenceJoining)
        val native = FakeNative().apply { texts += listOf(a, b) }
        val heard = collect(engine(native, CzechSpeechConfig(), two()))
        assertEquals(listOf(a, b), heard.transcripts.map { it.text })
        assertTrue(heard.transcripts.all { it.sentenceJoin == null })
        assertEquals("no language model", 0, native.lmInits.get())
        assertEquals(2, native.greedyCalls.get())
        assertTrue(native.lmThreads.isEmpty())
    }

    @Test
    fun aPauseInTheMiddleOfASentenceIsJoinedAndOnlyTheSecondTranscriptWaitsForIt() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val hooks = LmCallsAtEachUtterance(native)
        val heard = collect(engine(native, joining, two(), hooks))
        val (first, second) = heard.transcripts
        assertEquals("the text is Whisper's", listOf(a, b), listOf(first.text, second.text))
        assertNull(first.sentenceJoin)
        assertEquals(joined, second.sentenceJoin)
        assertEquals("the first transcript was sent before the language model ran", listOf(0, 1), hooks.calls.toList())
        assertEquals("the four candidates in one call", 4, native.lmScoreCalls.single().size)
        // greedy decoding: sentence joining does not need variant selection
        assertEquals(2, native.greedyCalls.get())
        assertTrue(native.beamSizes.isEmpty())
        assertEquals("loaded once, with the model", 1, native.lmInits.get())
        assertEquals("freed with it", 0, native.liveLanguageModels)
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
    }

    @Test
    fun withVariantSelectionBothUseTheOneLanguageModel() {
        // variant selection needs the language model for the first transcript already: it loads with the model
        val loadedAtRecognition = Collections.synchronizedList(mutableListOf<Int>())
        val native = FakeNative()
        native.joinsWithSmallLetter()
        native.onTranscribe = { i ->
            loadedAtRecognition += native.lmInits.get()
            native.hypotheses = listOf(listOf(Hyp(" " + (if (i == 0) a else b), -1.0, 6, best = true)))
        }
        val config = CzechSpeechConfig(
            variantSelection = VariantSelection(TestEnv.tinyLm),
            sentenceJoining = SentenceJoining(TestEnv.tinyLm),
        )
        val heard = collect(engine(native, config, two()))
        assertEquals(listOf(a, b), heard.transcripts.map { it.text })
        assertEquals(joined, heard.transcripts[1].sentenceJoin)
        assertEquals(listOf(5, 5), native.beamSizes)
        assertEquals("loaded before the first recognition", listOf(1, 1), loadedAtRecognition.toList())
        assertEquals("one language model for both", 1, native.lmInits.get())
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun theTwoFeaturesTakeOneLanguageModelFile() {
        val e = assertThrows(IllegalArgumentException::class.java) {
            CzechSpeechConfig(
                variantSelection = VariantSelection(File("one.gguf")),
                sentenceJoining = SentenceJoining(File("other.gguf")),
            )
        }
        assertTrue(e.message, e.message!!.contains("languageModelFile"))
        CzechSpeechConfig(variantSelection = VariantSelection(File("one.gguf")), sentenceJoining = SentenceJoining(File("one.gguf")))
        CzechSpeechConfig(variantSelection = VariantSelection(languageModelWeight = 0f), sentenceJoining = SentenceJoining(File("one.gguf")))
    }

    @Test
    fun noJoinAcrossAPauseOfMoreThanThreeSeconds() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val heard = collect(engine(native, joining, two(pauseS = 3.5)))
        assertTrue(heard.transcripts[1].startMs - heard.transcripts[0].endMs > 3_000)
        assertNull(heard.transcripts[1].sentenceJoin)
        assertTrue("no language model run", native.lmScoreCalls.isEmpty())
    }

    @Test
    fun withoutTheLanguageModelTheTextsComeAsTheyAre() {
        // it does not load: the session goes on, without sentence joining
        val failing = FakeNative().apply {
            texts += listOf(a, b)
            failLmInit = true
        }
        val heard = collect(engine(failing, joining, two()))
        assertEquals(listOf(a, b), heard.transcripts.map { it.text })
        assertTrue(heard.transcripts.all { it.sentenceJoin == null })
        assertTrue("no error", heard.events.none { it is SpeechEvent.Error })
        // it fails on the boundary
        val native = FakeNative().apply {
            texts += listOf(a, b)
            lmScoreOf = null
        }
        val again = collect(engine(native, joining, two()))
        assertEquals(listOf(a, b), again.transcripts.map { it.text })
        assertTrue(again.transcripts.all { it.sentenceJoin == null })
        assertEquals(1, native.lmScoreCalls.size)
    }

    @Test
    fun aFailedRecognitionInBetweenBreaksTheChain() {
        val native = FakeNative().apply {
            texts += listOf(a, b)                     // the failing recognition takes no text
            onTranscribe = { if (it == 1) failNext = 1 }
            joinsWithSmallLetter()
        }
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(0.8), Signals.tone(1.0), Signals.zeros(0.8), Signals.tone(1.0), Signals.zeros(1.5))
        val heard = collect(engine(native, joining, audio))
        assertEquals(1, heard.events.count { it is SpeechEvent.Error })
        assertEquals(2, heard.transcripts.size)
        assertNull("the speech between was lost", heard.transcripts[1].sentenceJoin)
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun skippedSpeechInBetweenBreaksTheChain() {
        // Four short utterances, 0.7 s apart. The first one's recognition is held: the second waits in the queue (3 s
        // bound), the third does not fit and is skipped. Then the queue drains before the fourth comes, which is
        // within 3 s of the second but must not continue it across the skipped speech.
        val release = CountDownLatch(1)
        val recognised = AtomicInteger()
        val native = FakeNative().apply {
            texts += listOf(a, b, b)
            onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) }
            joinsWithSmallLetter()
        }
        val hooks = object : SpeechEngine.Hooks {
            override fun utteranceRecognised() {
                recognised.incrementAndGet()
            }
        }
        val gap = Signals.zeros(0.7)
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0),
            Signals.tone(1.0), gap, Signals.tone(1.0), gap, Signals.tone(1.0), Signals.zeros(1.5),
        )
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 100) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "the first not in recognition" }
            if (n == 200) release.countDown()
            if (n == 201) {
                val deadline = System.nanoTime() + 10_000_000_000L
                while (recognised.get() < 2) {
                    check(System.nanoTime() < deadline) { "the queue never drained" }
                    Thread.sleep(1)
                }
            }
        })
        val engine = fakeEngine(native, hooks, maxQueuedAudioMs = 3_000, config = joining) { source }
        val heard = collect(engine)
        assertEquals(1, engine.skippedUtterances)
        val t = heard.transcripts
        assertEquals(3, t.size)
        assertEquals(joined, t[1].sentenceJoin)
        assertTrue("within 3 s", t[2].startMs - t[1].endMs <= 3_000)
        assertNull("not across the skipped speech", t[2].sentenceJoin)
        assertEquals(1, native.lmScoreCalls.size)
    }

    @Test
    fun anUtteranceWithoutWordsInBetweenBreaksTheChain() {
        // the measurement never paired transcripts across a part without words: nothing is known of that speech
        val native = FakeNative().apply {
            texts += listOf(a, "", b)
            joinsWithSmallLetter()
        }
        val gap = Signals.zeros(0.8)
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), gap, Signals.tone(1.0), gap, Signals.tone(1.0), Signals.zeros(1.5))
        val heard = collect(engine(native, joining.copy(retryEmptyFinals = false), audio))        // no repeat of the empty one
        val t = heard.transcripts
        assertEquals(listOf(a, b), t.map { it.text })
        assertTrue("within 3 s", t[1].startMs - t[0].endMs <= 3_000)
        assertNull(t[1].sentenceJoin)
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun aTextRescuedByTheRetryOfAnEmptyFinalIsJoinedLikeANormalFinal() {
        // the second utterance (1.2 s of clear speech) comes out empty on its first run and is rescued by the retry
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.2), Signals.zeros(1.0), Signals.tone(1.2), Signals.zeros(1.5))
        val native = FakeNative().apply {
            texts += listOf(a, "", b)
            joinsWithSmallLetter()
        }
        val heard = collect(engine(native, joining, audio))
        assertEquals(listOf(a, b), heard.transcripts.map { it.text })
        assertEquals("the first run, the retry and the first utterance", 3, native.transcribeCalls.get())
        assertNull(heard.transcripts[0].sentenceJoin)
        assertEquals(joined, heard.transcripts[1].sentenceJoin)
        assertTrue("the language model decided the boundary of the rescued text", native.lmScoreCalls.isNotEmpty())
    }

    @Test
    fun theLanguageModelLoadsOnlyAfterTheFirstTranscriptOfTheSession() {
        // sentence joining alone: the first transcript never waits for the language model to load
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val loadsWhenSent = Collections.synchronizedList(mutableListOf<Int>())
        val hooks = object : SpeechEngine.Hooks {
            override fun utteranceRecognised() {
                loadsWhenSent += native.lmThreads.size
            }
        }
        val heard = collect(engine(native, joining, two(), hooks))
        assertEquals("no load before the first transcript was sent, one before the second", listOf(0, 1), loadsWhenSent.toList())
        assertEquals(joined, heard.transcripts[1].sentenceJoin)
        assertEquals(1, native.lmInits.get())
        assertEquals(0, native.liveLanguageModels)
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
    }

    @Test
    fun noLanguageModelWhenNoOtherUtteranceCanCome() {
        // dictating one sentence and pressing stop: loading 135 MB to free it right away would only delay the end
        val native = FakeNative().apply { texts += listOf(a) }
        lateinit var engine: SpeechEngine
        native.onTranscribe = { engine.stop() }                 // stop() while the only utterance is recognised
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))
        engine = fakeEngine(native, config = joining) { ArrayAudioSource(audio, tail = { 0f }) }     // silence after it
        val heard = collect(engine)
        assertEquals(listOf(a), heard.transcripts.map { it.text })
        assertEquals("not loaded", 0, native.lmInits.get())
        assertTrue(native.lmThreads.isEmpty())
    }

    @Test
    fun theLanguageModelLoadsWhenAnUtteranceStillWaitsAfterStop() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        lateinit var engine: SpeechEngine
        lateinit var source: ArrayAudioSource
        native.onTranscribe = { i ->
            if (i == 0) {
                // the second utterance (ready at frame ~128) is queued before stop()
                val deadline = System.nanoTime() + 10_000_000_000L
                while (!(source.framesRead >= 140)) {
                    check(System.nanoTime() < deadline) { "capture never got past the second utterance" }
                    Thread.sleep(1)
                }
                engine.stop()
            }
        }
        source = ArrayAudioSource(two())
        engine = fakeEngine(native, config = joining) { source }
        val heard = collect(engine)
        assertEquals(1, native.lmInits.get())
        assertEquals(joined, heard.transcripts[1].sentenceJoin)
    }

    @Test
    fun theLanguageModelLoadsWhileCaptureGoesOn() {
        // one utterance so far, and capture still running (held at frame 100 until the load was tried)
        val native = FakeNative().apply { texts += listOf(a) }
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(3.0))
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 100) {
                val deadline = System.nanoTime() + 10_000_000_000L
                while (!(native.lmThreads.size >= 1)) {
                    check(System.nanoTime() < deadline) { "the language model was never loaded" }
                    Thread.sleep(1)
                }
            }
        })
        collect(fakeEngine(native, config = joining) { source })
        assertEquals(1, native.lmInits.get())
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun aLanguageModelLoadThatThrowsDoesNotEndTheSession() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            lmInitError = IllegalStateException("native failure")
        }
        val heard = collect(engine(native, joining, two()))
        assertEquals(listOf(a, b), heard.transcripts.map { it.text })
        assertTrue(heard.transcripts.all { it.sentenceJoin == null })
        assertTrue("no error", heard.events.none { it is SpeechEvent.Error })
        assertEquals(1, native.lmThreads.size)
    }

    @Test
    fun aLanguageModelThatDidNotLoadIsTriedAgainInTheNextSession() {
        // keepModelLoaded: the model stays, the language model file was not there yet in the first session
        val native = FakeNative().apply {
            texts += listOf(a, b, a, b)
            joinsWithSmallLetter()
            failLmInit = true
        }
        val engine = fakeEngine(native, keepModelLoaded = true, config = joining) { ArrayAudioSource(two()) }
        val first = collect(engine)
        assertTrue(first.transcripts.all { it.sentenceJoin == null })
        native.failLmInit = false
        val second = collect(engine)
        assertNull("a new session continues nothing", second.transcripts[0].sentenceJoin)
        assertEquals(joined, second.transcripts[1].sentenceJoin)
        assertEquals("the model stayed loaded", 1, native.whisperInits.get())
        assertEquals("a failed attempt, then the load", 2, native.lmThreads.size)
        engine.close()
        runBlocking { engine.awaitClosed() }
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun aNewSessionContinuesNothing() {
        for (keep in listOf(false, true)) {
            val native = FakeNative().apply {
                texts += listOf(a, b, b)
                joinsWithSmallLetter()
            }
            val audio = ArrayDeque(listOf(two(), Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.5))))
            val engine = fakeEngine(native, keepModelLoaded = keep, config = joining) { ArrayAudioSource(audio.removeFirst()) }
            assertEquals("keep=$keep", joined, collect(engine).transcripts[1].sentenceJoin)
            val next = collect(engine).transcripts.single()
            assertEquals(b, next.text)
            assertNull("keep=$keep", next.sentenceJoin)
            engine.close()
            runBlocking { engine.awaitClosed() }
            assertEquals(0, native.liveLanguageModels)
        }
    }

    @Test
    fun closeBeforeTheDecisionLeavesTheLanguageModelAlone() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val boundaries = AtomicInteger()
        lateinit var engine: SpeechEngine
        val hooks = object : SpeechEngine.Hooks {
            override fun sentenceBoundaryComing() {
                if (boundaries.incrementAndGet() == 2) engine.close()      // the decision on the second transcript is next
            }
        }
        engine = engine(native, joining, two(), hooks)
        val outcome = runCatching { collect(engine) }
        assertTrue("$outcome", outcome.exceptionOrNull() is CancellationException)
        runBlocking { engine.awaitClosed() }
        assertEquals(2, boundaries.get())
        assertTrue("no language model after close()", native.lmScoreCalls.isEmpty())
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun closeDuringTheDecisionFreesTheLanguageModelOnlyAfterIt() = runBlocking {
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
            holdLmScore = release
        }
        val engine = engine(native, joining, two())
        val collected = async(Dispatchers.Default) { runCatching { engine.listen().collect { } } }
        assertTrue("the decision runs", native.lmScoreStarted.await(10, TimeUnit.SECONDS))
        engine.close()
        val outcome = collected.awaitWithin(2_000, "the collection closed during the decision")
        assertTrue("$outcome", outcome.exceptionOrNull() is CancellationException)
        assertEquals("not freed under the running call", 1, native.liveLanguageModels)
        release.countDown()
        engine.awaitClosed()
        assertEquals(0, native.liveLanguageModels)
        assertEquals(0, native.lmFreedWhileScoring.get())
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
    }

    @Test
    fun transcribeNeverJoins() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val engine = engine(native, joining, two())
        val texts = runBlocking { listOf(engine.transcribe(Signals.tone(1.0)), engine.transcribe(Signals.tone(1.0))) }
        assertEquals(listOf(a, b), texts)
        assertTrue(native.lmScoreCalls.isEmpty())
        assertEquals("not even loaded", 0, native.lmInits.get())
    }

    @Test
    fun theLogNeverHoldsTheText() {
        val native = FakeNative().apply {
            texts += listOf(a, b)
            joinsWithSmallLetter()
        }
        val logs = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        val heard = try {
            collect(engine(native, joining, two()))
        } finally {
            CsLog.sink = sink
        }
        assertEquals(joined, heard.transcripts[1].sentenceJoin)
        assertTrue(logs.any { it.startsWith("sentence boundary: join") })
        for (word in listOf("Jasně", "nějak", "tebou", "spojit")) assertTrue(word, logs.none { word in it })
        assertTrue("toString hides the texts", "tebou" !in heard.transcripts[1].toString())
    }
}
