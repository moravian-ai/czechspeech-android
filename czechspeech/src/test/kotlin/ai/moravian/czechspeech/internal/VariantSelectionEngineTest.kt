package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechLanguageModel
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.rules.TestRule
import org.junit.Test
import java.io.File

/**
 * Variant selection in the pipeline, on the fake native layer: off by default (then the engine makes exactly the calls
 * it made before), and when configured, beam search, the language model and the choice by the formula.
 */
class VariantSelectionEngineTest {
    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val speech = Signals.tone(1.0)
    private val tokenizer by lazy { Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm)) }

    private fun selection(file: File = TestEnv.tinyLm, beam: Int = 5) = VariantSelection(file, beamSize = beam)

    private fun engine(native: FakeNative, selection: VariantSelection?, keep: Boolean = false) =
        fakeEngine(native, keepModelLoaded = keep, config = CzechSpeechConfig(variantSelection = selection)) {
            ArrayAudioSource(Signals.concat(speech, Signals.zeros(2.0)))
        }

    /** The sequence the language model gets for [text]: the start token and the text's tokens. */
    private fun seq(text: String) = intArrayOf(tokenizer.startToken) + tokenizer.encode(text)

    @Test
    fun offByDefault() {
        assertNull(CzechSpeechConfig().variantSelection)
    }

    @Test
    fun withoutVariantSelectionOnlyTheGreedyCallRuns() {
        val native = FakeNative()
        val text = runBlocking { engine(native, selection = null).transcribe(speech) }
        assertEquals("dobrý den", text)
        assertEquals(1, native.greedyCalls.get())
        assertTrue(native.beamSizes.isEmpty())
        assertEquals(0, native.lmInits.get())
        assertTrue(native.lmThreads.isEmpty())
        assertTrue(native.returned.single().all { it == 0.toByte() })
    }

    @Test
    fun theLanguageModelPicksAmongTheBeams() {
        val native = FakeNative()
        native.hypotheses = listOf(
            listOf(
                Hyp(" Viděl jsem velký psy.", -2.0, 9, best = true, score = -0.22),
                Hyp(" Viděl jsem velké psy.", -2.3, 9, score = -0.25),
            ),
        )
        val good = seq("Viděl jsem velké psy.")
        native.lmScoreOf = { if (it.contentEquals(good)) -40.0 else -50.0 }
        val text = runBlocking { engine(native, selection()).transcribe(speech) }
        assertEquals("Viděl jsem velké psy.", text)
        assertEquals(listOf(5), native.beamSizes)
        assertEquals(0, native.greedyCalls.get())
        val call = native.lmScoreCalls.single()
        assertEquals(2, call.size)
        assertArrayEquals(seq("Viděl jsem velký psy."), call[0])
        assertArrayEquals(good, call[1])
        assertTrue("the hypotheses are zeroed", native.returned.single().all { it == 0.toByte() })
    }

    @Test
    fun theWeightsComeFromTheConfiguration() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" krátká", -1.0, 3, best = true), Hyp(" krátká věta", -1.5, 5)))
        native.lmScoreOf = { 0.0 }
        val noBonus = VariantSelection(TestEnv.tinyLm, beamSize = 3, languageModelWeight = 0.3f, lengthBonus = 0f)
        assertEquals("krátká", runBlocking { engine(native, noBonus).transcribe(speech) })
        assertEquals("krátká věta", runBlocking { engine(native, selection()).transcribe(speech) })
        assertEquals(listOf(3, 5), native.beamSizes)
    }

    @Test
    fun aZeroWeightTurnsTheLanguageModelOff() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" krátká", -1.0, 3, best = true), Hyp(" krátká věta", -1.5, 5)))
        val beamOnly = VariantSelection(File("/nonexistent/lm.gguf"), languageModelWeight = 0f)
        assertEquals("krátká věta", runBlocking { engine(native, beamOnly).transcribe(speech) })
        assertEquals(listOf(5), native.beamSizes)
        assertEquals(0, native.lmInits.get())
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun windowsAreJoinedLikeWhisperJoinsSegments() {
        val native = FakeNative()
        native.hypotheses = listOf(
            listOf(Hyp(" První okno.", -1.0, 4, best = true)),
            listOf(Hyp(" Druhé okno.", -1.0, 4, best = true), Hyp(" Druhý okno.", -1.1, 4)),
        )
        native.lmScoreOf = { if (it.contentEquals(seq("Druhé okno."))) -10.0 else -30.0 }
        assertEquals("První okno. Druhé okno.", runBlocking { engine(native, selection()).transcribe(speech) })
        assertEquals("one window needed the language model", 1, native.lmScoreCalls.size)
    }

    @Test
    fun noLanguageModelWhenTheBeamsDifferOnlyInCaseOrPunctuation() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" Ahoj, Petře.", -1.2, 5), Hyp(" Ahoj Petře.", -1.0, 4, best = true)))
        assertEquals("Ahoj Petře.", runBlocking { engine(native, selection()).transcribe(speech) })
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun aFailingLanguageModelLeavesWhispersChoice() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" Ahoj Petře.", -1.0, 4, best = true), Hyp(" Ahoj Petr.", -1.1, 4)))
        native.lmScoreOf = null
        assertEquals("Ahoj Petře.", runBlocking { engine(native, selection()).transcribe(speech) })
        assertEquals(1, native.lmScoreCalls.size)
    }

    @Test
    fun theLanguageModelLoadsAndIsFreedWithTheModel() {
        val native = FakeNative()
        val engine = engine(native, selection())
        runBlocking { engine.transcribe(speech) }
        assertEquals(1, native.lmInits.get())
        assertEquals(1, native.lmFrees.get())
        assertFalse(engine.isModelLoaded)
        assertTrue(native.lmThreads.all { it == FakeNative.WHISPER_THREAD })
        assertEquals(listOf("whisperFree", "lmFree"), native.events.filter { it.endsWith("Free") })
    }

    @Test
    fun keptWithTheModelUntilItIsReleased() {
        val native = FakeNative()
        val engine = engine(native, selection(), keep = true)
        runBlocking {
            engine.transcribe(speech)
            engine.transcribe(speech)
        }
        assertEquals(1, native.lmInits.get())
        assertEquals(0, native.lmFrees.get())
        engine.releaseModel()
        runBlocking { engine.awaitWhisperThread() }
        assertEquals(1, native.lmFrees.get())
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun aListenSessionUsesItToo() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" Ahoj Petře.", -1.0, 4, best = true), Hyp(" Ahoj Petr.", -1.1, 4)))
        native.lmScoreOf = { if (it.contentEquals(seq("Ahoj Petr."))) -1.0 else -30.0 }
        val events = runBlocking { engine(native, selection()).listen().toList() }
        val transcripts = events.filterIsInstance<SpeechEvent.Transcript>()
        assertEquals("Ahoj Petr.", transcripts.single().transcript.text)
        assertEquals(1, native.lmInits.get())
        assertEquals(1, native.lmFrees.get())
    }

    @Test
    fun aMissingLanguageModelFileFailsLikeAMissingModel() {
        val native = FakeNative()
        val e = failure<CzechSpeechException> {
            runBlocking { engine(native, selection(File("/nonexistent/lm.gguf"))).transcribe(speech) }
        }
        assertTrue(e.message, e.message!!.contains("Language model file not found"))
        assertEquals(0, native.whisperInits.get())
    }

    @Test
    fun aLanguageModelThatDoesNotLoadFails() {
        val native = FakeNative()
        native.failLmInit = true
        val e = failure<CzechSpeechException> { runBlocking { engine(native, selection()).transcribe(speech) } }
        assertTrue(e.message, e.message!!.contains("Could not load the language model"))
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun aVocabularyThatDoesNotMatchTheModelFails() {
        val native = FakeNative()
        native.lmVocab = 50257
        val e = failure<CzechSpeechException> { runBlocking { engine(native, selection()).transcribe(speech) } }
        assertTrue(e.message, e.message!!.contains("vocabulary"))
        assertEquals("freed again", 0, native.liveLanguageModels)
    }

    @Test
    fun aFileThatIsNotGgufFails() {
        val native = FakeNative()
        val e = failure<CzechSpeechException> {
            runBlocking { engine(native, selection(TestEnv.vadAsset)).transcribe(speech) }
        }
        assertTrue(e.message, e.message!!.contains("Could not load the language model"))
        assertEquals(0, native.liveLanguageModels)
    }

    @Test
    fun cancellationAbortsTheBeamSearch() {
        val native = FakeNative()
        native.blockUntilAborted = true
        val engine = engine(native, selection())
        runBlocking {
            val job = launch(Dispatchers.Default) { engine.transcribe(speech) }
            check(native.transcribeStarted.await(10, java.util.concurrent.TimeUnit.SECONDS))
            job.cancel()
            awaitCondition("model freed") { !engine.isModelLoaded && native.lmFrees.get() == 1 }
        }
        assertEquals(1, engine.abortedRecognitions)
        assertEquals(listOf(5), native.beamSizes)
    }

    /** Whisper's best says there is no speech; another beam has 21 tokens of text it turned down. */
    private val noSpeechWindow = listOf(
        listOf(
            Hyp("", -0.2, 1, best = true, score = -0.2),
            Hyp(" a to je text, který Whisper zamítl a jiný paprsek ho přesto napsal až do konce", -5.0, 21),
        ),
    )

    @Test
    fun whenWhisperChoseNoTextNoOtherBeamOverridesIt() {
        // λ 0, β 0.3: by the formula the text would win (-5.0 + 0.3 * 21 = 1.3 > -0.2 + 0.3 = 0.1)
        val native = FakeNative()
        native.hypotheses = noSpeechWindow
        assertEquals("", runBlocking { engine(native, VariantSelection(languageModelWeight = 0f)).transcribe(speech) })
        // a large length bonus with the language model: neither the language model nor the formula is run
        val big = VariantSelection(TestEnv.tinyLm, languageModelWeight = 0.3f, lengthBonus = 5f)
        assertEquals("", runBlocking { engine(native, big).transcribe(speech) })
        assertEquals("", runBlocking { engine(native, selection()).transcribe(speech) })
        assertTrue("no language model for a window without speech", native.lmScoreCalls.isEmpty())
    }

    @Test
    fun anEmptyVariantNeverWins() {
        // the language model gives an empty text 0, more than any sentence gets
        val native = FakeNative()
        native.hypotheses = listOf(
            listOf(
                Hyp(" Dobrý den, posílám vám zprávu z vlaku.", -5.0, 12, best = true, score = -0.4),
                Hyp("", -4.0, 1, score = -4.0),
            ),
        )
        assertEquals("Dobrý den, posílám vám zprávu z vlaku.", runBlocking { engine(native, selection()).transcribe(speech) })
        assertTrue("one candidate left: no language model", native.lmScoreCalls.isEmpty())
    }

    @Test
    fun withTheLanguageModelOffCaseOrPunctuationOnlyKeepsWhispersChoice() {
        val native = FakeNative()
        // by length alone the second would win (-1.05 + 0.3 * 5 > -1.0 + 0.3 * 4)
        native.hypotheses = listOf(listOf(Hyp(" Ahoj Petře.", -1.0, 4, best = true), Hyp(" Ahoj, Petře.", -1.05, 5)))
        val beamOnly = VariantSelection(File("/nonexistent/lm.gguf"), languageModelWeight = 0f)
        assertEquals("Ahoj Petře.", runBlocking { engine(native, beamOnly).transcribe(speech) })
    }

    @Test
    fun aWindowWithoutWhispersChoiceIsStillChosen() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" Ahoj Petře.", -1.0, 4, score = -0.3), Hyp(" Ahoj Petr.", -1.1, 4, score = -0.2)))
        native.lmScoreOf = { if (it.contentEquals(seq("Ahoj Petře."))) -5.0 else -30.0 }
        assertEquals("Ahoj Petře.", runBlocking { engine(native, selection()).transcribe(speech) })
    }

    @Test
    fun textLongerThanTheLanguageModelsContextIsScoredOnItsBeginning() {
        val native = FakeNative()
        native.lmContext = 16
        val long = " " + List(40) { "Viděl jsem velké psy" }.joinToString(" ")
        native.hypotheses = listOf(listOf(Hyp(long, -9.0, 200, best = true), Hyp(long.replace("velké", "velký"), -9.5, 200)))
        runBlocking { engine(native, selection()).transcribe(speech) }
        val call = native.lmScoreCalls.single()
        assertEquals(listOf(16, 16), call.map { it.size })
        assertArrayEquals(seq(long.trim()).copyOf(16), call[0])
    }

    @Test
    fun aBeamSearchCancelledBeforeTheEncoderCountsAsAborted() {
        val native = FakeNative()
        native.blockUntilAborted = true
        native.abortedBeforeEncoder = true
        val engine = engine(native, selection())
        runBlocking {
            val job = launch(Dispatchers.Default) { engine.transcribe(speech) }
            check(native.transcribeStarted.await(10, java.util.concurrent.TimeUnit.SECONDS))
            job.cancel()
            awaitCondition("model freed") { !engine.isModelLoaded }
        }
        assertEquals(1, engine.abortedRecognitions)
        assertTrue(native.returned.single().all { it == 0.toByte() })
    }

    @Test
    fun aWindowDecodedAtAFallbackTemperatureKeepsWhispersChoice() {
        // its log-probabilities are of logits / T: not the scale λ and β were tuned for
        val native = FakeNative()
        native.hypotheses = listOf(
            listOf(
                Hyp(" Ahoj Petře.", -9.0, 4, best = true, temperature = 0.2f),
                Hyp(" Ahoj Petr.", -8.0, 4, temperature = 0.2f),
            ),
        )
        native.lmScoreOf = { if (it.contentEquals(seq("Ahoj Petr."))) -1.0 else -30.0 }
        assertEquals("Ahoj Petře.", runBlocking { engine(native, selection()).transcribe(speech) })
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun theLanguageModelFileIsOptionalWithoutTheLanguageModel() {
        val native = FakeNative()
        native.hypotheses = listOf(listOf(Hyp(" krátká", -1.0, 3, best = true), Hyp(" krátká věta", -1.5, 5)))
        assertEquals(
            "krátká věta",
            runBlocking { engine(native, VariantSelection(languageModelWeight = 0f)).transcribe(speech) },
        )
        assertEquals(0, native.lmInits.get())
        val e = failure<IllegalArgumentException> { VariantSelection() }
        assertTrue(e.message, e.message!!.contains("languageModelFile"))
    }

    @Test
    fun configurationIsChecked() {
        val lm = TestEnv.tinyLm
        assertEquals(5, VariantSelection(lm).beamSize)
        assertEquals(0.3f, VariantSelection(lm).languageModelWeight, 0f)
        assertEquals(0.3f, VariantSelection(lm).lengthBonus, 0f)
        failure<IllegalArgumentException> { VariantSelection(lm, beamSize = 1) }
        failure<IllegalArgumentException> { VariantSelection(lm, beamSize = 9) }
        failure<IllegalArgumentException> { VariantSelection(lm, languageModelWeight = -0.1f) }
        failure<IllegalArgumentException> { VariantSelection(lm, languageModelWeight = Float.NaN) }
        failure<IllegalArgumentException> { VariantSelection(lm, lengthBonus = Float.POSITIVE_INFINITY) }
        VariantSelection(lm, beamSize = 2, languageModelWeight = 0f, lengthBonus = -1f)
    }

    @Test
    fun theLanguageModelIsDescribedLikeTheModel() {
        val source = CzechLanguageModel.source("https://models.example.com/lm.gguf")
        assertEquals(CzechLanguageModel.FILE_NAME, source.fileName)
        assertEquals(CzechLanguageModel.SIZE_BYTES, source.sizeBytes)
        assertEquals(CzechLanguageModel.SHA256, source.sha256)
        assertTrue(CzechLanguageModel.FILE_NAME.endsWith(".gguf"))
    }
}
