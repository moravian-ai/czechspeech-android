package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechModel
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
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections

/**
 * The language of a session without streaming: the language a collection is in starts as the first language of the list,
 * changes with every transcript that was sent, and gives each recognition its thresholds ([LanguageRule]); `transcribe`
 * has no session; a forced language and a list without a switch keep their own rules.
 *
 * Lengths are the length of the audio the recogniser gets (`utterance(ms)`: speech plus the 100 ms of silence that
 * ends it, with no pre-roll), and are at least 200 ms from the minimums of 1 000 and 2 000 ms.
 */
class LanguageSessionTest {
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

    private val nan = Float.NaN
    private val three = listOf("cs", "sk", "en")
    private val switch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)
    private val config = CzechSpeechConfig(minEncoderWindow = 384, preRollMs = 0, minSilenceMs = 100, retryEmptyFinals = false)

    private val czech = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val english = floatArrayOf(0.002f, 0.001f, 0.997f)
    private val silence = floatArrayOf(0.70f, 0f, 0.30f)

    /** Speech that the recogniser gets as about [audioMs] of audio (the 100 ms silence frame-rounded make up the rest). */
    private fun utterance(audioMs: Int) = Signals.tone((audioMs - 130) / 1000.0)

    private fun audio(vararg audioMs: Int): FloatArray = Signals.concat(
        Signals.zeros(0.5),
        *audioMs.map { Signals.concat(utterance(it), Signals.zeros(0.6)) }.toTypedArray(),
    )

    /** Shares by the index of the recognition (each recognition is one call here). */
    private fun FakeNative.sharesInOrder(vararg shares: FloatArray) {
        sharesOf = { index, _, _ -> shares[index] }
    }

    private fun th(vararg values: Float) = values

    private fun assertThresholds(expected: List<FloatArray?>, native: FakeNative) {
        val actual = native.thresholdsPassed.toList()
        assertEquals("thresholds of ${actual.size} calls: ${actual.map { it?.toList() }}", expected.size, actual.size)
        for (i in expected.indices) {
            val e = expected[i]
            val a = actual[i]
            if (e == null) assertNull("call $i", a) else assertArrayEquals("call $i: ${a?.toList()}", e, a, 0f)
        }
    }

    private fun session(
        native: FakeNative,
        audio: FloatArray,
        languages: List<String> = three,
        languageSwitch: CzechSpeechModel.LanguageSwitch? = switch,
        config: CzechSpeechConfig = this.config,
        language: String? = null,
    ): List<SpeechEvent> = runBlocking {
        val engine = fakeEngine(native, config = config, languages = languages, languageSwitch = languageSwitch) { ArrayAudioSource(audio) }
        withContext(Dispatchers.Default) { engine.listen(language).toList() }
    }

    private fun List<SpeechEvent>.transcripts() = filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
    private fun List<SpeechEvent>.languages() = transcripts().map { it.language }

    @Test
    fun theTestAudioHasTheLengthItClaims() {
        val native = FakeNative().apply { keepCopies = true }
        session(native, audio(3_000, 1_500, 800), languages = listOf("cs"), languageSwitch = null)
        val lengths = native.copiesPassed.map { it.size / 16 }
        assertEquals(3, lengths.size)
        for ((got, want) in lengths.zip(listOf(3_000, 1_500, 800))) assertTrue("$got ms for $want", got in want - 60..want + 60)
    }

    @Test
    fun aOneLanguageModelPassesOneCodeNoThresholdsAndNeverAsks() {
        val native = FakeNative()
        val events = session(native, audio(3_000, 1_500), languages = listOf("cs"), languageSwitch = null)
        assertEquals(2, events.transcripts().size)
        assertEquals(listOf("cs", "cs"), native.languagesPassed.toList())
        assertThresholds(listOf(null, null), native)
        assertEquals(0, native.whisperLanguageCalls.get())
        assertEquals(0, native.sharesCalls.get())
    }

    @Test
    fun aLongEnglishUtteranceSwitchesTheSessionAndAShortOneStaysEnglish() {
        val native = FakeNative().apply { sharesInOrder(english, english) }
        val events = session(native, audio(3_000, 800))
        assertThresholds(listOf(th(nan, nan, 0.9f), th(nan, nan, 0f)), native)
        assertEquals(listOf("en", "en"), events.languages())
        assertEquals(listOf("cs,sk,en", "cs,sk,en"), native.languagesPassed.toList())
    }

    @Test
    fun aShortEnglishUtteranceInACzechSessionStaysCzech() {
        val native = FakeNative().apply { sharesInOrder(english) }
        val events = session(native, audio(1_500))
        assertThresholds(listOf(th(nan, nan, nan)), native)
        assertEquals(listOf("cs"), events.languages())
    }

    @Test
    fun aCzechUtteranceFromTheReturnMinimumReturnsTheSession() {
        val native = FakeNative().apply { sharesInOrder(english, czech, english) }
        val events = session(native, audio(3_000, 1_200, 1_500))
        assertThresholds(listOf(th(nan, nan, 0.9f), th(nan, nan, 0.9f), th(nan, nan, nan)), native)
        assertEquals(listOf("en", "cs", "cs"), events.languages())
    }

    @Test
    fun anEmptyResultLeavesTheSessionLanguage() {
        val native = FakeNative().apply {
            sharesInOrder(english, silence, english)
            texts += listOf("hello there", "", "and more")
        }
        val events = session(native, audio(3_000, 1_500, 800))
        assertEquals(listOf("hello there", "and more"), events.transcripts().map { it.text })
        assertEquals(listOf("en", "en"), events.languages())
        assertEquals(3, native.thresholdsPassed.size)
        assertArrayEquals(th(nan, nan, 0f), native.thresholdsPassed[2], 0f)
    }

    @Test
    fun everyCollectionStartsInTheFirstLanguage() {
        val native = FakeNative().apply { sharesInOrder(english, english) }
        val audios = listOf(audio(3_000), audio(1_500))
        var next = 0
        val engine = fakeEngine(native, config = config, languages = three, languageSwitch = switch) { ArrayAudioSource(audios[next++]) }
        val first = runBlocking { withContext(Dispatchers.Default) { engine.listen().toList() } }
        val second = runBlocking { withContext(Dispatchers.Default) { engine.listen().toList() } }
        assertEquals(listOf("en"), first.languages())
        assertEquals(listOf("cs"), second.languages())
        assertThresholds(listOf(th(nan, nan, 0.9f), th(nan, nan, nan)), native)
    }

    @Test
    fun transcribeHasNoSession() {
        val native = FakeNative().apply { sharesInOrder(english, english) }
        val engine = fakeEngine(native, config = config, languages = three, languageSwitch = switch) { ArrayAudioSource(FloatArray(0)) }
        runBlocking {
            engine.transcribe(Signals.tone(3.0))
            engine.transcribe(Signals.tone(1.5))
        }
        assertThresholds(listOf(th(nan, nan, 0.9f), th(nan, nan, nan)), native)
    }

    @Test
    fun aForcedLanguageIsOneCodeWithoutThresholdsSharesOrRescue() {
        for (withPreviews in listOf(false, true)) {
            val native = FakeNative().apply { sharesInOrder(english, english, english, english) }
            val engine = fakeEngine(native, config = config, languages = three, languageSwitch = switch) { ArrayAudioSource(audio(3_000, 1_500)) }
            val flow: Flow<ListenEvent> = if (withPreviews) engine.listenWithPreviews(PreviewConfig(), "en") else engine.listen("en")
            val events = runBlocking { withContext(Dispatchers.Default) { flow.toList() } }
            assertEquals(listOf("en", "en"), events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.language })
            assertTrue(native.languagesPassed.isNotEmpty() && native.languagesPassed.all { it == "en" })
            assertTrue(native.thresholdsPassed.all { it == null })
            assertEquals(0, native.sharesCalls.get())
        }
    }

    @Test
    fun withoutASwitchSeveralLanguagesKeepTheMostProbableRule() {
        val native = FakeNative().apply { sharesInOrder(floatArrayOf(0.3f, 0.7f)) }
        val events = session(native, audio(1_500), languages = listOf("cs", "en"), languageSwitch = null)
        assertEquals(listOf("cs,en"), native.languagesPassed.toList())
        assertThresholds(listOf(null), native)
        assertEquals(listOf("en"), events.languages())
    }

    @Test
    fun previewsUseTheSessionLanguageAndNeverChangeIt() {
        // an utterance that is long enough for a preview, which is English while the session is Czech, and whose final comes
        // out empty (so it does not move the session); the next one, 1.5 s, then has no candidate. The long one comes
        // first: a preview is not asked for while the final of an earlier utterance is pending.
        val native = FakeNative().apply {
            sharesOf = { _, _, endMs -> if (endMs < 7_500) english else czech }
            texts += listOf("hello preview", "")
        }
        val events = Collections.synchronizedList(mutableListOf<ListenEvent>())
        val hooks = object : SpeechEngine.Hooks {
            var requests = 0
            override fun previewRequested(samples: FloatArray) {
                requests++
                val deadline = System.nanoTime() + 10_000_000_000L
                while (events.none { it is SpeechPreview }) {
                    check(System.nanoTime() < deadline) { "no preview arrived" }
                    Thread.sleep(1)
                }
            }
        }
        val engine = fakeEngine(native, hooks, config = config, languages = three, languageSwitch = switch) {
            ArrayAudioSource(audio(8_000, 1_500))
        }
        runBlocking { withContext(Dispatchers.Default) { engine.listenWithPreviews(PreviewConfig()).collect { events += it } } }
        assertEquals(1, hooks.requests)
        assertEquals(listOf("hello preview"), events.filterIsInstance<SpeechPreview>().map { it.text }.filter { it.isNotEmpty() })
        assertEquals(listOf("cs"), events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.language })
        assertThresholds(listOf(th(nan, nan, 0.9f), th(nan, nan, 0.9f), th(nan, nan, nan)), native)
    }

    private fun joinsWithSmallLetter(native: FakeNative, first: String, second: String) {
        val tokenizer = Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm))
        fun seq(text: String) = (intArrayOf(tokenizer.startToken) + tokenizer.encode(text)).toList()
        val score = mapOf(
            SentenceBoundary.Choice.JOIN to -40.0, SentenceBoundary.Choice.JOIN_KEEP to -50.0,
            SentenceBoundary.Choice.SPLIT to -52.6, SentenceBoundary.Choice.COMMA to -47.0,
        )
        val bySequence = SentenceBoundary.languageModelTexts(first, second).entries.associate { (choice, text) -> seq(text) to score.getValue(choice) }
        native.lmScoreOf = { bySequence[it.toList()] ?: -100.0 }
    }

    @Test
    fun sentenceJoiningHappensOnlyBetweenTranscriptsInTheFirstLanguage() {
        val a = "Jasně, ale můžeš nějak."
        val b = "s tebou spojit."
        val joined = SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, b, "S tebou spojit.")
        val joining = config.copy(sentenceJoining = SentenceJoining(TestEnv.tinyLm))

        // as before: two Czech transcripts join
        val control = FakeNative().apply {
            sharesInOrder(czech, czech)
            texts += listOf(a, b)
            joinsWithSmallLetter(this, a, b)
        }
        val joinedEvents = session(control, audio(3_000, 1_500), config = joining).transcripts()
        assertEquals(joined, joinedEvents[1].sentenceJoin)

        // an English one in between: no decision for it, and the Czech after it is not joined to the Czech before it
        val native = FakeNative().apply {
            sharesInOrder(czech, english, czech)
            texts += listOf(a, "this is English", b)
            joinsWithSmallLetter(this, a, b)
        }
        val heard = session(native, audio(1_500, 3_000, 1_200), config = joining).transcripts()
        assertEquals(listOf("cs", "en", "cs"), heard.map { it.language })
        assertEquals(listOf(a, "this is English", b), heard.map { it.text })
        assertTrue("no sentence decision: ${heard.map { it.sentenceJoin }}", heard.all { it.sentenceJoin == null })
        assertTrue("the language model scored nothing", native.lmScoreCalls.isEmpty())
    }

    @Test
    fun variantSelectionChoosesWithTheLanguageModelOnlyInTheFirstLanguage() {
        val selecting = config.copy(variantSelection = VariantSelection(TestEnv.tinyLm))
        val tokenizer = Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm))
        val worse = " Viděl jsem velký psy."
        val better = " Viděl jsem velké psy."
        val native = FakeNative().apply {
            sharesInOrder(english, czech)
            hypotheses = listOf(listOf(Hyp(worse, -2.0, 9, best = true, score = -0.22), Hyp(better, -2.3, 9, score = -0.25)))
            lmScoreOf = { if (it.contentEquals(intArrayOf(tokenizer.startToken) + tokenizer.encode(better.trim()))) -40.0 else -50.0 }
        }
        val heard = session(native, audio(3_000, 1_200), config = selecting).transcripts()
        assertEquals(listOf("en", "cs"), heard.map { it.language })
        assertEquals("English keeps Whisper's own choice", worse.trim(), heard[0].text)
        assertEquals("the language model chooses in Czech", better.trim(), heard[1].text)
        assertEquals("one run of the language model, for the Czech one", 1, native.lmScoreCalls.size)
    }

    @Test
    fun logLinesCarryLanguageAndSharesButNoText() {
        val native = FakeNative().apply {
            sharesInOrder(english)
            texts += "supersecretwords"
        }
        session(native, audio(3_000))
        val line = logs.single { it.startsWith("utterance:") && it.contains("transcript kept") }
        assertTrue(line, line.contains("language en (cs 0.002, sk 0.001, en 0.997)"))
        assertTrue("the end of the session: $logs", logs.contains("language: 1 switches, 0 rescues"))
        assertTrue("no text in any line", logs.none { it.contains("supersecretwords") })

        logs.clear()
        session(FakeNative(), audio(1_500), language = "en")
        val forced = logs.single { it.startsWith("utterance:") && it.contains("transcript kept") }
        assertTrue(forced, forced.contains("language en (forced)"))
    }
}
