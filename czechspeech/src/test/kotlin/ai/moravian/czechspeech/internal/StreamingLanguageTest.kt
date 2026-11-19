package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger

/**
 * Streaming transcription together with the model's language list: every recognition of a streaming session (the
 * passes, the final recognition of a piece, the previews) takes the same language handling as a session without
 * streaming. One language is forced and nothing is asked afterwards (the path of a one-language model is unchanged);
 * several are passed as a list and the recogniser's pick is the language of the chunks.
 */
class StreamingLanguageTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val streaming = CzechSpeechConfig(streaming = StreamingConfig())

    /** 20 s of clock speech from 500 ms (words `slovo0`, `slovo1`, ...), as in [StreamingSessionTest]. */
    private val speech = Signals.concat(Signals.zeros(0.5), Signals.clock(500, 20.0), Signals.zeros(1.5))
    private val spoken = (0 until 40).map { "slovo$it" }

    private val clockWords: (Long, Long) -> String = { startMs, endMs ->
        (0 until 40).map { 500L + 500L * it }
            .filter { it + 400 <= endMs && it >= startMs - 50 }
            .joinToString(" ") { "slovo${(it - 500) / 500}" }
    }

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }
    private val words: List<String> get() = transcripts.flatMap { it.text.split(' ') }.filter { it.isNotEmpty() }

    /** Capture goes on only once each pass has been absorbed, so that the passes happen whatever the timing. */
    private class Paced : SpeechEngine.Hooks {
        val requested = AtomicInteger()
        val finished = AtomicInteger()

        override fun streamingPassRequested(samples: FloatArray) {
            val n = requested.incrementAndGet()
            val deadline = System.nanoTime() + 10_000_000_000L
            while (finished.get() < n) {
                if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for pass $n")
                Thread.sleep(1)
            }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }
    }

    private suspend fun collect(flow: Flow<ListenEvent>) = withContext(Dispatchers.Default) { flow.collect { events += it } }

    private fun engine(native: FakeNative, hooks: Paced, languages: List<String>) =
        fakeEngine(native, hooks, config = streaming, languages = languages) { ArrayAudioSource(speech) }

    /** The recognitions that ran: passes, previews and finals all go through whisperPreview or whisperTranscribe. */
    private fun recognitions(native: FakeNative) = native.previewCalls.get() + native.greedyCalls.get()

    @Test
    fun oneLanguageIsForcedInEveryPassAndFinalAndNeverAsked() = runBlocking {
        val native = FakeNative().apply { clockWords = this@StreamingLanguageTest.clockWords }
        val hooks = Paced()
        collect(engine(native, hooks, listOf("cs")).listen())
        assertTrue("passes ran: ${hooks.requested.get()}", hooks.requested.get() >= 2)
        assertTrue("a final ran", native.greedyCalls.get() >= 1)
        assertEquals("every recognition is forced to cs", List(recognitions(native)) { "cs" }, native.languagesPassed.toList())
        assertEquals("no detection, nothing asked afterwards", 0, native.whisperLanguageCalls.get())
        assertEquals(spoken, words)
        assertTrue("several chunks", transcripts.size >= 2)
        assertTrue(transcripts.all { it.language == "cs" })
    }

    @Test
    fun theOnlyLanguageOfAModelIsForcedWhateverItIs() = runBlocking {
        val native = FakeNative().apply { clockWords = this@StreamingLanguageTest.clockWords }
        collect(engine(native, Paced(), listOf("pl")).listen())
        assertTrue(native.languagesPassed.isNotEmpty() && native.languagesPassed.all { it == "pl" })
        assertEquals(0, native.whisperLanguageCalls.get())
        assertTrue(transcripts.isNotEmpty() && transcripts.all { it.language == "pl" })
    }

    @Test
    fun severalLanguagesAreDetectedInEveryRecognitionAndTheChunksSayWhich() = runBlocking {
        val native = FakeNative().apply {
            clockWords = this@StreamingLanguageTest.clockWords
            detectedLanguage = "en"
        }
        collect(engine(native, Paced(), listOf("cs", "sk", "en")).listen())
        assertEquals(List(recognitions(native)) { "cs,sk,en" }, native.languagesPassed.toList())
        assertEquals("the pick is read after every recognition", recognitions(native), native.whisperLanguageCalls.get())
        assertEquals(spoken, words)
        assertTrue("chunks of the passes and of the final", transcripts.size >= 2)
        assertTrue(transcripts.all { it.language == "en" })
    }

    @Test
    fun aChosenLanguageOfSeveralIsForcedInTheStreamingToo() = runBlocking {
        val native = FakeNative().apply { clockWords = this@StreamingLanguageTest.clockWords }
        collect(engine(native, Paced(), listOf("cs", "sk", "en")).listen("sk"))
        assertTrue(native.languagesPassed.isNotEmpty() && native.languagesPassed.all { it == "sk" })
        assertEquals(0, native.whisperLanguageCalls.get())
        assertTrue(transcripts.isNotEmpty() && transcripts.all { it.language == "sk" })
    }

    @Test
    fun previewsOfAStreamingSessionTakeTheChosenLanguageToo() = runBlocking {
        val native = FakeNative().apply { clockWords = this@StreamingLanguageTest.clockWords }
        collect(engine(native, Paced(), listOf("cs", "en")).listenWithPreviews(PreviewConfig(), "en"))
        assertTrue(native.previewCalls.get() >= 1)
        assertTrue(native.languagesPassed.all { it == "en" })
        assertTrue(transcripts.all { it.language == "en" })
    }

    @Test
    fun aPickOutsideTheListIsAFailureAndNoLanguageIsMadeUp() = runBlocking {
        val native = FakeNative().apply {
            clockWords = this@StreamingLanguageTest.clockWords
            detectedLanguage = "de"
        }
        collect(engine(native, Paced(), listOf("cs", "en")).listen())
        assertTrue("an Error was reported", events.any { it is SpeechEvent.Error })
        assertTrue("no transcript carries a language outside the list", transcripts.all { it.language in listOf("cs", "en") })
        assertTrue("no chunk was written in a language that was never detected", transcripts.isEmpty())
    }

    // ---- the language rule with streaming ---------------------------------------------------------------------------

    private val nan = Float.NaN
    private val czech = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val english = floatArrayOf(0.002f, 0.001f, 0.997f)
    private val switch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)
    private val ruled = CzechSpeechConfig(streaming = StreamingConfig(), minEncoderWindow = 384)

    /** Czech words carry the prefix `x`, English ones do not: the text of a chunk says which language it was recognised in. */
    private val prefixInCzech: (String, String) -> String = { language, text ->
        if (language == "cs") text.split(' ').filter { it.isNotEmpty() }.joinToString(" ") { "x$it" } else text
    }

    private fun ruledEngine(native: FakeNative, hooks: SpeechEngine.Hooks, audio: FloatArray = speech, switch: CzechSpeechModel.LanguageSwitch = this.switch) =
        fakeEngine(native, hooks, config = ruled, languages = listOf("cs", "sk", "en"), languageSwitch = switch) { ArrayAudioSource(audio) }

    /**
     * English speech in a Czech session: the passes shorter than the switch's minLeaveMs have no candidate but Czech, so they
     * run in Czech over English speech and write nonsense (here the `x` prefix), which two such passes agree on. With
     * minLeaveMs of 3 s there are several of them, so only the rule that a pass shorter than that does not decide
     * (and does not let its piece commit) keeps that nonsense from being written.
     */
    @Test
    fun englishSpeechInACzechSessionNeverCommitsCzechWords() = runBlocking {
        val native = FakeNative().apply {
            clockWords = this@StreamingLanguageTest.clockWords
            sharesOf = { _, _, _ -> english }
            textIn = prefixInCzech
        }
        val longLeave = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 3_000, minReturnMs = 1_000)
        collect(ruledEngine(native, Paced(), switch = longLeave).listen())
        val shortCzechPasses = native.thresholdsPassed.count { it != null && it.all { t -> t.isNaN() } }
        assertTrue("several passes ran in Czech over English speech: $shortCzechPasses", shortCzechPasses >= 2)
        assertTrue("no Czech word was written: $words", words.none { it.startsWith("x") })
        assertEquals(spoken, words)
        assertTrue("several chunks: ${transcripts.size}", transcripts.size >= 2)
        assertTrue(transcripts.all { it.language == "en" })
    }

    /**
     * One pass in English among Czech ones: the words are the same in both languages, so only the rule that a pass commits
     * when it and the pass before ran in the same language keeps the deviating pass from writing a chunk in English.
     */
    @Test
    fun oneDeviatingPassChangesNothing() = runBlocking {
        val native = FakeNative().apply {
            clockWords = this@StreamingLanguageTest.clockWords
            sharesOf = { index, _, _ -> if (index == 3) english else czech }       // the fourth pass only
        }
        collect(ruledEngine(native, Paced()).listen())
        // not vacuous: the deviating pass really was a candidate for English (a pass too short to leave Czech would change nothing
        // whatever the rules were)
        val deviating = native.thresholdsPassed.getOrNull(3)
        assertTrue("the fourth pass had an English threshold: ${native.thresholdsPassed.map { it?.toList() }}", deviating != null && !deviating[2].isNaN())
        assertEquals(spoken, words)
        assertTrue(transcripts.size >= 2)
        assertTrue("all chunks are Czech: ${transcripts.map { it.language }}", transcripts.all { it.language == "cs" })
    }

    @Test
    fun aPieceThatCommittedWordsHasItsFinalForcedInThePieceLanguage() = runBlocking {
        val native = FakeNative().apply {
            clockWords = this@StreamingLanguageTest.clockWords
            sharesOf = { _, _, _ -> english }
            textIn = prefixInCzech
        }
        collect(ruledEngine(native, Paced()).listen())
        assertTrue("the passes wrote words before the final", transcripts.size >= 2)
        assertEquals("the final is forced in the language of the piece", "en", native.languagesPassed.last())
        assertNull("without thresholds: no recognition, no rescue", native.thresholdsPassed.last())
        assertTrue("the passes had thresholds", native.thresholdsPassed.dropLast(1).all { it != null })
    }

    /** Thresholds and list of the first pass of the second piece of two (the first one ends the session in the language of [shares]). */
    private fun firstPassOfTheNextPiece(shares: FloatArray): Pair<FloatArray, String> = runBlocking {
        events.clear()
        val audio = Signals.concat(Signals.zeros(0.5), Signals.clock(500, 6.0), Signals.zeros(1.0), Signals.clock(7_500, 3.0), Signals.zeros(1.5))
        val native = FakeNative().apply {
            clockWords = { from, to -> (0 until 40).map { 500L + 500L * it }.filter { it + 400 <= to && it >= from - 50 }.joinToString(" ") { "slovo$it" } }
            sharesOf = { _, _, _ -> shares }
        }
        // the second piece starts only once the first is done with, so that its first pass is the 1.3 s one
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 230) {
                val deadline = System.nanoTime() + 10_000_000_000L
                while (events.none { it is SpeechEvent.UtteranceDone }) {
                    if (System.nanoTime() > deadline) throw AssertionError("the first piece never ended")
                    Thread.sleep(1)
                }
            }
        })
        collect(fakeEngine(native, Paced(), config = ruled, languages = listOf("cs", "sk", "en"), languageSwitch = switch) { source }.listen())
        val thresholds = native.thresholdsPassed.toList()
        val finalOfTheFirstPiece = thresholds.indexOfFirst { it == null }
        assertTrue("the first piece's final is forced: ${thresholds.map { it?.toList() }}", finalOfTheFirstPiece in 1 until thresholds.size - 1)
        thresholds[finalOfTheFirstPiece + 1]!! to native.languagesPassed[finalOfTheFirstPiece + 1]
    }

    @Test
    fun theSessionLanguageCarriesToTheNextPiece() {
        val (thresholds, languages) = firstPassOfTheNextPiece(english)
        assertArrayEquals("the first pass of the next piece: the session is in English", floatArrayOf(nan, nan, 0.9f), thresholds, 0f)
        assertEquals("cs,sk,en", languages)
        // in a Czech session the same 1.3 s have no candidate
        assertArrayEquals(floatArrayOf(nan, nan, nan), firstPassOfTheNextPiece(czech).first, 0f)
    }
}
