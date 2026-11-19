package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections

/**
 * The context of a cut inside speech ([CutSeam]) in a session with several languages and a language rule (the dictation app's:
 * cs, sk, en, English only from a share of 0.9): the context run is forced in the language of the piece before, so a chain of cut
 * pieces stays in the language of its first piece, and a language change is seen again after a silence (a new first run, which the
 * rule decides). A rescued piece hands no context on.
 */
class CutSeamLanguageSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

    private val three = listOf("cs", "sk", "en")
    private val switch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)
    private val czech = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val english = floatArrayOf(0.002f, 0.001f, 0.997f)

    private class Seen : SpeechEngine.Hooks {
        val released: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
        @Volatile var stats: StreamingStats? = null
        override fun contextReleased(samples: FloatArray) {
            released += samples
        }

        override fun streamingEnded(stats: StreamingStats) {
            this.stats = stats
        }
    }

    private fun run(native: FakeNative, hooks: Seen, audio: FloatArray, language: String? = null): List<String> {
        val config = CzechSpeechConfig(minEncoderWindow = 384, streaming = StreamingConfig(cutSeam = CutSeam.CONTEXT))
        val engine = fakeEngine(native, hooks, config = config, languages = three, languageSwitch = switch) { ArrayAudioSource(audio) }
        engine.streamingGuard!!.recorded(6_000, 3_000)
        engine.streamingGuard!!.recordedFinal(6_000, 2_250)
        val lines = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        try {
            runBlocking { withContext(Dispatchers.Default) { engine.listen(language).collect { events += it } } }
        } finally {
            CsLog.sink = sink
        }
        return lines.toList()
    }

    /** The words `slovo<k>` at 500 + 500k ms, for the clock speech of [audio]: 9 s, 1.5 s of silence, 3 s more. */
    private val clockWords: (Long, Long) -> String = { startMs, endMs ->
        ((0 until 18) + (21 until 27)).map { 500L + 500L * it }
            .filter { it + 400 <= endMs && it >= startMs - 50 }
            .joinToString(" ") { "slovo${(it - 500) / 500}" }
    }

    private val chainThenSilenceThenMore = Signals.concat(
        Signals.zeros(0.5), Signals.clock(500, 9.0), Signals.zeros(1.5), Signals.clock(11_000, 3.0), Signals.zeros(1.5),
    )

    @Test
    fun aChainOfCutPiecesStaysInTheLanguageOfItsFirstPieceAndTheRuleDecidesAgainAfterASilence() {
        // the first piece's own run is English; every later run would be Czech if the rule decided it
        val native = FakeNative().apply {
            clockWords = this@CutSeamLanguageSessionTest.clockWords
            sharesOf = { index, _, _ -> if (index == 0) english else czech }
        }
        val hooks = Seen()
        run(native, hooks, chainThenSilenceThenMore)
        val before = transcripts.filter { it.startMs < 9_500 }
        val after = transcripts.filter { it.startMs >= 9_500 }
        assertTrue("a chain of at least two pieces before the silence: ${before.size}", before.size >= 2)
        assertTrue("every piece of the chain is in English: ${before.map { it.language }}", before.all { it.language == "en" })
        val passed = native.languagesPassed.toList()
        assertEquals("the first run decides among the languages", "cs,sk,en", passed.first())
        assertEquals("the context run of the second piece is forced in the first piece's language", "en", passed[1])
        assertTrue("the context runs are forced: $passed", passed.drop(1).take(before.size - 1).all { it == "en" })
        assertTrue("the speech after the silence was recognised: $transcripts", after.isNotEmpty())
        assertEquals("after the silence the rule decides again: $passed", "cs,sk,en", passed[before.size])
        assertTrue("and finds Czech: ${after.map { it.language }}", after.first().language == "cs")
        assertTrue(hooks.released.all { a -> a.all { it == 0f } })
    }

    @Test
    fun anAskedLanguageForcesEveryRunIncludingTheContextRuns() {
        val native = FakeNative().apply {
            clockWords = this@CutSeamLanguageSessionTest.clockWords
            sharesOf = { _, _, _ -> english }
        }
        run(native, Seen(), chainThenSilenceThenMore, language = "sk")
        assertTrue("every run was forced: ${native.languagesPassed}", native.languagesPassed.all { it == "sk" })
        assertTrue(transcripts.all { it.language == "sk" })
    }

    @Test
    fun aPieceWhoseTextCameFromARescueHandsNoContextToTheNextOne() {
        // the first run over the first piece is empty and its language is Czech by the rule, but English has 0.6: the rescue in English writes the text
        val native = FakeNative().apply {
            sharesOf = { index, _, _ -> if (index == 0) floatArrayOf(0.4f, 0f, 0.6f) else czech }
            texts.addAll(listOf("", "a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 b3 b4 b5 b6"))
        }
        val hooks = Seen()
        val audio = Signals.concat(Signals.zeros(0.5), Signals.clock(500, 8.0), Signals.zeros(1.5))
        val lines = run(native, hooks, audio)
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 b3 b4 b5 b6"), transcripts.map { it.text })
        assertEquals(listOf("en", "cs"), transcripts.map { it.language })
        assertTrue(lines.toString(), lines.contains("streaming: seam context released unused (rescued)"))
        assertTrue("no context run: $lines", lines.none { it.startsWith("utterance: seam context") || it.startsWith("utterance: seam fallback") })
        assertEquals(0, hooks.stats!!.seamsWithContext)
        assertEquals(1, hooks.stats!!.seamsUnused)
        assertTrue(hooks.released.all { a -> a.all { it == 0f } })
        assertTrue(native.passedToWhisper.all { a -> a.all { it == 0f } })
    }
}
