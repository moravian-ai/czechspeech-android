package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger

/**
 * Sentence joining of a streaming session in a model with several languages: the language model of the sentence decision
 * is the first language's, so another language's chunks are never joined, never given to the language model, and no
 * transcript continues another across them. The text is punctuated (a capital and a period at each chunk, as Whisper
 * writes it): text without punctuation joins nothing and would hide a join that should not be.
 */
class StreamingSentenceJoiningLanguageTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm), streaming = StreamingConfig())
    private val three = listOf("cs", "sk", "en")
    private val czech = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val english = floatArrayOf(0.002f, 0.001f, 0.997f)

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

    private class Hooks : SpeechEngine.Hooks {
        val requested = AtomicInteger()
        val finished = AtomicInteger()
        val boundaries = AtomicInteger()

        override fun streamingPassRequested(samples: FloatArray) {
            val n = requested.incrementAndGet()
            waitOnCapture("pass $n") { finished.get() >= n }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }

        override fun sentenceBoundaryComing() {
            boundaries.incrementAndGet()
        }
    }

    private companion object {
        fun waitOnCapture(what: String, condition: () -> Boolean) {
            val deadline = System.nanoTime() + 10_000_000_000L
            while (!condition()) {
                if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for $what")
                Thread.sleep(1)
            }
        }
    }

    private suspend fun collect(flow: Flow<ListenEvent>) = withContext(Dispatchers.Default) { flow.collect { events += it } }

    private fun withLog(block: () -> Unit): List<String> {
        val lines = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        try {
            block()
        } finally {
            CsLog.sink = sink
        }
        return lines.toList()
    }

    /** Speech of 3 s from each start (ms), silence between them, then 1.5 s of silence. */
    private fun audio(vararg startsMs: Long): FloatArray = audioOf(*startsMs.map { it to 3.0 }.toTypedArray())

    /** Speech of the given length (s) from each start (ms), silence between them, then 1.5 s of silence. */
    private fun audioOf(vararg pieces: Pair<Long, Double>): FloatArray {
        val parts = mutableListOf(Signals.zeros(0.5))
        var at = 500L
        for ((start, seconds) in pieces) {
            if (start > at) parts += Signals.zeros((start - at) / 1000.0)
            parts += Signals.clock(start, seconds)
            at = start + (seconds * 1000).toLong()
        }
        parts += Signals.zeros(1.5)
        return Signals.concat(*parts.toTypedArray())
    }

    /** `slovoN` words with a capital and a period on each chunk, as the model writes a sentence. */
    private fun FakeNative.punctuated() {
        clockWords = { startMs, endMs ->
            (0 until 80).map { 500L + 500L * it }
                .filter { it + 400 <= endMs && it >= startMs - 50 }
                .joinToString(" ") { "slovo${(it - 500) / 500}" }
                .let { if (it.isEmpty()) it else it.replaceFirstChar { c -> c.uppercase() } + "." }
        }
    }

    /**
     * One session over [audio]: a recognition whose audio starts within [englishAt] is in English, any other in Czech. The
     * capture waits for the language model at the frames [waitForLmAt] (it is loaded after the first utterance).
     */
    private fun run(
        audio: FloatArray,
        englishAt: LongRange?,
        waitForLmAt: List<Int>,
        prepare: (SpeechEngine) -> Unit = {},
    ): Triple<FakeNative, Hooks, List<String>> {
        val native = FakeNative().apply {
            punctuated()
            sharesOf = { _, startMs, _ -> if (englishAt != null && startMs in englishAt) english else czech }
            // a model that likes the small letter, as in StreamingSessionTest
            lmScoreOf = { ids -> -2.0 * ids.size + 0.05 * ids.sum() }
        }
        val hooks = Hooks()
        val source = ArrayAudioSource(audio, onFrame = { n -> if (n in waitForLmAt) waitOnCapture("the language model") { native.lmInits.get() >= 1 } })
        val engine = fakeEngine(native, hooks, config = config, languages = three) { source }
        prepare(engine)
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        return Triple(native, hooks, lines)
    }

    private fun Transcript.describe() = "$language@$startMs join=$sentenceJoin"

    @Test
    fun anEnglishPieceAfterACzechOneIsNeitherJoinedNorGivenToTheLanguageModel() {
        // Czech 0.5-3.5 s ending with a period, English 4.5-7.5 s within 3 s of it
        val (native, hooks, lines) = run(audio(500, 4_500), englishAt = 4_000L..60_000L, waitForLmAt = listOf(150))
        val english = transcripts.filter { it.language == "en" }
        assertTrue("Czech, then English: ${transcripts.map { it.describe() }}", english.isNotEmpty() && transcripts.first().language == "cs")
        assertTrue("a Czech piece ending with a period: ${transcripts.last { it.language == "cs" }.text}", transcripts.last { it.language == "cs" }.text.endsWith("."))
        assertTrue("no join for English: ${english.map { it.describe() }}", english.all { it.sentenceJoin == null })
        assertTrue("the language model was not consulted: ${native.lmScoreCalls.size} runs", native.lmScoreCalls.isEmpty())
        assertTrue("no sentence decision was made: $lines", lines.none { it.startsWith("sentence boundary:") })
        assertEquals("a boundary only for the first Czech piece", 1, hooks.boundaries.get())
    }

    @Test
    fun englishSpeechCutInsideIsNotContinuedAcrossTheCut() {
        val audio = Signals.concat(Signals.zeros(0.5), Signals.clock(500, 13.0), Signals.zeros(1.5))
        val (native, _, lines) = run(audio, englishAt = 0L..60_000L, waitForLmAt = emptyList()) { engine ->
            engine.streamingGuard!!.recorded(6_000, 3_000)                  // passes are paused: chunk mode
            engine.streamingGuard!!.recordedFinal(6_000, 2_250)             // pieces of 4 s
        }
        assertTrue("forced cuts: ${lines.filter { it.startsWith("streaming: forced cut") }}", lines.count { it.startsWith("streaming: forced cut at ") } >= 2)
        assertTrue("English throughout: ${transcripts.map { it.describe() }}", transcripts.size >= 3 && transcripts.all { it.language == "en" })
        assertTrue("no chunk is joined: ${transcripts.map { it.describe() }}", transcripts.all { it.sentenceJoin == null })
        assertTrue("no decision after a cut: $lines", lines.none { it.startsWith("sentence boundary:") })
        assertTrue(native.lmScoreCalls.isEmpty())
    }

    @Test
    fun czechAfterCzechIsDecidedByTheLanguageModelAsBefore() {
        val (native, hooks, lines) = run(audio(500, 4_500), englishAt = null, waitForLmAt = listOf(150))
        assertTrue(transcripts.all { it.language == "cs" })
        assertEquals("one decision, for the first chunk of the second piece", 1, lines.count { it.startsWith("sentence boundary: ") })
        assertTrue("the language model decided: ${native.lmScoreCalls.size} runs", native.lmScoreCalls.isNotEmpty())
        assertEquals("a boundary for each piece", 2, hooks.boundaries.get())
    }

    @Test
    fun czechAfterAnEnglishPieceDoesNotJoinToTheEnglishText() {
        // Czech 0.5-3.5 s, a short English interjection 4.5-5.1 s, Czech 6.1-9.1 s: the two Czech pieces are 2.6 s apart, within
        // SentenceBoundary.MAX_PAUSE_MS, so only SentenceJoiner.reset() on the English chunk keeps the second from joining the first
        val (native, hooks, lines) = run(audioOf(500L to 3.0, 4_500L to 0.6, 6_100L to 3.0), englishAt = 4_000L..5_500L, waitForLmAt = listOf(150))
        val order = transcripts.map { it.language }.fold(listOf<String>()) { acc, l -> if (acc.lastOrNull() == l) acc else acc + l }
        assertEquals("the languages in the order spoken", listOf("cs", "en", "cs"), order)
        assertTrue("no join at all: ${transcripts.map { it.describe() }}", transcripts.all { it.sentenceJoin == null })
        assertTrue("the language model was not consulted: ${native.lmScoreCalls.size} runs", native.lmScoreCalls.isEmpty())
        assertTrue("no sentence decision: $lines", lines.none { it.startsWith("sentence boundary:") })
        assertEquals("a boundary for each Czech piece, none for the English one", 2, hooks.boundaries.get())
    }
}
