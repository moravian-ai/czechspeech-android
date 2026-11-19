package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.Dispatchers
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

/**
 * Pieces cut inside speech in a [SpeechEngine] session ([CutSeam]), on the fake native layer, in chunk mode (the guard
 * says passes are too dear, so the text comes from the finals of pieces of 4 to 5.5 s). "Speech" is [Signals.clock] from 500 ms:
 * the word `slovo<k>` over [500 + 500k, 900 + 500k) ms; the fake recogniser returns the words that lie in the audio it
 * gets, or, where a test scripts it, the next text of [FakeNative.texts].
 */
class CutSeamSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

    private fun speech(seconds: Double) = Signals.concat(Signals.zeros(0.5), Signals.clock(500, seconds), Signals.zeros(1.5))

    private fun clockWords(seconds: Double): (Long, Long) -> String = { startMs, endMs ->
        (0 until (seconds * 2).toInt()).map { 500L + 500L * it }
            .filter { it + 400 <= endMs && it >= startMs - 50 }
            .joinToString(" ") { "slovo${(it - 500) / 500}" }
    }

    private fun allWords(seconds: Double) = (0 until (seconds * 2).toInt()).map { "slovo$it" }

    private class Seen : SpeechEngine.Hooks {
        val kept: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
        val released: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
        @Volatile var stats: StreamingStats? = null

        override fun contextReleased(samples: FloatArray) {
            released += samples
        }

        override fun contextKept(samples: FloatArray) {
            kept += samples
        }

        override fun streamingEnded(stats: StreamingStats) {
            this.stats = stats
        }
    }

    private fun config(cutSeam: CutSeam, maxLagMs: Int = 15_000, selection: VariantSelection? = null) =
        CzechSpeechConfig(minEncoderWindow = 384, variantSelection = selection, streaming = StreamingConfig(maxLagMs = maxLagMs, cutSeam = cutSeam))

    /** Runs a session over [audio]; returns the log lines. The guard says: passes paused, pieces of 4 s. */
    private fun run(
        native: FakeNative, hooks: Seen, cutSeam: CutSeam, audio: FloatArray, maxLagMs: Int = 15_000, selection: VariantSelection? = null,
    ): List<String> {
        val engine = fakeEngine(native, hooks, config = config(cutSeam, maxLagMs, selection)) { ArrayAudioSource(audio) }
        engine.streamingGuard!!.recorded(6_000, 3_000)
        engine.streamingGuard!!.recordedFinal(6_000, 2_250)
        val lines = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        try {
            runBlocking { withContext(Dispatchers.Default) { engine.listen().collect { events += it } } }
        } finally {
            CsLog.sink = sink
        }
        return lines.toList()
    }

    private fun allZero(arrays: Collection<FloatArray>) = arrays.all { a -> a.all { it == 0f } }

    @Test
    fun aChainOfCutPiecesWritesEveryWordOnceAndInOrder() {
        val native = FakeNative().apply { clockWords = clockWords(13.0) }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(13.0))
        val texts = transcripts
        assertTrue("pieces were cut: ${texts.size}", texts.size >= 3)
        assertEquals(allWords(13.0), texts.flatMap { it.text.split(' ') }.filter { it.isNotEmpty() })
        val contexts = lines.filter { it.startsWith("utterance: seam context ") }
        assertEquals("every piece after a cut ran with the context: $lines", texts.size - 1, contexts.size)
        assertTrue(contexts.joinToString("\n"), contexts.all { it.contains(" ms + tail ") && it.contains(" anchor ") && it.contains("new text after") })
        assertTrue("no fallback: $lines", lines.none { it.contains("seam fallback") })
        val ended = lines.single { it.startsWith("streaming: ") && it.contains(" seams: ") }
        assertTrue(ended, ended.contains("seams: ${texts.size - 1} with context (${texts.size - 1} anchored, 0 fallback, 0 ms of fallback runs"))
        assertEquals(texts.size - 1, hooks.stats!!.seamsAnchored)
    }

    @Test
    fun noAudioOfThePiecesSurvivesTheSession() {
        val native = FakeNative().apply { clockWords = clockWords(13.0) }
        val hooks = Seen()
        run(native, hooks, CutSeam.CONTEXT, speech(13.0))
        val runs = transcripts.size - 1
        assertTrue("an array for the context and one for the joined run of every piece", hooks.released.size >= 2 * runs)
        assertTrue("every array the context lived in was zeroed", allZero(hooks.released))
        assertTrue("a context was kept for every continuation: ${hooks.kept.size} of $runs", hooks.kept.size >= runs)
        assertTrue("every context the tracker ever kept is zeroed at the end", allZero(hooks.kept))
        assertTrue("everything the recogniser got is zeroed", allZero(native.passedToWhisper))
        assertTrue("some run had the context in it: ${native.passedToWhisper.map { it.size }}", native.passedToWhisper.any { it.size >= 40_000 + 3 * 16_000 })
    }

    @Test
    fun withNoneNothingIsKeptAndTheRunsAreTheOldOnes() {
        val native = FakeNative().apply { clockWords = clockWords(13.0) }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.NONE, speech(13.0))
        assertTrue(transcripts.size >= 3)
        assertTrue("no context run: $lines", lines.none { it.contains("seam context") || it.contains("seam fallback") || it.contains("seams:") })
        assertTrue(hooks.released.isEmpty())
        assertEquals("one final per piece", transcripts.size, native.transcribeCalls.get())
        assertFalse("the session line says nothing of seams", lines.any { it.startsWith("streaming: ") && it.contains("seams:") })
    }

    @Test
    fun withoutAnAnchorTheTextIsTheOneOfTheTailOnlyRunAndTheSecondRunIsCounted() {
        val native = FakeNative().apply {
            texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "zz yy xx ww", "b1 b2 b3 b4 b5 b6"))
        }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(8.0))
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 b3 b4 b5 b6"), transcripts.map { it.text })
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (no anchor), tail-only run follows"))
        assertEquals("a context run and a tail run for the second piece, one run for the first", 3, native.transcribeCalls.get())
        assertEquals(1, hooks.stats!!.seamsFallback)
        assertEquals(0, hooks.stats!!.seamsAnchored)
        assertTrue(allZero(hooks.released))
        assertTrue(allZero(native.passedToWhisper))
    }

    @Test
    fun anAnchoredContextRunDeliversOnlyTheWordsAfterTheAnchor() {
        val native = FakeNative().apply {
            // the piece before sent a1 .. a8; the context covers the last words of it, then the new ones
            texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a6 a7 a8 b1 b2 b3"))
        }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(8.0))
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 b3"), transcripts.map { it.text })
        assertTrue(lines.toString(), lines.any { it.startsWith("utterance: seam context ") && it.contains("anchor 3 of 3 word(s) matched, new text after 3 word(s)") })
        assertEquals("one run per piece", 2, native.transcribeCalls.get())
    }

    @Test
    fun anEmptyContextRunFallsBackAndTheLadderSkips512ButTriesTheFullWindow() {
        val native = FakeNative().apply {
            // the piece before, the empty context run, the empty tail run, then the run on the full window finds the words
            texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "", "", "d1 d2 d3"))
        }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(8.0), maxLagMs = 30_000)
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "d1 d2 d3"), transcripts.map { it.text })
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (empty), tail-only run follows"))
        assertFalse("no repeat on 512 frames after the context run: ${native.minAudioCtxSet}", 512 in native.minAudioCtxSet)
        assertTrue("the repeat on the full window: ${native.minAudioCtxSet}", 1488 in native.minAudioCtxSet)
    }

    @Test
    fun aContextRunWithTheWrongWordsNeverMakesTheSecondPieceWriteTheFirstOnesWordsAgain() {
        val native = FakeNative().apply {
            // the context run hears only the old words: nothing new after the anchor
            texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a7 a8", "b1 b2 b3"))
        }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(8.0))
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 b3"), transcripts.map { it.text })
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (empty rest), tail-only run follows"))
    }

    @Test
    fun aFullWindowRepeatThatWouldEndPastTheLagLimitIsNotTried() {
        val native = FakeNative().apply {
            // the piece before, the empty context run, the empty tail run: the repeat on the whole window (estimated at 11 s and more) would end past 10 s
            texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "", "", "d1 d2 d3"))
        }
        val lines = run(native, Seen(), CutSeam.CONTEXT, speech(8.0), maxLagMs = 10_000)
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8"), transcripts.map { it.text })
        assertFalse("no repeat on the full window: ${native.minAudioCtxSet}", 1488 in native.minAudioCtxSet)
        assertFalse("nor on 512 frames: ${native.minAudioCtxSet}", 512 in native.minAudioCtxSet)
        assertEquals("the piece before, the context run and the tail run only", 3, native.transcribeCalls.get())
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (empty), tail-only run follows"))
    }

    @Test
    fun theNewLogLinesOfTheSeamsCarryNoTranscriptText() {
        val native = FakeNative().apply { clockWords = clockWords(13.0) }
        val lines = run(native, Seen(), CutSeam.CONTEXT, speech(13.0))
        assertTrue(lines.any { it.startsWith("utterance: seam context ") })
        assertTrue(lines.toString(), lines.none { "slovo" in it })
        val fallback = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "zz yy xx ww", "b1 b2 b3 b4 b5 b6")) }
        val fallbackLines = run(fallback, Seen(), CutSeam.CONTEXT, speech(8.0))
        assertTrue(fallbackLines.any { it.startsWith("utterance: seam fallback") })
        assertTrue(fallbackLines.toString(), fallbackLines.none { Regex("""\b(a[1-8]|b[1-6]|zz|yy|xx|ww)\b""").containsMatchIn(it) })
    }

    /**
     * Variant selection on: every run (the piece's, the context's) is chosen among the beams by the language model, and the anchor works
     * on the text that was chosen, not on Whisper's first choice.
     */
    @Test
    fun withVariantSelectionTheAnchorWorksOnTheChosenTextOfTheContextRun() {
        val tokenizer = Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.tinyLm))
        fun seq(text: String) = intArrayOf(tokenizer.startToken) + tokenizer.encode(text)
        val chosen = seq("a6 a7 a8 b1 b2 c3")
        val native = FakeNative().apply {
            hypothesesQueue.add(listOf(listOf(Hyp(" a1 a2 a3 a4 a5 a6 a7 a8", -1.0, 9, best = true, score = -0.1))))
            // Whisper prefers "b3", the language model "c3": the words after the anchor are the model's
            hypothesesQueue.add(listOf(listOf(Hyp(" a6 a7 a8 b1 b2 b3", -2.0, 7, best = true, score = -0.22), Hyp(" a6 a7 a8 b1 b2 c3", -2.3, 7, score = -0.25))))
            lmScoreOf = { if (it.contentEquals(chosen)) -40.0 else -50.0 }
        }
        val hooks = Seen()
        val lines = run(native, hooks, CutSeam.CONTEXT, speech(8.0), selection = VariantSelection(TestEnv.tinyLm, beamSize = 5))
        assertEquals(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "b1 b2 c3"), transcripts.map { it.text })
        assertEquals(listOf(5, 5), native.beamSizes)
        assertEquals(0, native.greedyCalls.get())
        assertTrue("the language model scored the beams of the context run", native.lmScoreCalls.isNotEmpty())
        assertTrue(lines.toString(), lines.any { it.startsWith("utterance: seam context ") && it.contains("new text after 3 word(s)") })
        assertTrue(allZero(hooks.released))
        assertTrue(allZero(hooks.kept))
    }
}
