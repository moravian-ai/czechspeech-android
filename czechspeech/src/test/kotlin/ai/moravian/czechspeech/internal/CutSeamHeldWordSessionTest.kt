package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections

/**
 * [CutSeam.CONTEXT_AND_HELD_WORD] in a [SpeechEngine] session on the fake native layer, chunk mode, 8 s of speech: two pieces, the
 * first cut inside speech. The recogniser says what the script says ([FakeNative.texts]): first the text of the first piece, then
 * the run with the context of the second, then (only after a fallback) the run on its own audio.
 */
class CutSeamHeldWordSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

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

    private fun run(native: FakeNative, hooks: Seen, seconds: Double = 8.0): List<String> {
        val audio = Signals.concat(Signals.zeros(0.5), Signals.clock(500, seconds), Signals.zeros(1.5))
        val config = CzechSpeechConfig(
            sentenceJoining = SentenceJoining(TestEnv.tinyLm),
            streaming = StreamingConfig(cutSeam = CutSeam.CONTEXT_AND_HELD_WORD),
        )
        val engine = fakeEngine(native, hooks, config = config) { ArrayAudioSource(audio) }
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

    private fun kinds(): List<String> = synchronized(events) {
        events.mapNotNull {
            when (it) {
                is SpeechEvent.Transcript -> "T:" + it.transcript.text
                is SpeechEvent.UtteranceDone -> "D"
                else -> null
            }
        }
    }

    @Test
    fun anAnchoredContextRunReplacesTheHeldWordAndTheDoneOfThePieceComesRightBeforeIt() {
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a6 a7 a8 b1 b2")) }
        val hooks = Seen()
        val lines = run(native, hooks)
        assertEquals(listOf("T:a1 a2 a3 a4 a5 a6 a7", "D", "T:a8 b1 b2", "D"), kinds())
        val second = transcripts[1]
        val join = second.sentenceJoin!!
        assertTrue(join.continuesPrevious)
        assertEquals(PreviousEnd.KEEP, join.previousEnd)
        assertEquals("a8 b1 b2", join.continuingText)
        assertEquals("one run per piece", 2, native.transcribeCalls.get())
        assertTrue(lines.toString(), lines.any { it.startsWith("utterance: seam context ") && it.contains("held word replaced") })
        assertEquals(1, hooks.stats!!.heldReplaced)
        assertTrue(hooks.released.all { a -> a.all { it == 0f } })
    }

    @Test
    fun theLogLinesOfTheHeldWordsCarryNoTranscriptText() {
        val words = Regex("""\b[ab][1-8]\b""")
        for (texts in listOf(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a6 a7 a8 b1 b2"), listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a7 a8 b1 b2"), listOf("a1 a2 a3 a4 a5 a6 a7 a8", "zz yy xx ww", "b1 b2 b3 b4"))) {
            val lines = run(FakeNative().apply { this.texts.addAll(texts) }, Seen())
            assertTrue(lines.any { it.startsWith("streaming: held word ") })
            assertTrue(lines.toString(), lines.none { words.containsMatchIn(it) || "zz" in it })
        }
    }

    @Test
    fun withoutAHeldAnchorTheAnchorAgainstAllTheWordsKeepsTheHeldWordAndCostsNoSecondRun() {
        // the held-word anchor needs two words of the delivered a1..a7 in front of a8: the run heard "a7 a8 b1 b2" only
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a7 a8 b1 b2")) }
        val hooks = Seen()
        val lines = run(native, hooks)
        assertEquals(listOf("T:a1 a2 a3 a4 a5 a6 a7", "T:a8", "D", "T:b1 b2", "D"), kinds())
        assertEquals("no run on the piece's own audio", 2, native.transcribeCalls.get())
        assertNotNull(transcripts[2].sentenceJoin)
        assertTrue(lines.toString(), lines.any { it.startsWith("streaming: held word kept (no anchor on the held word), delay ") })
        assertEquals(1, hooks.stats!!.heldKept)
        assertEquals(0, hooks.stats!!.seamsFallback)
    }

    @Test
    fun withNoAnchorAtAllTheHeldWordIsFlushedAndTheTailOnlyRunWritesTheRest() {
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "zz yy xx ww", "b1 b2 b3 b4")) }
        val hooks = Seen()
        val lines = run(native, hooks)
        assertEquals(listOf("T:a1 a2 a3 a4 a5 a6 a7", "T:a8", "D", "T:b1 b2 b3 b4", "D"), kinds())
        assertEquals(3, native.transcribeCalls.get())
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (no anchor), tail-only run follows"))
        assertEquals(1, hooks.stats!!.seamsFallback)
        assertTrue(hooks.released.all { a -> a.all { it == 0f } })
        assertTrue(native.passedToWhisper.all { a -> a.all { it == 0f } })
    }

    @Test
    fun aContextRunWhoseTextAfterTheAnchorIsOnlyTheHeldWordFallsBackAndKeepsTheWholePiece() {
        // Whisper stopped after the seam: the run heard "a6 a7 a8" and nothing of the piece's own speech. Path 1 would take the
        // held word as the new text and lose b1..b3 (the piece has seconds of speech of its own).
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a6 a7 a8", "b1 b2 b3")) }
        val hooks = Seen()
        val lines = run(native, hooks)
        assertEquals(listOf("T:a1 a2 a3 a4 a5 a6 a7", "T:a8", "D", "T:b1 b2 b3", "D"), kinds())
        assertEquals("the tail-only run followed", 3, native.transcribeCalls.get())
        assertTrue(lines.toString(), lines.contains("utterance: seam fallback (empty rest), tail-only run follows"))
        assertEquals(0, hooks.stats!!.heldReplaced)
        assertEquals(1, hooks.stats!!.heldKept)
        assertTrue(hooks.released.all { a -> a.all { it == 0f } })
    }

    @Test
    fun aPieceWithTwoWordsHoldsNothingAndItsDoneIsNotDeferred() {
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2", "a1 a2 b1 b2")) }
        val hooks = Seen()
        run(native, hooks)
        assertEquals(listOf("T:a1 a2", "D", "T:b1 b2", "D"), kinds())
        assertEquals(0, hooks.stats!!.heldReplaced + hooks.stats!!.heldKept)
    }

    @Test
    fun theSessionLineSaysHowTheHeldWordsEnded() {
        val native = FakeNative().apply { texts.addAll(listOf("a1 a2 a3 a4 a5 a6 a7 a8", "a6 a7 a8 b1 b2")) }
        val lines = run(native, Seen())
        val ended = lines.single { it.startsWith("streaming: ") && it.contains(" seams: ") }
        assertTrue(ended, ended.contains("held 1: replaced 1, kept 0"))
    }
}
