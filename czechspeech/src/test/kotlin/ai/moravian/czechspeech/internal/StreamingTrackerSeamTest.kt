package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.StreamingConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The seam context of the tracker ([CutSeam.CONTEXT]): the last 2.5 s of a piece cut inside speech are kept in RAM for the
 * piece that continues it, handed over in the final plan, and zeroed on every path. Played by hand, in virtual time.
 */
class StreamingTrackerSeamTest {
    private val events = mutableListOf<ListenEvent>()
    private val released = mutableListOf<FloatArray>()

    private fun tracker(cutSeam: CutSeam = CutSeam.CONTEXT, guard: StreamingGuard? = null): StreamingTracker {
        val config = StreamingConfig(cutSeam = cutSeam)
        val t = if (guard == null) {
            StreamingTracker(config, 384, false) { events += it }
        } else {
            StreamingTracker(config, 384, false, guard, null) { events += it }
        }
        t.onContextReleased = { released += it }
        return t
    }

    private fun frame(t: StreamingTracker, pieceMs: Long, audioStartMs: Long, positionMs: Long) =
        t.onFrame(pieceMs, audioStartMs, positionMs) { FloatArray(0) }

    /** Samples whose values say where they are (never zero), so a copy can be compared to the right slice. */
    private fun utt(startMs: Long, endMs: Long, quiet: Boolean = false) =
        Utterance(FloatArray(((endMs - startMs) * 16).toInt()) { 0.001f + (it % 997) / 1000f }, startMs, endMs, audioStartMs = startMs, quietCut = quiet)

    private fun words(n: Int, from: Int = 1) = (from until from + n).joinToString(" ") { "w$it" }

    /** The piece [startMs, endMs) ends (a cut inside speech unless [bySilence] or [quiet]) and its final delivers [text]. */
    private fun finishPiece(t: StreamingTracker, startMs: Long, endMs: Long, text: String, quiet: Boolean = false, bySilence: Boolean = false, recogniserMs: Long = 0): Utterance {
        frame(t, startMs, startMs, endMs)
        if (bySilence) t.pieceEnded()
        val u = utt(startMs, endMs, quiet)
        t.finalComing(u)
        val plan = t.finalPlan(u)
        t.absorbFinal(plan, text, recogniserMs)
        t.finalDone(u, true)
        return u
    }

    private fun assertAllZero(what: String, arrays: List<FloatArray>) {
        assertTrue("$what: ${arrays.size} arrays", arrays.isNotEmpty())
        assertTrue("$what zeroed", arrays.all { a -> a.all { it == 0f } })
    }

    @Test
    fun theContinuationOfACutInsideSpeechGetsTheLast2500MsOfThePieceBefore() {
        val t = tracker()
        val first = finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan = t.finalPlan(second)
        val context = assertNotNull(plan.context).let { plan.context!! }
        assertEquals(40_000, context.size)
        assertTrue("the end of the first piece", context.contentEquals(first.samples.copyOfRange(first.samples.size - 40_000, first.samples.size)))
        assertEquals(6, plan.seam!!.pieceWords)
        assertEquals(4_000L, plan.seam!!.pieceAudioMs)
        assertEquals(TEST_LANGUAGE, plan.seam!!.language)
        assertEquals("expected words in the context: 6 * 2.5 / 4 = 3.75 -> 4", 4, plan.seam!!.expected)
        val run = plan.contextRunSamples()
        assertEquals(40_000 + 64_000, run.size)
        assertEquals(run.size, plan.firstRunSamples)
        assertTrue(run.copyOfRange(0, 40_000).contentEquals(context))
        assertTrue(run.copyOfRange(40_000, run.size).contentEquals(second.samples))
        assertEquals("samples() is still the tail alone", 64_000, plan.samples().size)
    }

    @Test
    fun aShortPieceKeepsAllItsAudioAsContext() {
        val t = tracker()
        finishPiece(t, 1_000, 2_000, words(3))
        frame(t, 2_000, 2_000, 2_032)
        val second = utt(2_000, 6_000)
        t.finalComing(second)
        assertEquals(16_000, t.finalPlan(second).context!!.size)
    }

    @Test
    fun noContextAfterAQuietCutNorAfterSilenceNorWithNone() {
        for ((name, run) in listOf<Pair<String, (StreamingTracker) -> Unit>>(
            "quiet cut" to { t -> finishPiece(t, 1_000, 5_000, words(6), quiet = true) },
            "silence" to { t -> finishPiece(t, 1_000, 5_000, words(6), bySilence = true) },
        )) {
            val t = tracker()
            run(t)
            frame(t, 5_000, 5_000, 5_032)
            val second = utt(5_000, 9_000)
            t.finalComing(second)
            val plan = t.finalPlan(second)
            assertNull(name, plan.context)
            assertNull(name, plan.seam)
            assertEquals(name, 64_000, plan.firstRunSamples)
            assertTrue("$name: nothing kept, nothing to zero", released.isEmpty())
        }
        val none = tracker(CutSeam.NONE)
        finishPiece(none, 1_000, 5_000, words(6))
        frame(none, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        none.finalComing(second)
        assertNull(none.finalPlan(second).context)
        assertTrue(released.isEmpty())
    }

    @Test
    fun aPieceThatDeliveredFewerThanTwoWordsKeepsNothingUsable() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, "jedno")
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        assertNull(t.finalPlan(second).context)
        assertAllZero("one word is no anchor", released)
        assertEquals(1, released.size)
    }

    @Test
    fun aPieceThatDeliveredNothingKeepsNothing() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, "")
        assertTrue(released.isEmpty())
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        assertNull(t.finalPlan(second).context)
    }

    @Test
    fun theContextIsZeroedWhenTheNextFinalDoesNotContinueIt() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        val other = utt(7_000, 9_000)       // after silence: another start
        frame(t, 7_000, 7_000, 7_032)
        t.finalComing(other)
        assertNull(t.finalPlan(other).context)
        assertAllZero("not continued", released)
        assertEquals(1, released.size)
        assertEquals(1, t.stats().seamsUnused)
    }

    @Test
    fun theContextIsZeroedWhenTheContinuationIsSkipped() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        t.finalNotComing(second)
        assertAllZero("skipped", released)
    }

    @Test
    fun theContextIsZeroedAsSoonAsNoPieceContinuesIt() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        assertTrue("the continuation may still start", released.isEmpty())
        frame(t, 5_000, 5_000, 5_032)          // the continuation exists: kept
        assertTrue(released.isEmpty())
        frame(t, -1, 0, 5_064)                  // it vanished (dropped as too short): zeroed at this frame
        assertAllZero("dropped", released)
    }

    @Test
    fun aSeamNobodyContinuesIsZeroedAtOnceWhenThePieceInProgressIsAnother() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 8_000, 8_000, 8_032)
        assertAllZero("another piece", released)
    }

    @Test
    fun endOfCaptureZeroesTheContextOfAPieceNobodyContinuesButNotOneWhoseFinalWaits() {
        val lonely = tracker()
        finishPiece(lonely, 1_000, 5_000, words(6))
        lonely.end()
        assertAllZero("nobody continues", released)

        released.clear()
        val waiting = tracker()
        finishPiece(waiting, 1_000, 5_000, words(6))
        frame(waiting, 5_000, 5_000, 5_032)
        val second = utt(5_000, 7_000)
        waiting.finalComing(second)            // its final waits in the queue
        waiting.end()
        assertTrue("still wanted", released.isEmpty())
        val plan = waiting.finalPlan(second)
        assertNotNull(plan.context)
        waiting.releaseContext(plan.context)
        assertAllZero("used", released)
    }

    @Test
    fun teardownZeroesWhatIsLeft() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        t.discardRequests()
        assertAllZero("teardown", released)
    }

    @Test
    fun aFinalAbsorbedAfterTheTeardownKeepsNoAudio() {
        val t = tracker()
        frame(t, 1_000, 1_000, 5_000)
        val u = utt(1_000, 5_000)
        t.finalComing(u)
        val plan = t.finalPlan(u)
        t.discardRequests()                       // the session is torn down while the whisper task runs
        t.absorbFinal(plan, words(6), 0)         // ... and the task goes on to absorb its text
        assertEquals("no array of the piece's audio is kept", 0, t.retainedContextArrays())
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        assertNull(t.finalPlan(second).context)
        assertTrue("nothing was copied, so nothing is left unzeroed", released.isEmpty())
    }

    @Test
    fun theEngineZeroesThePlansContextThroughReleaseContext() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan = t.finalPlan(second)
        val array = plan.context!!
        val run = plan.contextRunSamples()
        t.releaseContext(run)
        t.releaseContext(array)
        assertAllZero("context run buffer and context", released)
        assertEquals(2, released.size)
        assertSame(run, released[0])
        assertSame(array, released[1])
        assertTrue(plan.context!!.all { it == 0f })
    }

    @Test
    fun aChainKeepsAContextFromEachPiecesOwnAudioAndTheDeliveredWordsOfTheLastOne() {
        val t = tracker()
        finishPiece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan2 = t.finalPlan(second)
        assertNotNull(plan2.context)
        t.absorbFinal(plan2, words(5, from = 7), 100)       // the words after the anchor
        t.releaseContext(plan2.context)
        t.finalDone(second, true)
        frame(t, 9_000, 9_000, 9_032)
        val third = utt(9_000, 12_000)
        t.finalComing(third)
        val plan3 = t.finalPlan(third)
        assertEquals("the words of the second piece alone", 5, plan3.seam!!.pieceWords)
        assertEquals(words(5, from = 7).split(' '), plan3.seam!!.delivered)
        assertEquals("the second piece's own audio", 4_000L, plan3.seam!!.pieceAudioMs)
        assertTrue("its own last 2.5 s, not the first piece's", plan3.context!!.contentEquals(second.samples.copyOfRange(second.samples.size - 40_000, second.samples.size)))
    }

    @Test
    fun seamRunsAreCountedForTheSessionLine() {
        val t = tracker()
        t.seamRun(anchored = true, fallbackMs = 0, window = 384)
        t.seamRun(anchored = false, fallbackMs = 2_300, window = 432)
        val s = t.stats()
        assertEquals(2, s.seamsWithContext)
        assertEquals(1, s.seamsAnchored)
        assertEquals(1, s.seamsFallback)
        assertEquals(2_300L, s.seamFallbackMs)
        assertEquals("windows above the floor of 384", 1, s.seamWindowsOverFloor)
    }

    @Test
    fun theGuardBooksTheRunWithTheContextAndTheStatisticStaysThePiecesOwnLength() {
        val guard = StreamingGuard(15_000, 384)
        val t = tracker(guard = guard)
        finishPiece(t, 1_000, 5_000, words(6), recogniserMs = 100)
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 11_000)             // 6 s own + 2.5 s context = 8.5 s: window 464 frames
        t.finalComing(second)
        val plan = t.finalPlan(second)
        assertEquals(40_000 + 96_000, plan.firstRunSamples)
        t.absorbFinal(plan, words(4, from = 7), 2_400)
        t.releaseContext(plan.context)
        assertEquals("the piece's own length", listOf(4_000L, 6_000L), t.stats().finalAudioMs)
        // booked as 8 500 ms of audio on a window of 464 frames = 9 280 ms: 2400 / 9280
        assertEquals(2_400.0 / 9_280.0, guard.finalCostPerWindowMs!!, 1e-9)
    }

    @Test
    fun theLagCutCountsTheContextOfAContinuationButNotOfAPieceAfterSilence() {
        for ((name, bySilence, cutExpected) in listOf(Triple("after a cut", false, true), Triple("after silence", true, false))) {
            val guard = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500) }
            val t = tracker(guard = guard)
            finishPiece(t, 1_000, 5_000, words(6), bySilence = bySilence)
            frame(t, 5_000, 5_000, 5_032)
            val step = frame(t, 5_000, 5_000, 5_000 + 11_300)        // 11 300 ms not committed: 14 675 ms without the context, 15 175 ms with it
            assertEquals(name, cutExpected, step == StreamingTracker.Step.Cut)
        }
    }
}
