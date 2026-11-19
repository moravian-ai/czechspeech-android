package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The safeguard of streaming transcription: pure logic, the times come in as arguments. */
class StreamingGuardTest {
    @Test fun beforeAnyMeasurementAPassOverSixSecondsIsEstimatedAt1536ms() {
        val g = StreamingGuard(15_000, 384)
        assertEquals(1_536, g.estimateMs(6_000)); assertFalse(g.passesPaused)
        assertEquals(2_048, StreamingGuard(15_000, 512).estimateMs(6_000))     // the floor counts
    }

    @Test fun theCostIsTheLargestOfTheLastThreeRecognitions() {
        val g = StreamingGuard(15_000, 384)
        g.recorded(6_000, 1_536); assertEquals(0.2, g.costPerWindowMs, 1e-9)
        g.recorded(6_000, 3_072); assertEquals(0.4, g.costPerWindowMs, 1e-9)
        g.recorded(6_000, 1_536); g.recorded(6_000, 1_536); assertEquals(0.4, g.costPerWindowMs, 1e-9)
        g.recorded(6_000, 1_536); assertEquals(0.2, g.costPerWindowMs, 1e-9)
        g.recorded(12_000, 2_560); assertEquals(0.2, g.costPerWindowMs, 1e-9)  // 640 frames = 12 800 ms of window
        g.recorded(6_000, 0); assertEquals(0.2, g.costPerWindowMs, 1e-9)       // did not run: not counted
    }

    @Test fun theLagCutComesWhereTheBufferAndItsClosingReachTheLimit() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500) }   // 0.1953125: a 1.5 s short pass
        assertFalse(g.mustCut(11_550)); assertTrue(g.mustCut(11_600))
    }

    @Test fun neverALagCutBelowEightSecondsAndAlwaysFromTwelve() {
        val slow = StreamingGuard(15_000, 384).apply { recorded(6_000, 15_000) }
        assertFalse(slow.mustCut(7_999)); assertTrue(slow.mustCut(8_000))
        val fast = StreamingGuard(15_000, 384).apply { recorded(6_000, 100) }
        assertFalse(fast.mustCut(11_999)); assertTrue(fast.mustCut(12_000))
    }

    @Test fun passesPauseAboveTwoSecondsAndResumeAt1700ms() {
        val g = StreamingGuard(15_000, 384)
        assertEquals(StreamingGuard.Change.PAUSED, g.recorded(6_000, 2_100)); assertTrue(g.passesPaused)
        repeat(3) { assertEquals(StreamingGuard.Change.NONE, g.recorded(6_000, 1_800)) }
        repeat(2) { assertEquals(StreamingGuard.Change.NONE, g.recorded(6_000, 1_700)) }
        assertEquals(StreamingGuard.Change.RESUMED, g.recorded(6_000, 1_700)); assertFalse(g.passesPaused)
    }

    @Test fun aDisabledGuardNeitherCutsNorPauses() {
        val g = StreamingGuard.disabled(384).apply { recorded(6_000, 60_000) }
        assertFalse(g.mustCut(27_000)); assertFalse(g.passesPaused)
    }

    // Boundary values, monotonicity and hysteresis beyond the plan's tests.

    @Test fun passesPauseOnlyStrictlyAboveTwoSeconds() {
        val at = StreamingGuard(15_000, 384)
        assertEquals(StreamingGuard.Change.NONE, at.recorded(6_000, 2_000)); assertFalse(at.passesPaused)
        val over = StreamingGuard(15_000, 384)
        assertEquals(StreamingGuard.Change.PAUSED, over.recorded(6_000, 2_001)); assertTrue(over.passesPaused)
    }

    @Test fun pausedPassesStayPausedBetweenTheThresholdsAndResumeExactlyAt1700() {
        val g = StreamingGuard(15_000, 384)
        g.recorded(6_000, 2_100)
        repeat(3) { assertEquals(StreamingGuard.Change.NONE, g.recorded(6_000, 1_701)) }   // in the gap
        assertTrue(g.passesPaused)
        repeat(2) { assertEquals(StreamingGuard.Change.NONE, g.recorded(6_000, 1_700)) }   // 1 701 still in the window
        assertEquals(StreamingGuard.Change.RESUMED, g.recorded(6_000, 1_700))
        repeat(3) { assertEquals(StreamingGuard.Change.NONE, g.recorded(6_000, 1_950)) }   // the gap does not pause again
        assertFalse(g.passesPaused)
    }

    @Test fun noFlappingInsideTheHysteresisGap() {
        val g = StreamingGuard(15_000, 384)
        val changes = (1..40).map { g.recorded(6_000, if (it % 2 == 0) 1_750L else 1_950L) }
        assertTrue(changes.all { it == StreamingGuard.Change.NONE }); assertFalse(g.passesPaused)
    }

    @Test fun theBufferCapCutsExactlyAtTwelveSecondsWhateverTheCost() {
        val g = StreamingGuard(30_000, 384).apply { recorded(6_000, 1) }
        assertFalse(g.mustCut(11_999)); assertTrue(g.mustCut(12_000))
        val custom = StreamingGuard(30_000, 384, maxBufferMs = 9_000)
        assertFalse(custom.mustCut(8_999)); assertTrue(custom.mustCut(9_000))
    }

    @Test fun theLagCutIsExactlyOnTheLimit() {
        val g = StreamingGuard(0, 384)
        val estimate = g.estimateMs(10_000)
        assertTrue(StreamingGuard(10_000 + estimate + 1_000, 384).mustCut(10_000))        // on the limit
        assertFalse(StreamingGuard(10_000 + estimate + 1_001, 384).mustCut(10_000))       // one ms below
    }

    @Test fun theBufferIsCutBeforeItsClosingCanOvershootTheLimit() {
        for (cost in listOf(100L, 800L, 1_536L, 3_000L, 6_000L)) {
            val g = StreamingGuard(15_000, 384).apply { recorded(6_000, cost) }
            var buffer = 0L
            while (!g.mustCut(buffer) && buffer < 60_000) buffer += 32
            assertTrue("cut at $buffer", buffer <= StreamingGuard.MAX_BUFFER_MS + 32)
            val previous = buffer - 32
            if (previous >= StreamingGuard.FORCED_CUT_MIN_MS) {
                // the frame before still ended within the limit: the cut is not later than it must be
                assertTrue(previous + g.estimateMs(previous) + StreamingGuard.LAG_MARGIN_MS < 15_000)
            }
            assertTrue(
                buffer >= StreamingGuard.MAX_BUFFER_MS ||
                    buffer + g.estimateMs(buffer) + StreamingGuard.LAG_MARGIN_MS >= 15_000 ||
                    buffer < StreamingGuard.FORCED_CUT_MIN_MS,
            )
        }
    }

    @Test fun mustCutIsMonotonicInTheBuffer() {
        for (cost in listOf(10L, 1_500L, 5_000L)) {
            val g = StreamingGuard(15_000, 384).apply { recorded(6_000, cost) }
            var cut = false
            for (b in 0L..30_000L step 32) {
                val now = g.mustCut(b)
                assertFalse("once a buffer must be cut, a longer one must as well (b=$b)", cut && !now)
                cut = cut || now
            }
            assertTrue(cut)
        }
    }

    @Test fun theEstimateNeverShrinksWithTheBufferAndGrowsWithTheCost() {
        val g = StreamingGuard(15_000, 384)
        var last = 0L
        for (b in 0L..40_000L step 160) { val e = g.estimateMs(b); assertTrue(e >= last); last = e }
        val before = g.estimateMs(9_000); g.recorded(6_000, 4_000); assertTrue(g.estimateMs(9_000) > before)
    }

    @Test fun aTighterLimitCutsEarlierAndASmallerFloorEstimatesLess() {
        val loose = StreamingGuard(30_000, 384).apply { recorded(6_000, 1_500) }
        val tight = StreamingGuard(10_000, 384).apply { recorded(6_000, 1_500) }
        assertFalse(loose.mustCut(11_000)); assertTrue(tight.mustCut(8_000))
        assertTrue(StreamingGuard(15_000, 256).estimateMs(2_000) < StreamingGuard(15_000, 512).estimateMs(2_000))
    }

    // Two costs: the greedy passes (they decide whether passes run) and the final recognitions (they decide the cut).

    @Test fun theCutIsDecidedOnTheCostOfTheFinalsAndPausingOnThatOfThePasses() {
        val g = StreamingGuard(15_000, 384)
        g.recorded(6_000, 1_500)                            // a pass: 0.1953 per ms of window
        assertEquals(g.estimateMs(10_000), g.closingEstimateMs(10_000))     // no final yet: the passes' cost stands in
        g.recordedFinal(12_000, 6_400)                      // a final over 640 frames (12 800 ms): 0.5
        assertEquals(0.1953125, g.costPerWindowMs, 1e-9)
        assertEquals(0.5, g.finalCostPerWindowMs!!, 1e-9)
        assertTrue(g.closingEstimateMs(10_000) > 2 * g.estimateMs(10_000))
        assertFalse(g.passesPaused)                         // the dear final paused nothing
        assertTrue(g.mustCut(9_100))                        // 9 100 + 4 960 + 1 000 >= 15 000 (by the passes alone it would not be)
    }

    @Test fun theTimeOfTheLanguageModelIsPartOfTheClosing() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500) }
        val before = g.closingEstimateMs(9_000)
        g.recordedLanguageModel(400)
        assertEquals(400L, g.languageModelMs)
        assertEquals(before + 400, g.closingEstimateMs(9_000))
        assertFalse(g.mustCut(11_200)); assertTrue(g.mustCut(11_300))      // 11 300 + 2 375 + 1 000 + 400 >= 15 000, without the 400 it is not
    }

    @Test fun withoutAnyPassMeasuredAFinalIsTheOnlyMeasurementAndCountsForBoth() {
        val g = StreamingGuard(15_000, 384)
        assertEquals(StreamingGuard.Change.PAUSED, g.recordedFinal(6_000, 2_100))
        assertTrue(g.passesPaused)
    }

    @Test fun pausedPassesResumeWhenTheFinalsGotAsMuchCheaperAsThePausingRequired() {
        val g = StreamingGuard(15_000, 384)
        g.recordedFinal(6_000, 1_400)                       // the finals cost 0.182 per window ms
        assertEquals(StreamingGuard.Change.PAUSED, g.recorded(6_000, 2_100))   // passes 2.1 s: paused
        repeat(3) { assertEquals(StreamingGuard.Change.NONE, g.recordedFinal(6_000, 1_400)) }   // as dear as then
        assertTrue(g.passesPaused)
        repeat(2) { assertEquals(StreamingGuard.Change.NONE, g.recordedFinal(6_000, 1_130)) }
        assertEquals(StreamingGuard.Change.RESUMED, g.recordedFinal(6_000, 1_130))              // 2 100 * 1 130/1 400 <= 1 700
        assertFalse(g.passesPaused)
    }

    @Test fun passesAtTwoSecondsAndFinalsOfAnotherSizeDoNotMakeItFlap() {
        // passes cost 2.1 s (slow), the finals over 12 s only 1.4 s-equivalent: the old mix resumed at once
        val g = StreamingGuard(15_000, 384)
        g.recordedFinal(12_000, 2_333)
        var changes = 0
        repeat(40) { if ((if (g.passesPaused) g.recordedFinal(12_000, 2_333) else g.recorded(6_000, 2_100)) != StreamingGuard.Change.NONE) changes++ }
        assertEquals("paused once and stayed", 1, changes)
    }

    // ---- the target length of a piece (passes paused: the text comes from the final recognitions of pieces) ----

    /**
     * A guard whose finals over 6 s of audio took [finalMs]: the encoder works on its lowest window then (384 frames =
     * 7.68 s; the window is the audio plus 32 frames, so audio up to 7.04 s stays on it).
     */
    private fun guardWithFinals(finalMs: Long) = StreamingGuard(15_000, 384).apply { recordedFinal(6_000, finalMs) }

    @Test fun theTargetIsTheShortestOneWhoseShortestPossiblePieceTakesAtMostSixtyPercentOfItsLength() {
        // the cut falls in the last 1.5 s before the target, so the piece may be that much shorter: the duty is of that piece
        assertEquals("2.25 s <= 0.6 x 4 s: the minimum piece of 4 s and the window of 1.5 s", 5_500L, guardWithFinals(2_250).targetPieceMs())
        assertEquals("3.1 s / 0.6 = 5.17 s of piece, 6.67 s of target", 6_700L, guardWithFinals(3_100).targetPieceMs())
        for (ms in listOf(1_000L, 2_250L, 3_100L, 3_600L, 4_000L, 4_300L)) {
            val g = guardWithFinals(ms)
            val t = g.targetPieceMs() ?: continue
            assertTrue("target $t is at least the 4 s minimum and the window", t >= StreamingGuard.MIN_PIECE_MS + StreamingGuard.CUT_SEARCH_MS)
            assertTrue("the shortest piece ${t - 1500} s: ${g.closingEstimateMs(t)}", g.closingEstimateMs(t) <= 0.6 * (t - StreamingGuard.CUT_SEARCH_MS))
            if (t > StreamingGuard.MIN_PIECE_MS + StreamingGuard.CUT_SEARCH_MS) {
                assertTrue(g.closingEstimateMs(t - 100) > 0.6 * (t - 100 - StreamingGuard.CUT_SEARCH_MS))
            }
        }
    }

    @Test fun thereIsNoTargetWhenEvenTheLongestPieceIsTooDearForTheDuty() {
        // beyond the window floor the cost is proportional to the window: 5.0 s per 7.68 s is 65 % at any length
        assertNull(guardWithFinals(5_000).targetPieceMs())
        assertNull(guardWithFinals(60_000).targetPieceMs())
        // 4.3 s would need a piece of 7.17 s: 8.67 s of target, beyond the lowest window, where it costs more
        assertNull(guardWithFinals(4_300).targetPieceMs())
    }

    @Test fun aLongPieceMeasuredOnceDoesNotKeepThePhoneFromTheShortOnesUnderTheWindow() {
        val g = StreamingGuard(15_000, 384).apply { recordedFinal(12_000, 8_000) }      // 0.63 per ms of window: too dear
        assertNull(g.targetPieceMs())
        g.recordedFinal(6_000, 2_250)                                                   // but a short one took 2.25 s
        assertEquals(5_500L, g.targetPieceMs())
        assertEquals(2_250L, g.closingEstimateMs(5_000))                                // flat under the window
        assertTrue(g.closingEstimateMs(12_000) >= 8_000)
    }

    @Test fun theTargetIncludesTheLanguageModelAndNeverLeavesTheBufferCap() {
        val g = guardWithFinals(2_250).apply { recordedLanguageModel(600) }       // 2.85 s: a piece of 4.75 s, a target of 6.25 s
        assertEquals(6_300L, g.targetPieceMs())
        val t = guardWithFinals(4_500).targetPieceMs()
        assertTrue("$t", t == null || t <= StreamingGuard.MAX_BUFFER_MS)
        assertEquals(5_500L, StreamingGuard(15_000, 384).targetPieceMs())          // before any measurement
    }

    @Test fun oneSlowOutlierAmongTheFinalsUnderTheWindowDoesNotRemoveTheTarget() {
        val g = guardWithFinals(2_300).apply { recordedFinal(6_000, 2_300) }
        assertEquals(5_500L, g.targetPieceMs())
        g.recordedFinal(6_000, 4_500)                                        // one outlier (a busy moment of the phone)
        assertEquals("the median of the last three is 2.3 s", 2_300L, g.closingEstimateMs(5_000))
        assertEquals(5_500L, g.targetPieceMs())
        g.recordedFinal(6_000, 2_300)
        assertEquals(5_500L, g.targetPieceMs())
    }

    @Test fun threeSlowFinalsUnderTheWindowRemoveTheTarget() {
        val g = guardWithFinals(2_300).apply { recordedFinal(6_000, 2_300) }
        repeat(3) { g.recordedFinal(6_000, 4_500) }
        assertEquals(4_500L, g.closingEstimateMs(5_000))
        assertNull(g.targetPieceMs())
    }

    @Test fun aRetryIsBookedApartAndChangesNoEstimate() {
        val g = guardWithFinals(2_300)
        val closing = g.closingEstimateMs(5_000)
        val target = g.targetPieceMs()
        val cost = g.finalCostPerWindowMs
        g.recordedRetry(9_000)
        g.recordedRetry(4_000)
        assertEquals(2, g.retryCount)
        assertEquals(13_000L, g.retryRecogniserMs)
        assertEquals(closing, g.closingEstimateMs(5_000))
        assertEquals(target, g.targetPieceMs())
        assertEquals(cost, g.finalCostPerWindowMs)
        g.recordedRetry(0)
        assertEquals("a run that did not finish is not counted", 2, g.retryCount)
    }
}
