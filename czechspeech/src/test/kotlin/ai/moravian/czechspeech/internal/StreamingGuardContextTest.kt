package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The audio of the seam context counts in the lag estimate (mustCut, closingEstimateMs), not in the target length. */
class StreamingGuardContextTest {
    @Test fun theContextLengthensTheClosingRecognitionInTheLagCut() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500) }     // 0.1953125
        assertFalse(g.mustCut(11_300))                                           // 11 300 + 2 375 + 1 000 < 15 000
        assertTrue(g.mustCut(11_300, contextMs = 2_500))                         // audio 13 800: 11 300 + 2 875 + 1 000
    }

    @Test fun theBufferCapAndTheEightSecondFloorDoNotMoveWithTheContext() {
        val fast = StreamingGuard(15_000, 384).apply { recorded(6_000, 100) }
        assertFalse(fast.mustCut(11_999, contextMs = 2_500)); assertTrue(fast.mustCut(12_000, contextMs = 2_500))
        val slow = StreamingGuard(15_000, 384).apply { recorded(6_000, 15_000) }
        assertFalse(slow.mustCut(7_999, contextMs = 2_500)); assertTrue(slow.mustCut(8_000, contextMs = 2_500))
    }

    @Test fun noContextIsTheOldEstimate() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500); recordedFinal(4_000, 2_200) }
        assertEquals(g.closingEstimateMs(5_000), g.closingEstimateMs(5_000, 0))
        assertEquals(g.mustCut(11_300), g.mustCut(11_300, 0))
    }

    @Test fun underTheLowestWindowTheClosingCostsWhatTheWindowCostsAlsoWithContext() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500); recordedFinal(4_000, 2_200) }
        assertEquals("4 000 + 2 500 ms still fit 384 frames", 2_200, g.closingEstimateMs(4_000, 2_500))
        assertEquals("5 000 + 2 500 ms need 416 frames: 2200 / 7680 * 8320", 2_384, g.closingEstimateMs(5_000, 2_500))
    }

    @Test fun theTargetPieceLengthIgnoresTheContext() {
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 3_000); recordedFinal(6_000, 2_250) }
        // the target is computed from the piece's own length alone (the measured 4 to 5.5 s pieces stay)
        assertEquals(5_500L, g.targetPieceMs())
    }

    @Test fun theFullWindowEstimateIsTheCostOfTheWholeWindow() {
        assertEquals("before any measurement: 0.2 * 30 000 ms, times 1.3", 7_800, StreamingGuard(15_000, 384).fullWindowEstimateMs())
        val g = StreamingGuard(15_000, 384).apply { recorded(6_000, 1_500); recordedFinal(4_000, 2_200) }
        assertEquals("2200 / 7680 * 30000 = 8594, times 1.3", 11_173, g.fullWindowEstimateMs())
    }
}
