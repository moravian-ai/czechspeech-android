package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The ladder of an empty final of a piece that was recognised with the context of the piece before it (CutSeam). */
class EmptyFinalRetryContinuationTest {
    private val samples = 100_000      // 6.25 s: 384 frames at the lowest window

    @Test fun anOrdinaryFinalStepsThrough512First() {
        assertEquals(512, EmptyFinalRetry.nextFloor(384, 384, samples, 0, 1, false))
    }

    @Test fun aContinuationWithContextSkips512AndGoesToTheFullWindow() {
        assertEquals(1488, EmptyFinalRetry.nextFloor(384, 384, samples, 0, 1, false, continuation = true, fullFitsLag = true))
    }

    @Test fun aContinuationDoesNotRepeatOnTheFullWindowWhenItWouldEndAfterTheLagLimit() {
        assertNull(EmptyFinalRetry.nextFloor(384, 384, samples, 0, 1, false, continuation = true, fullFitsLag = false))
    }

    @Test fun theFullWindowStillWaitsForOtherFinals() {
        assertNull(EmptyFinalRetry.nextFloor(384, 384, samples, 0, 2, false, continuation = true, fullFitsLag = true))
        assertNull(EmptyFinalRetry.nextFloor(384, 384, samples, 0, 1, true, continuation = true, fullFitsLag = true))
    }

    @Test fun atMostOneRepeatOfAContinuation() {
        assertNull(EmptyFinalRetry.nextFloor(384, 1488, samples, 1, 1, false, continuation = true, fullFitsLag = true))
    }

    @Test fun anOrdinaryFinalIgnoresTheLagFlag() {
        assertEquals(512, EmptyFinalRetry.nextFloor(384, 384, samples, 0, 1, false, continuation = false, fullFitsLag = false))
    }
}
