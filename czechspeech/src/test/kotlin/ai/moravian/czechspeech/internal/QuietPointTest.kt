package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Test

class QuietPointTest {
    @Test
    fun theCentreOfTheQuietestThirtyMillisecondsIsFound() {
        val x = FloatArray(16_000) { 0.5f }.also { it.fill(0f, 8_000, 8_480) }
        assertEquals(8_240, QuietPoint.find(x, 0, 16_000))
    }

    @Test
    fun theFirstOfEquallyQuietWindowsWins() {
        assertEquals(1_240, QuietPoint.find(FloatArray(16_000), 1_000, 9_000))
    }

    @Test
    fun aRangeShorterThanAWindowGivesItsStart() {
        assertEquals(100, QuietPoint.find(FloatArray(16_000), 100, 579))
    }

    @Test
    fun theRangeIsClampedToTheArray() {
        val x = FloatArray(1_000) { 0.5f }.also { it.fill(0f, 480, 960) }
        assertEquals(720, QuietPoint.find(x, -500, 2_000)) // windows start at 0, 160, 320, 480
    }
}
