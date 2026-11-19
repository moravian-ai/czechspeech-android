package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/** The rule inside the JNI library ([NativeBridge.nativePickLanguage]) against its copy in Kotlin; host library, no model. */
class NativeLanguagePickTest {
    @Before
    fun load() = TestEnv.assumeNativeLibrary()

    private val nan = Float.NaN

    @Test
    fun theCRuleMatchesKotlinOnAGrid() {
        val thresholdSets = listOf(
            null,
            floatArrayOf(nan, nan, .9f),
            floatArrayOf(nan, .95f, .9f),
            floatArrayOf(nan, nan, 0f),
            floatArrayOf(nan, nan, nan),
        )
        var checked = 0
        for (a in 0..20) for (b in 0..(20 - a)) {
            // shares in steps of 0.05 that sum to 1 (the last one is what is left)
            val shares = floatArrayOf(a * .05f, b * .05f, (20 - a - b) * .05f)
            for (thresholds in thresholdSets) {
                assertEquals(
                    "shares ${shares.toList()} thresholds ${thresholds?.toList()}",
                    pickLanguage(shares, thresholds),
                    NativeBridge.nativePickLanguage(shares, thresholds),
                )
                checked++
            }
        }
        assertEquals(231 * 5, checked)
    }

    @Test
    fun badThresholdsAreRefused() {
        val shares = floatArrayOf(.2f, .3f, .5f)
        val bad = listOf(
            "length" to floatArrayOf(nan, .9f),
            "above 1" to floatArrayOf(nan, nan, 1.5f),
            "below 0" to floatArrayOf(nan, nan, -.1f),
        )
        for ((what, thresholds) in bad) {
            val e = failure<IllegalArgumentException> { NativeBridge.nativePickLanguage(shares, thresholds) }
            assertTrue("$what: ${e.message}", e.message!!.isNotEmpty())
        }
        // 0 and 1 are in range
        assertEquals(2, NativeBridge.nativePickLanguage(shares, floatArrayOf(nan, nan, 0f)))
        assertEquals(0, NativeBridge.nativePickLanguage(shares, floatArrayOf(nan, nan, 1f)))
    }
}
