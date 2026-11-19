package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The language rule of one recognition ([pickLanguage]) and of one call ([LanguageRule]). */
class LanguageRuleTest {
    private val nan = Float.NaN

    private fun pick(shares: FloatArray, thresholds: FloatArray?) = pickLanguage(shares, thresholds)

    @Test
    fun pickTakesTheFirstUnlessACandidateReachesItsThreshold() {
        assertEquals(2, pick(floatArrayOf(.05f, .05f, .90f), floatArrayOf(nan, nan, .9f)))
        assertEquals(0, pick(floatArrayOf(.06f, .05f, .89f), floatArrayOf(nan, nan, .9f)))
        for (shares in listOf(floatArrayOf(.9f, .05f, .05f), floatArrayOf(.05f, .05f, .9f), floatArrayOf(.3f, .3f, .4f))) {
            assertEquals(0, pick(shares, floatArrayOf(nan, nan, nan)))
        }
        assertEquals(2, pick(floatArrayOf(.9f, .05f, .05f), floatArrayOf(nan, nan, 0f)))
        assertEquals(1, pick(floatArrayOf(.3f, .5f, .2f), null))
    }

    @Test
    fun amongSeveralCandidatesTheMostProbableIsTried() {
        val thresholds = floatArrayOf(nan, .6f, .6f)
        assertEquals(2, pick(floatArrayOf(.1f, .25f, .65f), thresholds))
        assertEquals(1, pick(floatArrayOf(.1f, .65f, .25f), thresholds))
    }

    // --- the rule of one call: thresholds, decisive pass, rescue (cs, sk, en with en 0.9, 2000 / 1000) ---

    private val switch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)
    private val rule = LanguageRule(listOf("cs", "sk", "en"), switch)
    private val czechShares = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val englishShares = floatArrayOf(0.002f, 0.001f, 0.997f)
    private val silenceShares = floatArrayOf(0.70f, 0f, 0.30f)
    private val slovakShares = floatArrayOf(0.2f, 0.8f, 0f)

    private fun assertThresholds(expected: FloatArray?, actual: FloatArray?) {
        if (expected == null || actual == null) {
            assertNull(actual)
            assertNull(expected)
            return
        }
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            if (expected[i].isNaN()) assertTrue("index $i is NaN, not ${actual[i]}", actual[i].isNaN())
            else assertEquals(expected[i], actual[i], 0f)
        }
    }

    @Test
    fun inCzechBelowTheMinimumNothingIsACandidate() {
        assertThresholds(floatArrayOf(nan, nan, nan), rule.thresholds("cs", 1_999))
    }

    @Test
    fun inCzechFromTheMinimumTheConfiguredThresholdsApply() {
        assertThresholds(floatArrayOf(nan, nan, 0.9f), rule.thresholds("cs", 2_000))
    }

    @Test
    fun slovakWithoutAThresholdIsNeverACandidate() {
        for (current in listOf("cs", "en")) {
            for (ms in listOf(0L, 999, 1_000, 1_999, 2_000, 30_000)) {
                assertTrue("$current $ms", rule.thresholds(current, ms)!![1].isNaN())
            }
        }
    }

    @Test
    fun inEnglishBelowTheReturnMinimumEnglishIsCertain() {
        assertThresholds(floatArrayOf(nan, nan, 0f), rule.thresholds("en", 999))
    }

    @Test
    fun inEnglishFromTheReturnMinimumEnglishMustReachItsThreshold() {
        assertThresholds(floatArrayOf(nan, nan, 0.9f), rule.thresholds("en", 1_000))
        assertThresholds(floatArrayOf(nan, nan, 0.9f), rule.thresholds("en", 2_000))
    }

    @Test
    fun decisiveFollowsTheTwoMinimums() {
        assertFalse(rule.decisive("cs", 1_999))
        assertTrue(rule.decisive("cs", 2_000))
        assertFalse(rule.decisive("en", 999))
        assertTrue(rule.decisive("en", 1_000))
    }

    @Test
    fun rescueNeedsAMajorityAndALanguageThatMaySwitch() {
        assertEquals("en", rule.rescue("cs", englishShares))
        assertNull(rule.rescue("cs", silenceShares))
        assertNull(rule.rescue("cs", slovakShares))
        assertEquals("cs", rule.rescue("en", czechShares))
        assertNull(rule.rescue("cs", null))
        assertNull(rule.rescue("en", floatArrayOf(0.49f, 0.02f, 0.49f)))
    }

    @Test
    fun theReasonForNoRescueNamesTheBestOtherLanguageAndItsShare() {
        assertEquals("best other en 0.312 < 0.5", rule.noRescueReason("cs", floatArrayOf(0.6f, 0.088f, 0.312f)))
        assertEquals("no shares", rule.noRescueReason("cs", null))
    }

    @Test
    fun withoutASwitchOrWithOneLanguageThereAreNoThresholdsAndNoRescue() {
        val plain = LanguageRule(listOf("cs", "sk", "en"), null)
        assertNull(plain.thresholds("cs", 5_000))
        assertNull(plain.thresholds("en", 500))
        assertNull(plain.rescue("cs", englishShares))
        assertTrue(plain.decisive("cs", 0))
        assertEquals("cs,sk,en", plain.argument(null))
        assertEquals("en", plain.argument("en"))
        val single = LanguageRule(listOf("cs"), null)
        assertEquals("cs", single.argument(null))
        assertNull(single.thresholds("cs", 5_000))
        assertEquals("cs", single.primary)
    }
}
