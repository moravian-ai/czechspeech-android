package ai.moravian.czechspeech

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.fail
import org.junit.Test

/** [CzechSpeechModel.LanguageSwitch]: the numbers of the rule for leaving the first language, checked at construction. */
class LanguageSwitchTest {
    private fun refused(block: () -> Unit) {
        try {
            block()
            fail("accepted an invalid switch")
        } catch (e: IllegalArgumentException) {
            assertFalse(e.message.isNullOrEmpty())
        }
    }

    @Test
    fun defaultsAre2000And1000() {
        val s = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f))
        assertEquals(2_000, s.minLeaveMs)
        assertEquals(1_000, s.minReturnMs)
        assertEquals(mapOf("en" to 0.9f), s.thresholds)
    }

    @Test
    fun theThresholdsCannotBeChangedAfterTheCheck() {
        // two entries: a one-entry map is already immutable in toMap()
        val source = mutableMapOf("en" to 0.9f, "sk" to 0.8f)
        val s = CzechSpeechModel.LanguageSwitch(source)
        source["en"] = 0.1f
        assertEquals("a copy", mapOf("en" to 0.9f, "sk" to 0.8f), s.thresholds)
        @Suppress("UNCHECKED_CAST")
        val writable = s.thresholds as MutableMap<String, Float>
        try {
            writable["en"] = 0.1f
            fail("the thresholds were changed")
        } catch (_: UnsupportedOperationException) {
        }
        try {
            writable["xx"] = 0.1f
            fail("a threshold was added")
        } catch (_: UnsupportedOperationException) {
        }
        assertEquals(mapOf("en" to 0.9f, "sk" to 0.8f), s.thresholds)
    }

    @Test
    fun aThresholdIsFromHalfToBelowOne() {
        CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.5f))
        CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.999f))
        for (bad in listOf(0.49f, 1.0f, Float.NaN)) refused { CzechSpeechModel.LanguageSwitch(mapOf("en" to bad)) }
    }

    @Test
    fun codesFollowTheLanguageRule() {
        refused { CzechSpeechModel.LanguageSwitch(mapOf("EN" to 0.9f)) }
        refused { CzechSpeechModel.LanguageSwitch(mapOf("e" to 0.9f)) }
    }

    @Test
    fun minimumsAreFrom0To30000() {
        CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), 0, 30_000)
        CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), 30_000, 0)
        for (bad in listOf(-1, 30_001)) {
            refused { CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = bad) }
            refused { CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minReturnMs = bad) }
        }
    }

    @Test
    fun theMapIsCopied() {
        val source = mutableMapOf("en" to 0.9f)
        val s = CzechSpeechModel.LanguageSwitch(source)
        source["sk"] = 0.95f
        source["en"] = 0.6f
        assertEquals(mapOf("en" to 0.9f), s.thresholds)
    }
}
