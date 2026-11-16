package ai.moravian.czechspeech

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class CatalogTest {
    @Test
    fun czechIsTheDefaultEntry() {
        val cs = CzechSpeechModel.CATALOG["cs"]!!
        assertEquals("cs", cs.id)
        assertEquals(CzechSpeechModel.DEFAULT, cs.source)
        assertEquals(listOf("cs"), cs.languages)
        assertEquals(25, cs.tokensPerSecond)
        assertEquals(setOf("cs"), CzechSpeechModel.CATALOG.keys)
    }

    @Test
    fun entryFindsByIdAndRejectsUnknownOnes() {
        assertEquals(CzechSpeechModel.CATALOG["cs"], CzechSpeechModel.entry("cs"))
        for (bad in listOf("xx", "", "CS")) {
            try {
                CzechSpeechModel.entry(bad)
                fail("accepted '$bad'")
            } catch (e: IllegalArgumentException) {
                assertEquals("unknown model: $bad", e.message)
            }
        }
    }

    @Test
    fun entryLanguageCodesAreTwoOrThreeLowercaseLetters() {
        val source = CzechSpeechModel.DEFAULT
        for (bad in listOf("czech", "CS", "c", "cs ", "", "c1")) {
            try {
                CzechSpeechModel.Entry("x", source, listOf("pl", bad), 25)
                fail("accepted '$bad'")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!, e.message!!.contains("\"$bad\""))
            }
        }
        assertEquals(listOf("cs", "yue"), CzechSpeechModel.Entry("x", source, listOf("cs", "yue"), 25).languages)
    }

    @Test
    fun entryValidatesItsFields() {
        val source = CzechSpeechModel.DEFAULT
        for (bad in listOf(
            { CzechSpeechModel.Entry("", source, listOf("cs"), 25) },
            { CzechSpeechModel.Entry("x", source, emptyList(), 25) },
            { CzechSpeechModel.Entry("x", source, listOf("cs"), 0) },
            { CzechSpeechModel.Entry("x", source, listOf("czech"), 25) },
        )) {
            try {
                bad()
                fail("accepted an invalid entry")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!.isNotEmpty())
            }
        }
    }
}
