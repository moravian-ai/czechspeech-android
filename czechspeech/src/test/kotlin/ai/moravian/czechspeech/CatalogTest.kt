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

    @Test
    fun aSwitchNeedsSeveralLanguagesAndNamesOnlyTheOthers() {
        val source = CzechSpeechModel.DEFAULT
        val en = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f))
        for (bad in listOf(
            { CzechSpeechModel.Entry("x", source, listOf("cs"), 25, en) },
            { CzechSpeechModel.Entry("x", source, listOf("cs", "en"), 25, CzechSpeechModel.LanguageSwitch(mapOf("cs" to 0.9f))) },
            { CzechSpeechModel.Entry("x", source, listOf("cs", "sk", "en"), 25, CzechSpeechModel.LanguageSwitch(mapOf("de" to 0.9f))) },
        )) {
            try {
                bad()
                fail("accepted a switch that does not fit the languages")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!.isNotEmpty())
            }
        }
        val entry = CzechSpeechModel.Entry("x", source, listOf("cs", "sk", "en"), 25, en)
        assertEquals(en, entry.languageSwitch)
        assertEquals(null, CzechSpeechModel.CATALOG.getValue("cs").languageSwitch)
    }

    /** What is wrong with an entry that the catalogue must not hold: an unpinned URL, an upper-case hash, an id that is not its key. */
    private fun problems(key: String, e: CzechSpeechModel.Entry): List<String> = buildList {
        if (e.id != key) add("id ${e.id} is not its key $key")
        if (!Regex("""https://huggingface\.co/moravianai/[^/]+/resolve/[0-9a-f]{40}/[^/]+""").matches(e.source.url)) add("url is not pinned to a revision: ${e.source.url}")
        if (!Regex("[0-9a-f]{64}").matches(e.source.sha256)) add("sha256 is not 64 lower-case hex digits")
        if (e.source.sizeBytes <= 0) add("size is not positive")
    }

    @Test
    fun everyCatalogueEntryIsPinnedAndWellFormedWhateverTheSetOfIds() {
        assertTrue("the default id is in the catalogue", CzechSpeechModel.DEFAULT_ID in CzechSpeechModel.CATALOG)
        assertEquals("cs", CzechSpeechModel.DEFAULT_ID)
        assertTrue(CzechSpeechModel.CATALOG.isNotEmpty())
        for ((key, entry) in CzechSpeechModel.CATALOG) {
            assertEquals("$key", emptyList<String>(), problems(key, entry))
            assertTrue(entry.languages.isNotEmpty() && entry.languages.distinct().size == entry.languages.size)
        }
    }

    @Test
    fun theCatalogueCheckSeesAnUnpinnedUrlAnUpperCaseHashAndAWrongKey() {
        val good = CzechSpeechModel.CATALOG.getValue("cs")
        fun with(source: CzechSpeechModel.Source) = good.copy(source = source)
        assertTrue(problems("cs", with(good.source.copy(url = "https://huggingface.co/moravianai/CzechSpeech/resolve/main/m.bin"))).isNotEmpty())
        assertTrue(problems("cs", with(good.source.copy(sha256 = "A".repeat(64)))).isNotEmpty())    // upper case: the catalogue pins lower case
        assertTrue(problems("other", good).isNotEmpty())
        assertEquals(emptyList<String>(), problems("cs", good))
    }
}
