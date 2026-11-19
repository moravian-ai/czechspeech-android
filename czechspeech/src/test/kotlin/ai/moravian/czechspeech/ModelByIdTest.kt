package ai.moravian.czechspeech

import android.content.Context
import android.content.ContextWrapper
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.RandomAccessFile

/** The model catalog by id: the id overloads of [CzechSpeechModel] and the [CzechSpeech] constructor for an entry. */
class ModelByIdTest {
    @get:Rule
    val tmp = TemporaryFolder()

    /** A context whose only job is to name the app's no-backup directory. */
    private class DirContext(private val dir: File) : ContextWrapper(null) {
        override fun getNoBackupFilesDir(): File = dir
        override fun getApplicationContext(): Context = this
    }

    private fun context() = DirContext(File(tmp.root, "app"))

    /** Puts a verified copy of the default model in place (sparse file plus the marker), so nothing is downloaded. */
    private fun seedDefaultModel(context: Context) {
        val file = CzechSpeechModel.file(context)
        file.parentFile!!.mkdirs()
        RandomAccessFile(file, "rw").use { it.setLength(CzechSpeechModel.DEFAULT.sizeBytes) }
        File(file.parentFile, file.name + ".verified").writeText(
            "sha256=${CzechSpeechModel.DEFAULT.sha256}\nsize=${CzechSpeechModel.DEFAULT.sizeBytes}\n" +
                "mtime=${file.lastModified()}\n",
        )
    }

    private fun unknown(block: () -> Unit) {
        try {
            block()
            fail("accepted an unknown id")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!, e.message!!.startsWith("unknown model: "))
        }
    }

    @Test
    fun fileByIdIsTheFileOfTheEntrysSource() {
        val context = context()
        val source = CzechSpeechModel.entry("cs").source
        assertEquals(CzechSpeechModel.file(context, source), CzechSpeechModel.file(context, "cs"))
        assertEquals(CzechSpeechModel.file(context), CzechSpeechModel.file(context, "cs"))
    }

    @Test
    fun anUnknownIdThrowsBeforeAnyIo() {
        val context = context()
        for (bad in listOf("xx", "", "CS")) {
            unknown { CzechSpeechModel.file(context, bad) }
            unknown { CzechSpeechModel.isDownloaded(context, bad) }
            unknown { runBlocking { CzechSpeechModel.download(context, bad) } }
            unknown { runBlocking { CzechSpeechModel.delete(context, bad) } }
        }
        assertFalse("nothing was created", CzechSpeechModel.directory(context).exists())
    }

    @Test
    fun theIdOverloadsAgreeWithTheDefaultSourceOnes() = runBlocking {
        val context = context()
        assertFalse(CzechSpeechModel.isDownloaded(context, "cs"))
        seedDefaultModel(context)
        assertTrue(CzechSpeechModel.isDownloaded(context))
        assertTrue(CzechSpeechModel.isDownloaded(context, "cs"))
        var done = -1L
        // No id still means DEFAULT: Czech is unchanged, and the verified file is found without a download.
        assertEquals(CzechSpeechModel.file(context), CzechSpeechModel.download(context) { d, _ -> done = d })
        assertEquals(CzechSpeechModel.DEFAULT.sizeBytes, done)
        assertEquals(CzechSpeechModel.file(context), CzechSpeechModel.download(context, "cs"))
        assertTrue(CzechSpeechModel.delete(context, "cs"))
        assertFalse(CzechSpeechModel.isDownloaded(context))
        assertFalse(CzechSpeechModel.delete(context, "cs"))
    }

    @Test
    fun anEntryGivesTheLanguagesOfTheCzechSpeechInstance() {
        val context = context()
        val file = File(tmp.root, "model.bin")
        val entry = CzechSpeechModel.Entry("x", CzechSpeechModel.DEFAULT, listOf("pl", "en"), 25)
        assertEquals(listOf("pl", "en"), CzechSpeech(context, file, entry).languages)
        assertEquals(listOf("cs"), CzechSpeech(context, file, CzechSpeechModel.entry("cs")).languages)
        val config = CzechSpeechConfig()
        val speech = CzechSpeech(context, file, entry, config, true)
        assertEquals(listOf("pl", "en"), speech.languages)
        assertEquals(config, speech.config)
        assertTrue(speech.keepModelLoaded)
        // The constructor with explicit languages stays for files outside the catalog.
        assertEquals(listOf("de"), CzechSpeech(context, file, languages = listOf("de")).languages)
    }

    @Test
    fun aCustomEntryWithALanguageSwitchPassesItsRuleToCzechSpeech() {
        // The README snippet for a self-hosted model: an Entry outside the catalog, passed to the constructor with the file.
        val source = CzechSpeechModel.Source("https://models.example.com/own.bin", 123_456L, "ab".repeat(32), "own.bin")
        val rule = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)
        val entry = CzechSpeechModel.Entry("own", source, listOf("cs", "sk", "en"), 25, rule)
        assertFalse("not in the catalogue", CzechSpeechModel.CATALOG.containsKey("own"))
        val speech = CzechSpeech(context(), File(tmp.root, "own.bin"), entry)
        assertEquals(rule, speech.languageSwitch)
        assertEquals(listOf("cs", "sk", "en"), speech.languages)
        // The engine applies that rule (thresholds, return minimum) in LanguageSessionTest, on the same values.
        assertEquals(null, CzechSpeech(context(), File(tmp.root, "own.bin"), CzechSpeechModel.entry("cs")).languageSwitch)
    }

    @Test
    fun duplicateLanguagesCountOnce() {
        val context = context()
        val file = File(tmp.root, "model.bin")
        assertEquals(listOf("cs"), CzechSpeech(context, file, languages = listOf("cs", "cs")).languages)
        assertEquals(
            listOf("pl", "en"),
            CzechSpeech(context, file, languages = listOf("pl", "en", "pl")).languages,
        )
    }

    @Test
    fun transcriptsTakesALanguageAndChecksItAtOnce() {
        val file = File(tmp.root, "model.bin")
        val speech = CzechSpeech(context(), file, languages = listOf("pl", "en"))
        // Like listen(language): a language outside the model throws from the call, not from the collection.
        for (bad in listOf("de", "EN")) {
            try {
                speech.transcripts(bad)
                fail("accepted '$bad'")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!, e.message!!.contains("\"$bad\""))
            }
        }
        speech.transcripts()
        speech.transcripts("en")
        speech.transcripts(null)
    }
}
