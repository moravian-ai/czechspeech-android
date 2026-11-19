package ai.moravian.czechspeech.internal

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Rule
import org.junit.Test
import org.junit.rules.Timeout
import java.io.File
import kotlin.math.abs
import kotlin.concurrent.thread

/**
 * The language chosen inside whisper_full (native/patches/whisper.cpp-language-select.patch): the recognition decodes
 * exactly as if its language had been forced, and the audio is encoded once.
 *
 * The one test of this change that needs a model. It needs the host library, CZECHSPEECH_MODEL (a cs+sk+en model) and
 * CZECHSPEECH_TEST_AUDIO with at least one `cs-*.wav` and one `en-*.wav` (16 kHz mono, a few seconds each); it is
 * skipped without them. Run by hand through the queue with the variables set.
 */
class LanguageSelectIntegrationTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(600)

    private var handle = 0L

    @After
    fun free() {
        if (handle != 0L) JniNativeApi.whisperFree(handle)
    }

    private fun recording(prefix: String): FloatArray {
        val files = TestEnv.audioDir.listFiles { f -> f.name.startsWith(prefix) && f.name.endsWith(".wav") }?.sortedBy(File::getName)
        Assume.assumeTrue("a $prefix*.wav in CZECHSPEECH_TEST_AUDIO", !files.isNullOrEmpty())
        return Wav.read(files!!.first())
    }

    @Test
    fun theLanguageChosenInsideWhisperFullDecodesExactlyAsForcedWithOneEncoderPass() {
        TestEnv.assumeNativeLibrary()
        TestEnv.assumeFile(TestEnv.model, "cs+sk+en model (CZECHSPEECH_MODEL)")
        TestEnv.assumeFile(TestEnv.audioDir, "test audio directory (CZECHSPEECH_TEST_AUDIO)")
        val czech = recording("cs-")
        val english = recording("en-")
        handle = JniNativeApi.whisperInit(TestEnv.model.absolutePath)
        assertTrue("model loaded", handle != 0L)
        val threads = TestEnv.threads
        val thresholds = floatArrayOf(Float.NaN, Float.NaN, 0.9f)
        var generation = 0

        fun encoderRuns() = NativeBridge.nativeWhisperEncodeRuns(handle)

        // runs the recognition, returns the text and how many encoder runs it took
        fun run(audio: FloatArray, languages: String, thresholds: FloatArray?): Pair<ByteArray, Int> {
            val before = encoderRuns()
            val bytes = JniNativeApi.whisperTranscribe(handle, ++generation, audio, audio.size, threads, languages, thresholds)
            assertNotNull("$languages decoded", bytes)
            return bytes!! to encoderRuns() - before
        }

        for ((audio, language) in listOf(czech to "cs", english to "en")) {
            val (chosen, chosenRuns) = run(audio, "cs,sk,en", thresholds)
            assertEquals(language, JniNativeApi.whisperLanguage(handle))
            val shares = JniNativeApi.whisperLanguageShares(handle)
            assertNotNull(shares)
            assertEquals(3, shares!!.size)
            assertEquals("shares sum to 1", 1.0, shares.sumOf { it.toDouble() }, 1e-4)
            if (language == "en") assertTrue("English share ${shares[2]}", shares[2] >= 0.9f)

            val (forced, forcedRuns) = run(audio, language, null)
            assertNull("no shares for a forced language", JniNativeApi.whisperLanguageShares(handle))
            assertArrayEquals("$language: the chosen language decodes as the forced one", forced, chosen)
            assertEquals("$language: one encoder pass with the selection", 1, chosenRuns)
            assertEquals("$language: one encoder pass forced", 1, forcedRuns)
        }

        // an abort 50 ms after the start: null well within a second, and the next call decodes
        val target = ++generation
        val killer = thread {
            Thread.sleep(50)
            JniNativeApi.whisperAbort(handle, target)
        }
        val t0 = System.nanoTime()
        val cancelled = JniNativeApi.whisperTranscribe(handle, target, english, english.size, threads, "cs,sk,en", thresholds)
        val ms = (System.nanoTime() - t0) / 1_000_000
        killer.join()
        assertNull("cancelled", cancelled)
        assertTrue("returned in $ms ms", ms < 1_000)
        assertNull("no language for a cancelled call", JniNativeApi.whisperLanguage(handle))
        assertNull("no shares for a cancelled call", JniNativeApi.whisperLanguageShares(handle))
        val (again, _) = run(english, "cs,sk,en", thresholds)
        assertTrue("decodes after a cancellation", again.isNotEmpty())
        assertTrue(abs(1.0 - JniNativeApi.whisperLanguageShares(handle)!!.sumOf { it.toDouble() }) < 1e-4)
    }
}
