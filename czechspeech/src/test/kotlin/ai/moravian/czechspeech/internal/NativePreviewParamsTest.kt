package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * A preview decodes once, greedily. whisper.cpp decodes a result that looks wrong again at higher temperatures (0.2
 * steps up to 1.0, best of 5); audio cut mid-word, as a preview's is, triggers that often, for text that is replaced
 * anyway, and the cost of a preview would swing with it. The final text keeps the fallback. The parameters live in C
 * and cannot be read back without a model, so the source is guarded, as in [NativeParamsTest].
 */
class NativePreviewParamsTest {
    private val source = File("../native/src/czechspeech_jni.c").takeIf { it.exists() }?.readText()

    private fun text() = source ?: error("native/src/czechspeech_jni.c not found from ${File(".").absolutePath}")

    @Test
    fun withoutFallbackTheTemperatureDoesNotRise() {
        val noFallback = Regex("""if\s*\(\s*!fallback\s*\)\s*\{?\s*params\.temperature_inc\s*=\s*0(\.0)?f?\s*;""")
        assertTrue(noFallback.containsMatchIn(text()))
    }

    @Test
    fun thePreviewExportHasNoFallbackAndTheTranscriptionKeepsIt() {
        fun passes(export: String, fallback: Boolean) = Regex(
            """Java_ai_moravian_czechspeech_internal_NativeBridge_$export\(.*?\)\s*\{[^}]*?""" +
                """return\s+cs_transcribe\([^;]*,\s*$fallback\s*,\s*languages\s*,\s*thresholds\s*\)\s*;""",
            RegexOption.DOT_MATCHES_ALL,
        ).containsMatchIn(text())
        assertTrue("nativeWhisperPreview: no fallback", passes("nativeWhisperPreview", false))
        assertTrue("nativeWhisperTranscribe: with fallback", passes("nativeWhisperTranscribe", true))
    }

    /**
     * Variant selection switches only the final text to beam search ([NativeApi.whisperTranscribeNBest]); the greedy
     * calls, the preview's included, take the shared greedy parameters and never set a beam search or touch the
     * language model.
     */
    @Test
    fun aPreviewIsGreedyWhateverTheFinalTextUses() {
        fun function(signature: String) =
            Regex("""${Regex.escape(signature)}\(.*?\n\}\n""", RegexOption.DOT_MATCHES_ALL).find(text())?.value
                ?: error("$signature not found")
        val shared = function("static struct whisper_full_params cs_full_params")
        assertTrue(shared.contains("whisper_full_default_params(WHISPER_SAMPLING_GREEDY)"))
        val greedy = function("static jbyteArray cs_transcribe")
        assertTrue(greedy.contains("cs_full_params(n_threads, n_samples)"))
        for (body in listOf(shared, greedy)) {
            for (forbidden in listOf("BEAM_SEARCH", "beam_search", "strategy", "cs_lm")) {
                assertFalse("$forbidden in\n$body", body.contains(forbidden))
            }
        }
    }

    /** The export is in the library (host build): without a context it fails like the transcription does. */
    @Test
    fun thePreviewExportIsInTheLibrary() {
        TestEnv.assumeNativeLibrary()
        val calls = listOf<() -> Unit>(
            { JniNativeApi.whisperPreview(0, 1, FloatArray(16), 16, 1, "cs", null) },
            { JniNativeApi.whisperTranscribe(0, 1, FloatArray(16), 16, 1, "cs", null) },
        )
        for (call in calls) {
            val e = failure<IllegalStateException> { call() }
            assertEquals("whisper context not loaded", e.message)
        }
    }
}
