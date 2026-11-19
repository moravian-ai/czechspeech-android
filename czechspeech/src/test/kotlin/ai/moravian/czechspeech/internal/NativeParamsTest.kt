package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The whisper_full parameters live in C and cannot be read back without a model, so this guards the source: whisper.cpp
 * keeps `suppress_blank` on by default, which forbids an empty transcript as the first token. A model that learned to
 * answer a segment without speech with an empty transcript (cs+sk+en) then has to write something into silence.
 *
 * It also guards the whisper.cpp patch the native build applies: refreshing it for a newer whisper.cpp must keep both
 * beam search fixes (the integration test sees their effect only with the real model).
 */
class NativeParamsTest {
    private val source = File("../native/src/czechspeech_jni.c").takeIf { it.exists() }?.readText()
    private val patch = File("../native/patches/whisper.cpp-czechspeech.patch").takeIf { it.exists() }?.readText()
    private val selectPatch = File("../native/patches/whisper.cpp-language-select.patch").takeIf { it.exists() }?.readText()

    @Test
    fun suppressBlankIsOff() {
        val text = source ?: error("native/src/czechspeech_jni.c not found from ${File(".").absolutePath}")
        assertTrue("params.suppress_blank = false;", Regex("""params\.suppress_blank\s*=\s*false\s*;""").containsMatchIn(text))
    }

    @Test
    fun greedyAndBeamSearchShareTheParameters() {
        val text = source ?: error("native/src/czechspeech_jni.c not found")
        assertEquals("one whisper_full_default_params, for both", 1, Regex("""whisper_full_default_params\(""").findAll(text).count())
        assertEquals(2, Regex("""cs_full_params\(n_threads, n_samples\)""").findAll(text).count())
    }

    @Test
    fun theBuildNeverReusesAnUnpatchedTree() {
        val script = File("../native/build-native.sh").takeIf { it.exists() }?.readText()
            ?: error("native/build-native.sh not found")
        // the patched header is checked when the tree is reused and before the stamp that allows reuse is written
        assertTrue("reuse checks the patch", Regex("""if \[ "\${'$'}stamp" = "\${'$'}want" \] && patched_ok""").containsMatchIn(script))
        val check = script.indexOf("patched_ok || die")
        val stamp = script.indexOf("echo \"\$want\" > \"\$PATCHED.stamp\"")
        assertTrue("checked ($check) before the stamp ($stamp)", check in 0 until stamp)
        assertTrue(Regex("""patched_ok\(\) \{[^}]*whisper_full_get_hypothesis""").containsMatchIn(script))
    }

    @Test
    fun beamSearchPatchTakesTheMostLikelyTokensAndDistinctFirstTokens() {
        val text = patch ?: error("native/patches/whisper.cpp-czechspeech.patch not found")
        assertTrue("top-k at temperature 0", text.contains("+        const auto id = sample ? dist(decoder.rng) : logits_id[i].second;"))
        assertTrue("sampling only above it", text.contains("whisper_sample_token_topk(*ctx, decoder, params.beam_search.beam_size, t_cur > 0.0f)"))
        assertTrue(
            "no duplicate beams in the first step",
            text.contains("-                        while (beam_candidates.size() > cur_c && whisper_sequence_tokens_equal(beam_candidates[cur_c].sequence, cur.sequence) && i > 0) {") &&
                text.contains("+                        while (beam_candidates.size() > cur_c && whisper_sequence_tokens_equal(beam_candidates[cur_c].sequence, cur.sequence)) {"),
        )
    }

    @Test
    fun languageSelectPatchSetsTheWindowBeforeTheDetectionAndReusesTheEncoder() {
        val text = selectPatch ?: error("native/patches/whisper.cpp-language-select.patch not found")
        assertTrue(text.contains("language_select_callback"))
        // the window (an added line) comes before the detection (a line the patch only has as context or changes)
        val window = text.indexOf("+        state->exp_n_audio_ctx = params.audio_ctx;")
        val detection = text.indexOf("whisper_lang_auto_detect_internal(")
        assertTrue("window set ($window) before the detection ($detection)", window in 0 until detection)
        assertTrue("the first window skips an encoder run the detection did", text.contains("seek == state->cs_encoded_seek"))
        assertTrue("and the count of encoder runs is exported", text.contains("whisper_full_n_encode"))
    }

    @Test
    fun theLanguageSelectPatchRefusesAnIdAboveTheLastLanguage() {
        val text = selectPatch ?: error("native/patches/whisper.cpp-language-select.patch not found")
        // the id the callback returns is checked on both sides, so whisper_lang_str can never return nullptr
        assertTrue(
            "an id above whisper_lang_max_id() aborts with -3 like a negative one",
            Regex("""\+\s+if \(lang_id < 0 \|\| lang_id > whisper_lang_max_id\(\)\) \{[^}]*return -3;""")
                .containsMatchIn(text),
        )
    }

    @Test
    fun theBuildChecksTheLanguageSelectPatch() {
        val script = File("../native/build-native.sh").takeIf { it.exists() }?.readText()
            ?: error("native/build-native.sh not found")
        assertTrue(Regex("""patched_ok\(\) \{[^}]*language_select_callback""").containsMatchIn(script))
    }

    @Test
    fun patchesApplyInNameOrderWithLanguageSelectLast() {
        val names = File("../native/patches").listFiles { f -> f.name.endsWith(".patch") }?.map { it.name }?.sorted()
            ?: error("native/patches not found from ${File(".").absolutePath}")
        assertEquals(
            listOf("whisper.cpp-czechspeech.patch", "whisper.cpp-encoder-abort.patch", "whisper.cpp-language-select.patch"),
            names,
        )
    }

    @Test
    fun oneLanguageNeverSetsTheSelectCallbackAndNoSecondStateIsMade() {
        val text = source ?: error("native/src/czechspeech_jni.c not found")
        assertFalse("no second whisper state", text.contains("whisper_init_state("))
        assertFalse("no detection of its own", text.contains("whisper_lang_auto_detect"))
        val sets = Regex("""language_select_callback\s*=""").findAll(text).toList()
        assertTrue("the callback is set", sets.isNotEmpty())
        for (set in sets) {
            // the nearest `if` before an assignment is the multi-language branch
            val before = text.substring(0, set.range.first)
            val branch = Regex("""if \(([^)]*)\)""").findAll(before).last().groupValues[1]
            assertEquals("n_languages > 1", branch)
        }
        assertEquals("both recognitions with a list use it", 2, sets.size)
    }
}
