package ai.moravian.czechspeech

import java.io.File

/**
 * Variant selection: Whisper proposes the [beamSize] best variants of each utterance (beam search) and a small Czech
 * language model picks one. Off unless set in [CzechSpeechConfig.variantSelection].
 *
 * It mostly fixes word endings, i/y and diacritics where the sound alone does not decide ("velký psy" for "velké
 * psy"). Measured on parliament speech, read sentences and speech in noise, it lowered the word error rate by 16 to
 * 20 % (0.34 to 1.60 points) and fixed a quarter of such confusions. The price is time and memory: beam search
 * repeats Whisper's decoder for every beam, and the language model ([CzechLanguageModel], about 135 MB) is loaded with
 * the CzechSpeech model and freed with it.
 *
 * The variants are compared by log P_Whisper + [languageModelWeight] · log P_LM + [lengthBonus] · length in Whisper
 * tokens; the defaults are the measured optimum for five beams. Whisper's own choice stays when the variants differ
 * only in case and punctuation (the language model is not run), when the language model fails, and in a window
 * Whisper decoded at a fallback temperature. A variant without any letter or digit is never chosen unless Whisper
 * chose it, and when Whisper chose one (no speech), no other beam's text overrides it.
 *
 * @property languageModelFile the language model, for example from
 *   `CzechSpeechModel.download(context, CzechLanguageModel.source(url))`, or your own copy; required unless
 *   [languageModelWeight] is 0. Its SHA-256 is checked when it is downloaded, as the model's is, not each time it is
 *   loaded; loading checks its structure and fails with a CzechSpeechException for a file it cannot run.
 * @property beamSize variants Whisper proposes, 2 to 8; default 5. Three cost less and give 60 to 95 % of the gain.
 * @property languageModelWeight λ, the weight of the language model's log-probability; default 0.3. 0 turns the
 *   language model off: none is loaded or run (no file needed), and Whisper's score with the length bonus chooses
 *   among the beams (beam search alone, which helps less: in noise, not on clean speech).
 * @property lengthBonus β, added per Whisper token, so that the language model does not favour short variants;
 *   default 0.3 (measured 0.75 for three beams).
 */
public data class VariantSelection(
    public val languageModelFile: File? = null,
    public val beamSize: Int = 5,
    public val languageModelWeight: Float = 0.3f,
    public val lengthBonus: Float = 0.3f,
) {
    init {
        require(beamSize in 2..8) { "beamSize must be in 2..8" }
        require(languageModelWeight.isFinite() && languageModelWeight >= 0f) { "languageModelWeight must be finite and >= 0" }
        require(lengthBonus.isFinite()) { "lengthBonus must be finite" }
        require(languageModelFile != null || languageModelWeight == 0f) {
            "languageModelFile is needed unless languageModelWeight is 0"
        }
    }
}

/**
 * The Czech language model of [VariantSelection]: GPT-2 small trained on Czech (lchaloupsky/czech-gpt2-oscar, MIT,
 * 124 M parameters), converted to GGUF q8_0 by `scripts/lm/convert_gpt2.py`. The conversion is reproducible, so the
 * file has a fixed size and SHA-256, and it is downloaded and checked like the CzechSpeech model (when it is downloaded,
 * not each time it is loaded):
 *
 * ```
 * val lm = CzechSpeechModel.download(context, CzechLanguageModel.source("https://models.example.com/$FILE_NAME"))
 * CzechSpeech(context, modelFile, CzechSpeechConfig(variantSelection = VariantSelection(lm)))
 * ```
 *
 * It is not published by Moravian AI yet, so there is no default URL: serve the file yourself, or push it to the
 * device with adb.
 */
public object CzechLanguageModel {
    /** File name of the converted model. */
    public const val FILE_NAME: String = "ggml-czech-gpt2-oscar-q8_0.gguf"

    /** Size of the converted model in bytes (137 MB). */
    public const val SIZE_BYTES: Long = 136_688_032L

    /** SHA-256 of the converted model. */
    public const val SHA256: String = "edc50ef42cd9497fb175533094bcce7253a698ad598da9181430651c27ae5701"

    /** The converted model served from [url], checked against [SIZE_BYTES] and [SHA256] when downloaded. */
    public fun source(url: String): CzechSpeechModel.Source = CzechSpeechModel.Source(url, SIZE_BYTES, SHA256, FILE_NAME)
}
