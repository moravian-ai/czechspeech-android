package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import java.io.File
import java.io.IOException

/**
 * The GPT-2 language model of variant selection: native weights ([NativeApi.lmInit]) and the Kotlin tokenizer read
 * from the same GGUF file. Owned by the engine's recognition thread, loaded and freed with the CzechSpeech model.
 */
internal class LanguageModel private constructor(
    private val native: NativeApi,
    private val handle: Long,
    private val tokenizer: Gpt2Tokenizer,
    private val contextLength: Int,
) {
    /**
     * Sum of the token log-probabilities of each text (after the start token), or null when the native call failed.
     * Texts longer than the model's context are scored on their beginning. The token arrays are zeroed afterwards,
     * and the tokenizer forgets the words it cached.
     */
    fun score(texts: List<String>, threads: Int): DoubleArray? {
        val seqs = try {
            texts.map { text ->
                val s = intArrayOf(tokenizer.startToken) + tokenizer.encode(text)
                if (s.size > contextLength) s.copyOf(contextLength).also { s.fill(0) } else s
            }
        } finally {
            tokenizer.forget()
        }
        val offsets = IntArray(seqs.size + 1)
        for (i in seqs.indices) offsets[i + 1] = offsets[i] + seqs[i].size
        val tokens = IntArray(offsets.last())
        seqs.forEachIndexed { i, s -> s.copyInto(tokens, offsets[i]) }
        try {
            return native.lmScore(handle, tokens, offsets, threads)
        } finally {
            tokens.fill(0)
            seqs.forEach { it.fill(0) }
        }
    }

    fun free() = native.lmFree(handle)

    companion object {
        /** @throws CzechSpeechException when the file is missing or is not a GPT-2 GGUF file that loads. */
        fun load(native: NativeApi, file: File): LanguageModel {
            if (!file.isFile) throw CzechSpeechException("Language model file not found: ${file.path}")
            val tokenizer = try {
                Gpt2Tokenizer(Gguf.readVocabulary(file))
            } catch (e: IOException) {
                throw CzechSpeechException("Could not load the language model from ${file.path}: ${e.message}", e)
            } catch (e: IllegalArgumentException) {
                throw CzechSpeechException("Could not load the language model from ${file.path}: ${e.message}", e)
            }
            val handle = native.lmInit(file.absolutePath)
            if (handle == 0L) throw CzechSpeechException("Could not load the language model from ${file.path}")
            val vocabulary = native.lmVocabSize(handle)
            if (vocabulary != tokenizer.vocabularySize) {
                native.lmFree(handle)
                throw CzechSpeechException(
                    "Could not load the language model from ${file.path}: its vocabulary has " +
                        "${tokenizer.vocabularySize} tokens, its weights $vocabulary",
                )
            }
            return LanguageModel(native, handle, tokenizer, native.lmContextLength(handle))
        }
    }
}
