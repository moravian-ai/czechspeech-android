package ai.moravian.czechspeech.internal

import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import java.io.IOException

/** The tokenizer of a GPT-2 GGUF file: tokens by id, BPE merges by rank ("a b"), and the start token. */
internal class GgufVocabulary(
    val architecture: String,
    val tokens: List<String>,
    val merges: List<String>,
    val startToken: Int,
)

/**
 * Reads the metadata of a GGUF file (version 2 or 3) up to the tensor list: the language model's tokenizer lives in
 * Kotlin, its weights are read by native code. Keys follow llama.cpp (tokenizer.ggml.*), as written by
 * scripts/lm/convert_gpt2.py.
 */
internal object Gguf {
    private const val MAX_STRING_BYTES = 1 shl 20
    private const val MAX_ARRAY = 10_000_000L
    private const val MAX_KEYS = 100_000L

    // value types
    private const val UINT8 = 0
    private const val INT8 = 1
    private const val UINT16 = 2
    private const val INT16 = 3
    private const val UINT32 = 4
    private const val INT32 = 5
    private const val FLOAT32 = 6
    private const val BOOL = 7
    private const val STRING = 8
    private const val ARRAY = 9
    private const val UINT64 = 10
    private const val INT64 = 11
    private const val FLOAT64 = 12

    /** @throws IOException when the file is not GGUF or has no GPT-2 tokenizer. */
    fun readVocabulary(file: File): GgufVocabulary =
        DataInputStream(BufferedInputStream(file.inputStream(), 1 shl 16)).use { input ->
            try {
                read(Reader(input))
            } catch (e: EOFException) {
                throw IOException("${file.name}: GGUF file ends early", e)
            }
        }

    private fun read(r: Reader): GgufVocabulary {
        val magic = ByteArray(4).also { r.input.readFully(it) }
        if (String(magic, Charsets.US_ASCII) != "GGUF") throw IOException("not a GGUF file")
        val version = r.u32()
        if (version != 2 && version != 3) throw IOException("GGUF version $version is not supported")
        r.u64()                                              // tensor count
        val keys = r.u64()
        if (keys !in 0..MAX_KEYS) throw IOException("GGUF: $keys metadata keys")
        var architecture: String? = null
        var model: String? = null
        var tokens: List<String>? = null
        var merges: List<String>? = null
        var start: Long? = null
        repeat(keys.toInt()) {
            val key = r.string()
            val type = r.u32()
            when {
                key == "general.architecture" && type == STRING -> architecture = r.string()
                key == "tokenizer.ggml.model" && type == STRING -> model = r.string()
                key == "tokenizer.ggml.tokens" && type == ARRAY -> tokens = r.stringArray()
                key == "tokenizer.ggml.merges" && type == ARRAY -> merges = r.stringArray()
                key == "tokenizer.ggml.bos_token_id" && (type == UINT32 || type == INT32) -> start = r.u32().toLong()
                else -> r.skip(type)
            }
        }
        val t = tokens ?: throw IOException("GGUF: no vocabulary (tokenizer.ggml.tokens)")
        val m = merges ?: throw IOException("GGUF: no BPE merges (tokenizer.ggml.merges)")
        if (model != "gpt2") throw IOException("GGUF: tokenizer '$model' is not gpt2")
        val s = start ?: t.indexOf("<|endoftext|>").toLong()
        if (s < 0 || s >= t.size) throw IOException("GGUF: no start token")
        return GgufVocabulary(architecture ?: "", t, m, s.toInt())
    }

    private class Reader(val input: DataInputStream) {
        private val scratch = ByteArray(8)

        fun u32(): Int {
            input.readFully(scratch, 0, 4)
            return (scratch[0].toInt() and 0xff) or ((scratch[1].toInt() and 0xff) shl 8) or
                ((scratch[2].toInt() and 0xff) shl 16) or ((scratch[3].toInt() and 0xff) shl 24)
        }

        fun u64(): Long = (u32().toLong() and 0xffffffffL) or (u32().toLong() shl 32)

        fun string(): String {
            val n = u64()
            if (n !in 0..MAX_STRING_BYTES.toLong()) throw IOException("GGUF: string of $n bytes")
            val bytes = ByteArray(n.toInt())
            input.readFully(bytes)
            return String(bytes, Charsets.UTF_8)
        }

        fun stringArray(): List<String> {
            val type = u32()
            val n = u64()
            if (type != STRING) throw IOException("GGUF: array of type $type where strings were expected")
            if (n !in 0..MAX_ARRAY) throw IOException("GGUF: array of $n entries")
            return List(n.toInt()) { string() }
        }

        fun skip(type: Int) {
            when (type) {
                UINT8, INT8, BOOL -> skipBytes(1)
                UINT16, INT16 -> skipBytes(2)
                UINT32, INT32, FLOAT32 -> skipBytes(4)
                UINT64, INT64, FLOAT64 -> skipBytes(8)
                STRING -> string()
                ARRAY -> {
                    val inner = u32()
                    val n = u64()
                    if (n !in 0..MAX_ARRAY || inner == ARRAY) throw IOException("GGUF: unsupported array")
                    for (i in 0 until n) skip(inner)
                }
                else -> throw IOException("GGUF: unknown value type $type")
            }
        }

        private fun skipBytes(n: Int) = input.readFully(scratch, 0, n)
    }
}
