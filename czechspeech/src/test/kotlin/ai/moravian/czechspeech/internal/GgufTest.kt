package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The GGUF metadata reader on hand-made files: every value type is skipped, and malformed files are refused. */
class GgufTest {
    /** Little-endian GGUF writer for the metadata section. */
    private class Writer(version: Int = 3, keys: Int) {
        private val out = ByteArrayOutputStream()

        init {
            out.write("GGUF".toByteArray())
            u32(version).u64(0).u64(keys.toLong())
        }

        fun u32(v: Int) = apply { out.write(le(4) { putInt(v) }) }
        fun u64(v: Long) = apply { out.write(le(8) { putLong(v) }) }
        fun str(s: String) = apply { s.toByteArray().let { u64(it.size.toLong()); out.write(it) } }
        fun raw(n: Int) = apply { out.write(ByteArray(n) { 7 }) }
        fun key(k: String, type: Int) = str(k).u32(type)
        fun strings(k: String, values: List<String>) = apply { key(k, 9).u32(8).u64(values.size.toLong()); values.forEach { str(it) } }

        fun file(): File = File.createTempFile("test", ".gguf").apply {
            deleteOnExit()
            writeBytes(out.toByteArray())
        }

        private fun le(n: Int, put: ByteBuffer.() -> Unit) = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN).apply(put).array()
    }

    private val tokens = listOf("a", "b", "ab", "<|endoftext|>")

    /** A tokenizer that reads, after [extraKeys] keys written by [extra]. */
    private fun tokenizerFile(extraKeys: Int = 0, model: String = "gpt2", bos: Boolean = true, extra: Writer.() -> Unit = {}) =
        Writer(keys = 3 + extraKeys + if (bos) 1 else 0).apply {
            extra()
            key("tokenizer.ggml.model", 8).str(model)
            strings("tokenizer.ggml.tokens", tokens)
            strings("tokenizer.ggml.merges", listOf("a b"))
            if (bos) key("tokenizer.ggml.bos_token_id", 4).u32(3)
        }.file()

    @Test
    fun everyValueTypeIsSkipped() {
        val file = tokenizerFile(extraKeys = 16) {
            key("general.architecture", 8).str("gpt2")
            key("u8", 0).raw(1)
            key("i8", 1).raw(1)
            key("u16", 2).raw(2)
            key("i16", 3).raw(2)
            key("u32", 4).raw(4)
            key("i32", 5).raw(4)
            key("f32", 6).raw(4)
            key("bool", 7).raw(1)
            key("string", 8).str("x")
            key("u64", 10).raw(8)
            key("i64", 11).raw(8)
            key("f64", 12).raw(8)
            key("ints", 9).u32(4).u64(3).raw(12)
            key("texts", 9).u32(8).u64(2).str("p").str("q")
            key("general.architecture", 4).raw(4)          // a known key with an unexpected type is skipped too
        }
        val v = Gguf.readVocabulary(file)
        assertEquals("gpt2", v.architecture)
        assertEquals(tokens, v.tokens)
        assertEquals(listOf("a b"), v.merges)
        assertEquals(3, v.startToken)
    }

    @Test
    fun theStartTokenDefaultsToTheEndOfTextToken() {
        assertEquals(3, Gguf.readVocabulary(tokenizerFile(bos = false)).startToken)
    }

    @Test
    fun version2IsRead() {
        val file = Writer(version = 2, keys = 3).apply {
            key("tokenizer.ggml.model", 8).str("gpt2")
            strings("tokenizer.ggml.tokens", tokens)
            strings("tokenizer.ggml.merges", listOf("a b"))
        }.file()
        assertEquals(4, Gguf.readVocabulary(file).tokens.size)
    }

    private fun refused(file: File, what: String) {
        val e = failure<IOException> { Gguf.readVocabulary(file) }
        assertTrue("${e.message} should mention $what", e.message!!.contains(what))
    }

    @Test
    fun malformedFilesAreRefused() {
        refused(Writer(version = 4, keys = 0).file(), "version 4")
        refused(Writer(keys = -1).file(), "metadata keys")
        refused(Writer(keys = 1).apply { key("x", 99) }.file(), "unknown value type 99")
        refused(Writer(keys = 1).apply { key("x", 9).u32(9).u64(1) }.file(), "unsupported array")
        refused(Writer(keys = 1).apply { key("x", 9).u32(4).u64(-1) }.file(), "unsupported array")
        refused(Writer(keys = 1).apply { key("x", 8).u64(1L shl 40) }.file(), "string of")
        refused(Writer(keys = 1).apply { key("tokenizer.ggml.tokens", 9).u32(4).u64(1).raw(4) }.file(), "array of type 4")
        refused(Writer(keys = 1).apply { key("tokenizer.ggml.tokens", 9).u32(8).u64(-5) }.file(), "array of")
        refused(Writer(keys = 1).apply { strings("tokenizer.ggml.tokens", tokens) }.file(), "merges")
        refused(tokenizerFile(model = "llama"), "is not gpt2")
        val noStart = Writer(keys = 3).apply {
            key("tokenizer.ggml.model", 8).str("gpt2")
            strings("tokenizer.ggml.tokens", listOf("a", "b"))
            strings("tokenizer.ggml.merges", listOf("a b"))
        }.file()
        refused(noStart, "start token")
    }
}
