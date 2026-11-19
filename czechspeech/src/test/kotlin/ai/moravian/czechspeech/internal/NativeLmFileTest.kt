package ai.moravian.czechspeech.internal

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin

/**
 * Language model files that are well-formed GGUF but not a model this code can run must be refused (lmInit gives 0),
 * never abort the process in a ggml assertion: wrong tensor types, more layers than one graph holds, a cut file.
 * The models are tiny and written here.
 */
class NativeLmFileTest {
    private val handles = mutableListOf<Long>()

    @Before
    fun load() = TestEnv.assumeNativeLibrary()

    @After
    fun free() = handles.forEach { JniNativeApi.lmFree(it) }

    /** A GPT-2 GGUF file: [types] overrides tensor types (0 f32, 1 f16, 2 q4_0, 8 q8_0), [drop] leaves tensors out. */
    private class Model(
        val layers: Int = 1,
        val embd: Int = 32,
        val heads: Int = 4,
        val blockCount: Int = layers,
        val vocab: Int = 8,
        val ctx: Int = 16,
        /** The embedding size the metadata claims (the tensors have [embd]). */
        val claimedEmbd: Int = embd,
        val types: Map<String, Int> = emptyMap(),
        val drop: Set<String> = emptySet(),
    ) {
        private val tensors: List<Triple<String, Long, Long>> = buildList {
            add(Triple("token_embd.weight", embd.toLong(), vocab.toLong()))
            add(Triple("position_embd.weight", embd.toLong(), ctx.toLong()))
            for (i in 0 until layers) {
                val e = embd.toLong()
                add(Triple("blk.$i.attn_norm.weight", e, 1L))
                add(Triple("blk.$i.attn_norm.bias", e, 1L))
                add(Triple("blk.$i.attn_qkv.weight", e, 3 * e))
                add(Triple("blk.$i.attn_qkv.bias", 3 * e, 1L))
                add(Triple("blk.$i.attn_output.weight", e, e))
                add(Triple("blk.$i.attn_output.bias", e, 1L))
                add(Triple("blk.$i.ffn_norm.weight", e, 1L))
                add(Triple("blk.$i.ffn_norm.bias", e, 1L))
                add(Triple("blk.$i.ffn_up.weight", e, 4 * e))
                add(Triple("blk.$i.ffn_up.bias", 4 * e, 1L))
                add(Triple("blk.$i.ffn_down.weight", 4 * e, e))
                add(Triple("blk.$i.ffn_down.bias", e, 1L))
            }
            add(Triple("output_norm.weight", embd.toLong(), 1L))
            add(Triple("output_norm.bias", embd.toLong(), 1L))
        }.filter { it.first !in drop }

        private val out = ByteArrayOutputStream()
        private fun le(n: Int, put: ByteBuffer.() -> Unit) = out.write(ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN).apply(put).array())
        private fun u32(v: Int) = le(4) { putInt(v) }
        private fun u64(v: Long) = le(8) { putLong(v) }
        private fun str(s: String) = s.toByteArray().let { u64(it.size.toLong()); out.write(it) }
        private fun kvU32(k: String, v: Int) { str(k); u32(4); u32(v) }
        private fun kvStr(k: String, v: String) { str(k); u32(8); str(v) }
        private fun kvStrings(k: String, v: List<String>) { str(k); u32(9); u32(8); u64(v.size.toLong()); v.forEach { str(it) } }

        private fun bytes(type: Int, n: Long): Int = when (type) {
            0 -> 4 * n
            1 -> 2 * n
            2 -> n / 32 * 18
            8 -> n / 32 * 34
            else -> error("type $type")
        }.toInt()

        fun file(cutTo: Int? = null): File {
            out.write("GGUF".toByteArray()); u32(3); u64(tensors.size.toLong()); u64(11)
            kvStr("general.architecture", "gpt2")
            kvU32("gpt2.context_length", ctx)
            kvU32("gpt2.embedding_length", claimedEmbd)
            kvU32("gpt2.feed_forward_length", 4 * embd)
            kvU32("gpt2.block_count", blockCount)
            kvU32("gpt2.attention.head_count", heads)
            str("gpt2.attention.layer_norm_epsilon"); u32(6); le(4) { putFloat(1e-5f) }
            kvStr("tokenizer.ggml.model", "gpt2")
            kvStrings("tokenizer.ggml.tokens", List(vocab) { "t$it" })
            kvStrings("tokenizer.ggml.merges", listOf("t 0"))
            kvU32("tokenizer.ggml.bos_token_id", 0)
            var offset = 0L
            val data = ByteArrayOutputStream()
            for ((name, ne0, ne1) in tensors) {
                val type = types[name] ?: 0
                str(name); u32(2); u64(ne0); u64(ne1); u32(type); u64(offset)
                val n = bytes(type, ne0 * ne1)
                val buf = ByteBuffer.allocate(n + (-n).mod(32)).order(ByteOrder.LITTLE_ENDIAN)
                if (type == 0) for (i in 0 until ne0 * ne1) buf.putFloat((0.05 * sin(i * 0.7 + name.length)).toFloat())
                data.write(buf.array())
                offset += buf.capacity()
            }
            out.write(ByteArray((-out.size()).mod(32)))
            out.write(data.toByteArray())
            val all = out.toByteArray()
            return File.createTempFile("lmfile", ".gguf").apply {
                deleteOnExit()
                writeBytes(if (cutTo == null) all else all.copyOf(cutTo))
            }
        }
    }

    private fun refused(model: Model, cutTo: Int? = null) =
        assertEquals("refused", 0L, JniNativeApi.lmInit(model.file(cutTo).absolutePath))

    private fun loadsAndScores(model: Model) {
        val lm = JniNativeApi.lmInit(model.file().absolutePath)
        assertNotEquals("loaded", 0L, lm)
        handles += lm
        val got = JniNativeApi.lmScore(lm, intArrayOf(0, 1, 2, 3), intArrayOf(0, 4), 2) ?: throw AssertionError("score")
        assertTrue("finite ${got[0]}", got[0].isFinite() && got[0] < 0)
    }

    @Test
    fun aWellFormedTinyModelLoads() {
        loadsAndScores(Model())
        loadsAndScores(Model(types = mapOf("blk.0.attn_qkv.weight" to 1, "token_embd.weight" to 8, "blk.0.ffn_up.weight" to 8)))
    }

    @Test
    fun normsBiasesAndPositionsMustBeF32() {
        refused(Model(types = mapOf("blk.0.attn_norm.weight" to 1)))
        refused(Model(types = mapOf("blk.0.ffn_up.bias" to 1)))
        refused(Model(types = mapOf("output_norm.bias" to 1)))
        refused(Model(types = mapOf("position_embd.weight" to 1)))
    }

    @Test
    fun matricesMustBeF32F16OrQ8() {
        refused(Model(types = mapOf("blk.0.attn_qkv.weight" to 2)))
        refused(Model(types = mapOf("token_embd.weight" to 2)))
    }

    @Test
    fun moreLayersThanOneGraphHoldsAreRefused() {
        loadsAndScores(Model(layers = 48))
        refused(Model(layers = 49))
        refused(Model(layers = 1, blockCount = 1_000_000))
    }

    /**
     * An embedding size the tensors do not have is refused at the first tensor whose shape does not match (the token
     * embedding). This does not reach the 64-bit shape products of cs_lm_check (3 x 1,431,655,766 overflows 32 bits):
     * a file that got that far would have to describe tensors of several GB, so that arithmetic is left to review.
     */
    @Test
    fun anEmbeddingSizeTheTensorsDoNotHaveIsRefused() {
        refused(Model(heads = 2, claimedEmbd = 1_431_655_766))
        refused(Model(heads = 1, claimedEmbd = Int.MAX_VALUE))
    }

    @Test
    fun brokenStructureIsRefused() {
        refused(Model(blockCount = 2))                              // a layer's tensors are missing
        refused(Model(heads = 5))                                   // 32 does not split into 5 heads
        refused(Model(drop = setOf("output_norm.bias")))
        val whole = Model().file().length().toInt()
        refused(Model(), cutTo = whole - 100)                       // tensor data cut short
    }
}
