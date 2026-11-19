package ai.moravian.czechspeech.internal

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * GPT-2 on ggml in the host build of the JNI library against PyTorch (scripts/lm/make_reference.py): a tiny two-layer
 * model with random weights always, the real Czech GPT-2 when CZECHSPEECH_LM points at it.
 */
class NativeLmTest {
    private val handles = mutableListOf<Long>()

    @Before
    fun load() = TestEnv.assumeNativeLibrary()

    @After
    fun free() = handles.forEach { JniNativeApi.lmFree(it) }

    private fun open(file: File): Long = JniNativeApi.lmInit(file.absolutePath).also {
        assertNotEquals("loaded $file", 0L, it)
        handles += it
    }

    private fun flatten(seqs: List<IntArray>): Pair<IntArray, IntArray> {
        val offsets = IntArray(seqs.size + 1)
        for (i in seqs.indices) offsets[i + 1] = offsets[i] + seqs[i].size
        return seqs.fold(IntArray(0)) { acc, s -> acc + s } to offsets
    }

    private fun score(handle: Long, seqs: List<IntArray>, threads: Int = 2): DoubleArray {
        val (tokens, offsets) = flatten(seqs)
        return JniNativeApi.lmScore(handle, tokens, offsets, threads) ?: throw AssertionError("lmScore failed")
    }

    private fun tinyRows() = LmReference.read(File(TestEnv.lmResources, "reference-tiny.tsv"))

    private fun start(file: File) = Gguf.readVocabulary(file).startToken

    @Test
    fun f32MatchesPyTorch() {
        val lm = open(TestEnv.tinyLm)
        val bos = start(TestEnv.tinyLm)
        for (r in tinyRows()) {
            val got = score(lm, listOf(intArrayOf(bos) + r.ids))[0]
            // ggml's GELU goes through an f16 table; the rest is f32 as in PyTorch
            assertEquals(r.text, r.logprob, got, 2e-3 * abs(r.logprob) + 0.02)
        }
    }

    @Test
    fun q8MatchesPyTorchWithinTheQuantisationError() {
        val lm = open(TestEnv.tinyLmQ8)
        val bos = start(TestEnv.tinyLmQ8)
        for (r in tinyRows()) {
            val got = score(lm, listOf(intArrayOf(bos) + r.ids))[0]
            assertEquals(r.text, r.logprob, got, 0.03 * abs(r.logprob) + 0.1)
        }
    }

    @Test
    fun sequencesScoredTogetherShareTheirPrefixesWithoutChangingTheResult() {
        val lm = open(TestEnv.tinyLm)
        val bos = start(TestEnv.tinyLm)
        val seqs = tinyRows().map { intArrayOf(bos) + it.ids } + listOf(intArrayOf(bos))   // and an empty text
        val together = score(lm, seqs)
        val alone = seqs.map { score(lm, listOf(it))[0] }
        for (i in seqs.indices) assertEquals("sequence $i", alone[i], together[i], 1e-3)
        assertEquals("only the start token", 0.0, together.last(), 0.0)
        // the same sequence twice, and a prefix of another one
        val v = seqs[10]
        val twice = score(lm, listOf(v, v.copyOf(v.size - 2), v))
        assertEquals(twice[0], twice[2], 0.0)
        assertEquals(alone[10], twice[0], 1e-3)
    }

    @Test
    fun longSequencesSpanSeveralChunks() {
        // more tokens than one graph evaluates (32), so the cache carries over between chunks
        val lm = open(TestEnv.tinyLm)
        val bos = start(TestEnv.tinyLm)
        val rows = tinyRows()
        val long = intArrayOf(bos) + rows[0].ids + rows[1].ids + rows[2].ids
        assertTrue(long.size > 64)
        val whole = score(lm, listOf(long))[0]
        val threads1 = score(lm, listOf(long), threads = 1)[0]
        assertEquals(whole, threads1, 1e-3)
        assertTrue(whole < score(lm, listOf(intArrayOf(bos) + rows[0].ids))[0])
    }

    @Test
    fun theComputeBufferStaysAfterScoring() {
        val lm = open(TestEnv.tinyLm)
        val before = NativeBridge.nativeLmBufferBytes(lm)
        assertEquals("no compute buffer before the first call", 0L, before[1])
        assertTrue("weights ${before[0]}", before[0] in 100_000L..TestEnv.tinyLm.length())
        score(lm, tinyRows().map { intArrayOf(start(TestEnv.tinyLm)) + it.ids })
        assertTrue("compute buffer kept", NativeBridge.nativeLmBufferBytes(lm)[1] > 0)
    }

    @Test
    fun invalidInputIsRejected() {
        val lm = open(TestEnv.tinyLm)
        val bos = start(TestEnv.tinyLm)
        assertEquals(420, JniNativeApi.lmVocabSize(lm))
        assertEquals(128, JniNativeApi.lmContextLength(lm))
        assertNull("token out of range", JniNativeApi.lmScore(lm, intArrayOf(bos, 420), intArrayOf(0, 2), 1))
        assertNull("negative token", JniNativeApi.lmScore(lm, intArrayOf(bos, -1), intArrayOf(0, 2), 1))
        assertNull("empty sequence", JniNativeApi.lmScore(lm, intArrayOf(bos), intArrayOf(0, 0, 1), 1))
        assertNull("offsets past the tokens", JniNativeApi.lmScore(lm, intArrayOf(bos), intArrayOf(0, 2), 1))
        assertNull("longer than the context", JniNativeApi.lmScore(lm, IntArray(129) { bos }, intArrayOf(0, 129), 1))
        assertEquals(0, JniNativeApi.lmScore(lm, IntArray(0), intArrayOf(0), 1)!!.size)
        failure<IllegalArgumentException> { JniNativeApi.lmScore(lm, IntArray(0), IntArray(0), 1) }
    }

    @Test
    fun filesThatAreNotAGpt2ModelAreNotLoaded() {
        assertEquals(0L, JniNativeApi.lmInit(TestEnv.vadAsset.absolutePath))
        assertEquals(0L, JniNativeApi.lmInit(File(TestEnv.lmResources, "reference-tiny.tsv").absolutePath))
        assertEquals(0L, JniNativeApi.lmInit("/nonexistent/model.gguf"))
    }

    @Test
    fun czechGpt2MatchesPyTorch() {
        TestEnv.assumeFile(TestEnv.languageModel, "Czech GPT-2 language model (CZECHSPEECH_LM)")
        val lm = open(TestEnv.languageModel)
        val tok = Gpt2Tokenizer(Gguf.readVocabulary(TestEnv.languageModel))
        val rows = LmReference.read(File(TestEnv.lmResources, "reference-czech-gpt2.tsv"))
        val seqs = rows.map { intArrayOf(tok.startToken) + tok.encode(it.text) }
        val t0 = System.nanoTime()
        val got = score(lm, seqs, threads = TestEnv.threads)
        val ms = (System.nanoTime() - t0) / 1_000_000
        println("czech-gpt2 q8_0: ${seqs.size} sentences, ${seqs.sumOf { it.size }} tokens in $ms ms")
        val (weights, compute) = NativeBridge.nativeLmBufferBytes(lm).let { it[0] to it[1] }
        println("czech-gpt2 q8_0: weights %.1f MB, compute buffer kept between calls %.1f MB".format(weights / 1e6, compute / 1e6))
        assertTrue("compute buffer $compute", compute in 1L..32_000_000L)
        for ((i, r) in rows.withIndex()) {
            println("  %9.3f  PyTorch %9.3f  %s".format(got[i], r.logprob, "(${r.ids.size} tokens)"))
            assertEquals(r.text, r.logprob, got[i], 0.02 * abs(r.logprob) + 0.3)
        }
        // the variants of a sentence keep their order: what variant selection relies on
        for (group in listOf(10..13, 14..16)) {
            val ref = group.sortedBy { rows[it].logprob }
            val ours = group.sortedBy { got[it] }
            assertEquals("order of $group", ref, ours)
        }
        assertNotNull(got)
    }
}
