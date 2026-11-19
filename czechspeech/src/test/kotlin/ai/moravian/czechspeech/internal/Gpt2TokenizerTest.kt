package ai.moravian.czechspeech.internal

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The language model's tokenizer must give exactly the token ids of the Hugging Face tokenizer the measurement used
 * (AutoTokenizer of lchaloupsky/czech-gpt2-oscar), or the log-probabilities would be of other sequences. Checked on the
 * tiny test vocabulary always, and on the real model's vocabulary when CZECHSPEECH_LM points at it.
 */
class Gpt2TokenizerTest {
    private fun tokenizer(file: File) = Gpt2Tokenizer(Gguf.readVocabulary(file))

    @Test
    fun readsTheVocabularyOfAGpt2File() {
        val v = Gguf.readVocabulary(TestEnv.tinyLm)
        assertEquals("gpt2", v.architecture)
        assertEquals(420, v.tokens.size)
        assertEquals("<|endoftext|>", v.tokens[v.startToken])
        assertTrue(v.merges.isNotEmpty())
        assertTrue(v.merges.all { it.split(' ').size == 2 })
    }

    @Test
    fun matchesHuggingFaceOnTheTestVocabulary() {
        val tok = tokenizer(TestEnv.tinyLm)
        for (r in LmReference.read(File(TestEnv.lmResources, "reference-tiny.tsv"))) {
            assertArrayEquals(r.text, r.ids, tok.encode(r.text))
        }
    }

    @Test
    fun matchesHuggingFaceOnCzechGpt2() {
        TestEnv.assumeFile(TestEnv.languageModel, "Czech GPT-2 language model (CZECHSPEECH_LM)")
        val rt = Runtime.getRuntime()
        fun used(): Long {
            repeat(3) { System.gc() }
            return rt.totalMemory() - rt.freeMemory()
        }
        val before = used()
        val tok = tokenizer(TestEnv.languageModel)
        println("tokenizer heap: %.1f MB (JVM)".format((used() - before) / 1e6))
        assertEquals(50257, tok.vocabularySize)
        for (r in LmReference.read(File(TestEnv.lmResources, "reference-czech-gpt2.tsv"))) {
            assertArrayEquals(r.text, r.ids, tok.encode(r.text))
        }
    }

    @Test
    fun everyByteHasAToken() {
        val tok = tokenizer(TestEnv.tinyLm)
        val text = String((0 until 0x80).map { it.toChar() }.toCharArray()) + "ěščřžýáíéúůťďňó€😀　 "
        val ids = tok.encode(text)
        assertTrue(ids.isNotEmpty())
        assertTrue(ids.all { it in 0 until tok.vocabularySize })
    }

    @Test
    fun emptyTextHasNoTokens() {
        assertEquals(0, tokenizer(TestEnv.tinyLm).encode("").size)
    }

    @Test
    fun theEndOfTextMarkerIsTheStartToken() {
        val tok = tokenizer(TestEnv.tinyLm)
        val ids = tok.encode("a<|endoftext|>a")
        assertArrayEquals(tok.encode("a") + tok.startToken + tok.encode("a"), ids)
    }

    @Test
    fun repeatedWordsGiveTheSameTokens() {
        // the word cache must not change the result
        val tok = tokenizer(TestEnv.tinyLm)
        val once = tok.encode("Viděl jsem velké psy u řeky.")
        repeat(3) { assertArrayEquals(once, tok.encode("Viděl jsem velké psy u řeky.")) }
    }

    @Test
    fun aFileThatIsNotGgufIsRejected() {
        val e = failure<IOException> { Gguf.readVocabulary(TestEnv.vadAsset) }
        assertTrue(e.message, e.message!!.contains("GGUF"))
    }

    @Test
    fun aTruncatedFileIsRejected() {
        val bytes = TestEnv.tinyLm.readBytes()
        val cut = File.createTempFile("cut", ".gguf").apply { deleteOnExit() }
        cut.writeBytes(bytes.copyOf(3000))
        failure<IOException> { Gguf.readVocabulary(cut) }
    }

    @Test
    fun aGgufFileWithoutAVocabularyIsRejected() {
        val file = File.createTempFile("novocab", ".gguf").apply { deleteOnExit() }
        val key = "general.architecture".toByteArray()
        val value = "gpt2".toByteArray()
        val buf = ByteBuffer.allocate(4 + 4 + 8 + 8 + 8 + key.size + 4 + 8 + value.size).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("GGUF".toByteArray()).putInt(3).putLong(0).putLong(1)
        buf.putLong(key.size.toLong()).put(key).putInt(8).putLong(value.size.toLong()).put(value)
        file.writeBytes(buf.array())
        val e = failure<IOException> { Gguf.readVocabulary(file) }
        assertTrue(e.message, e.message!!.contains("vocabulary"))
    }
}
