package ai.moravian.czechspeech.internal

import java.util.regex.Pattern

/**
 * GPT-2's byte-level BPE tokenizer, giving the same token ids as the Hugging Face tokenizer of a GPT-2 model
 * (GPT2Tokenizer and GPT2TokenizerFast): the text is split by GPT-2's pattern, each piece's UTF-8 bytes are mapped to
 * printable characters, and the pair with the lowest merge rank is merged, everywhere in the piece, until none is
 * left. "<|endoftext|>" in the text is the start token, as Hugging Face treats the special token. Not thread-safe
 * (word cache, see [forget]); the engine uses it on its recognition thread only.
 */
internal class Gpt2Tokenizer(vocabulary: GgufVocabulary) {
    val startToken: Int = vocabulary.startToken
    val vocabularySize: Int = vocabulary.tokens.size

    private val ids = HashMap<String, Int>(vocabulary.tokens.size * 2).apply {
        vocabulary.tokens.forEachIndexed { i, token -> put(token, i) }
    }

    /** (left id, right id) -> (rank, merged id), for merges whose parts and result are tokens. */
    private val merges = HashMap<Long, Long>(vocabulary.merges.size * 2).apply {
        vocabulary.merges.forEachIndexed { rank, merge ->
            val space = merge.indexOf(' ')
            if (space <= 0) return@forEachIndexed
            val a = ids[merge.substring(0, space)] ?: return@forEachIndexed
            val b = ids[merge.substring(space + 1)] ?: return@forEachIndexed
            val merged = ids[merge.substring(0, space) + merge.substring(space + 1)] ?: return@forEachIndexed
            putIfAbsent(pair(a, b), pair(rank, merged))
        }
    }

    /** Byte -> its token id (the byte's printable stand-in character). */
    private val byteIds = IntArray(256) { b ->
        ids[BYTE_CHARS[b].toString()] ?: throw IllegalArgumentException("the vocabulary has no token for byte $b")
    }

    private val cache = object : LinkedHashMap<String, IntArray>(256, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, IntArray>) = size > CACHE_WORDS
    }

    /** Drops the cached words: they are pieces of transcripts, kept only while one utterance is scored. */
    fun forget() = cache.clear()

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length / 2 + 4)
        var from = 0
        while (true) {
            val special = text.indexOf(END_OF_TEXT, from)
            encodePiece(text.substring(from, if (special < 0) text.length else special), out)
            if (special < 0) break
            out += startToken
            from = special + END_OF_TEXT.length
        }
        return out.toIntArray()
    }

    private fun encodePiece(text: String, out: MutableList<Int>) {
        val m = PATTERN.matcher(text)
        while (m.find()) {
            val word = m.group()
            val tokens = cache[word] ?: bpe(word).also { cache[word] = it }
            for (t in tokens) out += t
        }
    }

    private fun bpe(word: String): IntArray {
        val bytes = word.toByteArray(Charsets.UTF_8)
        var symbols = IntArray(bytes.size) { byteIds[bytes[it].toInt() and 0xff] }
        var n = symbols.size
        while (n > 1) {
            var bestRank = Int.MAX_VALUE
            var bestA = 0
            var bestB = 0
            var merged = 0
            for (i in 0 until n - 1) {
                val m = merges[pair(symbols[i], symbols[i + 1])] ?: continue
                val rank = (m ushr 32).toInt()
                if (rank < bestRank) {
                    bestRank = rank
                    bestA = symbols[i]
                    bestB = symbols[i + 1]
                    merged = m.toInt()
                }
            }
            if (bestRank == Int.MAX_VALUE) break
            // every occurrence of the pair, left to right and without overlap
            val next = IntArray(n)
            var k = 0
            var i = 0
            while (i < n) {
                if (i < n - 1 && symbols[i] == bestA && symbols[i + 1] == bestB) {
                    next[k++] = merged
                    i += 2
                } else {
                    next[k++] = symbols[i++]
                }
            }
            symbols = next
            n = k
        }
        return symbols.copyOf(n)
    }

    private companion object {
        const val END_OF_TEXT = "<|endoftext|>"
        const val CACHE_WORDS = 4096

        fun pair(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

        /** GPT-2's split, see [SdkPatterns.GPT2_SPLIT]. */
        val PATTERN: Pattern = Pattern.compile(SdkPatterns.GPT2_SPLIT)

        /** GPT-2's bytes_to_unicode: printable bytes stand for themselves, the others for 256, 257, ... */
        val BYTE_CHARS = CharArray(256).also { chars ->
            val printable = (('!'.code)..('~'.code)) + (0xA1..0xAC) + (0xAE..0xFF)
            var extra = 0
            for (b in 0 until 256) {
                chars[b] = if (b in printable) b.toChar() else (256 + extra++).toChar()
            }
        }
    }
}
