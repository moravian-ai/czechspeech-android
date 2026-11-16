package ai.moravian.czechspeech.internal

/**
 * Repetition loops in recognition results. Greedy decoding sometimes falls into a loop when the audio ends abruptly
 * in the middle of a word (stop(), or speech cut at maxUtteranceMs): "… v roce 2012 až se státní podporou v roce 2012
 * až se státní podporou …" until the token limit. whisper.cpp's temperature fallback does not prevent it.
 * [SpeechEngine] applies this to every result, after the letters/digits filter; the web SDK's src/repetition.js
 * implements the same rule.
 *
 * The rule:
 * - Words are separated by whitespace and compared without case and punctuation (only letters and decimal digits
 *   count; a token without any, such as a dash, belongs to the word before it). The text kept is the original.
 * - A phrase of 2 to 12 words that occurs at least 3 times in immediate succession keeps its first occurrence; the
 *   repeats are dropped. When they end the text, a trailing partial copy (a prefix of the phrase, its last word
 *   possibly cut off, as "se st") is dropped too.
 * - Single repeated words ("ne, ne, ne") and phrases repeated only twice stay. Scanning from the left, the smallest
 *   phrase that explains a run wins; a phrase that is itself a repetition of a shorter one ("ne ne") does not count,
 *   so "ne ne ne ne ne ne" stays as well.
 * - The rule is applied again until nothing changes (a loop of loops).
 */
internal object RepetitionGuard {
    const val MIN_WORDS = 2
    const val MAX_WORDS = 12
    const val MIN_COPIES = 3

    /** A loop removed: its phrase length, the copies dropped after the first, whether a partial copy at the end was dropped. */
    data class Loop(val words: Int, val repeats: Int, val partial: Boolean)

    /** The text without the loops (the same string when there are none). [toString] does not print the text. */
    class Result(val text: String, val loops: List<Loop>) {
        override fun toString(): String = "Result(${text.length} chars, $loops)"
    }

    fun clean(text: String): Result {
        var words = splitWords(text)
        val loops = mutableListOf<Loop>()
        while (true) {
            val pass = collapse(words)
            if (pass.loops.isEmpty()) break
            loops += pass.loops
            words = pass.words
        }
        if (loops.isEmpty()) return Result(text, emptyList())
        val out = StringBuilder(text.length)
        words.forEachIndexed { i, w ->
            if (i > 0) out.append(w.sep)
            out.append(text, w.start, w.end)
        }
        return Result(out.toString(), loops)
    }

    /** The log line for removed loops: counts only, never text. */
    fun describe(loops: List<Loop>): String {
        fun one(l: Loop) = "${l.words} words, ${l.repeats} repeats" + if (l.partial) ", partial copy at the end" else ""
        return if (loops.size == 1) {
            "repetition loop removed (${one(loops[0])})"
        } else {
            "${loops.size} repetition loops removed (${loops.joinToString("; ") { one(it) }})"
        }
    }

    /** A word: its place in the text, the whitespace before it and its comparison key. */
    private class Word(val start: Int, var end: Int, val sep: String, val key: String)

    private class Pass(val words: List<Word>, val loops: List<Loop>)

    private fun splitWords(text: String): List<Word> {
        val words = mutableListOf<Word>()
        var pendingStart = -1
        var lastEnd = 0
        var i = 0
        while (i < text.length) {
            if (text[i].isWhitespace()) {
                i++
                continue
            }
            val start = i
            while (i < text.length && !text[i].isWhitespace()) i++
            val key = key(text.substring(start, i))
            if (key.isEmpty()) {
                // punctuation on its own: part of the word before it (or, at the start, of the word after it)
                if (words.isNotEmpty()) {
                    words.last().end = i
                    lastEnd = i
                } else if (pendingStart < 0) {
                    pendingStart = start
                }
                continue
            }
            val first = if (pendingStart >= 0) pendingStart else start
            words += Word(first, i, text.substring(lastEnd, first), key)
            pendingStart = -1
            lastEnd = i
        }
        return words
    }

    /** Lower case, letters and decimal digits only (by code point, as the web SDK's /[\p{L}\p{Nd}]/u). */
    private fun key(token: String): String {
        val lower = token.lowercase()
        val key = StringBuilder(lower.length)
        var i = 0
        while (i < lower.length) {
            val cp = lower.codePointAt(i)
            if (Character.isLetterOrDigit(cp)) key.appendCodePoint(cp)
            i += Character.charCount(cp)
        }
        return key.toString()
    }

    /** One pass from the left: every run collapsed to its first copy. */
    private fun collapse(words: List<Word>): Pass {
        val out = ArrayList<Word>(words.size)
        val loops = mutableListOf<Loop>()
        val n = words.size
        var i = 0
        while (i < n) {
            val run = findRun(words, i)
            if (run == null) {
                out += words[i++]
                continue
            }
            val (k, copies) = run
            for (x in i until i + k) out += words[x]
            var next = i + k * copies
            val partial = next < n && isTrailingPartial(words, i, k, next)
            if (partial) next = n
            loops += Loop(k, copies - 1, partial)
            i = next
        }
        return Pass(out, loops)
    }

    /** The run starting at word [i] with the smallest phrase length: (k, copies), or null. */
    private fun findRun(words: List<Word>, i: Int): Pair<Int, Int>? {
        val n = words.size
        var k = MIN_WORDS
        while (k <= MAX_WORDS && i + MIN_COPIES * k <= n) {
            if (isPrimitive(words, i, k)) {
                var copies = 1
                while (i + (copies + 1) * k <= n && samePhrase(words, i, i + copies * k, k)) copies++
                if (copies >= MIN_COPIES) return k to copies
            }
            k++
        }
        return null
    }

    private fun samePhrase(words: List<Word>, a: Int, b: Int, k: Int): Boolean =
        (0 until k).all { words[a + it].key == words[b + it].key }

    /** False when the [k] words at [i] are a shorter phrase repeated ("ne ne", "a b a b"): that one explains the run. */
    private fun isPrimitive(words: List<Word>, i: Int, k: Int): Boolean =
        (1 until k).none { d -> k % d == 0 && (d until k).all { x -> words[i + x].key == words[i + x - d].key } }

    /** Whether the words from [from] to the end are a prefix of the phrase at [i] ([k] words), the last one possibly cut. */
    private fun isTrailingPartial(words: List<Word>, i: Int, k: Int, from: Int): Boolean {
        val m = words.size - from
        if (m < 1 || m > k) return false
        for (x in 0 until m - 1) if (words[from + x].key != words[i + x].key) return false
        return words[i + m - 1].key.startsWith(words[from + m - 1].key)
    }
}
