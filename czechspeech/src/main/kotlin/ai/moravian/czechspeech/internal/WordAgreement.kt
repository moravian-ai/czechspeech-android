package ai.moravian.czechspeech.internal

/**
 * Word-level text comparison of streaming transcription: splitting a transcript into words, comparing them without
 * punctuation and case, and finding how the words of a new pass line up with the words already committed. A port of
 * `key_word`, `lcp`, `split_prefix` and `anchor_overlap` of the measurement's simulation, with the same results.
 *
 * The alignment is a heuristic and not exact: when the words around a cut are heard differently in two passes, a
 * word can be taken twice or lost at the seam ([anchorOverlap]); tests pin two such cases.
 *
 * Two words *agree* only when they are equal letter for letter, with case and punctuation; the keys ([key]) serve
 * only to line words up, never to confirm them.
 */
internal object WordAgreement {
    /** Default of how far the number of committed words kept in the buffer may differ from the expected one. */
    const val ANCHOR_SLACK = 2

    /** Words of [text], split at ASCII whitespace only (space, tab, newline, CR, VT, FF), empty pieces dropped. */
    fun words(text: String): List<String> {
        val result = ArrayList<String>()
        var start = -1
        for (i in text.indices) {
            val c = text[i]
            val white = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\u000B' || c == '\u000C'
            if (white) {
                if (start >= 0) result.add(text.substring(start, i))
                start = -1
            } else if (start < 0) {
                start = i
            }
        }
        if (start >= 0) result.add(text.substring(start))
        return result
    }

    /**
     * [word] in lower case with only letters and digits kept (what Python's `isalnum` keeps: letters and the
     * numeric categories Nd, Nl and No).
     */
    fun key(word: String): String {
        val lower = word.lowercase()
        val sb = StringBuilder(lower.length)
        var i = 0
        while (i < lower.length) {
            val cp = lower.codePointAt(i)
            if (Character.isLetter(cp) || isNumber(cp)) sb.appendCodePoint(cp)
            i += Character.charCount(cp)
        }
        return sb.toString()
    }

    private fun isNumber(cp: Int): Boolean {
        val type = Character.getType(cp)
        return type == Character.DECIMAL_DIGIT_NUMBER.toInt() ||
            type == Character.LETTER_NUMBER.toInt() ||
            type == Character.OTHER_NUMBER.toInt()
    }

    /** The number of leading elements [a] and [b] have in common. */
    fun commonPrefix(a: List<String>, b: List<String>): Int {
        var n = 0
        while (n < a.size && n < b.size && a[n] == b[n]) n++
        return n
    }

    private fun keys(words: List<String>): List<String> = words.map(::key)

    /**
     * How many leading words of the new pass [hyp] correspond to the [committed] words still in the buffer: the
     * prefix of the least edit distance from [committed] (on keys). Ties go to the prefix whose rest follows the
     * not yet committed end of the previous pass, [expectedTail], best (up to 3 words), then to one whose last word
     * is the last committed word, then to the length nearest to the number of committed words; the first of equals
     * wins. An empty [hyp] or [committed] has nothing to assign (0).
     */
    fun splitPrefix(hyp: List<String>, committed: List<String>, expectedTail: List<String>): Int {
        if (committed.isEmpty() || hyp.isEmpty()) return 0
        val c = keys(committed)
        val h = keys(hyp)
        val e = keys(expectedTail)
        var prev = IntArray(c.size + 1) { it }
        val dist = IntArray(h.size + 1)
        dist[0] = prev[c.size]
        for (j in 1..h.size) {
            val cur = IntArray(c.size + 1)
            cur[0] = j
            for (i in 1..c.size) {
                val substitution = prev[i - 1] + if (c[i - 1] != h[j - 1]) 1 else 0
                cur[i] = minOf(prev[i] + 1, cur[i - 1] + 1, substitution)
            }
            dist[j] = cur[c.size]
            prev = cur
        }
        var bestM = 0
        var bestA = Double.POSITIVE_INFINITY
        var bestB = Int.MAX_VALUE
        var bestC = Int.MAX_VALUE
        for (m in 0..h.size) {
            val follow = minOf(commonPrefix(h.subList(m, h.size), e), 3)
            val last = if (m > 0 && h[m - 1] == c.last()) 1 else 0
            val a = dist[m] - 0.5 * follow
            val b = -last
            val d = Math.abs(m - c.size)
            val better = a < bestA || (a == bestA && (b < bestB || (b == bestB && d < bestC)))
            if (better) {
                bestA = a
                bestB = b
                bestC = d
                bestM = m
            }
        }
        return bestM
    }

    /**
     * The total size of the matching blocks of Python's `difflib.SequenceMatcher(a, b, autojunk=False)`: the longest
     * common contiguous block (the one starting first in [a], then first in [b]), then the same left and right of
     * it. This is not the longest common subsequence, which can be longer.
     */
    fun matchedWords(a: List<String>, b: List<String>): Int {
        var total = 0
        val stack = ArrayDeque<IntArray>()
        stack.addLast(intArrayOf(0, a.size, 0, b.size))
        while (stack.isNotEmpty()) {
            val range = stack.removeLast()
            val alo = range[0]
            val ahi = range[1]
            val blo = range[2]
            val bhi = range[3]
            var bestI = alo
            var bestJ = blo
            var bestSize = 0
            var previous = IntArray(b.size + 1)
            for (i in alo until ahi) {
                val current = IntArray(b.size + 1)
                for (j in blo until bhi) {
                    if (a[i] != b[j]) continue
                    val k = (if (j > blo) previous[j - 1] else 0) + 1
                    current[j] = k
                    if (k > bestSize) {
                        bestI = i - k + 1
                        bestJ = j - k + 1
                        bestSize = k
                    }
                }
                previous = current
            }
            if (bestSize == 0) continue
            total += bestSize
            if (alo < bestI && blo < bestJ) stack.addLast(intArrayOf(alo, bestI, blo, bestJ))
            if (bestI + bestSize < ahi && bestJ + bestSize < bhi) {
                stack.addLast(intArrayOf(bestI + bestSize, ahi, bestJ + bestSize, bhi))
            }
        }
        return total
    }

    /**
     * Where a new pass starts after a cut without word times: [inBuffer] is how many of the last committed words are
     * still in the (shortened) buffer and [prefix] how many leading words of the new pass correspond to them.
     */
    data class Anchor(val inBuffer: Int, val prefix: Int)

    /**
     * The overlap of the end of [committed] with the start of the new pass [hyp] after a cut without word times.
     * Tries the counts of kept words around [expected] (within [slack]) and takes the one whose committed end and
     * pass start match best: matching words minus edit distance, plus how well the rest follows [expectedTail];
     * ties go to the count nearest to [expected], then to the smaller one. An empty [hyp] changes nothing.
     */
    fun anchorOverlap(
        hyp: List<String>,
        committed: List<String>,
        expected: Int,
        expectedTail: List<String>,
        slack: Int = ANCHOR_SLACK,
    ): Anchor {
        val exp = expected.coerceIn(0, committed.size)
        if (hyp.isEmpty()) return Anchor(exp, 0)
        val hk = keys(hyp)
        val ek = keys(expectedTail)
        var best = Anchor(exp, 0)
        var bestScore = Double.POSITIVE_INFINITY
        var bestDistance = Int.MAX_VALUE
        for (k in maxOf(0, exp - slack)..minOf(committed.size, exp + slack)) {
            val c = if (k > 0) committed.subList(committed.size - k, committed.size) else emptyList()
            val m = if (k > 0) splitPrefix(hyp, c, expectedTail) else 0
            val ck = keys(c)
            val match = matchedWords(ck, hk.subList(0, m))
            val dist = maxOf(ck.size, m) - match
            val follow = minOf(commonPrefix(hk.subList(m, hk.size), ek), 3)
            val score = -(match - dist + 0.5 * follow)
            val distance = Math.abs(k - exp)
            if (score < bestScore || (score == bestScore && distance < bestDistance)) {
                bestScore = score
                bestDistance = distance
                best = Anchor(k, m)
            }
        }
        return best
    }
}
