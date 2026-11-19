package ai.moravian.czechspeech.internal

/**
 * Which part of a preview two passes in a row agree on (LocalAgreement-2 of whisper-streaming, Macháček et al. 2023).
 *
 * A preview is recognised from audio that ends wherever the speaker happens to be, often in the middle of a word, and
 * the recogniser's last words for such audio are unreliable: the next pass, with more audio, often writes them
 * differently. The words at the start, which the next pass repeats, rarely change again. So of each preview, the
 * longest run of words from its start that the previous preview of the same utterance also started with counts as
 * stable; the rest is the uncertain end.
 *
 * Words are compared without case and without the punctuation around them: a comma that comes or goes, or a capital
 * that turns small because the sentence now continues, is not a different word. Pure logic, no state: the caller
 * keeps the previous text.
 */
internal object PreviewAgreement {
    /**
     * The length of the stable part of [current], in chars: [current] up to the end of its last word that, like every
     * word before it, is the same word at the same place in [previous]. 0 when the first words differ or either text
     * is empty. The stable part ends at the end of a word (its punctuation included), never inside one.
     */
    fun stableLength(previous: String, current: String): Int {
        var p = 0
        var c = 0
        var stable = 0
        while (true) {
            p = skipSpace(previous, p)
            c = skipSpace(current, c)
            if (p >= previous.length || c >= current.length) return stable
            val pEnd = wordEnd(previous, p)
            val cEnd = wordEnd(current, c)
            if (!sameWord(previous, p, pEnd, current, c, cEnd)) return stable
            stable = cEnd
            p = pEnd
            c = cEnd
        }
    }

    private fun skipSpace(s: String, from: Int): Int {
        var i = from
        while (i < s.length && s[i].isWhitespace()) i++
        return i
    }

    private fun wordEnd(s: String, from: Int): Int {
        var i = from
        while (i < s.length && !s[i].isWhitespace()) i++
        return i
    }

    /** The same letters and digits, whatever their case and the punctuation around them; without any, the same chars. */
    private fun sameWord(a: String, aFrom: Int, aTo: Int, b: String, bFrom: Int, bTo: Int): Boolean {
        val x = core(a, aFrom, aTo)
        val y = core(b, bFrom, bTo)
        if (x.isEmpty() || y.isEmpty()) return a.regionMatches(aFrom, b, bFrom, maxOf(aTo - aFrom, bTo - bFrom)) && aTo - aFrom == bTo - bFrom
        return x.equals(y, ignoreCase = true)
    }

    /** The word without the punctuation before and after it. */
    private fun core(s: String, from: Int, to: Int): String {
        var i = from
        var j = to
        while (i < j && !s[i].isLetterOrDigit()) i++
        while (j > i && !s[j - 1].isLetterOrDigit()) j--
        return s.substring(i, j)
    }
}
