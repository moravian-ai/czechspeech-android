package ai.moravian.czechspeech.internal

import kotlin.math.min

/**
 * Where the text of a piece that continues a cut inside speech starts: pure rules over words, no audio and no clock.
 *
 * The final recognition of such a piece runs over the last [CONTEXT_MS] of the piece before it and the piece's own audio.
 * Its text then starts with words that were already delivered for the piece before. A two-word text anchor finds where the
 * new text starts: the last words delivered ([expected] of them, give or take [WordAgreement.ANCHOR_SLACK]) are lined up with
 * the first words of the run's text ([WordAgreement.anchorOverlap]), and the anchor is accepted when at least two words are
 * assigned and at least two of them are the same word (letters and digits, no case, no punctuation). This is the measurement's
 * rule for the arm "L2.5" (and "H1" with `held` words kept back), word for word: [CutSeamReplayTest] replays its stored decisions.
 */
internal object SeamAnchor {
    /** How much audio of the end of a piece cut inside speech is kept for the piece after it, in ms. */
    const val CONTEXT_MS = 2_500L

    /** The same in samples (16 kHz). */
    const val CONTEXT_SAMPLES = 40_000

    /** A text anchor needs at least this many words assigned, and as many of them matching. */
    const val MIN_ANCHOR_WORDS = 2

    /**
     * What [decide] found: how many of the delivered words [inBuffer] the context run repeats, how many leading words of
     * its text [prefix] they take, how many of those match ([matched]), and whether the anchor is [ok].
     */
    class Decision(val inBuffer: Int, val prefix: Int, val matched: Int, val ok: Boolean)

    /**
     * How many of the last delivered words of a piece of [pieceWords] words over [pieceAudioMs] ms the context of
     * [CONTEXT_MS] covers: the words in proportion to the audio, rounded half to even like Python's `round` (so 2.5 gives 2
     * and 3.5 gives 4; `Math.round` would differ), less the [held] words kept back, never below 0.
     */
    fun expected(pieceWords: Int, pieceAudioMs: Long, held: Int = 0): Int {
        val lengthS = pieceAudioMs / 1000.0
        val share = min(1.0, 2.5 / maxOf(lengthS, 1e-6))
        return maxOf(0, Math.rint(pieceWords * share).toInt() - held)
    }

    /** The anchor of [hyp], the words of the context run, in the [delivered] words of the piece before; [expected] from [expected]. */
    fun decide(hyp: List<String>, delivered: List<String>, expected: Int): Decision {
        val a = WordAgreement.anchorOverlap(hyp, delivered, expected, emptyList())
        val covered = if (a.inBuffer > 0) delivered.subList(delivered.size - a.inBuffer, delivered.size) else emptyList()
        // a token without a letter or a digit (a dash, an ellipsis) has the key "" and would match another one: not counted
        val matched = WordAgreement.matchedWords(
            covered.map(WordAgreement::key).filter { it.isNotEmpty() },
            hyp.subList(0, a.prefix).map(WordAgreement::key).filter { it.isNotEmpty() },
        )
        return Decision(a.inBuffer, a.prefix, matched, a.prefix >= MIN_ANCHOR_WORDS && matched >= MIN_ANCHOR_WORDS)
    }

    /** The number of leading words of [hyp] that repeat delivered words (the new text starts after them), or null without an anchor. */
    fun continuation(hyp: List<String>, delivered: List<String>, expected: Int): Int? =
        decide(hyp, delivered, expected).takeIf { it.ok }?.prefix

    /** [words] has a letter or a digit: an anchored run whose rest has none wrote nothing new. */
    fun hasText(words: List<String>): Boolean = words.any { w -> w.any { it.isLetterOrDigit() } }
}
