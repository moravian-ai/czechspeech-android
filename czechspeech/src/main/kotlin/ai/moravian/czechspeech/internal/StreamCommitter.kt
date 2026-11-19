package ai.moravian.czechspeech.internal

/**
 * The state of one piece of speech (one utterance) of streaming transcription: which words are committed, which are
 * still pending, where the buffer of audio the next pass recognises starts, and the cut that shortens that buffer. A
 * port of `absorb`, `choose_pause_pass` and the cut of `simulate_utt` of the measurement's simulation, with
 * milliseconds instead of samples. Pure: no audio, no clock, no threads; the caller serialises the calls and injects
 * the quiet point of a cut.
 *
 * A pass commits the words it and the pass before agree on; the final recognition of the piece commits all that
 * remains. Words are compared exactly ([WordAgreement.commonPrefix]) to commit and by keys to line up. Text is never
 * printed by [toString].
 *
 * What is guaranteed and what is not: the text once committed never changes. That the joins of the buffer after a
 * cut add no word twice and lose none is not guaranteed: when the committed end is heard differently the second
 * time, [WordAgreement.anchorOverlap] can take the wrong number of words as already committed (a word repeated or
 * lost at the seam). This is the simulation's behaviour, kept so that its measured numbers carry over (in the
 * measurement's data a seam with the anchor at 0 kept words was 16 of 390 seams, 13 of them in noise). The final
 * recognition has no agreement check at all, so it is exposed to the same limit.
 *
 * @property audioStartMs session clock of the first sample of the piece's audio (pre-roll included)
 */
internal class StreamCommitter(val audioStartMs: Long) {
    /** Start of the buffer the next pass recognises: [audioStartMs], moved forward by cuts. */
    var bufferStartMs: Long = audioStartMs
        private set

    /** Number of words committed so far. */
    val committedCount: Int get() = committed.size

    /** The uncommitted end of the last hypothesis (empty after an empty one). */
    val pending: List<String> get() = pendingWords

    /** Audio end of the last absorbed pass, -1 before the first one. */
    var previousPassEndMs: Long = -1
        private set

    private val committed = ArrayList<String>()
    private var inBuffer = 0
    private var anchorExpected: Int? = null
    private var finished = false
    private var pendingWords: List<String> = emptyList()

    /** (audio end ms, words of text after the pass) of the passes on the current buffer. */
    private val history = ArrayList<Pair<Long, Int>>()

    /**
     * Absorbs the result of a pass over `[startMs, endMs)`: aligns [text] to the committed words still in the buffer,
     * commits the words this pass and the one before agree on, and decides the cut, asking [quietPoint] for the
     * quietest place (in ms) of `[loMs, hiMs)` of this pass's audio when there is something to cut.
     *
     * With [mayCommit] false (the pass or the one before it was not in the piece's language, or the piece has had no
     * pass that could decide its language yet: [StreamingTracker]) nothing is committed: the whole uncommitted end of
     * the text stays pending, to be agreed on by the next passes.
     *
     * @throws IllegalArgumentException when [startMs] is not [bufferStartMs]
     * @throws IllegalStateException after [absorbFinal]
     */
    fun absorbPass(
        startMs: Long,
        endMs: Long,
        text: String,
        quietPoint: (loMs: Long, hiMs: Long) -> Long,
        mayCommit: Boolean = true,
    ): PassOutcome {
        check(!finished) { "the final recognition of the piece was absorbed" }
        require(startMs == bufferStartMs) { "pass starts at $startMs, the buffer at $bufferStartMs" }
        val agreedEndMs = previousPassEndMs
        val absorbed = absorb(text, final = false, mayCommit = mayCommit)
        previousPassEndMs = endMs
        history.add(endMs to committed.size + pendingWords.size)
        val cut = decideCut(endMs, quietPoint)
        return PassOutcome(
            prefix = absorbed.prefix,
            committed = absorbed.committedWords,
            pending = pendingWords,
            agreedEndMs = agreedEndMs,
            cut = cut,
            seamWord = absorbed.seamWord,
        )
    }

    /** [absorbPass] with the cut function last, so that it can be written as a trailing lambda. */
    fun absorbPass(
        startMs: Long,
        endMs: Long,
        text: String,
        mayCommit: Boolean = true,
        quietPoint: (loMs: Long, hiMs: Long) -> Long,
    ): PassOutcome = absorbPass(startMs, endMs, text, quietPoint, mayCommit)

    /**
     * Absorbs the final recognition of the piece (from [bufferStartMs] to its end): everything after the committed
     * words is committed. The piece is closed by it: nothing is absorbed afterwards.
     *
     * @throws IllegalStateException when the final was absorbed already
     */
    fun absorbFinal(text: String): FinalOutcome {
        check(!finished) { "the final recognition of the piece was absorbed" }
        finished = true
        val absorbed = absorb(text, final = true)
        return FinalOutcome(absorbed.prefix, absorbed.committedWords, absorbed.seamWord)
    }

    private class Absorbed(val prefix: Int, val committedWords: List<String>, val seamWord: Int?)

    private fun absorb(text: String, final: Boolean, mayCommit: Boolean = true): Absorbed {
        val hyp = WordAgreement.words(text)
        var seamWord: Int? = null
        val prefix: Int
        val expected = anchorExpected
        if (expected != null) {
            val anchor = WordAgreement.anchorOverlap(hyp, committed, expected, pendingWords)
            inBuffer = anchor.inBuffer
            prefix = anchor.prefix
            seamWord = committed.size - inBuffer
            anchorExpected = null
        } else {
            prefix = if (inBuffer > 0) {
                WordAgreement.splitPrefix(hyp, committed.subList(committed.size - inBuffer, committed.size), pendingWords)
            } else {
                0
            }
        }
        val tail = hyp.subList(prefix, hyp.size)
        val n = if (final) tail.size else if (mayCommit) WordAgreement.commonPrefix(pendingWords, tail) else 0
        val newlyCommitted = tail.subList(0, n).toList()
        committed.addAll(newlyCommitted)
        inBuffer += n
        pendingWords = tail.subList(n, tail.size).toList()
        return Absorbed(prefix, newlyCommitted, seamWord)
    }

    private fun decideCut(endMs: Long, quietPoint: (Long, Long) -> Long): Cut? {
        if (endMs - bufferStartMs <= MIN_BUFFER_FOR_CUT_MS) return null
        var best: Pair<Long, Int>? = null
        for (entry in history) {
            val (e, w) = entry
            if (w > committed.size - 1 || e < bufferStartMs + ANCHOR_MIN_MS || e > endMs - KEEP_MS) continue
            val b = best
            if (b == null || e > b.first || (e == b.first && w > b.second)) best = entry
        }
        val (e, w) = best ?: return null
        val at = quietPoint(e - QUIET_BEFORE_MS, e + QUIET_AFTER_MS)
        if (at <= bufferStartMs) return null
        val before = endMs - bufferStartMs
        bufferStartMs = at
        anchorExpected = committed.size - w + 1
        history.removeAll { it.first <= at + ANCHOR_MIN_MS }
        return Cut(atMs = at, bufferBeforeMs = before, bufferAfterMs = endMs - at)
    }

    override fun toString(): String =
        "StreamCommitter(audioStartMs=$audioStartMs, bufferStartMs=$bufferStartMs, " +
            "committed=${committed.size}, pending=${pendingWords.size})"

    companion object {
        /** A buffer is cut only when it is strictly longer than this. */
        const val MIN_BUFFER_FOR_CUT_MS = 6_000L

        /**
     * The pass whose end the cut follows ends at least this far before the end of the pass that cuts; the quiet point
     * may lie up to 85 ms after it, so at least 2 915 ms of audio stay in the buffer.
     */
        const val KEEP_MS = 3_000L

        /** The pass whose end the cut follows lies at least this far from the buffer start. */
        const val ANCHOR_MIN_MS = 1_000L

        /** The quiet point is searched this long before the end of the chosen pass... */
        const val QUIET_BEFORE_MS = 400L

        /** ...and this long after it. */
        const val QUIET_AFTER_MS = 100L
    }
}

/**
 * What one absorbed pass decided: [prefix] leading hypothesis words that were the committed words still in the buffer,
 * the newly [committed] words, the [pending] rest, [agreedEndMs] (the end of the earlier agreeing pass, -1 for the
 * first), the [cut] of the buffer if any, and [seamWord], the index among the piece's committed words of the first
 * word of the new buffer when this pass resolved the anchor after a cut (else null).
 */
internal class PassOutcome(
    val prefix: Int,
    val committed: List<String>,
    val pending: List<String>,
    val agreedEndMs: Long,
    val cut: Cut?,
    val seamWord: Int?,
) {
    override fun toString(): String =
        "PassOutcome(prefix=$prefix, committed=${committed.size}, pending=${pending.size}, cut=$cut)"
}

/** A cut of the buffer at [atMs]; the buffer was [bufferBeforeMs] long and is [bufferAfterMs] long now. */
internal data class Cut(val atMs: Long, val bufferBeforeMs: Long, val bufferAfterMs: Long)

/** What the final recognition decided: [prefix] words of the buffer already committed, the [words] left, [seamWord] as in [PassOutcome]. */
internal class FinalOutcome(val prefix: Int, val words: List<String>, val seamWord: Int?) {
    override fun toString(): String = "FinalOutcome(prefix=$prefix, words=${words.size})"
}
