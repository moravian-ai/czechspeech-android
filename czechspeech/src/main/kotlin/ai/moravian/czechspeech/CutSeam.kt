package ai.moravian.czechspeech

/**
 * What the SDK does at a cut inside speech ([StreamingConfig.cutSeam]): a piece of speech cut where the person is still
 * speaking, because it grew too long for the phone to recognise at once, may split a word, and the piece after it starts
 * with no sound before it.
 */
public enum class CutSeam {
    /** The piece after a cut is recognised from its own audio alone, as it was before this setting existed. */
    NONE,

    /**
     * The final recognition of the piece after a cut also gets the last 2.5 s of audio of the piece before it, so the
     * words at the seam are heard with their surroundings. The words that repeat what was already sent are found by
     * comparing the text with the text sent before (two words must agree) and are left out. Nothing already sent changes; that no
     * word is sent twice is a heuristic (a misaligned anchor can repeat or drop a word of the seam: measured on a laptop, 0 repeated and
     * 4 lost words in 196 seams). When that comparison finds nothing, the piece is recognised from its own audio alone, as with [NONE]
     * (only the retries of an empty result differ: no step on 512 frames, the full window only if it still fits the lag limit).
     * Costs one longer recognition per cut (the encoder window is the same up to 7 s of audio and larger above, and the decoder writes
     * about 2.5 s more words: about 7 % more time on a laptop) and, when the comparison fails (about 5 % of the cuts), a second
     * recognition; a final that covers up to 14.5 s of audio and a second recognition are exceptions to the delay guarantee of
     * [StreamingConfig]. The recognition is in the language of the piece before it: in a session with several languages a change of
     * language inside continuous speech is noticed again only after a pause, a quiet cut or an anchor fallback (a piece whose text came from
     * a rescue in another language hands no context on). Those 2.5 s of audio stay in memory only, until the recognition of the next
     * piece has ended, and are overwritten with zeros then, also when no piece continues them and when the session ends or is cancelled;
     * they are never written anywhere.
     */
    CONTEXT,

    /**
     * The default of [StreamingConfig.cutSeam]. [CONTEXT], and a piece cut inside speech is delivered without its last word while the phone is in chunk mode (the passes are
     * paused) and the piece has at least three words: that word is the one a cut may have split. The recognition of the next piece
     * (with the context) decodes it again, and its text then starts with the new version of the word; the old one is never sent. So a
     * piece can arrive without its last word, which comes at the start of the first [SpeechEvent.Transcript] of the next utterance (about
     * one piece later), re-recognised, and [SpeechEvent.UtteranceDone] of the cut piece waits for that: the text of an utterance is final at
     * its `UtteranceDone`. Where the word cannot be replaced (no text anchor, the next piece was dropped or skipped, the next speech
     * starts after silence, the session ends, or the replacement would come later than [StreamingConfig.maxLagMs] allows), it is sent as it
     * was, as one more `Transcript` of the cut piece (same `startMs`, `endMs` of the piece, `processingMs` 0, no `sentenceJoin`), once, before
     * anything else of the next piece. A held word is never a preview. The delay guarantee of [StreamingConfig.maxLagMs] holds: a replacement that would come later
     * is refused and the word is sent as it was (simulated on a phone's measured curve: at most 0.5 s more for the largest delay of a word). A cancelled session sends nothing, as for any text still waiting.
     * Use [CONTEXT] to opt out of the holding.
     */
    CONTEXT_AND_HELD_WORD,
}
