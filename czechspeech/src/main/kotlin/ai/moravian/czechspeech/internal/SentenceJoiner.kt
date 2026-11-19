package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Sentence joining in one [SpeechEngine.listen] session: the previous transcript and the rules of when a transcript may
 * continue it at all ([SentenceBoundary] has the decision). Only the session's recognition calls [next] and [reset], one
 * at a time (on the whisper thread and in the recognition loop); [lost] comes from capture.
 *
 * The previous text is Whisper's, as the measurement took it, not the form a client may have inserted.
 */
internal class SentenceJoiner {
    /** The outcome of [next]: what to tell the client, and the language model's time on the boundary. */
    class Step(val join: SentenceJoin?, val languageModelMs: Long)

    @Volatile
    private var previousText: String? = null

    @Volatile
    private var previousEndMs = 0L

    /** Ends of utterances whose speech was lost (skipped) and that have not been passed yet. */
    private val lostEnds = ConcurrentLinkedQueue<Long>()

    /** Words held back from a piece cut inside speech that were sent as they were, from any thread, not applied yet (see [continuedLater]). */
    private val laterWords = ConcurrentLinkedQueue<Pair<String, Long>>()

    /**
     * The word held back from the previous piece (a cut inside speech, `CutSeam.CONTEXT_AND_HELD_WORD`) was sent as it was, as one
     * more transcript of that piece: it extends the previous text, as [continued] does, ending at [endMs]. Callable from any thread
     * (some flushes happen on the capture thread); it is applied at the start of the next call of the whisper thread's, before any
     * decision, so the decision sees the text the client has. Nothing after [reset] or before the first transcript.
     */
    fun continuedLater(text: String, endMs: Long) {
        laterWords += text to endMs
    }

    private fun applyLater() {
        while (true) {
            val (text, endMs) = laterWords.poll() ?: return
            val previous = previousText ?: continue
            previousText = "$previous $text"
            previousEndMs = endMs
        }
    }

    /** Capture: the utterance ending at [endMs] was skipped, so no transcript continues across it. */
    fun lost(endMs: Long) {
        lostEnds += endMs
    }

    /**
     * A later chunk of the same utterance (streaming transcription sends one that has more than one): [text] extends the
     * previous text, separated by a space, and [endMs] is where the utterance now ends. Never a decision and never the
     * language model; the boundary is decided once, against the whole utterance, by the next [next]. Nothing after
     * [reset] or before the first transcript. Called on the whisper thread, like [next].
     */
    fun continued(text: String, endMs: Long) {
        applyLater()
        val previous = previousText ?: return
        previousText = "$previous $text"
        previousEndMs = endMs
    }

    /**
     * The utterance of the previous text ended at [endMs] after all (streaming transcription: its final recognition
     * found no more words, so the last chunk's end was only an upper bound of where the words end). Nothing before the
     * first transcript or after [reset].
     */
    fun endedAt(endMs: Long) {
        applyLater()
        if (previousText != null) previousEndMs = endMs
    }

    /** The recognition of an utterance failed: its speech is lost, so the next transcript continues nothing. */
    fun reset() {
        applyLater()
        previousText = null
    }

    /**
     * The transcript [text] of the utterance [startMs]..[endMs] (speech, on the session's clock) against the previous
     * transcript of the session, which it then replaces. No join for the first transcript, after [reset], across a
     * pause longer than [SentenceBoundary.MAX_PAUSE_MS] or across [lost] speech, nor without a language model
     * ([score] null) or when it fails.
     *
     * [quiet]: the utterance is a piece that starts where a forced cut fell on a quiet frame of continuous speech, not
     * after real silence: the decision leans to going on ([SentenceBoundary.decide] with `quiet`), and the log line says
     * which variant won and that it was a quiet cut: `join|comma|split (quiet cut)`. Never any text.
     */
    fun next(text: String, startMs: Long, endMs: Long, score: ((List<String>) -> DoubleArray?)?): Step = next(text, startMs, endMs, score, false)

    fun next(text: String, startMs: Long, endMs: Long, score: ((List<String>) -> DoubleArray?)?, quiet: Boolean): Step {
        applyLater()
        val previous = previousText
        val previousEnd = previousEndMs
        previousText = text
        previousEndMs = endMs
        val lostBetween = lostEnds.any { it > previousEnd && it <= startMs }
        lostEnds.removeAll { it <= startMs }
        if (previous == null || score == null || lostBetween) return NONE
        if (startMs - previousEnd !in 0..SentenceBoundary.MAX_PAUSE_MS) return NONE
        val t0 = System.nanoTime()
        val choice = try {
            SentenceBoundary.decide(previous, text, score, quiet, startMs - previousEnd < SentenceBoundary.ELLIPSIS_PAUSE_LIMIT_MS)
        } catch (e: Exception) {
            // the exception is not logged: its message could quote the text
            CsLog.w("sentence boundary: the language model failed (${e.javaClass.simpleName}); the transcript stays")
            SentenceBoundary.Choice.RAW
        }
        val ms = (System.nanoTime() - t0) / 1_000_000
        CsLog.i("sentence boundary: ${SentenceBoundary.label(previous, choice, quiet)}, language model $ms ms")
        return Step(SentenceBoundary.sentenceJoin(previous, text, choice), ms)
    }

    /**
     * The first chunk of a piece that starts where a cut inside speech ended (streaming transcription: a forced cut or the
     * length cap, not a pause): [text] continues the previous text. The SDK knows the sentence does not end there, so
     * is no sentence decision: the period that the model wrote at the end of the previous piece goes
     * ([PreviousEnd.REMOVE_PERIOD]; a question or exclamation mark stays, and then nothing is changed) and the first
     * word loses the capital that only the start of the piece gave it, by the rule of a joined sentence
     * ([SentenceBoundary.lowerFirstWord]: an acronym keeps its letters). With a language model ([score]) it is asked once
     * about that capital ([SentenceBoundary.continuationTexts], only when the first word has a capital to lower):
     * the first word keeps its capital unless the small letter wins by more than [SentenceBoundary.CASE_MARGIN_CONTINUED] (a name
     * stays capitalised because the model does not prefer the small letter that much); when it fails, or
     * without [score], the word is lowered as it always was. Its time is [Step.languageModelMs]. No join across lost
     * speech, after [reset] or before the first transcript, as [next]. Whisper thread, like [next].
     */
    fun afterCut(text: String, startMs: Long, endMs: Long, score: ((List<String>) -> DoubleArray?)? = null): Step {
        applyLater()
        val previous = previousText
        val previousEnd = previousEndMs
        previousText = text
        previousEndMs = endMs
        val lostBetween = lostEnds.any { it > previousEnd && it <= startMs }
        lostEnds.removeAll { it <= startMs }
        if (previous == null || lostBetween || startMs - previousEnd !in 0..SentenceBoundary.MAX_PAUSE_MS) return NONE
        val last = SentenceBoundary.words(previous).lastOrNull() ?: return NONE
        val end = SentenceBoundary.endPunct(last)
        if (end == "?" || end == "!" || end.startsWith("?") || end.startsWith("!")) return NONE
        val previousEndEdit = when {
            SentenceBoundary.periodEnd(previous) -> PreviousEnd.REMOVE_PERIOD
            SentenceBoundary.ellipsisEnd(previous) -> PreviousEnd.REMOVE_ELLIPSIS      // an artefact of the cut
            else -> PreviousEnd.KEEP
        }
        val standalone = SentenceBoundary.upperFirstWord(text)
        var lower = true
        var asked = false
        var ms = 0L
        if (score != null) {
            val texts = SentenceBoundary.continuationTexts(previous, text)
            if (texts.isNotEmpty()) {
                asked = true
                val t0 = System.nanoTime()
                try {
                    val values = score(texts.values.toList())
                    if (values != null && values.size == texts.size && values.all { it.isFinite() }) {
                        lower = SentenceBoundary.lowerAfterCut(texts.keys.zip(values.toList()).toMap())
                    }
                } catch (e: Exception) {
                    // the exception is not logged: its message could quote the text
                    CsLog.w("sentence boundary: the language model failed (${e.javaClass.simpleName}); the first word is lowered")
                }
                ms = (System.nanoTime() - t0) / 1_000_000
            }
        }
        CsLog.i(
            "sentence boundary: continued (cut)" + if (asked) ", ${if (lower) "lower-case" else "capital kept"}, language model $ms ms" else "",
        )
        val continuing = if (lower) SentenceBoundary.lowerFirstWord(text) else standalone
        return Step(SentenceJoin(true, previousEndEdit, continuing, standalone), ms)
    }

    /**
     * The first chunk of a piece that starts where a cut inside speech ended and whose text starts with the new version of the word
     * held back from the previous piece (`CutSeam.CONTEXT_AND_HELD_WORD`): [text] continues the previous text, which ends one word
     * earlier, without the held word and without the period the cut gave it. The text was recognised with the audio on both sides of
     * the seam, so its case and punctuation there are Whisper's, not artefacts of the cut: the first word keeps its case, nothing of the
     * previous end changes ([PreviousEnd.KEEP]), and the language model is not asked (no time in [Step.languageModelMs]). No join
     * across lost speech, after [reset], before the first transcript or across a pause longer than [SentenceBoundary.MAX_PAUSE_MS],
     * as [afterCut]. Whisper thread.
     */
    fun afterContext(text: String, startMs: Long, endMs: Long): Step {
        applyLater()
        val previous = previousText
        val previousEnd = previousEndMs
        previousText = text
        previousEndMs = endMs
        val lostBetween = lostEnds.any { it > previousEnd && it <= startMs }
        lostEnds.removeAll { it <= startMs }
        if (previous == null || lostBetween || startMs - previousEnd !in 0..SentenceBoundary.MAX_PAUSE_MS) return NONE
        return Step(SentenceJoin(true, PreviousEnd.KEEP, text, SentenceBoundary.upperFirstWord(text)), 0)
    }

    private companion object {
        val NONE = Step(null, 0)
    }
}
