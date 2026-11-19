package ai.moravian.czechspeech

/**
 * One recognised utterance.
 *
 * With [CzechSpeechConfig.streaming] a transcript can be a chunk of an utterance (concatenate the chunks with one space;
 * they share [startMs], see [StreamingConfig]).
 *
 * [toString] does not print [text]: transcripts are personal data and should not end up in logs by accident.
 *
 * @property text The transcript, in the model's language, never blank.
 * @property startMs Start of speech, in milliseconds of audio since [CzechSpeech.listen] started collecting.
 * @property endMs End of speech, on the same clock.
 * @property processingMs Time spent on this utterance by the speech check and the recogniser (wall clock), and by the
 *   language model when it decided [sentenceJoin].
 * @property language The language of the transcript: the one asked for, or the one the recogniser chose among the
 *   languages of the model (by the model's `LanguageSwitch`, else the most probable one); the language of a rescue run
 *   when an empty result was rescued in another language. `"cs"` for the Czech model.
 * @property sentenceJoin With [CzechSpeechConfig.sentenceJoining]: how this transcript continues the previous one of
 *   the session ([SentenceJoin]), or null when it stays as Whisper wrote it. Always null without sentence joining.
 *   [text] is the same either way.
 */
public data class Transcript(
    public val text: String,
    public val startMs: Long,
    public val endMs: Long,
    public val processingMs: Long,
    public val language: String,
    public val sentenceJoin: SentenceJoin? = null,
) {
    override fun toString(): String =
        "Transcript(text=<${text.length} chars>, startMs=$startMs, endMs=$endMs, processingMs=$processingMs, " +
            "language=$language, sentenceJoin=$sentenceJoin)"
}

/**
 * What [CzechSpeech.listenWithPreviews] emits, in order: the [SpeechEvent]s of the session, exactly as
 * [CzechSpeech.listen] emits them, and [SpeechPreview]s of the utterance in progress between them.
 */
public sealed interface ListenEvent

/**
 * Events of a [CzechSpeech.listen] session (or of [CzechSpeech.listenWithPreviews]), in order.
 *
 * Every [SpeechStarted] is followed by a [SpeechEnded]. The [Transcript] of that speech comes after [SpeechEnded]
 * (recognition runs once the utterance is complete), or not at all when the recogniser finds no words in it.
 * Speech longer than [CzechSpeechConfig.maxUtteranceMs] gives several [Transcript]s between one pair. With
 * [CzechSpeechConfig.streaming] one utterance gives several [Transcript]s while it is spoken, the last ones after its
 * [SpeechEnded] (a chunk found just before the end can come after it); see [StreamingConfig]. Each utterance ends with
 * [UtteranceDone].
 */
public sealed interface SpeechEvent : ListenEvent {
    /** Speech started at [atMs] milliseconds of audio since the session started. */
    public data class SpeechStarted(public val atMs: Long) : SpeechEvent

    /** Speech ended at [atMs]; emitted after [CzechSpeechConfig.minSilenceMs] of silence. */
    public data class SpeechEnded(public val atMs: Long) : SpeechEvent

    /** A recognised utterance. */
    public data class Transcript(public val transcript: ai.moravian.czechspeech.Transcript) : SpeechEvent

    /**
     * Only with [CzechSpeechConfig.streaming]: the utterance (piece of speech) that started at [startMs] is complete,
     * no more text of it comes. Emitted exactly once for every utterance that has a final recognition or wrote words,
     * after its last [Transcript]; also when its final recognition found no words, failed (the [Error] comes first) or
     * was skipped because recognition fell behind. Never emitted without streaming, and not after the session was
     * cancelled.
     *
     * - [startMs] is exactly the [Transcript.startMs] of every [Transcript] of the utterance (compare for equality).
     * - For the first utterance of a speech it is also the [SpeechStarted.atMs] of that speech. A speech that the length
     *   cap ([CzechSpeechConfig.maxUtteranceMs]) or the lag limit ([StreamingConfig.maxLagMs]) cut is several
     *   utterances: each after the first starts where the one before ended (that utterance's last [Transcript.endMs]
     *   is its [startMs]) and has no [SpeechStarted] of its own.
     * - Order: after the last [Transcript] of its utterance, but not necessarily before the next speech: the
     *   [SpeechStarted] of the next speech can come first when recognition lags behind (the events of one utterance
     *   come in order, those of different utterances may interleave with the speech events).
     * - With [CutSeam.CONTEXT_AND_HELD_WORD] the [UtteranceDone] of a piece cut inside speech waits until the word held back
     *   from it is resolved: replaced by its new version at the start of the next utterance's first [Transcript] (the
     *   [UtteranceDone] then comes right before that [Transcript]) or sent as one more [Transcript] of this utterance (same
     *   [startMs]) when it cannot be replaced. So the text of an utterance is final at its [UtteranceDone], and it may come about
     *   one piece (a few seconds) later than the last [Transcript] that was sent for it.
     */
    public data class UtteranceDone(public val startMs: Long) : SpeechEvent

    /**
     * A stretch of clear speech that could not be written: the final recognition of the utterance that started at
     * [startMs] found no letter or digit, also after the retries of [CzechSpeechConfig.retryEmptyFinals] (or with the retry
     * turned off), although the endpointer had seen at least one second of speech in it. Emitted once for that utterance,
     * so that an app can show that something was said and is missing. Not emitted for a stretch without clear speech (a
     * click, a cough, noise: under a second), for one that the recogniser never ran on (the gate found no speech) or
     * that was skipped (an [Error] says so), and, with [CzechSpeechConfig.streaming], not for a piece that already wrote
     * words while it was spoken (what remained of it was empty). Emitted by every session, with streaming or without.
     *
     * - [startMs] is the [SpeechStarted.atMs] of the speech for its first utterance, and exactly the
     *   [Transcript.startMs] a [Transcript] of that utterance would have had (as for [UtteranceDone]).
     * - There is no [Transcript] of that utterance. The event comes after the audio of that stretch has ended, in the place
     *   its [Transcript] would have had: when the speech ended with a pause, after its [SpeechEnded]; for a piece that a
     *   forced cut ([StreamingConfig.maxLagMs]) or the length cap ([CzechSpeechConfig.maxUtteranceMs]) ended inside
     *   speech, **before** the [SpeechEnded] of the whole utterance (the speech goes on and ends later). With streaming it
     *   comes before the [UtteranceDone] of the same [startMs] (which is then still emitted, as for any utterance whose final
     *   found no words). Nothing else is guaranteed about its order with the next [SpeechStarted].
     * - Also emitted when [CzechSpeech.stop] kept a retry from starting (speech never disappears silently); not after the
     *   session was cancelled.
     * - Also emitted when the recognition of such a stretch failed, with whatever it threw (an [Error] comes first): the
     *   speech is missing all the same.
     */
    public data class UtteranceNotWritten(public val startMs: Long) : SpeechEvent

    /**
     * Recognition of one utterance failed, or the utterance was skipped because recognition fell behind the speech
     * (a [CzechSpeechException] saying which speech was skipped); the session goes on. Errors that end the session
     * (no microphone permission, the microphone or the model cannot be opened) are thrown from the flow instead.
     * [cause] is whatever the recognition threw, an `Error` too (a class that failed to initialise): nothing the work of
     * one utterance throws ends the process of the app, only this event reports it.
     */
    public data class Error(public val cause: Throwable) : SpeechEvent
}

/**
 * A preview of the utterance still being spoken, from [CzechSpeech.listenWithPreviews]: what the recogniser makes of
 * its audio so far.
 *
 * **Unstable text.** Show it as provisional (in grey, as composing text) and never insert it as finished text: the
 * words of a preview can change in the next preview and in the final text, which is recognised from the whole
 * utterance. Each preview replaces the previous one of the same utterance.
 *
 * **The stable part.** A preview's audio ends wherever the speaker is, often inside a word, so its last words are the
 * least reliable. [stableText] is the start of [text] that the previous preview of the same utterance also started
 * with (the same words, whatever their case and punctuation); [unstableText] is the rest. An app can show the two
 * differently, say the uncertain end in a lighter grey. The stable part is still a preview: it usually only grows, but
 * it can shrink, and the final text can differ from it. The first preview of an utterance has none.
 *
 * It ends in one of two ways, which always come after it:
 * - the [SpeechEvent.Transcript] whose [Transcript.startMs] equals [startMs] replaces it with the final text;
 * - a preview with the same [startMs] and empty [text] withdraws it: the utterance gave no transcript (no words in
 *   it, its recognition failed, or it was skipped; the [SpeechEvent.Error] comes too in the last two cases).
 *
 * A preview with text comes while its utterance is in progress: after the [SpeechEvent.SpeechStarted] of its speech
 * and before its [SpeechEvent.SpeechEnded], so never after the Transcript of its utterance. A flow that completes
 * normally ([CzechSpeech.stop], the end of the audio) has ended every preview; a cancelled or failed one may leave
 * one shown, which the app clears.
 *
 * With [CzechSpeechConfig.streaming] a preview carries only the words after the last [SpeechEvent.Transcript] of its
 * utterance, [stableLength] is 0 and all of it is uncertain (see [StreamingConfig]).
 *
 * [toString] does not print [text]: like transcripts, previews are personal data.
 *
 * @property startMs Identifies the utterance: its start, on the clock of [Transcript.startMs].
 * @property text The provisional text, or "" when the preview is withdrawn.
 * @property stableLength How many chars of [text], from its start, the previous preview agreed on: the length of
 *   [stableText]. It ends at the end of a word. 0 for the first preview of an utterance and for a withdrawn one.
 */
public data class SpeechPreview @JvmOverloads constructor(
    public val startMs: Long,
    public val text: String,
    public val stableLength: Int = 0,
) : ListenEvent {
    init {
        require(stableLength in 0..text.length) { "stableLength must be in 0..${text.length}" }
    }

    /** The start of [text] that two previews in a row agreed on; "" when there is none. */
    public val stableText: String get() = text.substring(0, stableLength)

    /** The rest of [text]: its uncertain end. */
    public val unstableText: String get() = text.substring(stableLength)

    override fun toString(): String =
        "SpeechPreview(startMs=$startMs, text=<${text.length} chars>, stableLength=$stableLength)"
}

/** Thrown when the model, the microphone or the native library cannot be used. */
public class CzechSpeechException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
