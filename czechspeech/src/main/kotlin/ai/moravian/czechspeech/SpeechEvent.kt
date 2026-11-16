package ai.moravian.czechspeech

/**
 * One recognised utterance.
 *
 * [toString] does not print [text]: transcripts are personal data and should not end up in logs by accident.
 *
 * @property text The transcript, in the model's language, never blank.
 * @property startMs Start of speech, in milliseconds of audio since [CzechSpeech.listen] started collecting.
 * @property endMs End of speech, on the same clock.
 * @property processingMs Time spent on this utterance by the speech check and the recogniser (wall clock).
 * @property language The language of the transcript: the one asked for, or the one the recogniser detected among the
 *   languages of the model. `"cs"` for the Czech model.
 */
public data class Transcript(
    public val text: String,
    public val startMs: Long,
    public val endMs: Long,
    public val processingMs: Long,
    public val language: String,
) {
    override fun toString(): String =
        "Transcript(text=<${text.length} chars>, startMs=$startMs, endMs=$endMs, processingMs=$processingMs, language=$language)"
}

/**
 * Events of a [CzechSpeech.listen] session, in order.
 *
 * Every [SpeechStarted] is followed by a [SpeechEnded]. The [Transcript] of that speech comes after [SpeechEnded]
 * (recognition runs once the utterance is complete), or not at all when the recogniser finds no words in it.
 * Speech longer than [CzechSpeechConfig.maxUtteranceMs] gives several [Transcript]s between one pair.
 */
public sealed interface SpeechEvent {
    /** Speech started at [atMs] milliseconds of audio since the session started. */
    public data class SpeechStarted(public val atMs: Long) : SpeechEvent

    /** Speech ended at [atMs]; emitted after [CzechSpeechConfig.minSilenceMs] of silence. */
    public data class SpeechEnded(public val atMs: Long) : SpeechEvent

    /** A recognised utterance. */
    public data class Transcript(public val transcript: ai.moravian.czechspeech.Transcript) : SpeechEvent

    /**
     * Recognition of one utterance failed, or the utterance was skipped because recognition fell behind the speech
     * (a [CzechSpeechException] saying which speech was skipped); the session goes on. Errors that end the session
     * (no microphone permission, the microphone or the model cannot be opened) are thrown from the flow instead.
     */
    public data class Error(public val cause: Throwable) : SpeechEvent
}

/** Thrown when the model, the microphone or the native library cannot be used. */
public class CzechSpeechException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
