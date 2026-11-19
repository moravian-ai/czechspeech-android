package ai.moravian.czechspeech

/**
 * Streaming transcription during continuous speech: the text of words is sent while the person is still speaking,
 * instead of after the pause that ends the utterance. Turn it on with [CzechSpeechConfig.streaming]; null, the
 * default, keeps every behaviour and event as without it.
 *
 * How it works. While a piece of speech is in progress, the SDK transcribes its not yet committed audio again and
 * again, one pass at a time (the first one second after the speech started, each next one as soon as the previous
 * has finished and one more second of audio has come). The words on which two consecutive passes agree exactly
 * (letters, case and punctuation) are committed and sent at once as a [SpeechEvent.Transcript]; the SDK never
 * changes or withdraws them. When the not yet committed audio is longer than 6 s, its start moves forward to the
 * quietest place just before words that are all committed, so that passes stay short. When the piece ends (a pause,
 * `maxUtteranceMs`, `stop()`), the normal final recognition covers only the audio after the committed words.
 *
 * Events with streaming on:
 * - The text of one utterance can come in several [SpeechEvent.Transcript]s, the first ones while it is spoken
 *   (before its [SpeechEvent.SpeechEnded]), the last ones after it (a chunk found just before the end of the speech
 *   is sent when the recogniser has it, and the final recognition adds the rest). Concatenate them with one space: a chunk can end
 *   inside a sentence and the next one then starts inside it, in Whisper's own case and punctuation.
 * - A chunk is final: the SDK never changes or withdraws it. (With [SentenceJoining], the first chunk of the next
 *   utterance may still ask to edit the end of the previous one, as without streaming.)
 * - Chunks of one utterance share its `startMs`; `endMs` never decreases and never goes past the end of the utterance
 *   (the last chunk's `endMs` is that end when the final recognition added words, else the audio its last words
 *   lie in).
 * - When everything was committed while speaking, the final recognition adds no [SpeechEvent.Transcript] (as for
 *   speech without words). The end of every utterance is [SpeechEvent.UtteranceDone].
 * - A speech that the length cap or the lag limit cut is several utterances in a row (a cut never falls before the
 *   audio whose words were sent, so no word is written twice), each with its own `startMs` and [SpeechEvent.UtteranceDone].
 * - In `listenWithPreviews`, a [SpeechPreview] (the same `startMs`) carries only the words after the last chunk, with
 *   `stableLength` 0, all of it uncertain. Any [SpeechEvent.Transcript] replaces the preview shown, and a new preview
 *   may follow it. A preview with empty text withdraws it when the utterance ends without a last chunk (it can come
 *   after [SpeechEvent.SpeechEnded]). Previews with text come after a pass, never after [SpeechEvent.SpeechEnded].
 * - `listen()` gets the same [SpeechEvent.Transcript]s, without previews.
 *
 * Guarantee. Every word's text is sent at most [maxLagMs] after the word was spoken, and after the speech ends the
 * rest comes within [CzechSpeechConfig.minSilenceMs] plus one recognition of at most 12 s of audio, on a phone for
 * which a recognition of 8 s of audio takes at most [maxLagMs] minus 9 s (6 s with the default). On a phone that
 * is slower but still faster than real time, the delay stays bounded and does not grow. The SDK keeps this by
 * measuring the speed of the recogniser: when the not yet sent audio would get older than [maxLagMs], or reaches
 * 12 s, the piece is cut there and closed by its final recognition; on a phone where a pass over 6 s of audio would
 * take more than 2 s, no passes run and the text comes from the final recognitions of pieces, each cut as soon as the
 * phone keeps up with it: at the quietest frame before a target, the shortest length from 5.5 s (up to 12 s) at which the
 * recognition of the shortest possible piece (the target less 1.5 s, never under 4 s) is estimated at 60 % of it or less,
 * so 4 to 8 s of audio by the speed of the phone, longer as it heats up (a shorter piece would cost
 * about the same, as the encoder window has a floor, and be transcribed worse). A phone too slow for that is cut at the
 * lag limit only, after 8 s of audio at the earliest. A sudden slowdown within one recognition can overshoot the bound
 * once. The text therefore comes in pieces, not word by word. The margin is thin on a phone that is slow and also gets
 * slower as it heats up: in the simulation of the author's phone at 2.2 times its measured speed the largest delay was
 * 15.0 s of
 * the default 15 s; beyond that the bound does not hold, and a larger [maxLagMs] gives room.
 *
 * The retry of an empty final ([CzechSpeechConfig.retryEmptyFinals]) is an exception to the guarantee: a stretch of clear
 * speech whose final recognition came out without a letter or a digit is recognised again, so its text can come later
 * than [maxLagMs] for that one stretch: on a phone like the author's (2.3 s on the lowest window) by about 4 s for the
 * repeat on 512 frames and, only when nothing else waits, by up to 12 to 16 s more for the repeat on the full window, so
 * by up to about 20 s in all; a warm phone is 1.4 to 1.9 times slower. The repeats are not counted in the guard's estimates.
 * With [cutSeam] other than [CutSeam.NONE] there are two more exceptions: the final of a piece after a cut inside speech covers up to
 * 14.5 s of audio (the context of 2.5 s), not 12 s, so the rest after the speech ends does not always come within [CzechSpeechConfig.minSilenceMs]
 * plus one recognition of at most 12 s; and when the text anchor is not found the piece is recognised a second time (about 2 s on a phone),
 * which can exceed [maxLagMs] in the same way as the retry. The target length of the pieces does not count the context, so the real duty of
 * such a final is above the 60 %.
 *
 * Interactions: [EarlyStart] and the timing and `fromPauses` of [PreviewConfig] are ignored while streaming is on;
 * `transcribe()` is not affected. Cost: while passes run, the recogniser computes about four times as much as
 * without streaming, so battery and heat grow with the time spent speaking. It was measured with the cs+sk+en model
 * at the window floor 384 ([CzechSpeechConfig.minEncoderWindow]); with a model that needs the floor 512 passes are
 * dearer, and on slower phones more of the speech is closed by the safeguards.
 *
 * @property maxLagMs The longest time, in ms, between a word being spoken and its text being sent (see the
 *   guarantee above), 10 000 to 30 000. Smaller values make the SDK cut pieces sooner, which costs accuracy at the
 *   cuts; larger ones allow more delay on slow phones.
 * @property cutSeam What is done at a cut inside speech ([CutSeam]); [CutSeam.CONTEXT_AND_HELD_WORD] by default. [CutSeam.NONE] keeps the text
 *   of every piece as its own audio alone gives it, [CutSeam.CONTEXT] adds the last 2.5 s of the piece before to the recognition
 *   of a piece after a cut (no change in the events), [CutSeam.CONTEXT_AND_HELD_WORD] also holds back the last word of a piece
 *   cut inside speech (its text comes without it and its `UtteranceDone` later, right before the next piece's text; a held word is
 *   never a preview: see [CutSeam.CONTEXT_AND_HELD_WORD]). Pass `cutSeam = CutSeam.CONTEXT` to opt out of the holding and keep the
 *   plain order of the events.
 */
public data class StreamingConfig @JvmOverloads constructor(
    public val maxLagMs: Int = 15_000,
    public val cutSeam: CutSeam = CutSeam.CONTEXT_AND_HELD_WORD,
) {
    init {
        require(maxLagMs in 10_000..30_000) { "maxLagMs must be in 10000..30000" }
    }
}
