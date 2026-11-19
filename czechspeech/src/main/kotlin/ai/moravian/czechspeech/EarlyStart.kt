package ai.moravian.czechspeech

/**
 * Starts the final recognition of an utterance before its end is certain, so that its [Transcript] comes sooner.
 *
 * An utterance ends after [CzechSpeechConfig.minSilenceMs] of silence (600 ms), and only then is it recognised. With
 * this, the recogniser starts once the pause has lasted [silenceMs]: the same recognition as for the final text, with
 * everything the configuration asks for. When the pause grows into the end of the utterance, that recognition gives the
 * transcript, earlier by up to minSilenceMs − silenceMs (about 350 ms with the defaults). When the speech goes on, the
 * utterance is recognised at its end as usual, and the early recognition never becomes a [Transcript]:
 * - in [CzechSpeech.listen] it is aborted and dropped, and nothing of it reaches the app;
 * - in [CzechSpeech.listenWithPreviews] it finishes and its text comes as a [SpeechPreview] of the utterance in
 *   progress ([PreviewConfig.fromPauses], on by default), unless the next pause or the end of the utterance comes
 *   first, which aborts it. With `fromPauses = false` it is dropped as in `listen`.
 *
 * Two things to know:
 * - **The audio of the pause.** At the early start the rest of the pause has not been recorded yet, so the recogniser
 *   gets zeros in its place, and so does every utterance ended by silence while this is on, whether its early
 *   recognition was used or not: the text never depends on timing. It can differ from the text without early start,
 *   where the recogniser gets the microphone's own silence for those last ~350 ms. The difference has to be measured
 *   on real recordings before this is turned on for users.
 * - **The cost.** A recognition dropped because the speech went on is recogniser time and battery spent for nothing;
 *   every pause between [silenceMs] and minSilenceMs inside an utterance costs one. The logcat lines `utterance:` and
 *   `early start:` (tag `CzechSpeech`) count the early recognitions used, too late to gain anything (the recogniser
 *   was busy until the utterance had ended), shown as a preview, and dropped, and their times.
 *
 * @property silenceMs The recognition starts after this much silence, rounded up to 32 ms frames (256 ms is 8
 *   frames). It must be at least one frame less than [CzechSpeechConfig.minSilenceMs], which is rounded the same way
 *   (600 ms is 19 frames, so at most 576 ms with the default): [CzechSpeechConfig] throws IllegalArgumentException
 *   otherwise, because a start on the frame that ends the utterance would never start anything early. Shorter saves
 *   more time and drops more recognitions.
 */
public data class EarlyStart(
    public val silenceMs: Int = 256,
) {
    init {
        require(silenceMs in 96..5_000) { "silenceMs must be in 96..5000" }
    }
}
