package ai.moravian.czechspeech

/**
 * When [CzechSpeech.listenWithPreviews] previews the utterance in progress. Times are audio, on the clock of
 * [Transcript.startMs].
 *
 * Each preview runs the recogniser over the whole utterance so far, so previews cost recogniser time and battery: with
 * the defaults, a 28 s utterance gets up to 5 previews over about 75 s of audio, 2.7 times the audio of its final
 * recognition. A longer interval costs less.
 *
 * @property firstAfterMs The first timed preview of an utterance is computed once it has lasted this long. Shorter
 *   utterances get none by the clock: their final text comes about as soon. A preview from a pause ([fromPauses])
 *   does not wait for it: it comes whenever the speech runs over a pause, also in the utterance's first seconds.
 * @property intervalMs The next preview of the same utterance comes at the earliest this much audio after the previous
 *   one started, and never while the previous one runs.
 * @property fromPauses Only with [CzechSpeechConfig.earlyStart]: a recognition started in a pause that the speech then
 *   ran over is not aborted but finishes, and its text comes as a preview: it was started anyway, it is a full
 *   recognition, and its audio ends at a pause, not in the middle of a word. Timed previews then count from that
 *   pause, so they only come when no pause has come for [intervalMs]. Such a recognition is aborted when the next
 *   pause comes or its utterance ends before it has finished: the recognition that may give the final text starts at
 *   once and never waits for a preview longer than the abort takes. False: it is aborted as soon as the speech goes
 *   on, as in [CzechSpeech.listen].
 */
public data class PreviewConfig @JvmOverloads constructor(
    public val firstAfterMs: Int = 5_000,
    public val intervalMs: Int = 5_000,
    public val fromPauses: Boolean = true,
) {
    init {
        require(firstAfterMs in 1_000..29_000) { "firstAfterMs must be in 1000..29000" }
        require(intervalMs in 1_000..29_000) { "intervalMs must be in 1000..29000" }
    }

    public companion object {
        /**
         * Previews as densely as the recogniser allows: the first once the utterance has lasted 2 s, the next as soon
         * as the previous one has ended and at least 1 s of new audio has come. On a phone that takes 2 to 3.5 s for a
         * pass, the text grows every 2 to 3.5 s while someone speaks without pausing, and the recogniser works nearly
         * all the time the speech lasts: several times the battery and heat of the default, which previews every 5 s.
         * Final text still comes first: a pass is aborted when its utterance ends.
         */
        public fun continuous(): PreviewConfig = PreviewConfig(firstAfterMs = 2_000, intervalMs = 1_000)
    }
}
