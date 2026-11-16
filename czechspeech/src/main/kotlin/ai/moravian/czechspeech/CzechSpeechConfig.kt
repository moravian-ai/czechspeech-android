package ai.moravian.czechspeech

/**
 * Speech detection settings. The defaults are the validated ones; most apps never change them.
 *
 * Speech is detected per 32 ms frame by Silero VAD v6.2.0, which gives a speech probability for each frame.
 *
 * @property threshold An utterance starts at the first frame with probability at least this value. The same value
 *   gates Whisper: an utterance (or the audio passed to [CzechSpeech.transcribe]) in which Silero VAD finds no
 *   speech segment at this threshold is not transcribed at all. 0.25 rather than Silero's usual 0.5: on 2,181 test
 *   recordings with speech, 0.25 dropped none, 0.5 dropped 5 of 729 recordings with reverberation and babble noise;
 *   all synthetic non-speech (silence, white/pink/brown noise, 50 Hz hum, clicks, bursts, beeps) was still rejected.
 * @property negThreshold Once started, the utterance continues while the probability stays at or above this value.
 * @property minSilenceMs The utterance ends after this much audio below [negThreshold].
 * @property preRollMs Audio kept from before the start of speech and passed to the recogniser with the utterance,
 *   so that the first syllable is not cut off.
 * @property minSpeechMs Utterances with less than this much audio at or above [threshold] are discarded (clicks,
 *   coughs, short noises).
 * @property maxUtteranceMs Longest audio passed to the recogniser in one piece (Whisper's window is 30 s). Longer
 *   speech is cut at the frame with the lowest speech probability within the last 2 s before the limit and
 *   continues as the next utterance.
 * @property threads Recogniser threads; 0 means the number of big CPU cores (at least 2, at most 8).
 */
public data class CzechSpeechConfig(
    public val threshold: Float = 0.25f,
    public val negThreshold: Float = 0.10f,
    public val minSilenceMs: Int = 600,
    public val preRollMs: Int = 300,
    public val minSpeechMs: Int = 250,
    public val maxUtteranceMs: Int = 28_000,
    public val threads: Int = 0,
) {
    init {
        require(threshold > 0f && threshold < 1f) { "threshold must be in (0, 1)" }
        require(negThreshold >= 0f && negThreshold <= threshold) { "negThreshold must be in [0, threshold]" }
        require(minSilenceMs in 32..10_000) { "minSilenceMs must be in 32..10000" }
        require(preRollMs in 0..2_000) { "preRollMs must be in 0..2000" }
        require(minSpeechMs in 0..10_000) { "minSpeechMs must be in 0..10000" }
        require(maxUtteranceMs in 4_000..29_000) { "maxUtteranceMs must be in 4000..29000" }
        require(threads in 0..64) { "threads must be in 0..64 (0 = number of big cores)" }
    }
}
