package ai.moravian.czechspeech

/**
 * Speech detection and recognition settings. The defaults are the validated ones; most apps never change them.
 *
 * Speech is detected per 32 ms frame by Silero VAD v6.2.0, which gives a speech probability for each frame.
 *
 * @property threshold An utterance starts at the first frame with probability at least this value. The same value
 *   gates Whisper: an utterance (or the audio passed to [CzechSpeech.transcribe]) in which Silero VAD finds no
 *   speech segment at this threshold is not transcribed at all. 0.25 rather than Silero's usual 0.5: on 2,181 test
 *   recordings with speech, 0.25 dropped none, 0.5 dropped 5 of 729 recordings with reverberation and babble noise;
 *   all synthetic non-speech (silence, white/pink/brown noise, 50 Hz hum, clicks, bursts, beeps) was still rejected.
 * @property negThreshold Once started, the utterance continues while the probability stays at or above this value.
 * @property minSilenceMs The utterance ends after this much audio below [negThreshold] (default 600 ms; accepted
 *   32..10000, sensible 400..2000). The trade-off: a longer pause means fewer sentences split where the speaker only
 *   paused to think (a pause of 900 ms is two utterances at 600 and one at 1200), and the text of every utterance arrives
 *   later by the same amount, as it is recognised only when the pause is over; a shorter one gives the text sooner and
 *   splits more. With [EarlyStart] the utterance is recognised already after its `silenceMs` (a quarter of a second) of
 *   that pause and used when the pause lasts to [minSilenceMs], so a longer [minSilenceMs] makes more of those starts
 *   wasted (one for every pause between `silenceMs` and [minSilenceMs] inside speech) and the start must stay at least one
 *   32 ms frame below it. With streaming ([StreamingConfig]) a pause shorter than this is inside the speech: a cut on it
 *   is a quiet cut, and the sentence decision after it leans to going on. In chunk mode (a slow phone) a piece is cut as soon
 *   as a quiet stretch of 256 ms comes after the floor of its cut window, so there the text does not come later with a larger
 *   value; the setting then only changes which way the next piece leans (a pause under it is a quiet cut, one over it ends the
 *   utterance and is decided as after a pause).
 * @property preRollMs Audio kept from before the start of speech and passed to the recogniser with the utterance,
 *   so that the first syllable is not cut off.
 * @property minSpeechMs Utterances with less than this much audio at or above [threshold] are discarded (clicks,
 *   coughs, short noises).
 * @property maxUtteranceMs Longest audio passed to the recogniser in one piece (Whisper's window is 30 s). Longer
 *   speech is cut at the frame with the lowest speech probability within the last 2 s before the limit and
 *   continues as the next utterance, so such a piece is between the limit less 2 s and the limit long: with the
 *   smallest limit, 4000, it can be 2 s (documented, not clamped: a limit of 4000 asks for pieces of that size).
 * @property threads Most recogniser threads; 0 means the number of big CPU cores (at least 2, at most 8). With
 *   [autoThreads] (the default) each recognition may use fewer, as the cores the app may use at that moment allow;
 *   without it, exactly this many.
 * @property variantSelection Beam search and a Czech language model choosing among the variants (see
 *   [VariantSelection]); null, the default, keeps greedy decoding, unchanged.
 * @property sentenceJoining The Czech language model decides whether a transcript continues the sentence of the
 *   previous one across a pause ([SentenceJoining], [Transcript.sentenceJoin]); null, the default, leaves
 *   [Transcript.sentenceJoin] null. Independent of [variantSelection]; with both, they take the same language model
 *   file, loaded once.
 * @property autoThreads Before every recognition (final text, preview, language model), choose the recogniser threads
 *   from the cores Android lets the app use at that moment, which change with its state (on screen, a service under
 *   another app's window, screen off). The recogniser splits its work evenly, so one thread on a slow core holds back
 *   the others: when the allowed fast cores (faster than the device's little cores) do at least 1.5 times better than
 *   [threads] threads placed by Android, judged by threads × the speed of the slowest core a thread ends up on, the
 *   recogniser runs on them, pinned for the call and unpinned afterwards; otherwise [threads] threads, placed by
 *   Android, as before. Never more threads than allowed cores. When the pinned cores did not all run under load
 *   (isolated for heat, say), pinning pauses for 30 s. Devices whose cores the SDK cannot tell apart (no cpu_capacity
 *   or cpufreq, as in an emulator) keep [threads] threads. False, for diagnostics: [threads] threads in every
 *   recognition, placed by Android, as before this option existed. The logcat line of each recognition (tag
 *   `CzechSpeech`) gives the threads and the core masks either way.
 * @property minEncoderWindow The shortest window the recogniser's encoder is run on, in frames of 20 ms (whisper.cpp's
 *   audio_ctx): **a property of the model file you pass, not a tuning knob.** The encoder's time grows with its window,
 *   so the SDK trims the window to the utterance, but a model fails on windows much shorter than it was trained on: it
 *   invents and repeats sentences in short utterances. 512 (10.24 s), the default, is what the published CzechSpeech
 *   model needs: with 384 it gets worse, from 256 down it invents text. The Czech+Slovak+English model bears 384, which took
 *   27 % off the recognition time of utterances up to 7 s in the measurement; use 384 only with that model. A multiple
 *   of 16 from 256 to 1488. Applies to the final text, previews and early recognitions alike. It lives here because
 *   this version has no description of a model beyond its file; a model catalogue should carry it instead.
 * @property earlyStart Start the final recognition of an utterance already during the pause that may end it, so that
 *   its transcript comes sooner ([EarlyStart]); null, the default, starts it when the utterance has ended, as before.
 *   Off by default until its two costs are measured on phones: the recogniser then gets zeros for the last part of the
 *   closing pause, which may change the text, and a recognition dropped because the speech went on costs battery.
 * @property streaming Send the text of words while the person is still speaking, in several
 *   [SpeechEvent.Transcript]s per utterance, instead of one after the pause ([StreamingConfig]): the words two
 *   consecutive recognitions of the speech so far agree on are sent at once and never changed; the rest comes in the
 *   last transcript after the utterance. A chunk can end inside a sentence, so concatenate the transcripts of one
 *   utterance with one space. [earlyStart] and the timing of [PreviewConfig] are ignored with it. It costs about
 *   four times the computation while speech is recognised, and was measured with the cs+sk+en model at the window
 *   floor 384; null, the default, keeps every behaviour and event as before this option existed.
 * @property retryEmptyFinals Repeat a final recognition that came out without a letter or a digit, when the endpointer saw
 *   clear speech in it (at least one second of frames above [threshold]), with a longer encoder window: first 512 frames
 *   when the first run was below that, then the longest window the library takes (1488 frames, as good as the whole one),
 *   and keep the first result with words. Whether a model writes nothing for a stretch of speech depends on the window
 *   (measured: the next longer window gave text for 12 of 16 empty results of the cs+sk+en model, and on audio without
 *   speech it still writes nothing), but each retry costs one more recognition (on a phone about 4 s at 512 frames and
 *   about 12 s at the full window, when the first run took 2.3 s at 384), so the text of that utterance comes later. In
 *   streaming mode the full-window retry is skipped while two finals are waiting. True by default; false: an empty
 *   result is dropped at once, as before. Either way [SpeechEvent.UtteranceNotWritten] tells the app about a stretch
 *   of clear speech that ends up without text.
 *   Early start ([earlyStart]) recognitions are not repeated. The full-window repeat
 *   runs only when nothing else waits for the recogniser (an empty queue, no other final pending, no stop()), and a repeat
 *   whose window is less than 25 % longer than the first run's is skipped; a streamed piece that already wrote words is
 *   repeated only when words were still pending. With several languages and a [CzechSpeechModel.LanguageSwitch] a rescue in
 *   another language comes before these repeats and does not depend on this option or on the second of clear speech: when
 *   the result is empty and another language of the switch (or the first language) has at least half of the shares, one more
 *   recognition runs forced in it, on the same window; the repeats then run in the language of the last attempt.
 */
public data class CzechSpeechConfig @JvmOverloads constructor(
    public val threshold: Float = 0.25f,
    public val negThreshold: Float = 0.10f,
    public val minSilenceMs: Int = 600,
    public val preRollMs: Int = 300,
    public val minSpeechMs: Int = 250,
    public val maxUtteranceMs: Int = 28_000,
    public val threads: Int = 0,
    public val variantSelection: VariantSelection? = null,
    public val sentenceJoining: SentenceJoining? = null,
    public val autoThreads: Boolean = true,
    public val earlyStart: EarlyStart? = null,
    public val minEncoderWindow: Int = 512,
    public val streaming: StreamingConfig? = null,
    public val retryEmptyFinals: Boolean = true,
) {
    init {
        require(threshold > 0f && threshold < 1f) { "threshold must be in (0, 1)" }
        require(negThreshold >= 0f && negThreshold <= threshold) { "negThreshold must be in [0, threshold]" }
        require(minSilenceMs in 32..10_000) { "minSilenceMs must be in 32..10000" }
        require(preRollMs in 0..2_000) { "preRollMs must be in 0..2000" }
        require(minSpeechMs in 0..10_000) { "minSpeechMs must be in 0..10000" }
        require(maxUtteranceMs in 4_000..29_000) { "maxUtteranceMs must be in 4000..29000" }
        require(threads in 0..64) { "threads must be in 0..64 (0 = number of big cores)" }
        require(minEncoderWindow in 256..1500 && minEncoderWindow % 16 == 0) {
            "minEncoderWindow must be a multiple of 16 from 256 to 1488 (frames of 20 ms), not $minEncoderWindow"
        }
        // Silences are counted in 32 ms frames, rounded up: an early start that falls on the frame that ends the
        // utterance could never start anything. Refused rather than ignored: the caller asked for something else.
        require(earlyStart == null || (earlyStart.silenceMs + 31) / 32 < (minSilenceMs + 31) / 32) {
            "earlyStart.silenceMs (${earlyStart?.silenceMs}) must be at least one 32 ms frame less than minSilenceMs " +
                "($minSilenceMs): both are rounded up to 32 ms frames"
        }
        val selectionFile = variantSelection?.languageModelFile
        val joiningFile = sentenceJoining?.languageModelFile
        require(selectionFile == null || joiningFile == null || selectionFile.absoluteFile == joiningFile.absoluteFile) {
            "variantSelection and sentenceJoining take one language model: give both the same languageModelFile"
        }
    }
}
