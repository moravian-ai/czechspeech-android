package ai.moravian.czechspeech.internal

/**
 * When a final recognition that came out without a letter or a digit is repeated with a longer encoder window, and
 * with which. Whether the model writes anything for a stretch of speech is a flip on the edge that depends on the
 * window: in the measurement the next longer window gave text for 12 of 16 empty results, whisper.cpp's temperature
 * fallback gave none, and on audio without speech the model writes nothing at 384 frames or more, so a retry does not
 * invent text. Pure rules; the engine runs them (see `SpeechEngine.recognizeFinal`).
 */
internal object EmptyFinalRetry {
    /** At most this many repeats of one final: the window of 512 frames, then the longest one. */
    const val MAX_RETRIES = 2

    /** The retry is only for clear speech: this much of the utterance scored as speech by the endpointer, in ms. */
    const val CLEAR_SPEECH_MS = 1_000L

    /** The first retry's floor, when the run was below it: the default floor of the library. */
    const val FIRST_RETRY_FLOOR = EncoderWindow.DEFAULT_MIN_FRAMES

    /**
     * The longest floor the library takes (a multiple of 16 up to 1488): the window is then 1488 frames, or the full
     * 1500 when the audio needs it, i.e. the whole window within 0.8 %.
     */
    const val FULL_WINDOW_FLOOR = 1488

    /** In streaming mode the full-window retry (the dear one, about 12 s on a phone) waits while this many finals are pending. */
    const val PENDING_FINALS_BLOCKING_FULL_RETRY = 2

    /** A retry must make the window at least this much longer than the window of the first run, else it is skipped (1440 -> 1488 is pointless). */
    const val MIN_WINDOW_GROWTH = 1.25

    /**
     * The floor for the next repeat of a final whose first run was on a window of [firstWindow] frames and whose last run, on
     * [lastWindow], gave nothing, after [retriesDone] repeats, or null for no more. [audioSamples]: the audio, which sets the window
     * a floor gives. [pendingFinals]: the finals waiting or running (streaming mode, else 0; this one included). [othersQueued]:
     * other utterances wait for the recogniser. A candidate whose window is not longer than the last run's, or grows less
     * than [MIN_WINDOW_GROWTH] over the first run's, is skipped for the next one; the full-window candidate (the dear one) also
     * needs nothing else to wait: no other pending final and an empty queue. The caller asks again right before every retry,
     * and starts none after stop().
     */
    fun nextFloor(
        firstWindow: Int,
        lastWindow: Int,
        audioSamples: Int,
        retriesDone: Int,
        pendingFinals: Int,
        othersQueued: Boolean,
        /**
         * The final is that of a piece after a cut inside speech whose first run had the audio of the piece before in it (CutSeam):
         * that run replaced the repeat on 512 frames, so only the full window is left, and only when [fullFitsLag].
         */
        continuation: Boolean = false,
        /** The full-window repeat would still end within the lag limit ([StreamingGuard.fullWindowEstimateMs]); read only for a [continuation]. */
        fullFitsLag: Boolean = true,
    ): Int? {
        if (retriesDone >= MAX_RETRIES) return null
        for (floor in intArrayOf(FIRST_RETRY_FLOOR, FULL_WINDOW_FLOOR)) {
            if (continuation && (floor == FIRST_RETRY_FLOOR || !fullFitsLag)) continue
            val window = EncoderWindow.frames(audioSamples, floor)
            if (window <= lastWindow || window < firstWindow * MIN_WINDOW_GROWTH) continue
            if (floor == FULL_WINDOW_FLOOR && (pendingFinals >= PENDING_FINALS_BLOCKING_FULL_RETRY || othersQueued)) continue
            return floor
        }
        return null
    }
}
