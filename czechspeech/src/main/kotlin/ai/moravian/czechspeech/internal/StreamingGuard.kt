package ai.moravian.czechspeech.internal

import kotlin.math.ceil

/**
 * The safeguard of streaming transcription: bounds the delay of the text on any phone. Pure logic without a clock;
 * the times it needs come in as arguments. One instance serves all sessions of an engine (the speed of the phone does
 * not start from the default in every session), so its methods are synchronized: a late final recognition of a
 * cancelled session reports on the whisper thread while the capture thread of the next one reads.
 *
 * It keeps two estimates of what a recognition costs, in ms of recogniser time per ms of encoder window (the encoder's
 * cost follows the window, not the audio, see [EncoderWindow]), each the largest of the last [COST_SAMPLES]
 * measurements of its kind: [costPerWindowMs] of the greedy passes (it decides whether passes run) and
 * [finalCostPerWindowMs] of the final recognitions (with the language model's time, it decides when a buffer must be
 * cut: the recognition that closes it is a final one). [DEFAULT_COST] stands in before any pass was measured.
 *
 * - [estimateMs]: what a pass over a buffer takes; [closingEstimateMs]: what the final recognition of a buffer and the
 *   language model's decision take.
 * - [passesPaused]: stopped when a pass over [PROBE_MS] is estimated above [slowPassMs]. Resumed at or below
 *   [resumePassMs], judged on the same kind of recognition as the stop: no pass runs while they are paused, so the
 *   finals decide, scaled by what the passes cost against the finals when they were paused (per ms of window a final
 *   is not as dear as a pass over a smaller window: unscaled, finals would resume passes that the passes would stop
 *   again at once, 3 to 8 times in 2 to 5 minutes on a phone near the limit).
 * - [mustCut]: the buffer must be cut now (the lag limit).
 * - [targetPieceMs]: with the passes paused, how long a piece should be: the shortest one that the phone manages with
 *   [DUTY_TARGET] of its time, from [MIN_PIECE_MS]; the lag limit stays above it.
 */
internal class StreamingGuard(
    private val maxLagMs: Long,
    private val minEncoderWindow: Int,
    private val slowPassMs: Long = SLOW_PASS_MS,
    private val resumePassMs: Long = RESUME_PASS_MS,
    private val maxBufferMs: Long = MAX_BUFFER_MS,
) {
    /** What a measurement did to [passesPaused]. */
    enum class Change { NONE, PAUSED, RESUMED }

    private val passCosts = ArrayDeque<Double>()
    private val finalCosts = ArrayDeque<Double>()

    /**
     * The recogniser ms of the last [COST_SAMPLES] finals that ran at the lowest window (the audio fits under it): their
     * median counts, so one slow outlier (a busy moment of the phone) does not remove the target, and three slow finals do.
     */
    private val finalFlatMs = ArrayDeque<Long>()
    private val languageModelTimes = ArrayDeque<Long>()
    private var passMeasured = false

    /** What a pass over [PROBE_MS] was estimated at, and the final cost, when the passes were paused. */
    private var pausedProbeMs = 0L
    private var pausedFinalCost: Double? = null

    /** Recogniser ms per ms of encoder window of the greedy passes: the largest of the last [COST_SAMPLES]. */
    @Volatile
    var costPerWindowMs: Double = DEFAULT_COST
        private set

    /** The same for the final recognitions; null before the first one. */
    @Volatile
    var finalCostPerWindowMs: Double? = null
        private set

    /** The time the language model takes for the sentence decision of a chunk: the largest of the last [COST_SAMPLES], 0 without one. */
    @Volatile
    var languageModelMs: Long = 0
        private set

    /** Final recognitions repeated on a longer window ([EmptyFinalRetry]): how many, and their recogniser ms; no estimate uses them. */
    @Volatile
    var retryCount: Int = 0
        private set

    @Volatile
    var retryRecogniserMs: Long = 0
        private set

    /** Passes do not run: a pass over [PROBE_MS] is estimated above [slowPassMs] and has not become cheap enough since. */
    @Volatile
    var passesPaused: Boolean = false
        private set

    /** The encoder window of [bufferMs] of audio, in ms (20 ms per frame). */
    private fun windowMs(bufferMs: Long): Long =
        EncoderWindow.frames((bufferMs.coerceIn(0, MAX_MEASURED_MS) * 16).toInt(), minEncoderWindow) * 20L

    /** The middle of [values]; of an even count the larger of the two (the cautious one). */
    private fun median(values: Collection<Long>): Long = values.sorted()[values.size / 2]

    private fun estimate(cost: Double, bufferMs: Long): Long = ceil(cost * windowMs(bufferMs) - 1e-9).toLong()

    /** The time a pass over [bufferMs] of audio is estimated to take, rounded up. */
    fun estimateMs(bufferMs: Long): Long = estimate(costPerWindowMs, bufferMs)

    /**
     * The time the final recognition that closes a buffer of [bufferMs] of audio is estimated to take, with the
     * language model's. Before the first final it is the passes' cost that stands in.
     */
    @Synchronized
    fun closingEstimateMs(bufferMs: Long, contextMs: Long = 0): Long {
        // [contextMs]: the audio of the piece before, which a piece after a cut inside speech is recognised with (CutSeam).
        val audioMs = bufferMs + contextMs
        // Under the lowest window a final costs what it costs at the window, as measured, not what the per-window cost
        // of a longer piece projects (the cost grows faster than the window beyond it: the projection would keep a
        // phone that once recognised a long piece away from the short ones, which are cheaper).
        val flat = if (windowMs(audioMs) == windowMs(0) && finalFlatMs.isNotEmpty()) median(finalFlatMs) else null
        return (flat ?: estimate(finalCostPerWindowMs ?: costPerWindowMs, audioMs)) + languageModelMs
    }

    /**
     * What a final recognition on the whole encoder window (the last retry of an empty final) is estimated to take, with the
     * language model's. The per-window cost is measured near the floor and grows faster than the window, so the projection is
     * raised by [FULL_WINDOW_FACTOR] (the repeat was measured at about 12 s on a phone where it projects 9 s).
     */
    @Synchronized
    fun fullWindowEstimateMs(): Long =
        ceil(estimate(finalCostPerWindowMs ?: costPerWindowMs, FULL_WINDOW_MS) * FULL_WINDOW_FACTOR - 1e-9).toLong() + languageModelMs

    /**
     * A greedy pass over [audioMs] of audio took [recogniserMs]; one that did not run to its end ([recogniserMs] <= 0)
     * is not counted. Updates the passes' cost and the pausing of passes (judged on the estimate for [PROBE_MS]).
     */
    @Synchronized
    fun recorded(audioMs: Long, recogniserMs: Long): Change {
        if (recogniserMs <= 0) return Change.NONE
        passCosts.addLast(recogniserMs.toDouble() / windowMs(audioMs))
        while (passCosts.size > COST_SAMPLES) passCosts.removeFirst()
        costPerWindowMs = passCosts.max()
        passMeasured = true
        val probe = estimateMs(PROBE_MS)
        return when {
            !passesPaused && probe > slowPassMs -> {
                passesPaused = true
                pausedProbeMs = probe
                pausedFinalCost = finalCostPerWindowMs
                Change.PAUSED
            }
            passesPaused && probe <= resumePassMs -> {
                passesPaused = false
                Change.RESUMED
            }
            else -> Change.NONE
        }
    }

    /**
     * A final recognition over [audioMs] of audio took [recogniserMs]. Decides when to cut; and, while passes are
     * paused, whether they may run again (see the class comment). Before any pass was measured it is the only
     * measurement there is and counts as a pass too.
     */
    @Synchronized
    fun recordedFinal(audioMs: Long, recogniserMs: Long): Change {
        if (recogniserMs <= 0) return Change.NONE
        if (windowMs(audioMs) == windowMs(0)) {
            finalFlatMs.addLast(recogniserMs)
            while (finalFlatMs.size > COST_SAMPLES) finalFlatMs.removeFirst()
        }
        finalCosts.addLast(recogniserMs.toDouble() / windowMs(audioMs))
        while (finalCosts.size > COST_SAMPLES) finalCosts.removeFirst()
        val cost = finalCosts.max()
        finalCostPerWindowMs = cost
        if (!passMeasured) return recorded(audioMs, recogniserMs)
        if (!passesPaused) return Change.NONE
        val atPause = pausedFinalCost
        val equivalentMs = if (atPause == null) estimate(cost, PROBE_MS) else ceil(pausedProbeMs * cost / atPause - 1e-9).toLong()
        if (equivalentMs > resumePassMs) return Change.NONE
        passesPaused = false
        return Change.RESUMED
    }

    /**
     * A repeat of an empty final on a longer window took [recogniserMs] (not counted when it did not run to its end). Kept
     * apart: its time is for another window than the audio's own, so it must not enter the final cost nor the flat cost
     * under the lowest window.
     */
    @Synchronized
    fun recordedRetry(recogniserMs: Long) {
        if (recogniserMs <= 0) return
        retryCount++
        retryRecogniserMs += recogniserMs
    }

    /** The language model took [ms] for the sentence decision of a chunk (0: there was none, not counted). */
    @Synchronized
    fun recordedLanguageModel(ms: Long) {
        if (ms <= 0) return
        languageModelTimes.addLast(ms)
        while (languageModelTimes.size > COST_SAMPLES) languageModelTimes.removeFirst()
        languageModelMs = languageModelTimes.max()
    }

    /**
     * Chunk mode (passes paused: the text comes from the final recognitions of the pieces): how long the buffer should be
     * before it is cut. The cut is sought in the last [CUT_SEARCH_MS] before the target, so a piece is anywhere between
     * the target less that window and the target; the target is the shortest length from [MIN_PIECE_MS] + [CUT_SEARCH_MS]
     * (the cut never makes a piece shorter than [MIN_PIECE_MS]) up to [maxBufferMs], in steps of [TARGET_STEP_MS], for
     * which the closing ([closingEstimateMs]) takes at most [DUTY_TARGET] of the shortest piece the search may produce,
     * `target - CUT_SEARCH_MS`: the phone is busy with the pieces at most that part of the time, whatever the cut, so it
     * never builds up a backlog and has room to warm up. Under the encoder's lowest window a recognition costs what it
     * does at the window, so a shorter piece than [MIN_PIECE_MS] saves nothing and only adds cuts, load and split words.
     * Null when the phone is slower than that for every length: then the lag limit ([mustCut]) decides alone.
     */
    @Synchronized
    fun targetPieceMs(): Long? {
        var length = MIN_PIECE_MS + CUT_SEARCH_MS
        val most = minOf(maxBufferMs, MAX_MEASURED_MS)
        while (length <= most) {
            if (closingEstimateMs(length) <= DUTY_TARGET * (length - CUT_SEARCH_MS)) return length
            length += TARGET_STEP_MS
        }
        return null
    }

    /**
     * The buffer of [bufferMs] must be cut now: it has reached [maxBufferMs], or (from [FORCED_CUT_MIN_MS]) the
     * recognition that would close it, started now, would end within [LAG_MARGIN_MS] of the allowed lag.
     */
    @Synchronized
    fun mustCut(bufferMs: Long, contextMs: Long = 0): Boolean =
        bufferMs >= maxBufferMs ||
            (bufferMs >= FORCED_CUT_MIN_MS && bufferMs + closingEstimateMs(bufferMs, contextMs) + LAG_MARGIN_MS >= maxLagMs)

    companion object {
        const val FORCED_CUT_MIN_MS = 8_000L
        const val MAX_BUFFER_MS = 12_000L
        const val LAG_MARGIN_MS = 1_000L

        /** The projection of the cost of a final over the whole encoder window is raised by this (it is measured near the window floor). */
        const val FULL_WINDOW_FACTOR = 1.3
        const val COST_SAMPLES = 3
        const val DEFAULT_COST = 0.2
        const val SLOW_PASS_MS = 2_000L
        const val RESUME_PASS_MS = 1_700L
        const val PROBE_MS = 6_000L

        /**
         * Chunk mode: no piece is cut shorter than this (the target is at least this and [CUT_SEARCH_MS] more). The final recognition of anything under the encoder's lowest
         * window (384 frames = 7.68 s on the author's phone) costs what the lowest window does (measured: 2 s of audio
         * 2.0 to 2.9 s, 7 s the same), so a shorter piece saves nothing, adds cuts and load, and a cut inside speech
         * may split a word.
         */
        const val MIN_PIECE_MS = 4_000L

        /**
         * Chunk mode: the share of its time the phone may spend on the final recognitions of the pieces, of the shortest
         * piece the cut may produce. At 60 % it
         * never builds up a backlog and keeps a reserve for a phone that warms up (it then runs 1.4 to 1.9 times
         * slower); 100 % would be the line where the queue grows.
         */
        const val DUTY_TARGET = 0.6

        /** Chunk mode: the cut is sought as the quietest frame in this long a window before the target length. */
        const val CUT_SEARCH_MS = 1_500L

        /**
         * Chunk mode: once the buffer has reached the floor of that window (the target less [CUT_SEARCH_MS], and not under
         * [MIN_PIECE_MS]), the piece is cut at once on a quiet stretch: this many frames in a row (32 ms each) below the
         * endpointer's negThreshold, 256 ms, the same pause with which stop() decides that "the word has finished"
         * ([STOP_TAIL_SILENCE_MS]); a speaker who hesitates between a preposition and its word, or one frame that dipped inside a
         * word, does not trigger a cut. The cut falls INSIDE that stretch ([Endpointer.cutAtQuietStretch]), keeping
         * [QUIET_CUT_TAIL_MS] of it in the piece. Without such a stretch the cut is made at the target as before.
         */
        const val QUIET_HOLD_FRAMES = (STOP_TAIL_SILENCE_MS + 31) / 32

        /** Chunk mode: the target length is found in steps of this. */
        const val TARGET_STEP_MS = 100L

        /** Beyond this the window is the whole one anyway; keeps the sample count in an Int. */
        private const val MAX_MEASURED_MS = 120_000L

        /** The audio that makes the whole encoder window (1500 frames of 20 ms). */
        private const val FULL_WINDOW_MS = 30_000L

        /** A guard that never cuts and never pauses (the simulation's "off" configuration). */
        fun disabled(minEncoderWindow: Int): StreamingGuard {
            val never = Long.MAX_VALUE / 4
            return StreamingGuard(never, minEncoderWindow, never, never, never)
        }
    }
}
