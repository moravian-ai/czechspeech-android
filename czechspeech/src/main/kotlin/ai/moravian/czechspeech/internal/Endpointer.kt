package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import java.util.concurrent.atomic.AtomicBoolean

internal const val SAMPLE_RATE = 16_000

/** Silero VAD v6 window at 16 kHz (n_window in the ggml model header): one probability per 512 samples. */
internal const val FRAME_SAMPLES = 512
internal const val FRAME_MS = 32L

/** Search window for the cut when an utterance reaches [CzechSpeechConfig.maxUtteranceMs]. */
internal const val CAP_SEARCH_MS = 2_000

/**
 * stop() with speech in progress lets the word in progress finish: capture goes on until a pause of this long (frames
 * below negThreshold; a pause already under way counts) ...
 */
internal const val STOP_TAIL_SILENCE_MS = 256

/** ... or until this much audio after stop(), whichever comes first. An abrupt cut makes Whisper invent words. */
internal const val STOP_TAIL_MAX_MS = 1_000

/**
 * A quiet-point cut ([Endpointer.cutAtQuietStretch]) keeps this much of the quiet stretch in the piece: the tails of words that score
 * below negThreshold. The rest of the stretch goes to the remainder. 128 ms, between 96 and 160.
 */
internal const val QUIET_CUT_TAIL_MS = 128

/** [QUIET_CUT_TAIL_MS] in frames. */
internal const val QUIET_CUT_TAIL_FRAMES = (QUIET_CUT_TAIL_MS / FRAME_MS).toInt()

/** [STOP_TAIL_MAX_MS] in frames, rounded down: at most 1 s. */
internal const val STOP_TAIL_MAX_FRAMES = (STOP_TAIL_MAX_MS / FRAME_MS).toInt()

/**
 * A complete utterance for the recogniser. The receiver owns [samples] and zeroes them after use. With [early], it was
 * handed out before its end was certain ([EndpointerEvent.UtteranceEarly]) and [early] tells how that turned out.
 */
internal class Utterance(
    val samples: FloatArray,
    val startMs: Long,
    val endMs: Long,
    val early: EarlyRun? = null,
    /** Session-clock ms of [samples]`[0]`: the start of the pre-roll or of the rest after a cut. */
    val audioStartMs: Long = startMs,
    /**
     * A piece cut inside speech (the length cap, [Endpointer.cutNow]) at a frame below the endpointer's negThreshold: the cut
     * fell on a pause between sentences that was shorter than minSilenceMs, not on a word or a gap inside a phrase.
     */
    val quietCut: Boolean = false,
    /** How much of the audio the endpointer scored as speech (frames at or above the threshold), in ms: clear speech is a second of it. */
    val speechMs: Long = 0,
) {
    /** Counted in the audio waiting in an [UtteranceQueue]. */
    val queued = AtomicBoolean(false)
}

internal sealed interface EndpointerEvent {
    /** Speech confirmed ([CzechSpeechConfig.minSpeechMs] reached); [atMs] is where it started. */
    data class SpeechStarted(val atMs: Long) : EndpointerEvent

    /** A run of speech ended at [atMs]. [confirmed] is false for runs too short to report (no [SpeechStarted]). */
    data class SpeechEnded(val atMs: Long, val confirmed: Boolean) : EndpointerEvent

    class UtteranceReady(val utterance: Utterance) : EndpointerEvent

    /**
     * Only with [CzechSpeechConfig.earlyStart]: the pause in progress has lasted [ai.moravian.czechspeech.EarlyStart.silenceMs]
     * and [utterance] is what the recogniser gets if the speech ends here: the audio so far and zeros up to
     * [CzechSpeechConfig.minSilenceMs] of silence, since the rest of the pause has not been recorded yet. The receiver
     * owns it from now on. Exactly one of two events follows: [UtteranceReady] with this very utterance (the speech
     * ended here), or [EarlyRetracted] (it went on, or the length cap cut it first).
     */
    class UtteranceEarly(val utterance: Utterance) : EndpointerEvent

    /** The [utterance] of an [UtteranceEarly] is not an utterance after all; its receiver drops and zeroes it. */
    class EarlyRetracted(val utterance: Utterance) : EndpointerEvent
}

/**
 * Turns per-frame speech probabilities into utterances. Pure Kotlin, one thread, no allocation per frame.
 *
 * - Speech starts at the first frame with probability >= threshold and stays on while frames are >= negThreshold.
 * - It ends after minSilenceMs below negThreshold. The utterance audio runs from preRollMs before the start to the
 *   end of that silence; startMs/endMs report the speech itself (trigger frame to last frame >= negThreshold).
 * - A run with fewer than minSpeechMs of frames >= threshold is discarded silently.
 * - At maxUtteranceMs of audio the utterance is cut after the frame with the lowest probability in the last
 *   [CAP_SEARCH_MS] and the rest continues as the next utterance of the same run.
 *
 * - With [CzechSpeechConfig.earlyStart], the utterance is handed out once already when the pause that may end it has
 *   lasted [ai.moravian.czechspeech.EarlyStart.silenceMs] ([EndpointerEvent.UtteranceEarly]), and an utterance ended by
 *   silence is that very one: its audio after that point is zeros, not the microphone's. What an utterance holds
 *   depends on the audio alone, never on when anybody took it.
 *
 * Every audio buffer is zeroed once it has been copied out or discarded. Emitted [Utterance.samples] are copies
 * owned by the receiver.
 */
internal class Endpointer(config: CzechSpeechConfig, private val sink: (EndpointerEvent) -> Unit) {
    private val threshold = config.threshold
    private val negThreshold = config.negThreshold
    private val preRollFrames = ceilDiv(config.preRollMs.toLong(), FRAME_MS)
    private val minSilenceFrames = maxOf(1, ceilDiv(config.minSilenceMs.toLong(), FRAME_MS))
    private val minSpeechFrames = ceilDiv(config.minSpeechMs.toLong(), FRAME_MS)
    private val maxFrames = (config.maxUtteranceMs / FRAME_MS).toInt()
    private val capSearchFrames = minOf(maxFrames, ceilDiv(CAP_SEARCH_MS.toLong(), FRAME_MS))
    private val stopTailSilenceFrames = ceilDiv(STOP_TAIL_SILENCE_MS.toLong(), FRAME_MS)

    /** Frames of silence after which the utterance is handed out early; 0 = never (off, or not below minSilenceFrames). */
    private val earlyFrames =
        (config.earlyStart?.let { ceilDiv(it.silenceMs.toLong(), FRAME_MS) } ?: 0).let { if (it < minSilenceFrames) it else 0 }

    /** The utterance handed out early in the pause in progress, neither confirmed nor retracted yet. */
    private var early: Utterance? = null

    // Pre-roll ring buffer, filled while there is no speech.
    private val ring = Array(preRollFrames) { FloatArray(FRAME_SAMPLES) }
    private val ringProbs = FloatArray(preRollFrames)
    private var ringHead = 0
    private var ringSize = 0

    // Current utterance: pre-roll + speech + trailing silence, at most maxFrames.
    private val audio = FloatArray(maxFrames * FRAME_SAMPLES)
    private val probs = FloatArray(maxFrames)
    private var nFrames = 0
    private var bufferFirstFrame = 0L

    private var frameIndex = 0L
    private var inSpeech = false
    private var runStartFrame = 0L
    private var runConfirmed = false
    private var pieceStartMs = 0L
    private var pieceSpeechFrames = 0
    private var silenceRun = 0
    private var lastActiveFrame = 0L

    /** Feeds one frame of [FRAME_SAMPLES] samples and its speech probability. The frame is copied. */
    fun accept(frame: FloatArray, prob: Float) {
        require(frame.size == FRAME_SAMPLES) { "frame must have $FRAME_SAMPLES samples" }
        val index = frameIndex++
        if (!inSpeech) {
            if (prob >= threshold) startRun(index, frame, prob) else pushRing(frame, prob)
            return
        }
        if (nFrames == maxFrames) cut()
        append(frame, prob)
        if (prob >= threshold) pieceSpeechFrames++
        if (prob >= negThreshold) {
            silenceRun = 0
            lastActiveFrame = index
            retractEarly()              // the speech goes on
        } else {
            silenceRun++
        }
        confirmIfReady()
        if (silenceRun >= minSilenceFrames) endRun() else if (earlyFrames > 0 && silenceRun == earlyFrames) offerEarly()
    }

    /**
     * End of the audio stream: an utterance in progress ends as if silence followed. One handed out early in the pause
     * in progress is the utterance, zeros for the silence that never came included.
     */
    fun finish() {
        if (inSpeech) endRun()
    }

    /** Audio fed so far, in ms: the clock of [Utterance.startMs]. */
    val positionMs: Long get() = frameIndex * FRAME_MS

    /**
     * The start of the piece in progress, which will be its [Utterance.startMs], or -1 without confirmed speech in
     * progress. A piece ends at the end of its speech or at the length cap, where the next piece starts.
     */
    val pieceMs: Long get() = if (inSpeech && runConfirmed) pieceStartMs else -1

    /** Where the length cap cuts the piece in progress at the latest, on the clock of [positionMs]. */
    val capMs: Long get() = (bufferFirstFrame + maxFrames) * FRAME_MS

    /**
     * How many frames in a row at the end of the buffer are below negThreshold while speech is in progress (0 outside
     * speech, and once minSilenceMs is reached, which ends the utterance): the criterion of a quiet cut, which falls on a
     * frame below negThreshold ([Utterance.quietCut]).
     */
    val quietFrames: Int get() = if (inSpeech) silenceRun else 0

    /** The last frame was speech (at or above negThreshold), not a pause that may turn out to be the end. */
    val speaking: Boolean get() = inSpeech && silenceRun == 0

    /** A copy of the piece's audio so far (pre-roll included), for a preview; the receiver zeroes it. */
    fun snapshot(): FloatArray = audio.copyOf(nFrames * FRAME_SAMPLES)

    /** Session-clock ms of the first sample of the piece's audio (its pre-roll, or where the last cut left off). */
    val audioStartMs: Long get() = bufferFirstFrame * FRAME_MS

    /**
     * A copy of the piece's audio from [fromMs] (session clock, at sample precision, but not before [audioStartMs])
     * to its end; empty when [fromMs] is at or past the end. The receiver zeroes it.
     */
    fun snapshotFrom(fromMs: Long): FloatArray {
        val skip = (maxOf(fromMs, audioStartMs) - audioStartMs) * (SAMPLE_RATE / 1000)
        val total = nFrames.toLong() * FRAME_SAMPLES
        if (skip >= total) return FloatArray(0)
        return audio.copyOfRange(skip.toInt(), total.toInt())
    }

    /**
     * Capture thread only: ends the piece in progress now, exactly as the length cap does (after the quietest frame of
     * the last [CAP_SEARCH_MS], the rest continuing as the next piece), when speech is in progress and more than that
     * window is buffered. Streaming's lag cut. Returns whether it cut.
     */
    fun cutNow(): Boolean {
        if (!inSpeech || nFrames <= capSearchFrames) return false
        cut()
        return true
    }

    /**
     * Capture thread only: asked inside every cut (the length cap's and [cutNow]'s) for the earliest session ms the cut
     * may fall on; the cut is then after the quietest frame of the window that ends no earlier than that (the last
     * frame of the buffer when it is later still). 0 or null: anywhere in the last [CAP_SEARCH_MS]. Streaming sets it
     * so that no cut falls before the audio whose words were already written.
     */
    var cutNotBefore: (() -> Long)? = null

    /** No speech in progress, or it has paused for [STOP_TAIL_SILENCE_MS]: stop() need not wait for the word to end. */
    val atPause: Boolean get() = !inSpeech || silenceRun >= stopTailSilenceFrames

    /** Zeroes and drops all buffered audio (cancellation). */
    fun clear() {
        for (frame in ring) frame.fill(0f)
        ringProbs.fill(0f)
        ringHead = 0
        ringSize = 0
        audio.fill(0f)
        probs.fill(0f)
        nFrames = 0
        inSpeech = false
        runConfirmed = false
        pieceSpeechFrames = 0
        silenceRun = 0
        early = null                // its receiver owns it
    }

    /**
     * The pause has lasted [earlyFrames]: hands out the utterance this piece becomes if the speech ends here, under the
     * conditions of [endRun]. Not when the length cap would cut the piece before the pause is long enough.
     */
    private fun offerEarly() {
        if (!runConfirmed || pieceSpeechFrames < minSpeechFrames) return
        val frames = nFrames + minSilenceFrames - silenceRun
        if (frames > maxFrames) return
        val samples = FloatArray(frames * FRAME_SAMPLES)        // zeros for the silence still to come
        System.arraycopy(audio, 0, samples, 0, nFrames * FRAME_SAMPLES)
        val utterance = Utterance(samples, pieceStartMs, (lastActiveFrame + 1) * FRAME_MS, EarlyRun(), audioStartMs, speechMs = pieceSpeechFrames * FRAME_MS)
        early = utterance
        sink(EndpointerEvent.UtteranceEarly(utterance))
    }

    private fun retractEarly() {
        val utterance = early ?: return
        early = null
        sink(EndpointerEvent.EarlyRetracted(utterance))
    }

    private fun startRun(index: Long, frame: FloatArray, prob: Float) {
        inSpeech = true
        nFrames = 0
        bufferFirstFrame = index - ringSize
        for (k in 0 until ringSize) {
            val slot = (ringHead + k) % preRollFrames
            append(ring[slot], ringProbs[slot])
            ring[slot].fill(0f)
            ringProbs[slot] = 0f
        }
        ringHead = 0
        ringSize = 0
        append(frame, prob)
        runStartFrame = index
        runConfirmed = false
        pieceStartMs = index * FRAME_MS
        pieceSpeechFrames = 1
        silenceRun = 0
        lastActiveFrame = index
        confirmIfReady()
    }

    private fun confirmIfReady() {
        if (!runConfirmed && pieceSpeechFrames >= minSpeechFrames) {
            runConfirmed = true
            sink(EndpointerEvent.SpeechStarted(runStartFrame * FRAME_MS))
        }
    }

    private fun endRun() {
        val endMs = (lastActiveFrame + 1) * FRAME_MS
        val confirmed = runConfirmed
        sink(EndpointerEvent.SpeechEnded(endMs, confirmed))
        val handedOut = early
        early = null
        if (handedOut != null) {
            sink(EndpointerEvent.UtteranceReady(handedOut))     // offerEarly checked the conditions
        } else if (confirmed && pieceSpeechFrames >= minSpeechFrames) {
            emitPiece(nFrames, endMs)
        }

        // The trailing silence becomes the pre-roll of a quick next utterance.
        val keep = minOf(preRollFrames, nFrames)
        for (k in 0 until keep) {
            val src = nFrames - keep + k
            System.arraycopy(audio, src * FRAME_SAMPLES, ring[k], 0, FRAME_SAMPLES)
            ringProbs[k] = probs[src]
        }
        ringHead = 0
        ringSize = keep

        audio.fill(0f, 0, nFrames * FRAME_SAMPLES)
        probs.fill(0f, 0, nFrames)
        nFrames = 0
        inSpeech = false
        runConfirmed = false
        pieceSpeechFrames = 0
        silenceRun = 0
    }

    private fun cut() {
        retractEarly()              // defensive: offerEarly never hands out a piece a cut would split
        val floorMs = cutNotBefore?.invoke() ?: 0L
        var from = nFrames - capSearchFrames
        if (floorMs > 0) from = maxOf(from, (ceilDiv(floorMs, FRAME_MS) - bufferFirstFrame - 1).coerceIn(0, nFrames - 1L).toInt())
        var lowest = from
        for (k in from until nFrames) if (probs[k] < probs[lowest]) lowest = k
        cutAt(lowest + 1)
    }

    /**
     * Capture thread only: the cut of a quiet-point cut (streaming, chunk mode): inside the quiet stretch at the end of the buffer
     * ([quietFrames]: frames below negThreshold), never anywhere else. The piece keeps [QUIET_CUT_TAIL_MS] of that stretch (the
     * tails of words that score below negThreshold) and the rest of it goes to the next piece, so no frame of speech is ever left
     * in the remainder and no earlier dip of the window takes the cut. Never before the earliest cut ([cutNotBefore], as the
     * quietest-frame cut). Returns whether it cut: not without a quiet stretch or with too little audio buffered.
     */
    fun cutAtQuietStretch(): Boolean {
        if (!inSpeech || silenceRun < 1 || nFrames <= capSearchFrames) return false
        retractEarly()
        val floorMs = cutNotBefore?.invoke() ?: 0L
        val start = nFrames - silenceRun
        var cut = start + QUIET_CUT_TAIL_FRAMES
        if (floorMs > 0) cut = maxOf(cut, (ceilDiv(floorMs, FRAME_MS) - bufferFirstFrame).coerceIn(0, nFrames.toLong()).toInt())
        cut = cut.coerceIn(start + 1, nFrames)
        cutAt(cut)
        return true
    }

    /** Cuts after [cut] frames of the buffer: emits the piece up to there (when it has speech) and keeps the rest as the start of the next. */
    private fun cutAt(cut: Int) {
        val quiet = probs[cut - 1] < negThreshold
        var speech = 0
        for (k in 0 until cut) if (probs[k] >= threshold) speech++
        if (runConfirmed && speech >= minSpeechFrames) emitPiece(cut, (bufferFirstFrame + cut) * FRAME_MS, quiet)

        val rest = nFrames - cut
        System.arraycopy(audio, cut * FRAME_SAMPLES, audio, 0, rest * FRAME_SAMPLES)
        audio.fill(0f, rest * FRAME_SAMPLES, nFrames * FRAME_SAMPLES)
        System.arraycopy(probs, cut, probs, 0, rest)
        probs.fill(0f, rest, nFrames)
        nFrames = rest
        bufferFirstFrame += cut
        pieceStartMs = bufferFirstFrame * FRAME_MS
        pieceSpeechFrames = 0
        for (k in 0 until rest) if (probs[k] >= threshold) pieceSpeechFrames++
        if (!runConfirmed) runStartFrame = bufferFirstFrame
    }

    private fun emitPiece(frames: Int, endMs: Long, quietCut: Boolean = false) {
        val samples = audio.copyOf(frames * FRAME_SAMPLES)
        var speech = 0
        for (k in 0 until frames) if (probs[k] >= threshold) speech++
        sink(EndpointerEvent.UtteranceReady(Utterance(samples, pieceStartMs, endMs, audioStartMs = audioStartMs, quietCut = quietCut, speechMs = speech * FRAME_MS)))
    }

    private fun append(frame: FloatArray, prob: Float) {
        System.arraycopy(frame, 0, audio, nFrames * FRAME_SAMPLES, FRAME_SAMPLES)
        probs[nFrames] = prob
        nFrames++
    }

    private fun pushRing(frame: FloatArray, prob: Float) {
        if (preRollFrames == 0) return
        val slot: Int
        if (ringSize < preRollFrames) {
            slot = (ringHead + ringSize) % preRollFrames
            ringSize++
        } else {
            slot = ringHead
            ringHead = (ringHead + 1) % preRollFrames
        }
        System.arraycopy(frame, 0, ring[slot], 0, FRAME_SAMPLES)
        ringProbs[slot] = prob
    }

    private fun ceilDiv(a: Long, b: Long): Int = ((a + b - 1) / b).toInt()
}
