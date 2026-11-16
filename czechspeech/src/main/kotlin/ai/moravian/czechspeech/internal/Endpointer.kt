package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig

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

/** [STOP_TAIL_MAX_MS] in frames, rounded down: at most 1 s. */
internal const val STOP_TAIL_MAX_FRAMES = (STOP_TAIL_MAX_MS / FRAME_MS).toInt()

/** A complete utterance for the recogniser. The receiver owns [samples] and zeroes them after use. */
internal class Utterance(val samples: FloatArray, val startMs: Long, val endMs: Long)

internal sealed interface EndpointerEvent {
    /** Speech confirmed ([CzechSpeechConfig.minSpeechMs] reached); [atMs] is where it started. */
    data class SpeechStarted(val atMs: Long) : EndpointerEvent

    /** A run of speech ended at [atMs]. [confirmed] is false for runs too short to report (no [SpeechStarted]). */
    data class SpeechEnded(val atMs: Long, val confirmed: Boolean) : EndpointerEvent

    class UtteranceReady(val utterance: Utterance) : EndpointerEvent
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
        if (nFrames == maxFrames) cutAtCap()
        append(frame, prob)
        if (prob >= threshold) pieceSpeechFrames++
        if (prob >= negThreshold) {
            silenceRun = 0
            lastActiveFrame = index
        } else {
            silenceRun++
        }
        confirmIfReady()
        if (silenceRun >= minSilenceFrames) endRun()
    }

    /** End of the audio stream: an utterance in progress ends as if silence followed. */
    fun finish() {
        if (inSpeech) endRun()
    }

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
        if (confirmed && pieceSpeechFrames >= minSpeechFrames) emitPiece(nFrames, endMs)

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

    private fun cutAtCap() {
        val from = nFrames - capSearchFrames
        var lowest = from
        for (k in from until nFrames) if (probs[k] < probs[lowest]) lowest = k
        val cut = lowest + 1
        var speech = 0
        for (k in 0 until cut) if (probs[k] >= threshold) speech++
        if (runConfirmed && speech >= minSpeechFrames) emitPiece(cut, (bufferFirstFrame + cut) * FRAME_MS)

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

    private fun emitPiece(frames: Int, endMs: Long) {
        val samples = audio.copyOf(frames * FRAME_SAMPLES)
        sink(EndpointerEvent.UtteranceReady(Utterance(samples, pieceStartMs, endMs)))
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
