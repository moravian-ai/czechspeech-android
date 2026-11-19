package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SpeechPreview
import kotlinx.coroutines.channels.Channel

/**
 * The rules of previews in a [SpeechEngine.listenWithPreviews] session: when the utterance in progress (a piece, in
 * [Endpointer]'s words) is previewed, and which results may still reach the app. No threads or native code of its
 * own; three parties call it, ordered by its lock:
 * - the capture thread: every frame ([onFrame]), the end of speech ([pieceEnded]), each utterance going to the final
 *   recognition ([finalComing], [finalNotComing]; [finalStartingEarly] when it starts before the end is certain),
 *   stop() ([stop]) and the end of capture ([end]);
 * - the preview runner, which takes [requests], recognises them on the whisper thread and reports [finished];
 * - the recognition loop, after each final recognition ([finalDone]).
 *
 * Rules:
 * - a piece is previewed once it has lasted [PreviewConfig.firstAfterMs], then at the earliest
 *   [PreviewConfig.intervalMs] of audio after its previous preview started, from a copy of its audio so far; not in
 *   a pause, which may be its end, nor when the preview would still run at the length cap (see [onFrame]);
 * - finals first: no preview starts while an utterance waits for or is in its final recognition, nor while another
 *   preview runs, nor after stop(); the end of a piece, and stop(), cancel its preview (an abort reaches whisper.cpp
 *   when it is running);
 * - a result reaches the app ([emit], called under the lock) only while its piece is in progress, so never after the
 *   end of its speech or after its utterance has been handed to the final recognition, which sends its Transcript;
 * - a piece that was previewed and gives no Transcript (no words, a failed or skipped recognition, an utterance the
 *   endpointer drops) gets a preview with empty text, which withdraws it;
 * - of each preview, the start that the piece's previous preview agreed on is marked stable ([PreviewAgreement]),
 *   whether the two came by the clock or from a pause.
 */
internal class PreviewTracker(
    config: PreviewConfig,
    /** The shortest encoder window, in ms of audio (the model's floor): what the shortest recognition costs. */
    private val minWindowMs: Long = EncoderWindow.DEFAULT_MIN_FRAMES * 20L,
    /** Sends a preview to the app; called under the lock, so it must not block (trySend). */
    private val emit: (SpeechPreview) -> Unit,
) {
    private val firstAfterMs = config.firstAfterMs.toLong()
    private val intervalMs = config.intervalMs.toLong()

    /** [PreviewConfig.fromPauses]: the engine keeps early recognitions the speech ran over, for [fromPause]. */
    val fromPauses: Boolean = config.fromPauses

    /** Preview runs for the runner, at most one at a time. Audio a run cannot deliver is zeroed. */
    val requests: Channel<PreviewRun> = Channel(Channel.UNLIMITED, onUndeliveredElement = { it.samples.fill(0f) })

    /** A piece being previewed: its start (the utterance's startMs) and what has been shown of it. */
    private class Piece(val startMs: Long) {
        /** Speech (ms into the piece) when its last preview started; -1 before the first. */
        var lastAtMs = -1L
        var previewed = false
        var handedOver = false

        /** The text of its last preview, for the stable part of the next ([PreviewAgreement]); null before the first. */
        var lastText: String? = null
    }

    private var piece: Piece? = null

    /** A piece that ended during the last frame; at the next one, its fate is known (a final recognition or none). */
    private var closed: Piece? = null

    /** Previewed utterances in or waiting for their final recognition (startMs): withdrawn if they give no text. */
    private val awaitingFinal = HashSet<Long>()
    private var inFlight: PreviewRun? = null
    private var finalsPending = 0

    /** Recognitions started early in a pause that wait for the recogniser or run in it ([earlyStarted]). */
    private var earlyRuns = 0
    private var ended = false
    private var stopped = false

    /** Recogniser time per ms of encoder window, from the last preview that ran to its end; 0 before the first. */
    private var costPerWindowMs = 0.0

    /**
     * Capture thread, after each frame: [pieceMs] is the piece in progress ([Endpointer.pieceMs], -1 for none),
     * [positionMs] the audio so far, [capMs] where the length cap cuts the piece at the latest ([Endpointer.capMs]),
     * [speaking] whether the frame was speech. Returns the preview it requested, if any, with [snapshot]'s audio.
     *
     * Two more rules keep previews out of the way of the final text: none starts in a pause, which may be the end of
     * the utterance, nor when it would still run at the cap, judged by the speed of the last preview; either would
     * only be aborted after delaying the final recognition.
     */
    @Synchronized
    fun onFrame(
        pieceMs: Long,
        positionMs: Long,
        capMs: Long,
        speaking: Boolean,
        snapshot: () -> FloatArray,
    ): PreviewRun? {
        if (ended) return null
        settle()
        // A piece that changed without an utterance for it was dropped (too little speech before the length cap).
        if (piece != null && piece?.startMs != pieceMs) {
            close()
            settle()
        }
        if (pieceMs < 0) return null
        val p = piece ?: Piece(pieceMs).also { piece = it }
        if (stopped || !speaking || inFlight != null || finalsPending > 0 || earlyRuns > 0) return null
        val speechMs = positionMs - p.startMs
        val dueAtMs = if (p.lastAtMs < 0) firstAfterMs else p.lastAtMs + intervalMs
        if (speechMs < dueAtMs) return null
        if (positionMs + costPerWindowMs * windowMs(speechMs, minWindowMs) > capMs) return null
        p.lastAtMs = speechMs
        val run = PreviewRun(p.startMs, positionMs, snapshot())
        if (!requests.trySend(run).isSuccess) {
            // The runner is gone (the session is ending): no more previews.
            run.samples.fill(0f)
            ended = true
            return null
        }
        inFlight = run
        return run
    }

    /**
     * Capture thread: stop() was called. Its utterance ends within 1 s and its final text comes right after, so no
     * preview starts any more, and one that runs is cancelled now rather than when the utterance ends, so that nothing
     * holds up that final text. Previews shown so far are replaced or withdrawn as usual.
     */
    @Synchronized
    fun stop() {
        stopped = true
        inFlight?.cancel()
    }

    /**
     * Session teardown, any thread: runs still waiting in [requests] (the runner may never have started, or stopped
     * before taking them) are zeroed, and the channel is closed.
     */
    fun discardRequests() {
        requests.close()
        while (true) {
            val run = requests.tryReceive().getOrNull() ?: break
            run.samples.fill(0f)
        }
    }

    /**
     * Capture thread, with [ai.moravian.czechspeech.EarlyStart]: the pause in progress may be the end of the utterance,
     * and its final recognition starts now. Final text first: a timed preview that runs (or waits to) is cancelled, so
     * the recogniser is free for it; a preview would be dropped at the end of the utterance anyway. (The engine stops
     * a recognition of an earlier pause kept for a preview at the same moment.) When the speech goes on
     * instead, the piece is still in progress and its next preview comes when it is due.
     */
    @Synchronized
    fun finalStartingEarly() {
        inFlight?.cancel()
    }

    /**
     * Capture thread: the recognition of [utterance], handed out early, goes to the recogniser's queue. Until it has
     * ended, or the queue did not take it ([earlyEnded]), no timed preview starts: it is the final text, or else a preview itself ([fromPause]), and
     * then the piece's timed previews count from this pause.
     */
    @Synchronized
    fun earlyStarted(utterance: Utterance) {
        earlyRuns++
        val p = piece ?: return
        if (fromPauses && p.startMs == utterance.startMs) p.lastAtMs = maxOf(p.lastAtMs, utterance.endMs - p.startMs)
    }

    /** Recognition loop (or capture thread): a recognition counted by [earlyStarted] has ended, was aborted, or never ran. */
    @Synchronized
    fun earlyEnded() {
        earlyRuns--
    }

    /**
     * Recognition loop: [text] is what a recognition started in a pause made of the piece that started at [startMs],
     * and the speech went on. It is a preview like any other, while that piece is in progress. True when it was sent.
     */
    @Synchronized
    fun fromPause(startMs: Long, text: String): Boolean {
        val p = piece ?: return false
        if (ended || stopped || p.startMs != startMs || text.isEmpty()) return false
        show(p, text)
        return true
    }

    /** Capture thread: the speech in progress has ended (before its SpeechEnded is sent). Its previews stop here. */
    @Synchronized
    fun pieceEnded() {
        close()
    }

    /** Capture thread: [utterance] is about to be offered to the final recognition (it ends its piece). */
    @Synchronized
    fun finalComing(utterance: Utterance) {
        if (piece?.startMs == utterance.startMs) close()
        val c = closed
        if (c != null && c.startMs == utterance.startMs) {
            c.handedOver = true
            if (c.previewed) awaitingFinal += c.startMs
        }
        finalsPending++
    }

    /** Capture thread: the offer of [utterance] failed (a full queue: skipped; closed: the session is ending). */
    @Synchronized
    fun finalNotComing(utterance: Utterance) {
        finalsPending--
        if (awaitingFinal.remove(utterance.startMs)) withdraw(utterance.startMs)
    }

    /** Recognition loop: the final recognition of [utterance] has ended; [transcribed]: its Transcript was sent. */
    @Synchronized
    fun finalDone(utterance: Utterance, transcribed: Boolean) {
        finalsPending--
        if (awaitingFinal.remove(utterance.startMs) && !transcribed) withdraw(utterance.startMs)
    }

    /**
     * Runner: [run] has ended with [text] ("" for nothing: no words, cancelled or failed), after [recogniserMs] of
     * recognition (0 when it did not run to its end).
     */
    @Synchronized
    fun finished(run: PreviewRun, text: String, recogniserMs: Long = 0) {
        if (inFlight === run) inFlight = null
        if (recogniserMs > 0) costPerWindowMs = recogniserMs.toDouble() / windowMs(run.endMs - run.startMs, minWindowMs)
        val p = piece ?: return
        if (ended || run.isCancelled || p.startMs != run.startMs || text.isEmpty()) return
        show(p, text)
    }

    /** Sends [text] as the preview of [p], with the part the previous preview of the piece agreed on as stable. */
    private fun show(p: Piece, text: String) {
        val stable = p.lastText?.let { PreviewAgreement.stableLength(it, text) } ?: 0
        p.lastText = text
        emit(SpeechPreview(p.startMs, text, stable))
        p.previewed = true
    }

    /** Capture thread: capture has ended. The piece in progress ends, and nothing more is requested. */
    @Synchronized
    fun end() {
        if (ended) return
        close()
        settle()
        ended = true
        requests.close()
    }

    /** Ends the piece in progress: no result of it reaches the app any more, and its preview stops. */
    private fun close() {
        val p = piece ?: return
        piece = null
        inFlight?.takeIf { it.startMs == p.startMs }?.cancel()
        settle()
        closed = p
    }

    /** Decides about the piece that ended: without an utterance for the final recognition, its preview goes. */
    private fun settle() {
        val c = closed ?: return
        closed = null
        if (c.previewed && !c.handedOver) withdraw(c.startMs)
    }

    private fun withdraw(startMs: Long) = emit(SpeechPreview(startMs, ""))

    private companion object {
        /** Audio the encoder works on for [audioMs] of speech: what a recognition costs grows with it. */
        fun windowMs(audioMs: Long, minWindowMs: Long): Long = maxOf(audioMs, minWindowMs)
    }
}

/**
 * One preview: a copy of the audio of the piece that started at [startMs], up to [endMs], which whoever ends the run
 * zeroes, and the abort of its recognition.
 */
internal class PreviewRun(val startMs: Long, val endMs: Long, val samples: FloatArray) {
    private var token: AbortToken? = null
    private var cancelled = false
    private var cancelledNs = 0L

    /**
     * Since the abort reached the running recogniser, in ms (for the log: how long it took to stop); -1 when the run
     * was not cancelled, or cancelled while nothing native ran.
     */
    @Synchronized
    fun msSinceCancel(nowNs: Long): Long = if (cancelledNs == 0L) -1 else maxOf(0L, nowNs - cancelledNs) / 1_000_000

    val isCancelled: Boolean
        @Synchronized get() = cancelled

    /** Whisper thread, before the recogniser runs: false once the run is cancelled, which then never starts. */
    @Synchronized
    fun attach(token: AbortToken): Boolean {
        if (cancelled) return false
        this.token = token
        return true
    }

    /** Any thread: a run not started never starts; a running one stops at whisper.cpp's next check. */
    @Synchronized
    fun cancel() {
        cancelled = true
        // timed only when the abort reached whisper.cpp at work
        if (token?.cancel() == true && cancelledNs == 0L) cancelledNs = System.nanoTime()
    }
}
