package ai.moravian.czechspeech.internal

import kotlinx.coroutines.CompletableDeferred

/**
 * The fate of an utterance handed out before its end was certain ([EndpointerEvent.UtteranceEarly]): its recognition
 * may start at once, and the capture thread later decides what it was:
 * - [confirm]: the speech ended there, and the recognition is the utterance's final one ([Outcome.USED]);
 * - [retract]: the speech went on; the recognition is aborted and its result dropped ([Outcome.DROPPED]);
 * - [demote]: the speech went on, in a session with previews; the recognition runs on and its text is a preview of
 *   the utterance in progress ([Outcome.PREVIEW]), until [stop] aborts it because that utterance has ended and its
 *   final recognition comes first.
 *
 * Three parties, ordered by its lock:
 * - the capture thread decides, once, and may [stop] a demoted run;
 * - the whisper thread runs the recognition between [begin] and [end];
 * - the recognition loop waits for the decision ([awaitOutcome]) before anything of the result reaches the app.
 *
 * Times are System.nanoTime values, passed in.
 */
internal class EarlyRun {
    enum class Outcome { USED, DROPPED, PREVIEW }

    private enum class State { PENDING, CONFIRMED, RETRACTED, DEMOTED }

    private var state = State.PENDING
    private var token: AbortToken? = null
    private var began = false
    private var ended = false
    private var stopped = false
    private var abortSent = false
    private var startedNs = 0L
    private var endedNs = 0L
    private var decidedNs = 0L
    private var abortNs = 0L
    private var noted = false
    private val decision = CompletableDeferred<Outcome>()

    val isConfirmed: Boolean
        @Synchronized get() = state == State.CONFIRMED

    /** Never an utterance and no preview any more: retracted, or demoted and stopped. Its recognition never begins. */
    val isDead: Boolean
        @Synchronized get() = state == State.RETRACTED || stopped

    /** A demoted run was stopped before it ended: it gives no preview. */
    val wasStopped: Boolean
        @Synchronized get() = stopped

    /** Nothing of it runs or will run: its recognition has ended, or never begins. */
    val isOver: Boolean
        @Synchronized get() = ended || (!began && (state == State.RETRACTED || stopped))

    /** A [PreviewTracker] counts it as waiting or running ([PreviewTracker.earlyStarted]); capture thread, then loop. */
    @Volatile
    var tracked = false

    /**
     * Confirmed, and the recognition had begun before the confirmation: it ran ahead of the end of the utterance. False
     * for one that began only afterwards (the recogniser was busy, or the queue did not take the utterance early):
     * nothing was gained.
     */
    val hadHeadStart: Boolean
        @Synchronized get() = state == State.CONFIRMED && began && startedNs < decidedNs

    /** The recognition was started ([begin] returned true). */
    val hasBegun: Boolean
        @Synchronized get() = began

    /**
     * Whisper thread, before anything reads the utterance's audio: false once retracted or stopped, and then the
     * recognition never starts. [token] is what [retract] and [stop] abort.
     */
    @Synchronized
    fun begin(token: AbortToken, nowNs: Long): Boolean {
        if (state == State.RETRACTED || stopped) return false
        this.token = token
        began = true
        startedNs = nowNs
        return true
    }

    /** Whisper thread: the recognition has ended (or never began) and nothing reads the audio any more. */
    @Synchronized
    fun end(nowNs: Long) {
        token = null
        if (began && !ended) {
            ended = true
            endedNs = nowNs
        }
    }

    /** Capture thread: the speech ended where the utterance was handed out. */
    @Synchronized
    fun confirm(nowNs: Long) {
        if (state != State.PENDING) return
        state = State.CONFIRMED
        decidedNs = nowNs
        decision.complete(Outcome.USED)
    }

    /**
     * Capture thread (or the end of capture): the utterance is dropped. A running recognition is aborted. True when
     * the recognition never began, so nobody else will touch the audio and the caller zeroes it; otherwise the whisper
     * thread zeroes it when the recognition ends.
     */
    @Synchronized
    fun retract(nowNs: Long): Boolean {
        if (state != State.PENDING) return false
        state = State.RETRACTED
        decidedNs = nowNs
        abort(nowNs)
        decision.complete(Outcome.DROPPED)
        return !began
    }

    /**
     * Capture thread: the speech went on, and the recognition's text is wanted as a preview. Nothing is aborted: a
     * recognition that runs, or still waits for the recogniser, goes on until its end or [stop].
     */
    @Synchronized
    fun demote(nowNs: Long) {
        if (state != State.PENDING) return
        state = State.DEMOTED
        decidedNs = nowNs
        decision.complete(Outcome.PREVIEW)
    }

    /**
     * Capture thread: a demoted run gives way, because the utterance in progress has ended (or capture has). A running
     * recognition is aborted; one that has ended keeps its result. True as from [retract]: the caller zeroes the audio.
     */
    @Synchronized
    fun stop(nowNs: Long): Boolean {
        if (state != State.DEMOTED || ended || stopped) return false
        stopped = true
        abort(nowNs)
        return !began
    }

    private fun abort(nowNs: Long) {
        val running = token ?: return
        // Counted, and timed, only when it reached whisper.cpp at work: after the native call has returned (the
        // language model, the filters) there is nothing to stop.
        if (running.cancel()) {
            abortSent = true
            abortNs = nowNs
        }
    }

    /** Whisper thread: the abort came, but whisper.cpp returned its result all the same: it stopped nothing. */
    @Synchronized
    fun abortStoppedNothing() {
        abortSent = false
    }

    /** Suspends until the capture thread has decided. */
    suspend fun awaitOutcome(): Outcome = decision.await()

    /**
     * How much sooner the result was ready than with a recognition started at the confirmation, in ms: the part of the
     * recognition that ran before it. 0 when the recogniser was busy until then, the whole recognition when it had
     * ended by then. [nowNs] stands for the end of a recognition still running.
     */
    @Synchronized
    fun savedMs(nowNs: Long): Long {
        if (state != State.CONFIRMED || !began) return 0
        val end = if (ended) endedNs else nowNs
        return maxOf(0L, minOf(decidedNs, end) - startedNs) / 1_000_000
    }

    /** Recogniser time spent on a recognition, in ms; [nowNs] stands for the end of one still running. */
    @Synchronized
    fun recogniserMs(nowNs: Long): Long = if (!began) 0 else maxOf(0L, (if (ended) endedNs else nowNs) - startedNs) / 1_000_000

    /**
     * How long a running recognition took to stop after its abort, in ms: what the abort can delay the final text
     * behind it. -1 when no abort reached whisper.cpp at work, or it stopped nothing ([abortStoppedNothing]). [nowNs] stands for the end of one still running.
     */
    @Synchronized
    fun abortMs(nowNs: Long): Long =
        if (!abortSent) -1 else maxOf(0L, (if (ended) endedNs else nowNs) - abortNs) / 1_000_000

    /**
     * For the recognition's log line, when it is written: what became of the early start, if decided by then, and
     * then [wasNoted] is true; otherwise that it is not decided yet, and the decision gets a line of its own.
     */
    @Synchronized
    fun note(nowNs: Long): String {
        val stoppedAfter = if (abortSent) ", stopped ${abortMs(nowNs)} ms after the abort" else ""
        return when (state) {
            State.PENDING -> ", early start not decided yet"
            State.CONFIRMED ->
                (if (began && startedNs < decidedNs) ", early start used, ${savedMs(nowNs)} ms saved" else ", early start too late")
                    .also { noted = true }
            State.RETRACTED -> ", early start discarded$stoppedAfter".also { noted = true }
            State.DEMOTED ->
                (if (stopped) ", early start for a preview, aborted$stoppedAfter" else ", early start for a preview")
                    .also { noted = true }
        }
    }

    val wasNoted: Boolean
        @Synchronized get() = noted
}
