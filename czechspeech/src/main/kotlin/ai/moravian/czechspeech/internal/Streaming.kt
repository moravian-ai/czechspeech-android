package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.channels.Channel

/** The text of a pass or of a final recognition of a streamed piece, as the committer sees it. */
internal object PassText {
    /**
     * [raw] trimmed; "" when it has no letter or digit (punctuation only: an empty hypothesis, which clears the pending
     * words and commits nothing); else without repetition loops ([RepetitionGuard]). Applying it twice changes nothing.
     */
    fun clean(raw: String): String {
        val text = raw.trim()
        if (text.none { it.isLetterOrDigit() }) return ""
        return RepetitionGuard.clean(text).text
    }
}

/**
 * One pass over the audio of the buffer of a piece: a copy of it from [startMs] to [endMs] (session clock), which the
 * tracker zeroes when the pass is absorbed (or refused) and the queue zeroes when nobody takes it, and the abort of its
 * recognition.
 */
internal class StreamingPass(val pieceStartMs: Long, val startMs: Long, val endMs: Long, val samples: FloatArray) {
    private val run = PreviewRun(pieceStartMs, endMs, samples)

    /** The recogniser took the pass ([attach] returned true); written by the whisper thread, read under the tracker's lock. */
    @Volatile
    internal var attached = false

    /** [StreamingTracker.absorb] has dealt with it (guarded by the tracker's lock). */
    internal var settled = false

    /** Why the tracker dropped it when its result came (for the log line); null when it did not. Guarded by the tracker lock. */
    internal var dropReason: String? = null

    /** Whisper thread, before the recogniser runs: false once the pass is cancelled, which then never starts. */
    fun attach(token: AbortToken): Boolean = run.attach(token).also { if (it) attached = true }

    /** Any thread: a pass not started never starts; a running one stops at whisper.cpp's next check. */
    fun cancel() = run.cancel()

    val isCancelled: Boolean get() = run.isCancelled

    /** Since the abort reached the running recogniser, in ms; -1 when it was not cancelled or nothing native ran then. */
    fun msSinceCancel(nowNs: Long): Long = run.msSinceCancel(nowNs)

    val audioMs: Long get() = endMs - startMs

    override fun toString(): String = "StreamingPass(piece=$pieceStartMs, $startMs..$endMs)"
}

/**
 * What the recogniser made of a [StreamingPass]: [text] already cleaned by [PassText], the recogniser time (0 when it did
 * not run to its end), the [language] it ran in, and whether the pass was [decisive]: long enough to change the language of
 * its piece ([LanguageRule.decisive]; always true without a language rule).
 */
internal class PassResult(val text: String, val recogniserMs: Long, val language: String, val decisive: Boolean = true) {
    override fun toString(): String = "PassResult(<${text.length} chars>, ${recogniserMs} ms, $language, decisive=$decisive)"
}

/** Words to send as one [SpeechEvent.Transcript]. [toString] does not print the text. */
internal data class Chunk(
    val text: String,
    val startMs: Long,
    val endMs: Long,
    val processingMs: Long,
    /** The language of the recognition the words come from. */
    val language: String,
    val firstOfPiece: Boolean,
    /** The piece starts where a cut inside speech ended (a forced cut, the length cap), not after silence. */
    val continuesCut: Boolean = false,
    /**
     * The piece starts where a forced cut fell on a quiet frame of continuous speech (a pause shorter than minSilenceMs, not
     * real silence): the language model decides the boundary with a prior for going on.
     */
    val quietCut: Boolean = false,
) {
    override fun toString(): String = "Chunk(<${text.length} chars>, $startMs..$endMs, $processingMs ms, first=$firstOfPiece)"
}

/** The result of an absorbed pass: the committer's [outcome] and the [chunk] to send, if words were committed. */
internal class Absorbed(val pass: StreamingPass, val outcome: PassOutcome, val chunk: Chunk?)

/** The final recognition of an utterance: [samples] are the part of its audio that the committed words do not cover. */
internal class FinalPlan(
    val utterance: Utterance,
    val offset: Int,
    val hadChunks: Boolean,
    val hadPending: Boolean = false,
    /**
     * The language of the piece when the piece committed words and the session has a language rule: the final recognition is
     * forced in it, with no rescue. Null otherwise.
     */
    val language: String? = null,
    /** The language the piece is in (its first pass took the session's), which the thresholds of an unforced final are made from; null without a rule. */
    val current: String? = null,
    /**
     * The audio of the end of the piece before, when this piece continues a cut inside speech ([CutSeam]) and nothing of it was
     * written yet: the final recognition runs over it and the piece's audio, and the words it repeats are left out ([seam]).
     * The plan owns the array; the engine zeroes it through [StreamingTracker.releasePlanContext] right after the run that uses it.
     */
    val context: FloatArray? = null,
    /** What the words of the piece before say, for the anchor; null exactly when [context] is. */
    val seam: SeamInfo? = null,
) {
    /** The engine has zeroed [context] ([StreamingTracker.releasePlanContext]); guarded by the tracker's lock. */
    internal var contextReleased = false

    /** The samples of the audio from [offset] on (the tail), as they are; the caller zeroes it. */
    fun samples(): FloatArray = utterance.samples.copyOfRange(offset, utterance.samples.size)

    /** The size of what the first run of the final decodes: the context and the tail. */
    val firstRunSamples: Int = (context?.size ?: 0) + (utterance.samples.size - offset)

    /** The context followed by the tail, in a new array; the caller zeroes it. */
    fun contextRunSamples(): FloatArray {
        val c = checkNotNull(context) { "no context" }
        val out = FloatArray(c.size + utterance.samples.size - offset)
        System.arraycopy(c, 0, out, 0, c.size)
        System.arraycopy(utterance.samples, offset, out, c.size, utterance.samples.size - offset)
        return out
    }
}

/**
 * What the piece before a cut inside speech delivered, for the text anchor of the piece after it ([SeamAnchor]): [delivered]
 * its words as sent, [pieceWords] their number (the held word included), [pieceAudioMs] the length of its own audio, and the [language] of its last text.
 * [carriedIn]: how many of the words (0 or 1) were not spoken in the piece's own audio but came in from the piece before as the replacement of its held word.
 */
internal class SeamInfo(
    val delivered: List<String>,
    val heldWord: String?,
    val pieceAudioMs: Long,
    val language: String,
    val carriedIn: Int = 0,
) {
    /** The words of the piece's own audio, the held one included, a word carried in from the piece before not counted (its audio lies in that piece). */
    val pieceWords: Int get() = maxOf(0, delivered.size + (if (heldWord != null) 1 else 0) - carriedIn)

    /** Every word the piece wrote, the held one last. */
    val allWords: List<String> get() = if (heldWord != null) delivered + heldWord else delivered

    /** How many of the last of [allWords] the context covers (the stage 1 anchor). */
    val expected: Int get() = SeamAnchor.expected(pieceWords, pieceAudioMs)

    /** The same for the delivered words alone, the held one taken off (the anchor of the replacement). */
    val expectedWithHeld: Int get() = SeamAnchor.expected(pieceWords, pieceAudioMs, held = 1)

    override fun toString(): String = "SeamInfo($pieceWords words, held=${heldWord != null}, $pieceAudioMs ms, $language)"
}

/** What one piece of speech did, for its log line: counts only, never text. */
internal class PieceSummary(
    val passes: Int,
    val abortedPasses: Int,
    val wordsWhileSpeaking: Int,
    val wordsAfter: Int,
    val cuts: Int,
    val forcedCut: Boolean,
)

/**
 * What a session of streaming did, as counts and times only (never text). [finalSeams], [emptyFinals],
 * [lostPendingWords], [stalePasses] and [errors] are the counters of the situations the algorithm does not guard: a
 * seam decided by a final recognition (it has no agreement check), a final without words while words of the last pass
 * were still pending (they are lost), passes dropped because their buffer or piece was gone, and internal faults that
 * made a piece fall back to the final recognition alone.
 */
internal class StreamingStats(
    val passes: Int,
    val abortedPasses: Int,
    val passRecogniserMs: Long,
    val transcripts: Int,
    val wordsWhileSpeaking: Int,
    val wordsAfter: Int,
    val cuts: Int,
    val forcedCuts: Int,
    val pauses: Int,
    val maxBufferMs: Long,
    val chunkDelaysMs: List<Long>,
    val finalSeams: Int = 0,
    val emptyFinals: Int = 0,
    val lostPendingWords: Int = 0,
    val stalePasses: Int = 0,
    val errors: Int = 0,
    /** The audio of every final recognition, in ms: the length of the pieces. */
    val finalAudioMs: List<Long> = emptyList(),
    /** Times a piece changed its language (two passes in a row in another one). */
    val languageSwitches: Int = 0,
    /** Passes that committed nothing because of the language: the piece had no deciding pass yet, or this pass and the one before ran in different languages. */
    val passesHeldByLanguage: Int = 0,
    /** Final recognitions run with the context of the piece before ([CutSeam]), those whose text anchor was found, those that fell back to the piece's own audio, and the recogniser ms of those second runs. */
    val seamsWithContext: Int = 0,
    val seamsAnchored: Int = 0,
    val seamsFallback: Int = 0,
    val seamFallbackMs: Long = 0,
    /** Context runs whose encoder window was larger than the model's floor. */
    val seamWindowsOverFloor: Int = 0,
    /** Contexts kept and never used (no piece continued them), zeroed. */
    val seamsUnused: Int = 0,
    /** Held words ([CutSeam.CONTEXT_AND_HELD_WORD]): replaced by the next piece's text, sent as they were (and why: the reasons and their counts), and how long each was held, in ms of audio. */
    val heldReplaced: Int = 0,
    val heldKept: Int = 0,
    val heldKeptBy: Map<String, Int> = emptyMap(),
    val heldDelaysMs: List<Long> = emptyList(),
    /** Why the context runs found no anchor and fell back (the labels of the log line, with their counts). */
    val seamFallbackBy: Map<String, Int> = emptyMap(),
) {
    /** The median length of a piece (the audio of its final recognition), 0 before the first. */
    val pieceMedianMs: Long get() = if (finalAudioMs.isEmpty()) 0 else finalAudioMs.sorted()[finalAudioMs.size / 2]

    override fun toString(): String =
        "StreamingStats(passes=$passes, aborted=$abortedPasses, passMs=$passRecogniserMs, transcripts=$transcripts, " +
            "wordsWhileSpeaking=$wordsWhileSpeaking, wordsAfter=$wordsAfter, cuts=$cuts, forcedCuts=$forcedCuts, " +
            "pauses=$pauses, maxBufferMs=$maxBufferMs, delays=${chunkDelaysMs.size}, finalSeams=$finalSeams, " +
            "emptyFinals=$emptyFinals, lostPendingWords=$lostPendingWords, stalePasses=$stalePasses, errors=$errors)"
}

/**
 * The rules of a session with streaming transcription ([StreamingConfig]): when the buffer of the piece in progress is
 * recognised again, what its passes commit and send while the person is still speaking, when the buffer is cut, and
 * what remains for the final recognition. Pure rules over the pieces of [Endpointer], [StreamCommitter] and
 * [StreamingGuard]: no audio device, no native code, no clock (the times it needs are arguments), so tests drive it in
 * virtual time. Three parties call it, ordered by its lock, which is held across every access to the committers (they
 * are not thread safe) and across [emit]:
 * - the capture thread: every frame ([onFrame]), the end of speech ([pieceEnded]), each utterance going to the final
 *   recognition ([finalComing], [finalNotComing]), [stop] and [end];
 * - the whisper thread: each pass it took from [requests] ([absorb], then [publish]) and each final recognition
 *   ([finalPlan], [absorbFinal]);
 * - the recognition loop, after each final ([finalDone]).
 *
 * Rules:
 * - a piece gets its first pass [FIRST_PASS_AFTER_MS] after its speech started, the next ones [MIN_NEW_AUDIO_MS] of
 *   audio after the end of the previous one, one pass at a time, from a copy of the audio from the start of its buffer;
 *   none while a final waits or runs, after stop(), or while the guard has paused the passes;
 * - a result reaches the app only while its piece is in progress: a pass that ends after its piece ended is dropped
 *   (its chunk is not sent; the final recognises the words), whereas a pass already absorbed when the piece ends still
 *   sends its chunk, but no preview follows it;
 * - what the committer committed is never sent twice: the final recognition covers the audio after the buffer start and
 *   its words after the committed ones form the last chunk;
 * - the safeguard cuts the piece ([Step.Cut]) when its oldest uncommitted audio would otherwise get too old (the lag
 *   limit); and, while the passes are paused (chunk mode: the text comes from the final recognitions of the pieces),
 *   as soon as the piece reaches [StreamingGuard.targetPieceMs], the shortest length that the phone's final
 *   recognitions manage with 60 % of its time, recomputed every frame from the latest measurements (4 to 8 s of audio
 *   by the speed of the phone). A shorter piece would cost the same (the encoder window has a floor) and only add cuts;
 * - nothing that can go wrong in the tracker reaches its callers as an exception: a fault makes the piece fall back
 *   to the final recognition alone (today's behaviour) and is counted ([StreamingStats.errors]). What that costs: a
 *   broken piece that sent no chunk is recognised whole by the final recognition; one that sent chunks gets from its
 *   final the words after the committed ones, except when the fault is in absorbing the final itself, which sends the
 *   final's whole text (words already sent can repeat, none is lost). A fault inside absorbing a pass that had changed
 *   the committer already can lose the words that pass committed (they were never sent); the sent and the committed
 *   words are not counted apart, so such a loss shows only in [StreamingStats.errors];
 * - the cut of a piece ([cutNotBeforeMs]) never falls before the audio whose words were sent, and a forced one not
 *   before 8 s of buffer, the cut at the target length not before the last [StreamingGuard.CUT_SEARCH_MS] ms before it
 *   (the quietest frame of that window); the lag limit is kept whatever else is the matter with the piece (broken, stopped), and a
 *   forced cut the endpointer refused is asked for again after 2 s.
 *
 * With [previews], the words after the committed ones of the last pass are shown as a [SpeechPreview] (all of it
 * uncertain, `stableLength` 0); any Transcript replaces it, and one with empty text withdraws it.
 *
 * @param minEncoderWindow the model's encoder window floor in frames ([CzechSpeechConfig.minEncoderWindow]), which the
 *   cost estimate of the default guard depends on
 * @param emit sends an event to the app; called under the lock, so it must not block (trySend)
 */
internal class StreamingTracker(
    config: StreamingConfig,
    minEncoderWindow: Int,
    private val previews: Boolean,
    private val guard: StreamingGuard = StreamingGuard(config.maxLagMs.toLong(), minEncoderWindow),
    /** The session's language, when the model has a language rule: the pieces take their language from it and hold their words by it. Null: the tracker does not look at languages. */
    private val language: LanguageSession? = null,
    private val emit: (ListenEvent) -> Unit,
) {
    private val maxLagMs = config.maxLagMs
    private val cutSeam = config.cutSeam

    /** Called with every array of the seam context right after it was zeroed (a test seam, [SpeechEngine.Hooks.contextReleased]). */
    @Volatile
    internal var onContextReleased: (FloatArray) -> Unit = {}

    /** Called with every array the tracker keeps as a seam context, when it keeps it (a test seam, [SpeechEngine.Hooks.contextKept]). */
    @Volatile
    internal var onContextKept: (FloatArray) -> Unit = {}

    /** What the capture thread has to do after a frame. */
    sealed interface Step {
        object None : Step

        /** Hand [pass] to the recogniser (it is in [requests] already). */
        class Pass(val pass: StreamingPass) : Step

        /** Cut the piece now ([Endpointer.cutNow]): its closing recognition must start before it is too late. */
        object Cut : Step

        /** Cut the piece now inside the quiet stretch at the end of its buffer ([Endpointer.cutAtQuietStretch]): a quiet-point cut. */
        object CutQuiet : Step
    }

    /** Passes for the whisper thread, at most one at a time. Audio a pass cannot deliver is zeroed. */
    val requests: Channel<StreamingPass> = Channel(Channel.UNLIMITED, onUndeliveredElement = { it.samples.fill(0f) })

    /** A piece of speech the tracker knows: its committer and what has been sent of it. */
    private class Piece(val startMs: Long, val committer: StreamCommitter) {
        var open = true
        var handedOver = false

        /** A fault: no more passes, and the final recognition alone (see [StreamingTracker]). */
        var broken = false
        var forced = false

        /** The piece ended by silence (the endpointer's SpeechEnded), not by a cut inside speech. */
        var endedBySilence = false

        /** The piece starts where a cut inside speech ended ([cutEndMs]): the speech goes on, the sentence does not end there. */
        var continuesCut = false

        /** The piece starts where a cut inside speech ended on a quiet frame ([Utterance.quietCut]): the language model decides, leaning to going on. */
        var afterQuietCut = false

        /** The target length ([StreamingGuard.targetPieceMs]) the forced cut was decided at, 0 when it was the lag limit. */
        var cutTargetMs = 0L

        /** The endpointer asked for the cut of this piece ([cutNotBeforeMs]): no pass counts any more, the final recognises all. */
        var cutting = false

        /** After a refused forced cut: not asked again before this position. */
        var retryCutAtMs = -1L
        var finalAbsorbed = false

        /** End of the audio of the last pass requested, -1 before the first. */
        var lastPassEndMs = -1L
        var chunks = 0
        var lastChunkEndMs = -1L
        var previewShown = false

        /**
         * With a language rule: the language of the piece, taken from the session at its first pass ([languageFor]) and changed
         * when two passes in a row ran in another one. Null before the first pass.
         */
        var language: String? = null

        /** The language of the last absorbed pass; null before the first. */
        var lastPassLanguage: String? = null

        /** The piece has had a pass that could decide its language ([PassResult.decisive]): from then on its passes may commit. */
        var settled = false

        /** The language of the last chunk sent while speaking, null for none. */
        var lastChunkLanguage: String? = null

        /** The piece ended at a cut inside speech that fell on a word or a gap inside a phrase: not by silence, not on a quiet frame. */
        var endsInsideSpeech = false

        /** Every word sent for the piece (pass chunks and the final's), in order: what a piece after its cut is anchored against. */
        val sentWords = ArrayList<String>()

        // what the piece did, for its log line (counts only)
        var passes = 0
        var abortedPasses = 0
        var wordsWhileSpeaking = 0
        var wordsAfter = 0
        var cuts = 0
    }

    /**
     * The last [SeamAnchor.CONTEXT_MS] of the audio of a piece that ended at a cut inside speech, in RAM only, with what was
     * delivered for the piece, until the final recognition of the piece that continues it has taken it ([finalPlan]). The array is
     * zeroed when it is taken and used, and on every path where nobody takes it ([discardSeam]).
     */
    private class Seam(
        val cutMs: Long,
        var audio: FloatArray?,
        /** The words sent for the piece, the held one not among them. */
        val sent: List<String>,
        val heldWord: String?,
        val pieceAudioMs: Long,
        val language: String,
        /** The text of the piece came from a rescue in another language: the next piece is not continued in it (no context). */
        val rescued: Boolean = false,
        /** Words of [sent] (0 or 1) that came in from the piece before as the replacement of its held word. */
        val carriedIn: Int = 0,
    )

    private var seam: Seam? = null

    /** The session was torn down ([discardRequests]): no seam is kept or handed out any more (a whisper task that was already past its cancellation check could still absorb a final; its copy of the audio would never be zeroed). Guarded by the lock. */
    private var tornDown = false

    /**
     * The last word of a piece cut inside speech, withheld ([CutSeam.CONTEXT_AND_HELD_WORD]): [text] is what the final recognition
     * of the piece wrote, [cutMs] where the piece ended (the next piece starts there), [endMs] where its last chunk ends. It is
     * resolved exactly once, by [resolveHeld], which is the only thing that sends it or the [SpeechEvent.UtteranceDone] of its piece.
     * Until the final of the piece is over ([armed]) a resolution waits ([pending]).
     */
    private class HeldWord(
        val pieceStartMs: Long,
        val cutMs: Long,
        val text: String,
        val language: String,
        val endMs: Long,
        val sincePositionMs: Long,
    ) {
        var armed = false
        var pending: Pair<Boolean, String>? = null
    }

    private var held: HeldWord? = null

    /** Called with the held word of a piece and where its piece ends when it was sent as it was (any thread, under the tracker's lock): the sentence joiner's previous text then ends with it. */
    @Volatile
    internal var onHeldFlushed: (String, Long) -> Unit = { _, _ -> }
    private val pieces = HashMap<Long, Piece>()
    private var piece: Piece? = null

    /** A piece that ended during the last frame; at the next one its fate is known (a final recognition or none). */
    private var closed: Piece? = null
    private var inFlight: StreamingPass? = null
    private var finalsPending = 0
    private var ended = false
    private var stopped = false
    private var positionMs = 0L

    /** Where the last cut inside speech ended (the next piece starts there), -1 for none. */
    private var cutEndMs = -1L

    /** Where the last cut inside speech on a quiet frame ended, -1 for none. */
    private var quietCutEndMs = -1L

    private var passes = 0
    private var abortedPasses = 0
    private var passRecogniserMs = 0L
    private var transcripts = 0
    private var wordsWhileSpeaking = 0
    private var wordsAfter = 0
    private var cuts = 0
    private var forcedCuts = 0
    private var pauses = 0
    private var maxBufferMs = 0L
    private val chunkDelaysMs = ArrayList<Long>()
    private val finalAudioMs = ArrayList<Long>()
    private var finalSeams = 0
    private var emptyFinals = 0
    private var lostPendingWords = 0
    private var stalePasses = 0
    private var errors = 0
    private var languageSwitches = 0
    private var passesHeldByLanguage = 0
    private var seamsWithContext = 0
    private var seamsAnchored = 0
    private var seamsFallback = 0
    private var seamFallbackMs = 0L
    private val seamFallbackBy = java.util.TreeMap<String, Int>()
    private var seamWindowsOverFloor = 0
    private var seamsUnused = 0
    private var heldReplaced = 0
    private var heldKept = 0
    private val heldKeptBy = java.util.TreeMap<String, Int>()
    private val heldDelaysMs = ArrayList<Long>()
    private val floorFrames = minEncoderWindow

    /** Per piece: chunks absorbed but not yet published, and the pieces whose [SpeechEvent.UtteranceDone] waits for them. */
    private val unpublished = HashMap<Long, Int>()
    private val doneWaiting = HashSet<Long>()

    // ---- capture thread -------------------------------------------------------------------------------------------

    /**
     * Capture thread, after each frame: [pieceMs] is the piece in progress ([Endpointer.pieceMs], -1 for none),
     * [audioStartMs] where its audio starts ([Endpointer.audioStartMs]), [positionMs] the audio so far, [snapshotFrom]
     * a copy of the piece's audio from the given session ms ([Endpointer.snapshotFrom]; the pass owns it), [quietFrames]
     * how many frames in a row at the end of the buffer are quiet ([Endpointer.quietFrames]; chunk mode cuts on such a
     * stretch as soon as the cut window has opened, see [StreamingGuard.QUIET_HOLD_FRAMES]).
     */
    @Synchronized
    fun onFrame(pieceMs: Long, audioStartMs: Long, positionMs: Long, quietFrames: Int = 0, snapshotFrom: (Long) -> FloatArray): Step {
        if (ended) return Step.None
        this.positionMs = positionMs
        try {
            settle()
            // A piece that changed without an utterance for it was dropped (too little speech before the length cap).
            if (piece != null && piece?.startMs != pieceMs) {
                close()
                settle()
            }
            // a seam nobody continues (the piece after it was dropped, or never came) is zeroed now, not at the end of the session
            dropOrphanSeam(openPieceMs = pieceMs)
            if (pieceMs < 0) return Step.None
            val p = piece ?: newPiece(pieceMs, audioStartMs)
            val bufferMs = positionMs - p.committer.bufferStartMs
            if (bufferMs > maxBufferMs) maxBufferMs = bufferMs
            // the lag limit holds whatever else is the matter with the piece: broken, or the session stopping
            if (!p.forced && positionMs >= p.retryCutAtMs) {
                // chunk mode (passes paused): as soon as the phone keeps up with the piece, not as late as the limit allows
                val target = if (guard.passesPaused) guard.targetPieceMs() else null
                // not while two finals are pending: the cut would only add a third to the queue (the lag rule still backs it up)
                val canCut = target != null && finalsPending < MAX_FINALS_FOR_TARGET_CUT
                val atTarget = canCut && bufferMs >= target!!
                // a good cut point is used as soon as there is one: from the floor of the cut window on (the shortest piece the
                // duty was computed for), a quiet stretch of QUIET_HOLD_FRAMES frames; the cut then falls inside that stretch
                // (Step.CutQuiet), so never before the floor and never in front of speech. No quiet stretch: the cut at the
                // target, at the quietest frame of the window, as it was. A quiet stretch at the target itself is used too.
                val quietPoint = canCut &&
                    bufferMs >= maxOf(target!! - StreamingGuard.CUT_SEARCH_MS, StreamingGuard.MIN_PIECE_MS) &&
                    quietFrames >= StreamingGuard.QUIET_HOLD_FRAMES
                val byTarget = atTarget || quietPoint
                if (byTarget || guard.mustCut(bufferMs, contextMsOf(p))) {
                    p.forced = true
                    p.cutTargetMs = if (byTarget) target!! else 0L
                    forcedCuts++
                    val why = if (byTarget) {
                        // the duty of the shortest piece the cut may produce
                        "target $target ms${if (quietPoint) ", quiet point" else ""}, duty ${100 * guard.closingEstimateMs(target!!) / (target - StreamingGuard.CUT_SEARCH_MS)} %"
                    } else {
                        "lag limit $maxLagMs ms"
                    }
                    CsLog.i(
                        "streaming: forced cut at $positionMs ms, $bufferMs ms not committed, " +
                            "closing estimated at ${guard.closingEstimateMs(bufferMs, contextMsOf(p))} ms (lag limit $maxLagMs ms), reason: $why",
                    )
                    return if (quietPoint) Step.CutQuiet else Step.Cut
                }
            }
            if (stopped || p.broken) return Step.None
            val dueAtMs = if (p.lastPassEndMs < 0) p.startMs + FIRST_PASS_AFTER_MS else p.lastPassEndMs + MIN_NEW_AUDIO_MS
            if (inFlight != null || finalsPending > 0 || guard.passesPaused || positionMs < dueAtMs) return Step.None
            val from = p.committer.bufferStartMs
            val samples = snapshotFrom(from)
            if (samples.isEmpty()) return Step.None
            val pass = StreamingPass(p.startMs, from, positionMs, samples)
            if (!requests.trySend(pass).isSuccess) {
                // The runner is gone (the session is ending): no more passes.
                samples.fill(0f)
                ended = true
                return Step.None
            }
            inFlight = pass
            p.lastPassEndMs = positionMs
            return Step.Pass(pass)
        } catch (e: RuntimeException) {
            errors++
            return Step.None
        }
    }

    /** The audio of the piece before that the final recognition of [p] will carry, in ms (0: none; see [finalPlan]). */
    private fun contextMsOf(p: Piece): Long =
        if (cutSeam != CutSeam.NONE && p.continuesCut && p.chunks == 0) SeamAnchor.CONTEXT_MS else 0L

    private fun newPiece(startMs: Long, audioStartMs: Long): Piece =
        Piece(startMs, StreamCommitter(audioStartMs)).also {
            it.continuesCut = startMs == cutEndMs
            it.afterQuietCut = startMs == quietCutEndMs
            cutEndMs = -1
            quietCutEndMs = -1
            piece = it
            pieces[startMs] = it
        }

    /** Capture thread: the speech in progress has ended (before its SpeechEnded is sent). Its passes and previews stop here. */
    @Synchronized
    fun pieceEnded() {
        piece?.endedBySilence = true
        close()
    }

    /** Capture thread: [utterance] is about to be offered to the final recognition (it ends its piece). */
    @Synchronized
    fun finalComing(utterance: Utterance) {
        piece?.takeIf { it.startMs == utterance.startMs }?.let {
            // An utterance that ends a piece in progress without its SpeechEnded is a cut inside speech: the next piece starts
            // there and continues the sentence, unless the cut fell on a quiet frame (a pause between sentences shorter than
            // minSilenceMs): then the language model decides the boundary as after any pause.
            if (!it.endedBySilence) {
                if (utterance.quietCut) quietCutEndMs = utterance.endMs else cutEndMs = utterance.endMs
                it.endsInsideSpeech = !utterance.quietCut
            }
            close()
        }
        val c = closed
        if (c != null && c.startMs == utterance.startMs) c.handedOver = true
        pieces[utterance.startMs]?.handedOver = true
        finalsPending++
    }

    /** Capture thread: the offer of [utterance] failed (a full queue: skipped; closed: the session is ending). */
    @Synchronized
    fun finalNotComing(utterance: Utterance) {
        if (finalsPending > 0) finalsPending--
        forget(utterance.startMs, withdraw = true)
        if (seam?.cutMs == utterance.startMs) discardSeam("skipped")
        done(utterance.startMs)
    }

    /**
     * Capture thread, asked by the endpointer inside every cut of the piece in progress ([Endpointer.cutNotBefore]; the
     * length cap's and the forced one): the earliest session ms the cut may fall on. Never before the end of the
     * audio whose words were committed and sent (the next piece would write them again), and for the forced cut not
     * before [StreamingGuard.FORCED_CUT_MIN_MS] after the start of the buffer (the closing recognition of a shorter
     * piece costs about the same as of that one); a cut at the target length ([StreamingGuard.targetPieceMs]) not before
     * the last [StreamingGuard.CUT_SEARCH_MS] ms before the target, which replaces that floor. From now on no pass counts for the piece: a word committed after
     * this answer could lie beyond the cut; the final recognition has the audio.
     */
    @Synchronized
    fun cutNotBeforeMs(): Long {
        val p = piece ?: return 0
        p.cutting = true
        var floor = p.lastChunkEndMs
        if (p.forced) {
            // a cut at the target length is sought in the last CUT_SEARCH_MS before it and never makes a piece shorter than
            // MIN_PIECE_MS (the target is at least that and the window more); the cut at the lag limit not before FORCED_CUT_MIN_MS
            val after = if (p.cutTargetMs > 0) {
                maxOf(p.cutTargetMs - StreamingGuard.CUT_SEARCH_MS, StreamingGuard.MIN_PIECE_MS)
            } else {
                StreamingGuard.FORCED_CUT_MIN_MS
            }
            floor = maxOf(floor, p.committer.bufferStartMs + after)
        }
        return maxOf(floor, 0)
    }

    /** Capture thread: the endpointer did not carry out the forced cut ([Step.Cut]); it is asked for again 2 s later. */
    @Synchronized
    fun cutRefused() {
        val p = piece ?: return
        if (!p.forced) return
        p.forced = false
        p.retryCutAtMs = positionMs + CUT_RETRY_MS
        if (forcedCuts > 0) forcedCuts--
    }

    /** Capture thread: stop() was called; no pass starts any more and the one that runs is cancelled. */
    @Synchronized
    fun stop() {
        stopped = true
        inFlight?.cancel()
    }

    /** Capture thread: capture has ended. The piece in progress ends, and nothing more is requested. */
    @Synchronized
    fun end() {
        if (ended) return
        close()
        settle()
        ended = true
        inFlight?.cancel()
        requests.close()
        // a continuation whose final waits in the queue still takes the seam; one nobody continues goes now
        dropOrphanSeam(openPieceMs = -1)
    }

    /**
     * Session teardown, any thread: passes still waiting in [requests] (the runner may never have started, or stopped
     * before taking them) are zeroed, and the channel is closed.
     */
    fun discardRequests() {
        requests.close()
        while (true) {
            val pass = requests.tryReceive().getOrNull() ?: break
            pass.samples.fill(0f)
        }
        discardSeamAtTeardown()
    }

    /** Teardown, any thread: nothing will take the seam context any more. */
    @Synchronized
    private fun discardSeamAtTeardown() {
        tornDown = true
        discardSeam("session ended", flush = false)
        held = null         // a cancelled session sends nothing
    }

    /** Zeroes [audio] and tells the test seam. */
    private fun zero(audio: FloatArray) {
        audio.fill(0f)
        try {
            onContextReleased(audio)
        } catch (e: RuntimeException) {
            errors++
        }
    }

    /** Zeroes the seam context, if there is one, and forgets the seam (under the lock). [why] goes to the log, no text in it. */
    private fun discardSeam(why: String, flush: Boolean = true) {
        val s = seam
        seam = null
        val audio = s?.audio
        if (s != null && audio != null) {
            s.audio = null
            zero(audio)
            seamsUnused++
            CsLog.i("streaming: seam context released unused ($why)")
        }
        // a word held back that nobody will replace goes now (not at a teardown: nothing is sent then)
        if (flush) resolveHeld(replaced = false, why = why)
    }

    /**
     * The held word is resolved, exactly once ([held] is cleared here and nowhere else but by a teardown or a failed final that never
     * sent it): [replaced] the next piece's text starts with its new version, else it is sent as it was, as a [SpeechEvent.Transcript]
     * of its piece, and in both cases the [SpeechEvent.UtteranceDone] of the piece follows (it was held back for this). Until the final of
     * the piece has ended ([HeldWord.armed]) the resolution only waits; [finalDone] carries it out. Under the lock, from any thread.
     * [why] is a label for the log and the statistics, never text.
     */
    private fun resolveHeld(replaced: Boolean, why: String) {
        val h = held ?: return
        if (!h.armed) {
            if (h.pending == null) h.pending = replaced to why
            return
        }
        held = null
        if (replaced) {
            heldReplaced++
        } else {
            heldKept++
            heldKeptBy.merge(why, 1, Int::plus)
            send(SpeechEvent.Transcript(Transcript(h.text, h.pieceStartMs, h.endMs, 0, h.language, null)))
            transcripts++
            wordsAfter++
            if (!ended) chunkDelaysMs += maxOf(0L, positionMs - h.endMs)
            try {
                onHeldFlushed(h.text, h.endMs)
            } catch (e: RuntimeException) {
                errors++
            }
        }
        val delayMs = maxOf(0L, positionMs - h.sincePositionMs)
        heldDelaysMs += delayMs
        CsLog.i("streaming: held word ${if (replaced) "replaced" else "kept ($why)"}, delay $delayMs ms")
        done(h.pieceStartMs)
    }

    /** Whisper thread: the held word of the piece before is sent as it was ([why] a label), before anything of this piece. No effect when none is held. */
    @Synchronized
    fun flushHeld(why: String) = resolveHeld(replaced = false, why = why)

    /** Recognition loop, after the last final of the session: a word still held (its continuation never came) is sent. */
    @Synchronized
    fun finalsOver() {
        // every final is over: a held word whose piece's final never reported its end (it is unarmed) is not waited for any more
        held?.armed = true
        resolveHeld(replaced = false, why = "session end")
        discardSeam("session end")
    }

    /** A word is held back and not resolved yet. */
    @Synchronized
    fun heldWordPending(): Boolean = held != null

    /** Arrays of the seam context the tracker still holds (0 or 1): a test seam for "no audio survives". */
    @Synchronized
    fun retainedContextArrays(): Int = if (seam?.audio != null) 1 else 0

    /** The seam, when no piece continues it: not the piece in progress ([openPieceMs]) and not one waiting for or in its final. */
    private fun dropOrphanSeam(openPieceMs: Long) {
        val s = seam ?: return
        if (s.cutMs != openPieceMs && pieces[s.cutMs] == null) discardSeam("no continuation")
    }

    /**
     * The engine has finished with the context of [plan] (after the run that used it, or the final ended without it): zeroed.
     * The array of the combined run buffer goes through here too.
     */
    @Synchronized
    fun releaseContext(audio: FloatArray?) {
        if (audio != null) zero(audio)
    }

    /** The context of [plan] is zeroed, once: whoever ends up with the plan calls this, the run that used it first, the end of the final at the latest. */
    @Synchronized
    fun releasePlanContext(plan: FinalPlan) {
        val audio = plan.context ?: return
        if (plan.contextReleased) return
        plan.contextReleased = true
        zero(audio)
    }

    /** The repeat of an empty final on the whole encoder window, started now for a piece that began at [startMs], would still end within the lag limit. */
    @Synchronized
    fun fullFitsLag(startMs: Long): Boolean = positionMs - startMs + guard.fullWindowEstimateMs() + StreamingGuard.LAG_MARGIN_MS <= maxLagMs

    /**
     * Whisper thread, after the context run of a final: [anchored] says whether the text anchor was found, [fallbackMs] the
     * recogniser ms of the run on the piece's own audio that followed it when it was not, [window] the encoder window of the context run.
     */
    @Synchronized
    fun seamRun(anchored: Boolean, fallbackMs: Long, window: Int, reason: String? = null) {
        seamsWithContext++
        if (anchored) {
            seamsAnchored++
        } else {
            seamsFallback++
            if (reason != null) seamFallbackBy.merge(reason, 1, Int::plus)
        }
        seamFallbackMs += fallbackMs
        if (window > floorFrames) seamWindowsOverFloor++
    }

    /** Ends the piece in progress: no result of it reaches the app any more, and its pass is cancelled. */
    private fun close() {
        val p = piece ?: return
        piece = null
        p.open = false
        inFlight?.takeIf { it.pieceStartMs == p.startMs }?.cancel()
        settle()
        closed = p
    }

    /** The fate of the piece that ended is known: without an utterance for it, its preview goes and it is forgotten. */
    private fun settle() {
        val c = closed ?: return
        closed = null
        if (!c.handedOver && pieces[c.startMs] === c) {
            forget(c.startMs, withdraw = true)
            if (c.chunks > 0) done(c.startMs)     // it wrote words, so it ends; one that wrote none was never an utterance
        }
    }

    private fun forget(startMs: Long, withdraw: Boolean) {
        val p = pieces.remove(startMs) ?: return
        if (withdraw && p.previewShown) withdrawPreview(p)
        if (piece === p) piece = null
        if (closed === p) closed = null
    }

    // ---- whisper thread -------------------------------------------------------------------------------------------

    /**
     * Whisper thread, once for every pass it took from [requests], when the recogniser has finished with it:
     * [result] is null when the pass did not run (cancelled before it began, failed, no words). Returns what the
     * pass committed, or null when it counts for nothing (cancelled, its piece ended, its buffer was cut meanwhile,
     * absorbed already, a fault). The pass's audio is zeroed on every path; do not read it afterwards.
     */
    @Synchronized
    fun absorb(pass: StreamingPass, result: PassResult?): Absorbed? {
        try {
            if (pass.settled) return null
            pass.settled = true
            if (inFlight === pass) inFlight = null
            // the speed is known whenever the recogniser ran to its end, also when the result is not wanted any more
            val ran = result != null && result.recogniserMs > 0
            if (ran) recordPass(pass.audioMs, result!!.recogniserMs)
            if (result == null || pass.isCancelled) {
                if (result != null) pass.dropReason = "cancelled after it finished"
                if (pass.attached) {
                    abortedPasses++
                    pieces[pass.pieceStartMs]?.let { it.abortedPasses++ }
                }
                return null
            }
            if (ran) {
                passes++
                pieces[pass.pieceStartMs]?.let { it.passes++ }
                passRecogniserMs += result.recogniserMs
            }
            val p = pieces[pass.pieceStartMs]
            if (p == null || !p.open || p.broken || p.cutting || ended) {
                pass.dropReason = when {
                    ended -> "the session has ended"
                    p == null || !p.open -> "its utterance has ended"
                    p.broken -> "its utterance fell back to the final recognition"
                    else -> "its utterance was being cut"
                }
                stalePasses++
                return null
            }
            // Before the committer sees it: a pass over another buffer start would make it throw.
            if (pass.startMs != p.committer.bufferStartMs) {
                pass.dropReason = "its buffer was cut meanwhile"
                stalePasses++
                return null
            }
            // With a language rule a pass commits only when it and the pass before ran in the same language and the piece has had
            // a pass long enough to decide its language: a short pass over foreign speech forced in the wrong language writes
            // nonsense, which two such passes would agree on. Two passes in a row in another language change the piece's.
            var mayCommit = true
            if (language != null) {
                val same = result.language == p.lastPassLanguage
                val pieceLanguage = p.language ?: language.current
                if (result.language.isNotEmpty() && result.language != pieceLanguage && same) {
                    p.language = result.language
                    languageSwitches++
                } else if (p.language == null) {
                    p.language = pieceLanguage
                }
                if (result.decisive) p.settled = true
                mayCommit = p.settled && same
                p.lastPassLanguage = result.language
                if (!mayCommit) passesHeldByLanguage++
            }
            val outcome = try {
                p.committer.absorbPass(pass.startMs, pass.endMs, PassText.clean(result.text), quietPointOf(pass), mayCommit)
            } catch (e: RuntimeException) {
                breakPiece(p)
                return null
            }
            if (outcome.cut != null) {
                cuts++
                p.cuts++
            }
            val chunk = if (outcome.committed.isEmpty()) {
                null
            } else {
                chunkOf(p, outcome.committed, startMs = p.startMs, endMs = maxOf(outcome.agreedEndMs, p.startMs), result.recogniserMs, result.language)
            }
            // (defensive: a piece with the passes paused never has one) the continuation of a cut writes while a word of the piece before is
            // held: that word goes first
            if (chunk != null) held?.takeIf { it.cutMs == p.startMs }?.let { resolveHeld(replaced = false, why = "pass chunk") }
            if (chunk != null) unpublished.merge(pass.pieceStartMs, 1, Int::plus)
            return Absorbed(pass, outcome, chunk)
        } finally {
            pass.samples.fill(0f)
        }
    }

    /**
     * Whisper thread, before the recogniser runs a pass: the language of its piece, which the pass's thresholds are made from.
     * The first pass of a piece takes the session's language ([LanguageSession.current]; no pass runs while the final of the
     * piece before waits or runs, so it is up to date) and the piece keeps it until two passes in a row agree on another one.
     * "" when the tracker has no language session (the model has no language rule).
     */
    @Synchronized
    fun languageFor(pass: StreamingPass): String {
        val session = language ?: return ""
        val p = pieces[pass.pieceStartMs] ?: return session.current
        return p.language ?: session.current.also { p.language = it }
    }

    /** Whisper thread: the language of the last chunk the passes of the piece that started at [pieceStartMs] sent; null for none. */
    @Synchronized
    fun lastChunkLanguage(pieceStartMs: Long): String? = pieces[pieceStartMs]?.lastChunkLanguage

    /** The quiet point (session ms) of a range of the pass's audio; [QuietPoint] works on samples. */
    private fun quietPointOf(pass: StreamingPass): (Long, Long) -> Long = { lo, hi ->
        val size = pass.samples.size.toLong()
        val from = ((lo - pass.startMs) * SAMPLES_PER_MS).coerceIn(0, size).toInt()
        val to = ((hi - pass.startMs) * SAMPLES_PER_MS).coerceIn(0, size).toInt()
        pass.startMs + QuietPoint.find(pass.samples, from, to) / SAMPLES_PER_MS
    }

    /** Builds the chunk of [words] of [p]; its end never lies before the end of the chunk before it. */
    private fun chunkOf(p: Piece, words: List<String>, startMs: Long, endMs: Long, processingMs: Long, language: String): Chunk {
        val end = maxOf(endMs, p.lastChunkEndMs)
        val chunk = Chunk(words.joinToString(" "), startMs, end, processingMs, language, firstOfPiece = p.chunks == 0, continuesCut = p.chunks == 0 && p.continuesCut, quietCut = p.chunks == 0 && p.afterQuietCut)
        p.chunks++
        p.sentWords += words
        p.lastChunkEndMs = end
        p.lastChunkLanguage = language
        return chunk
    }

    /**
     * Whisper thread, right after [absorb] (with the sentence joiner's [join] for the chunk, if sentence joining
     * decided one, and the language model's time): sends the chunk, then the preview of the pending words. A chunk
     * always goes out, whatever has become of its piece; a preview only while the piece is in progress.
     */
    @Synchronized
    fun publish(absorbed: Absorbed, join: SentenceJoin?, languageModelMs: Long) {
        val p = pieces[absorbed.pass.pieceStartMs]
        guard.recordedLanguageModel(languageModelMs)
        absorbed.chunk?.let { chunk ->
            send(SpeechEvent.Transcript(Transcript(chunk.text, chunk.startMs, chunk.endMs, chunk.processingMs + languageModelMs, chunk.language, join)))
            transcripts++
            wordsWhileSpeaking += absorbed.outcome.committed.size
            p?.let { it.wordsWhileSpeaking += absorbed.outcome.committed.size }
            chunkDelaysMs += positionMs - chunk.endMs
            if (p != null) p.previewShown = false // the Transcript replaced it
        }
        if (absorbed.chunk != null) released(absorbed.pass.pieceStartMs)
        if (!previews || p == null || !p.open || stopped || ended) return
        if (absorbed.outcome.pending.isNotEmpty()) {
            send(SpeechPreview(p.startMs, absorbed.outcome.pending.joinToString(" "), 0))
            p.previewShown = true
        } else if (p.previewShown) {
            withdrawPreview(p)
        }
    }

    /** Whisper thread: the final recognition of [utterance]; what it has to recognise. */
    @Synchronized
    fun finalPlan(utterance: Utterance): FinalPlan {
        val p = pieces[utterance.startMs] ?: Piece(utterance.startMs, StreamCommitter(utterance.audioStartMs)).also {
            it.open = false
            it.handedOver = true
            pieces[utterance.startMs] = it
        }
        val size = utterance.samples.size
        if (p.committer.audioStartMs != utterance.audioStartMs) {
            // Never expected: the committer's audio is not this utterance's. Today's behaviour for it.
            breakPiece(p)
        }
        val offset = if (p.broken && p.chunks == 0) {
            0
        } else {
            ((p.committer.bufferStartMs - utterance.audioStartMs) * SAMPLES_PER_MS).coerceIn(0, size.toLong()).toInt()
        }
        // A piece that committed words has them in its language: the final of the rest is forced in it. Any other piece decides
        // like an utterance, from the language it is in.
        val forced = if (language != null && p.chunks > 0) p.language else null
        val current = language?.let { p.language ?: it.current }
        // The context of the piece before, when this piece continues its cut inside speech and has written nothing yet: the plan
        // takes the array over (it is zeroed by whoever ends up with it). Any other final settles the seam: nobody will use it.
        var context: FloatArray? = null
        var info: SeamInfo? = null
        val s = seam
        try {
            if (s != null) {
                val audio = s.audio
                if (cutSeam != CutSeam.NONE && audio != null && !s.rescued && utterance.startMs == s.cutMs && p.continuesCut && p.chunks == 0 &&
                    offset == 0 && s.sent.size >= MIN_SEAM_WORDS
                ) {
                    seam = null
                    s.audio = null
                    context = audio
                    // The word held back stays held only if the new piece can still replace it in time (its text would come later than the
                    // lag limit allows after the cut: then it goes now and the context is used the way CONTEXT does).
                    var heldText: String? = held?.takeIf { it.cutMs == s.cutMs }?.text
                    if (heldText != null) {
                        val tailMs = (size - offset) / SAMPLES_PER_MS.toLong()
                        if (positionMs - s.cutMs + guard.closingEstimateMs(tailMs, SeamAnchor.CONTEXT_MS) + StreamingGuard.LAG_MARGIN_MS > maxLagMs) {
                            resolveHeld(replaced = false, why = "lag")
                            heldText = null
                        }
                    }
                    // a word that was flushed already (or never held) counts as delivered: the stage-1 anchor then runs against all the words
                    val wasHeld = s.heldWord
                    val all = if (heldText == null && wasHeld != null) s.sent + wasHeld else s.sent
                    info = SeamInfo(all, heldText, s.pieceAudioMs, s.language, s.carriedIn)
                } else {
                    discardSeam(if (utterance.startMs != s.cutMs) "not continued" else if (s.rescued) "rescued" else "not usable")
                }
            }
            // a held word nobody can replace (no context for this final, another speech) goes before anything of this final
            if (info?.heldWord == null) resolveHeld(replaced = false, why = if (utterance.startMs == held?.cutMs) "not usable" else "not continued")
        } catch (e: Throwable) {
            // the array is taken out of the seam: if anything fails before the plan owns it, it is zeroed here
            context?.let { zero(it) }
            throw e
        }
        return FinalPlan(
            utterance, offset, p.chunks > 0, hadPending = p.committer.pending.isNotEmpty(), language = forced, current = current,
            context = context, seam = info,
        )
    }

    /**
     * Whisper thread, the end of [absorbFinal] of a piece that ended at a cut inside speech: keeps the last
     * [SeamAnchor.CONTEXT_SAMPLES] of its audio (a copy; [u]'s own is zeroed by the engine) for the piece that continues it.
     * Nothing is kept for a piece that sent no word, and none with [CutSeam.NONE].
     */
    private fun keepSeam(p: Piece, u: Utterance, heldWord: String? = null, rescued: Boolean = false, carriedIn: Int = 0) {
        if (cutSeam == CutSeam.NONE || tornDown || !p.endsInsideSpeech || p.sentWords.isEmpty()) return
        discardSeam("replaced")
        val n = minOf(SeamAnchor.CONTEXT_SAMPLES, u.samples.size)
        seam = Seam(
            u.endMs, u.samples.copyOfRange(u.samples.size - n, u.samples.size), p.sentWords.toList(), heldWord,
            u.samples.size / SAMPLES_PER_MS.toLong(), p.lastChunkLanguage.orEmpty(), rescued, carriedIn,
        )
        seam?.audio?.let { try { onContextKept(it) } catch (e: RuntimeException) { errors++ } }
        if (heldWord != null) {
            held = HeldWord(u.startMs, u.endMs, heldWord, p.lastChunkLanguage.orEmpty(), maxOf(u.endMs, p.lastChunkEndMs), positionMs)
        }
    }

    /**
     * Whether the last word of the final text of [p] is held back from the piece (the stage 2 of [CutSeam]): the piece ends at a cut
     * inside speech, the passes are paused (chunk mode: the next piece's final comes in the same way, with the context, soon), and
     * the text has at least [MIN_HELD_TEXT_WORDS] words, so that at least two delivered words lie in the piece's own audio before the held one.
     */
    private fun holdsLastWord(p: Piece, words: List<String>): Boolean =
        cutSeam == CutSeam.CONTEXT_AND_HELD_WORD && p.endsInsideSpeech && guard.passesPaused && words.size >= MIN_HELD_TEXT_WORDS

    /**
     * Whisper thread: [text] is what the final recognition of [plan] made of its audio after [recogniserMs]. Returns
     * the chunk to send (the words after the committed ones), or null when there is none.
     */
    @Synchronized
    fun absorbFinal(
        plan: FinalPlan, text: String, recogniserMs: Long, language: String, runSamples: Int = 0, replacedHeld: Boolean = false,
        rescued: Boolean = false,
    ): Chunk? {
        val u = plan.utterance
        if (recogniserMs > 0) {
            // the piece's own length is the statistic; the guard's cost is of what the run decoded, the context included
            // ([runSamples]: when the run was another one than the plan's first, the piece's own audio alone after a context run that did not run)
            finalAudioMs += ((u.samples.size - plan.offset) / SAMPLES_PER_MS).toLong()
            recordFinal(((if (runSamples > 0) runSamples else plan.firstRunSamples) / SAMPLES_PER_MS).toLong(), recogniserMs)
        }
        val p = pieces[u.startMs] ?: return null
        if (p.finalAbsorbed) return null
        p.finalAbsorbed = true
        val clean = PassText.clean(text)
        val words = try {
            if (p.broken && p.chunks == 0) {
                WordAgreement.words(clean)
            } else {
                val pendingBefore = p.committer.pending.size
                val outcome = p.committer.absorbFinal(clean)
                if (outcome.seamWord != null) finalSeams++
                if (outcome.words.isEmpty() && clean.isEmpty() && pendingBefore > 0) {
                    emptyFinals++
                    lostPendingWords += pendingBefore
                }
                outcome.words
            }
        } catch (e: RuntimeException) {
            errors++
            p.broken = true
            WordAgreement.words(clean)
        }
        // The word of the piece before that this final's text replaces (it starts the text) or, when there is no text, that goes after all.
        // Before anything of this piece is sent or held, so that at most one word is held at a time.
        if (words.isEmpty()) {
            if (plan.seam?.heldWord != null) resolveHeld(replaced = false, why = "empty")
            keepSeam(p, u)
            return null
        }
        // this piece's own last word, when it is held back (never when the piece's text is the replacement alone)
        val holds = holdsLastWord(p, words) && !rescued
        val sent = if (holds) words.dropLast(1) else words
        val chunk = chunkOf(p, sent, startMs = u.startMs, endMs = u.endMs, processingMs = recogniserMs, language = language)
            .copy(firstOfPiece = !plan.hadChunks, continuesCut = !plan.hadChunks && p.continuesCut, quietCut = !plan.hadChunks && p.afterQuietCut)
        if (!ended) chunkDelaysMs += maxOf(0L, positionMs - chunk.endMs)
        transcripts++
        wordsAfter += sent.size
        p.wordsAfter += sent.size
        if (plan.seam?.heldWord != null) resolveHeld(replaced = replacedHeld, why = "not replaced")
        keepSeam(p, u, if (holds) words.last() else null, rescued, if (replacedHeld) 1 else 0)
        return chunk
    }

    // ---- recognition loop -----------------------------------------------------------------------------------------

    /** Recognition loop: the final recognition of [utterance] has ended; [transcribed]: its Transcript was sent. */
    @Synchronized
    fun finalDone(utterance: Utterance, transcribed: Boolean): PieceSummary? {
        if (finalsPending > 0) finalsPending--
        val p = pieces[utterance.startMs]
        val summary = p?.let { PieceSummary(it.passes, it.abortedPasses, it.wordsWhileSpeaking, it.wordsAfter, it.cuts, it.forced) }
        forget(utterance.startMs, withdraw = !transcribed)
        // A piece that sent its text without its last word: its UtteranceDone waits until that word is resolved. Armed now that its
        // Transcript is out, a resolution asked for meanwhile (a flush, a replacement) is carried out. A piece whose text was never
        // sent (failed final) has no word to send: it is dropped silently, as its text is.
        var doneWaits = false
        val own = held?.takeIf { it.pieceStartMs == utterance.startMs }
        if (own != null) {
            if (transcribed) {
                own.armed = true
                doneWaits = true
                own.pending?.let { (replaced, why) -> resolveHeld(replaced, why) }
            } else {
                held = null
                discardSeam("final failed", flush = false)
            }
        }
        if (ended) dropOrphanSeam(openPieceMs = -1)
        // the final of the piece that was to replace a held word is over: whatever happened, the word is not waited for any more
        if (held?.cutMs == utterance.startMs) resolveHeld(replaced = false, why = "final ended without it")
        if (!doneWaits) done(utterance.startMs)
        return summary
    }

    /** Finals offered and not yet done (waiting or running): the full-window retry of an empty final needs fewer than two. */
    @Synchronized
    fun pendingFinals(): Int = finalsPending

    /** Whisper thread: a final was repeated on a longer window and the repeat took [ms]; booked apart in the guard. */
    @Synchronized
    fun retryTime(ms: Long) = guard.recordedRetry(ms)

    /** Whisper thread: the language model took [ms] for the sentence decision of the chunk of a final recognition. */
    @Synchronized
    fun languageModelTime(ms: Long) = guard.recordedLanguageModel(ms)

    /** Any thread: a fault outside the tracker (a pass whose recognition failed, say) that cost a pass or an event. */
    @Synchronized
    fun countError() {
        errors++
    }

    @Synchronized
    fun stats(): StreamingStats = StreamingStats(
        passes, abortedPasses, passRecogniserMs, transcripts, wordsWhileSpeaking, wordsAfter, cuts, forcedCuts, pauses,
        maxBufferMs, chunkDelaysMs.toList(), finalSeams, emptyFinals, lostPendingWords, stalePasses, errors, finalAudioMs.toList(),
        languageSwitches, passesHeldByLanguage, seamsWithContext, seamsAnchored, seamsFallback, seamFallbackMs, seamWindowsOverFloor,
        seamsUnused, heldReplaced, heldKept, heldKeptBy.toMap(), heldDelaysMs.toList(), seamFallbackBy.toMap(),
    )

    // ---- helpers --------------------------------------------------------------------------------------------------

    private fun recordPass(audioMs: Long, recogniserMs: Long) = changed(guard.recorded(audioMs, recogniserMs))

    private fun recordFinal(audioMs: Long, recogniserMs: Long) = changed(guard.recordedFinal(audioMs, recogniserMs))

    private fun changed(change: StreamingGuard.Change) {
        when (change) {
            StreamingGuard.Change.PAUSED -> {
                pauses++
                CsLog.i("streaming: passes paused, a pass over 6 s would take ${guard.estimateMs(StreamingGuard.PROBE_MS)} ms")
            }
            StreamingGuard.Change.RESUMED ->
                CsLog.i("streaming: passes resumed, ${guard.estimateMs(StreamingGuard.PROBE_MS)} ms")
            else -> Unit
        }
    }

    private fun breakPiece(p: Piece) {
        errors++
        p.broken = true
        inFlight?.takeIf { it.pieceStartMs == p.startMs }?.cancel()
    }

    /**
     * The piece that started at [startMs] is complete: [SpeechEvent.UtteranceDone], after its last chunk, also one that
     * was absorbed but is not published yet (the whisper thread publishes right after absorbing; the capture thread may
     * end the piece in between).
     */
    private fun done(startMs: Long) {
        if ((unpublished[startMs] ?: 0) > 0) doneWaiting += startMs else send(SpeechEvent.UtteranceDone(startMs))
    }

    private fun released(startMs: Long) {
        val left = (unpublished[startMs] ?: 0) - 1
        if (left > 0) unpublished[startMs] = left else unpublished.remove(startMs)
        if (left <= 0 && doneWaiting.remove(startMs)) send(SpeechEvent.UtteranceDone(startMs))
    }

    private fun withdrawPreview(p: Piece) {
        p.previewShown = false
        send(SpeechPreview(p.startMs, ""))
    }

    /** [emit] must not throw; if it does, the event is lost and counted, and the tracker's state stays consistent. */
    private fun send(event: ListenEvent) {
        try {
            emit(event)
        } catch (e: RuntimeException) {
            errors++
        }
    }

    companion object {
        /** The first pass of a piece comes when its speech is this old. */
        const val FIRST_PASS_AFTER_MS = 1_000L

        /** A pass starts when this much audio has come since the end of the pass before. */
        const val MIN_NEW_AUDIO_MS = 1_000L

        /** A forced cut the endpointer refused is asked for again after this long. */
        const val CUT_RETRY_MS = 2_000L

        /** Chunk mode: a cut at the target length is not asked for while this many finals are pending (running or waiting). */
        const val MAX_FINALS_FOR_TARGET_CUT = 2

        /** A piece before a cut needs this many delivered words for the next piece to use its audio: a two-word anchor cannot be made of fewer. */
        const val MIN_SEAM_WORDS = 2

        /** A piece's last word is held back only when its text has this many words: the anchor against the delivered ones needs two of them. */
        const val MIN_HELD_TEXT_WORDS = 3

        const val SAMPLES_PER_MS = 16
    }
}
