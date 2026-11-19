package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.Transcript
import ai.moravian.czechspeech.VariantSelection
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.channels.consumeEach
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadFactory
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

/**
 * Most audio that may wait for the recogniser in a session's queue: two utterances of the maximum length and then
 * some, about 3.8 MB. Beyond it utterances are skipped and reported (see [SpeechEngine.listen]).
 */
internal const val MAX_QUEUED_AUDIO_MS = 60_000L

/**
 * Longest audio [SpeechEngine.transcribe] takes in one call: 10 minutes. The call holds the array, a native copy and
 * whisper's mel spectrogram at once (about 110 MB at this length); longer audio is better split at pauses.
 */
internal const val MAX_TRANSCRIBE_MS = 600_000L
internal const val MAX_TRANSCRIBE_SAMPLES = (MAX_TRANSCRIBE_MS * SAMPLE_RATE / 1000).toInt()

/**
 * Largest sample magnitude [SpeechEngine.transcribe] accepts. Float audio is in [-1, 1] and may run a little hot;
 * 16-bit PCM passed without scaling has peaks in the hundreds or thousands for anything but silence.
 */
internal const val MAX_SAMPLE_MAGNITUDE = 2f

/**
 * From this magnitude up, a rejected sample is most likely 16-bit PCM passed without scaling (anything but the
 * quietest audio, -42 dBFS, peaks this high in int16 units); below it, float audio that clips.
 */
internal const val PCM16_HINT_MAGNITUDE = 256f

/**
 * The speech pipeline without Android dependencies: microphone (any [AudioSource]) -> Silero VAD per frame ->
 * [Endpointer] -> queue -> speech gate -> Whisper -> filter -> [RepetitionGuard]. [listenWithPreviews] also previews
 * the utterance in progress on the whisper thread, after any final recognition ([PreviewTracker] has the rules). With
 * sentence joining, a session's transcript is decided against the one before it right after its recognition, on the
 * whisper thread ([SentenceJoiner]); previews and [transcribe] never are. With [CzechSpeechConfig.earlyStart], an
 * utterance enters the queue, and its final recognition starts, during the pause that may end it; nothing of it reaches
 * the app before the capture thread confirms that the speech ended there, and when it went on instead, the recognition
 * is aborted and dropped ([EarlyRun]).
 *
 * Threads:
 * - capture: one thread per [listen] session reads audio, runs the streaming VAD and the endpointer;
 * - whisper: one thread per engine owns the whisper context and the gate VAD (neither is thread-safe) and runs
 *   every recognition, including [transcribe]. It is the only thread that touches [users], [kept], [whisperHandle]
 *   and [gateHandle], which makes it the lock for the model lifecycle; the model is only ever freed there. It exits
 *   when idle. For a recognition (or a run of the language model) [ThreadPlacement] may pin it to fast cores it may
 *   use at that moment; whisper.cpp's worker threads, which it starts, inherit them, and it is unpinned afterwards.
 *   It never resumes a coroutine while pinned, so no thread it starts for one inherits a pin.
 *
 * Model lifecycle: the model is loaded when a [listen] session starts (in parallel with opening the microphone)
 * and freed when the session ends; [transcribe] loads it only after the speech gate passes and frees it afterwards,
 * unless a session holds it. An idle ~0.5 GB model would be pushed to zram by Android, making the next
 * transcription take up to a minute. A session or call cancelled before its load starts never loads the model. A
 * cancelled session completes at once and never waits for the whisper thread: its recognition is aborted (it stops
 * within one step of whisper.cpp's computation, also inside the encoder), work that cannot be interrupted (a load, the
 * speech gate, a run of the language model, another caller's transcription) finishes, and the model is freed right
 * after.
 *
 * With [keepModelLoaded] the engine holds the model too ([kept]), from its load until [releaseModel] or [close]:
 * it outlives the session or call that loaded it, a cancelled one included.
 */
internal class SpeechEngine(
    private val modelFile: File,
    private val config: CzechSpeechConfig,
    private val native: NativeApi,
    vadModel: () -> ByteArray,
    private val audioSourceFactory: () -> AudioSource,
    threads: () -> Int,
    /** Keep the model loaded from its first load until [releaseModel] or [close] (see [kept]). */
    private val keepModelLoaded: Boolean = false,
    languages: List<String> = listOf("cs"),
    /** The rule for leaving the first language; null keeps the most probable language of the list. */
    val languageSwitch: CzechSpeechModel.LanguageSwitch? = null,
    /** Most audio that may wait for the recogniser in a session's queue; smaller in tests. */
    private val maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS,
    private val hooks: Hooks = Hooks.NONE,
    /** The cores' capacities for thread placement ([ThreadPlacement]); read once, on the whisper thread. */
    cpuTopology: () -> CpuTopology = { CpuTopology.UNKNOWN },
    /** Pauses pinning after a pinned computation whose cores did not all run ([ThreadPlacement]); tests set its clock. */
    private val pinGuard: PinGuard = PinGuard(),
    /** Compiles the SDK's regular expressions on this runtime when the engine is created ([SdkPatterns.verify]); tests replace it. */
    patternCheck: () -> Int = SdkPatterns::verify,
) : Closeable {
    /**
     * What the start-up check of the regular expressions found: the cause if one is not valid on this runtime's regex
     * engine (a build defect the desktop tests cannot see), else null. Reported by every [listen] session with an
     * [SpeechEvent.Error] at its start, and by [transcribe] as a failure, so that it shows in the first minute of a
     * run, not on the rare path that uses the pattern.
     */
    private val patternDefect: Throwable? = try {
        patternCheck()
        null
    } catch (e: Throwable) {
        CsLog.w("start-up check of the regular expressions failed", e)
        e
    }

    /** Test seams; the SDK uses [NONE]. */
    interface Hooks {
        /** A [listen] collection is starting and its session is not registered yet (on the collecting thread). */
        fun sessionStarting() {}

        /** A [listen] session has just been registered; nothing of it has started yet (on the collecting thread). */
        fun sessionRegistered() {}

        /** A [listen] session has dropped its hold on the model and is still registered (on the collecting thread). */
        fun sessionEnding() {}

        /** A [transcribe] call passed its first check and is not registered yet (on the caller's thread). */
        fun callStarting() {}

        /** A [transcribe] call has just been registered and is not queued on the whisper thread yet (caller's thread). */
        fun callRegistered() {}

        /** [releaseModel] holds the lock and has decided nothing yet (on the caller's thread). */
        fun releaseDeciding() {}

        /** An utterance was dropped without recognition (cancellation, a full queue); [samples] are already zeroed. */
        fun utteranceDiscarded(samples: FloatArray) {}

        /** A preview of the utterance in progress has been requested with a copy of its audio (capture thread). */
        fun previewRequested(samples: FloatArray) {}

        /** A preview has been absorbed by the preview tracker (its result shown, the next one may be requested); on the preview runner. */
        fun previewFinished() {}

        /** The final recognition of an utterance has ended, its events sent (on the recognition loop). */
        fun utteranceRecognised() {}

        /** An utterance was handed out early and offered to the queue ([queued]: taken) (capture thread). */
        fun earlyStartOffered(queued: Boolean) {}

        /**
         * The recognition of an utterance handed out early has ended, or never ran, and the loop now waits for the
         * capture thread's decision, if it has not come yet (on the recognition loop).
         */
        fun earlyStartAwaited() {}

        /** The early recognition of an utterance was dropped: the speech went on (on the recognition loop). */
        fun earlyStartDiscarded() {}

        /**
         * A session's utterance was recognised with text and its sentence-joining decision comes next (whisper thread,
         * only with sentence joining on).
         */
        fun sentenceBoundaryComing() {}

        /**
         * A streaming pass has been requested with a copy of the audio of its buffer ([samples], zeroed when the pass
         * has ended), right after the request (capture thread; only with [CzechSpeechConfig.streaming]).
         */
        fun streamingPassRequested(samples: FloatArray) {}

        /**
         * A streaming pass has ended: absorbed and published, or dropped (on the runner, only with
         * [CzechSpeechConfig.streaming]).
         */
        fun streamingPassFinished() {}

        /**
         * An array holding audio of the end of the piece before a cut inside speech ([ai.moravian.czechspeech.CutSeam]) has just been
         * zeroed and is let go: the retained context, or the buffer of the run that used it ([samples], all zeros here).
         */
        fun contextReleased(samples: FloatArray) {}

        /** The tracker kept [samples], a copy of the end of a piece cut inside speech, for the piece that continues it (it must be zeroed by the end of the session; a test seam). */
        fun contextKept(samples: FloatArray) {}

        /**
         * A streaming session is ending (its teardown, once): what it did, counts and times only (any thread; only with
         * [CzechSpeechConfig.streaming]).
         */
        fun streamingEnded(stats: StreamingStats) {}

        companion object {
            val NONE: Hooks = object : Hooks {}
        }
    }

    private val vadBytes: ByteArray by lazy(vadModel)

    /** The most recogniser threads (CzechSpeechConfig.threads, or the big core count): each recognition's limit. */
    private val threadCount: Int by lazy(threads)
    private val topology: CpuTopology by lazy(cpuTopology)

    /** The cores and their capacities have been logged since the current session started (any thread). */
    @Volatile
    private var cpusLogged = false

    private val whisperExecutor = ThreadPoolExecutor(
        1, 1, 5L, TimeUnit.SECONDS, LinkedBlockingQueue(), NamedThreadFactory("czechspeech-whisper"),
    ).apply { allowCoreThreadTimeOut(true) }

    // Owned by the whisper thread.
    private var users = 0
    private var whisperHandle = 0L
    private var gateHandle = 0L
    private var systemInfoLogged = false

    /**
     * The language model of variant selection and of sentence joining (one for both; whisper thread only): loaded with
     * the model for variant selection, after a session's first utterance for sentence joining alone; freed with the
     * model.
     */
    private var languageModel: LanguageModel? = null

    /**
     * The engine's own hold on the model with [keepModelLoaded]: taken when the model is loaded, dropped by
     * [releaseModel] and [close], so the model outlives the session or call that loaded it (whisper thread only).
     */
    private val kept = Lease()

    @Volatile
    var isModelLoaded: Boolean = false
        private set

    @Volatile
    var modelLoads: Int = 0
        private set

    /** Recognitions that whisper.cpp stopped early because of cancellation (diagnostics and tests). */
    @Volatile
    var abortedRecognitions: Int = 0
        private set

    /** Utterances skipped because the queue was full: recognition fell behind (diagnostics and tests). */
    @Volatile
    var skippedUtterances: Int = 0
        private set

    /** The most audio that ever waited in a session's queue, in ms (diagnostics and tests). */
    @Volatile
    var peakQueuedAudioMs: Long = 0
        private set

    /** The active [listen] session: its job, for [close], its stop flag, for [stop], and its end, for [awaitClosed]. */
    private class Session(val job: Job) {
        val stop = AtomicBoolean(false)
        val ended = CompletableDeferred<Unit>()
    }

    /**
     * Orders [close] and [releaseModel] against the start and the end of a session and of a [transcribe] call:
     * [closed], [session], [calls] and [releaseWhenIdle] are written only under it.
     */
    private val lock = Any()

    @Volatile
    private var closed = false

    @Volatile
    private var session: Session? = null

    /** [transcribe] calls not finished yet; [close] fails them. */
    private val calls = HashSet<CancellableContinuation<*>>()

    /**
     * [releaseModel] came while a session or call was running: [kept] is dropped once none is ([takeDueRelease]). A
     * session or call that registers meanwhile clears it: a release applies only to the work running when it came.
     */
    private var releaseWhenIdle = false

    val isListening: Boolean get() = session != null

    /** The model's languages (a copy, without repeats: later changes to the caller's list do not reach the engine). */
    private val languages: List<String> = languages.distinct()

    init {
        require(this.languages.isNotEmpty()) { "languages must not be empty" }
        this.languages.forEach(::requireLanguageCode)
        requireSwitchFits(this.languages, languageSwitch)
    }

    /** What the thresholds, the rescue and the first language of the list are made of (see [LanguageRule]). */
    private val rule = LanguageRule(this.languages, languageSwitch)

    /**
     * The state of a collection that decides the language by the rule, or null: the language is forced (asked for, or
     * the model's only one), or there is no switch and the recogniser takes the most probable language of the list.
     */
    private fun languageSessionFor(language: String?): LanguageSession? =
        if (language == null && languageSwitch != null && languages.size > 1) LanguageSession(rule) else null

    /** [language] must be one of the model's languages; null is allowed. */
    private fun checkLanguage(language: String?) {
        if (language != null) requireLanguageCode(language)
        require(language == null || language in languages) {
            "language \"$language\" is not one of this model's languages: ${languages.joinToString()}"
        }
    }

    /** Previews the recogniser has run, aborted ones included, and the audio they took (diagnostics and tests). */
    @Volatile
    var previewRecognitions: Int = 0
        private set

    @Volatile
    var previewAudioMs: Long = 0
        private set

    /** Tasks queued on the whisper thread behind the one it runs (diagnostics and tests). */
    val whisperTasksWaiting: Int get() = whisperExecutor.queue.size

    /**
     * The safeguard's estimate of the recogniser's speed, one for all sessions of this engine (diagnostics and tests):
     * a session does not start from the default estimate and spend a pass learning what the last one knew.
     */
    internal val streamingGuard: StreamingGuard? = config.streaming?.let { StreamingGuard(it.maxLagMs.toLong(), config.minEncoderWindow) }

    /** Previews that whisper.cpp stopped early because their utterance ended (diagnostics and tests). */
    @Volatile
    var abortedPreviews: Int = 0
        private set

    /** Early recognitions ([CzechSpeechConfig.earlyStart]) whose result became the transcript, and the time they saved. */
    @Volatile
    var earlyStartsUsed: Int = 0
        private set

    @Volatile
    var earlyStartSavedMs: Long = 0
        private set

    /**
     * Utterances handed out early whose recognition began only after they had ended (the recogniser was busy, or the
     * queue did not take them early): recognised as without early start, nothing gained, so not counted as used.
     */
    @Volatile
    var earlyStartsTooLate: Int = 0
        private set

    /** Early recognitions started and dropped because the speech went on, and the recogniser time they took. */
    @Volatile
    var earlyStartsDiscarded: Int = 0
        private set

    @Volatile
    var earlyStartDiscardedMs: Long = 0
        private set

    /** Early recognitions the speech ran over whose text came as a preview ([PreviewConfig.fromPauses]). */
    @Volatile
    var earlyStartsShown: Int = 0
        private set

    /** Early recognitions kept for a preview and aborted because their utterance ended first. */
    @Volatile
    var earlyStartsAborted: Int = 0
        private set

    /**
     * Aborts that stopped a running recognition (an early one or a preview) since the engine was created, and the
     * longest time one took to stop, in ms: the most an abort has held up the final text behind it. An abort that came
     * after whisper.cpp had returned, or that whisper.cpp did not heed before it finished, stopped nothing and is not
     * counted.
     */
    val aborts: Int get() = abortCount.get()

    val maxAbortMs: Long get() = abortMax.get()

    // Counted on the whisper thread (previews) and on the recognition loop (early recognitions).
    private val abortCount = AtomicInteger()
    private val abortMax = AtomicLong()

    /** The slowest abort of the session in progress, for its log line. */
    private val sessionAbortMax = AtomicLong()

    /**
     * A session without previews, which sends SpeechEvents only. Throws IllegalArgumentException at once for a
     * [language] outside the model's languages.
     */
    @Suppress("UNCHECKED_CAST")
    fun listen(language: String? = null): Flow<SpeechEvent> {
        checkLanguage(language)
        return session(previews = null, language) as Flow<SpeechEvent>
    }

    /** [listen] with previews of the utterance in progress, by the rules of [PreviewTracker]. */
    fun listenWithPreviews(previews: PreviewConfig, language: String? = null): Flow<ListenEvent> {
        checkLanguage(language)
        return session(previews, language)
    }

    private fun session(previews: PreviewConfig?, language: String?): Flow<ListenEvent> = channelFlow<ListenEvent> {
        hooks.sessionStarting()
        val current = Session(coroutineContext.job)
        // One step under the lock close() takes: close() either finds this session and cancels it, or this
        // collection finds the engine closed and fails. No session can start after close() has returned.
        synchronized(lock) {
            check(!closed) { CLOSED }
            check(session == null) { "listen() is already being collected; one session at a time" }
            session = current
            releaseWhenIdle = false     // a release still waiting does not apply to a session started after it
        }
        hooks.sessionRegistered()
        val stop = current.stop
        val lease = Lease()
        val utterances = UtteranceQueue(maxQueuedAudioMs * SAMPLE_RATE / 1000)
        // With streaming transcription its tracker replaces the preview tracker (it shows the pending words itself).
        val languageSession = languageSessionFor(language)
        val streaming = config.streaming?.let { StreamingTracker(it, config.minEncoderWindow, previews != null, streamingGuard!!, languageSession) { e -> trySend(e) } }
            ?.also { it.onContextReleased = { samples -> hooks.contextReleased(samples) }; it.onContextKept = { samples -> hooks.contextKept(samples) } }
        val tracker = if (streaming != null) null else previews?.let {
            PreviewTracker(it, config.minEncoderWindow * 20L) { preview -> trySend(preview) }
        }
        val joiner = config.sentenceJoining?.let { SentenceJoiner() }
        // a held word sent as it was (any thread): the joiner's previous text then ends with it, applied before its next decision
        streaming?.onHeldFlushed = { text, endMs -> joiner?.continuedLater(text, endMs) }
        val captureExecutor = Executors.newSingleThreadExecutor(NamedThreadFactory("czechspeech-capture"))
        val captureDispatcher = captureExecutor.asCoroutineDispatcher()
        CsLog.i("listen: session started")
        patternDefect?.let { trySend(SpeechEvent.Error(it)) }
        if (streaming != null && config.earlyStart != null) CsLog.i("listen: early start is off while streaming")
        cpusLogged = false          // logged again with the session's first recognition, on the whisper thread
        val earlyBefore =
            intArrayOf(earlyStartsUsed, earlyStartsDiscarded, earlyStartsShown, earlyStartsAborted, earlyStartsTooLate)
        val abortsBefore = abortCount.get()
        sessionAbortMax.set(0)
        val earlyMsBefore = longArrayOf(earlyStartSavedMs, earlyStartDiscardedMs)
        try {
            val source = audioSourceFactory()
            try {
                runSession(source, lease, utterances, captureDispatcher, stop, tracker, streaming, joiner, language, languageSession)
            } finally {
                // Capture closes the source as soon as it ends; this covers a session that ended before capture
                // started (a load that failed at once, say). AudioSource.close() is idempotent, like any Closeable.
                closeQuietly(source)
            }
        } finally {
            try {
                utterances.close()
                while (true) discard(utterances.poll() ?: break)
                tracker?.discardRequests()      // previews nobody took: the runner may never have started
                streaming?.discardRequests()    // likewise passes
                if (utterances.peakMs > peakQueuedAudioMs) peakQueuedAudioMs = utterances.peakMs
                // A cancelled session never waits for the whisper thread: its own recognition is being aborted, which
                // takes a step of whisper.cpp's computation, and the thread may be busy with work that cannot be
                // interrupted at all (a load, the speech gate, a run of the language model, another caller's
                // transcribe()); the model is freed right after it. A session that ends normally (stop(), the end of
                // the audio) has freed it when it completes.
                val cancelled = !isActive
                when (lease.endSession()) {
                    Lease.LOAD_NOT_STARTED -> Unit      // nothing held, and the load task now returns at once
                    Lease.LOAD_RUNNING -> releaseLater(lease)
                    else -> if (cancelled) {
                        releaseLater(lease)
                    } else {
                        withContext(NonCancellable) { onWhisperThread { release(lease) } }
                    }
                }
                captureDispatcher.close()
                hooks.sessionEnding()
            } finally {
                // In the same step as the unregistration, so that a releaseModel() during the teardown is never lost:
                // it either still finds the session (and leaves the release to this point) or finds it gone.
                val releasing = synchronized(lock) {
                    if (session === current) session = null
                    takeDueRelease().also { if (it) releaseLater(kept) }
                }
                // A session that ends normally has freed what it holds when its flow completes, the model released
                // during it included. A cancelled one never waits for the whisper thread.
                if (releasing && isActive) withContext(NonCancellable) { awaitWhisperThread() }
                current.ended.complete(Unit)
                if (config.earlyStart != null && streaming == null) {
                    CsLog.i(
                        "early start: ${earlyStartsUsed - earlyBefore[0]} used, " +
                            "${earlyStartSavedMs - earlyMsBefore[0]} ms saved, " +
                            "${earlyStartsTooLate - earlyBefore[4]} too late; " +
                            "${earlyStartsShown - earlyBefore[2]} shown as a preview; " +
                            "${earlyStartsDiscarded - earlyBefore[1]} discarded, " +
                            "${earlyStartsAborted - earlyBefore[3]} aborted, " +
                            "${earlyStartDiscardedMs - earlyMsBefore[1]} ms of recogniser time",
                    )
                }
                if (streaming != null) logStreamingEnded(streaming)
                languageSession?.let {
                    // with streaming also what the pieces did: language changes of a piece and passes that wrote nothing for the language's sake
                    val pieces = streaming?.stats()?.let { s -> ", ${s.languageSwitches} change(s) of a piece's language, ${s.passesHeldByLanguage} pass(es) held by the language" }.orEmpty()
                    CsLog.i("language: ${it.switches} switches, ${it.rescues} rescues$pieces")
                }
                val sessionAborts = abortCount.get() - abortsBefore
                if (sessionAborts > 0) {
                    CsLog.i("aborts: $sessionAborts, the slowest stopped the recogniser after ${sessionAbortMax.get()} ms")
                }
                CsLog.i("listen: session ended")
            }
        }
    }.buffer(Channel.UNLIMITED)

    /**
     * Recognises [samples]. Fails with IllegalStateException([CLOSED]) when the engine is closed, also when [close]
     * comes during the call: the call then ends at once, and a running recognition is aborted. Fails with
     * IllegalArgumentException, before any work, for input it cannot transcribe (see [requireTranscribable]).
     */
    suspend fun transcribe(samples: FloatArray, language: String? = null): String {
        check(!closed) { CLOSED }
        patternDefect?.let { throw it }
        checkLanguage(language)
        requireTranscribable(samples)
        if (samples.isEmpty()) return ""
        val lease = Lease()
        return onWhisperThread(failOnClose = true) { token ->
            if (token.isCancelled) return@onWhisperThread ""
            acquire(lease)
            try {
                val first = recognize(token, samples, "transcribe", language = language)
                (if (first.dropped) rescueRun(token, samples, "transcribe", first, language, config.minEncoderWindow, deferDropped = false) else null)?.text
                    ?: first.text
            } finally {
                release(lease)
            }
        }
    }

    /**
     * Ends the active [listen] session gracefully: capture stops at the next frame, or with speech in progress once
     * the word in progress has finished (a pause of [STOP_TAIL_SILENCE_MS], at most [STOP_TAIL_MAX_MS]); the
     * utterance in progress ends as if silence followed, everything queued is transcribed, then the flow completes.
     * No-op without a session.
     */
    fun stop() {
        session?.stop?.set(true)
    }

    /**
     * Suspends until the work [close] could not stop at once has ended: the session's teardown (its microphone is
     * closed by then), every native call already running or queued on the whisper thread (a model load, the speech
     * gate, a recognition being aborted), and the freeing of the model. Returns at once when nothing was running.
     */
    suspend fun awaitClosed() {
        check(closed) { "awaitClosed() needs close() first" }
        session?.ended?.await()
        // The whisper thread runs its tasks in order: once this one has run, so has everything queued before it, the
        // release of the session's model and of a kept one included. A call failed by close() queues nothing native
        // any more.
        awaitWhisperThread()
    }

    /**
     * Drops the hold of [keepModelLoaded] once the sessions and [transcribe] calls running now have ended. When none
     * is running, the release is queued on the whisper thread at once (after the native work already there, a
     * cancelled session's load say); otherwise the last of them to end does it: a session that ends normally before
     * its flow completes, a call before it returns, a cancelled session or call right after, in the background.
     * Never blocks. Does nothing without [keepModelLoaded], when no model is loaded and nothing is running, and after
     * [close].
     *
     * Decided and queued under [lock], which also registers sessions and calls, so the outcome depends only on the
     * order of the two. A session or call that registers after this keeps the model, as the first one did: it loads
     * the model again if the release has freed it, and it drops a release still waiting ([releaseWhenIdle]). One
     * registered before keeps the model until it ends.
     */
    fun releaseModel() {
        if (!keepModelLoaded) return
        synchronized(lock) {
            if (closed) return                  // close() has dropped the hold
            hooks.releaseDeciding()
            releaseWhenIdle = true
            if (takeDueRelease()) releaseLater(kept)
        }
    }

    /** Under [lock]: whether a [releaseModel] is due, nothing running any more. It is then no longer pending. */
    private fun takeDueRelease(): Boolean {
        if (!releaseWhenIdle || session != null || calls.isNotEmpty()) return false
        releaseWhenIdle = false
        return true
    }

    /** Suspends until every task queued on the whisper thread before this call has run ([awaitClosed], tests). */
    suspend fun awaitWhisperThread() {
        suspendCancellableCoroutine<Unit> { cont ->
            try {
                whisperExecutor.execute { cont.resume(Unit) }
            } catch (e: RejectedExecutionException) {
                cont.resume(Unit)
            }
        }
    }

    /**
     * Closes the engine without blocking. Once it returns: no new session or [transcribe] call starts (they fail with
     * IllegalStateException([CLOSED]), a collection or call starting concurrently included), the active session is
     * cancelled, and [transcribe] calls in progress resume at once with that exception. The microphone closes at the
     * end of the frame being read. A running recognition is aborted and stops within one step of whisper.cpp's
     * computation (also inside the encoder); native work that cannot be interrupted (a model load, the speech gate, a
     * run of the language model) finishes in the background, and the model is freed right after, a model kept by
     * [keepModelLoaded] too; [awaitClosed] waits for that. Idempotent.
     */
    override fun close() {
        val active: Session?
        val pending: List<CancellableContinuation<*>>
        synchronized(lock) {
            if (closed) return
            closed = true
            active = session
            pending = calls.toList()
            calls.clear()
        }
        active?.job?.cancel(CancellationException("CzechSpeech closed"))
        // Resumes each caller with the exception (not a CancellationException: the caller was not cancelled); the
        // cancellation handler of onWhisperThread aborts the recognition.
        for (call in pending) call.cancel(IllegalStateException(CLOSED))
        // The hold of keepModelLoaded goes too. A load still running ends before this release, which then drops the
        // hold the load took; every other whisper task, queued before or after, belongs to a session or call
        // cancelled above, which no longer loads the model.
        if (keepModelLoaded) releaseLater(kept)
    }

    /**
     * The two halves of a session: model load then recognition, and capture; with [previews], the previews too.
     * Returns when all have ended.
     */
    private suspend fun ProducerScope<ListenEvent>.runSession(
        source: AudioSource,
        lease: Lease,
        utterances: UtteranceQueue,
        captureDispatcher: CoroutineDispatcher,
        stop: AtomicBoolean,
        previews: PreviewTracker?,
        streaming: StreamingTracker?,
        joiner: SentenceJoiner?,
        language: String?,
        languageSession: LanguageSession?,
    ) = coroutineScope {
        // The model loads while the microphone opens; utterances wait in the queue meanwhile. Undispatched, so the
        // load is queued on the whisper thread before capture starts.
        launch(start = CoroutineStart.UNDISPATCHED) {
            onWhisperThread { token ->
                // A session that ended before the whisper thread got here (cancelled, or failed at once) must not
                // load ~0.5 GB for nothing.
                if (token.isCancelled || !lease.startLoad()) return@onWhisperThread
                try {
                    acquire(lease)
                    ensureWhisper()
                } finally {
                    lease.loadDone()
                }
            }
            var languageModelAsked = joiner == null
            utterances.forEach {
                // An utterance handed out early and retracted was never an utterance: no final recognition has ended.
                val transcribed = transcribeUtterance(it, joiner, previews, streaming, utterances, stop, language, languageSession) ?: return@forEach
                try {
                    previews?.finalDone(it, transcribed)
                    streaming?.finalDone(it, transcribed)?.let { summary -> logPiece(it, summary) }
                    hooks.utteranceRecognised()
                    if (!languageModelAsked && anotherUtteranceMayCome(utterances, stop)) {
                        languageModelAsked = true
                        loadLanguageModelForJoining(lease)
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Throwable) {
                    if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                    // the bookkeeping after an utterance is the host's process's business no more than its recognition is
                    CsLog.w("utterance bookkeeping failed", e)
                    send(SpeechEvent.Error(e))
                }
            }
            // the queue is drained: a word still held back (its continuation never came) is sent now
            streaming?.finalsOver()
        }
        if (previews != null) launch { runPreviews(previews, lease, language, languageSession) }
        if (streaming != null) launch { runStreamingPasses(streaming, lease, joiner, language, languageSession) }
        launch(captureDispatcher) {
            try {
                capture(source, utterances, stop, previews, streaming, joiner)
            } finally {
                utterances.close()
                previews?.end()
                streaming?.end()
            }
        }
    }

    /**
     * Runs the previews [previews] requests, one at a time, on the whisper thread, behind whatever is queued there.
     * Their audio is zeroed when they end; what the runner never took is zeroed by the session's teardown
     * ([PreviewTracker.discardRequests]).
     */
    private suspend fun runPreviews(previews: PreviewTracker, lease: Lease, language: String?, languageSession: LanguageSession?) {
        val nothing = Recognition("", 0)
        previews.requests.consumeEach { run ->
            val result = try {
                onWhisperThread { token ->
                    try {
                        // Not after the end of its utterance, and only while the session holds the model, which a
                        // preview never loads (recognizePreview).
                        val runs = !token.isCancelled && lease.held && run.attach(token)
                        if (runs) recognizePreview(token, run, language, languageSession) else nothing
                    } finally {
                        run.samples.fill(0f)
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                CsLog.w("preview failed", e)
                nothing
            }
            previews.finished(run, result.text, result.processingMs)
            hooks.previewFinished()
        }
    }

    /**
     * Runs the passes [streaming] requests, one at a time, on the whisper thread like previews, behind whatever is
     * queued there: a final recognition that is waiting comes first (the tracker asks for no pass while one waits or
     * runs, and cancels the one in flight when its piece ends). Each pass is one whisper task that recognises it, has
     * the tracker absorb the result while its audio still exists (the quiet point needs it), decides the sentence
     * joining of its chunk and publishes it, so everything of a pass happens in order on the one thread. The tracker's
     * [StreamingTracker.absorb] is called exactly once for every pass taken, with null when it did not run (cancelled,
     * failed). Nothing is sent after the session was cancelled; a pass that fails costs the session nothing but that
     * pass, and what it would have shown is recognised by the final recognition.
     */
    private suspend fun runStreamingPasses(streaming: StreamingTracker, lease: Lease, joiner: SentenceJoiner?, language: String?, languageSession: LanguageSession?) {
        val session = currentCoroutineContext().job
        streaming.requests.consumeEach { pass ->
            try {
                onWhisperThread { token ->
                    var result: PassResult? = null
                    val head = StringBuilder()      // the start of the pass's log line, once the recogniser ran to its end
                    try {
                        // Not after the end of its piece, and only while the session holds the model, which a pass
                        // never loads (recognizePass).
                        val runs = !token.isCancelled && lease.held && pass.attach(token)
                        if (runs) result = recognizePass(token, pass, head, language, streaming, languageSession)
                    } catch (e: Throwable) {
                        if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                        CsLog.w("streaming pass failed", e)
                        streaming.countError()
                    }
                    val absorbed = streaming.absorb(pass, result)
                    if (absorbed != null && session.isActive) publishPass(streaming, joiner, absorbed)
                    // the line comes when the pass has been dealt with completely, published as well
                    if (head.isNotEmpty()) CsLog.i(head.append(passOutcome(absorbed, pass)).toString())
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                // The task never ran or ended with an exception of its own: the pass counts for nothing.
                CsLog.w("streaming pass failed", e)
                streaming.countError()
            } finally {
                // Whatever happened, an Error included: a pass that did not get to absorb is settled now (absorb does
                // nothing for one it has settled already), so the tracker asks for more and the audio is zeroed.
                streaming.absorb(pass, null)
                hooks.streamingPassFinished()
            }
        }
    }

    /** Whisper thread: sentence joining of the chunk [absorbed] committed, then what the tracker sends. */
    private fun publishPass(streaming: StreamingTracker, joiner: SentenceJoiner?, absorbed: Absorbed) {
        var step: SentenceJoiner.Step? = null
        try {
            step = absorbed.chunk?.let { chunk -> joiner?.let { joinChunk(it, chunk) } }
        } finally {
            // the chunk goes out whatever the joiner did
            try {
                streaming.publish(absorbed, step?.join, step?.languageModelMs ?: 0)
            } catch (e: Throwable) {
                if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                CsLog.w("streaming pass could not be published", e)
                streaming.countError()
            }
        }
    }

    private fun closeQuietly(source: AudioSource) {
        try {
            source.close()
        } catch (e: Throwable) {
            if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
            CsLog.w("closing the audio source failed", e)
        }
    }

    /** Drops an utterance that will not be recognised: its audio is zeroed at once. */
    private fun discard(utterance: Utterance) {
        utterance.samples.fill(0f)
        hooks.utteranceDiscarded(utterance.samples)
    }

    /**
     * Capture thread: an utterance that does not fit in the queue is skipped and reported, never dropped silently.
     * Capture does not wait for room instead: a microphone that is not read loses audio without any signal.
     */
    private fun ProducerScope<SpeechEvent>.skip(utterance: Utterance, queuedMs: Long, joiner: SentenceJoiner?) {
        joiner?.lost(utterance.endMs)       // no transcript continues another across the speech lost here
        discard(utterance)
        skippedUtterances++
        CsLog.w("utterance skipped: recognition is falling behind, $queuedMs ms of audio waiting")
        trySend(
            SpeechEvent.Error(
                CzechSpeechException(
                    "Recognition is falling behind: the speech from ${utterance.startMs} to ${utterance.endMs} ms " +
                        "was skipped, $queuedMs ms of audio is still waiting for the recogniser",
                ),
            ),
        )
    }

    /**
     * Recognises [utterance]; true when its Transcript was sent. With [joiner], the transcript is decided against the
     * session's previous one in the same whisper task, so only this transcript waits for the language model.
     *
     * An utterance handed out early ([Utterance.early]) is recognised the same way, by the same code, but nothing of it
     * leaves this function before the capture thread has decided: confirmed, it goes on as any utterance (the sentence
     * decision in a whisper task of its own when the recognition ended first); retracted, its result is dropped, no
     * event is sent, the joiner never hears of it, and the function returns null.
     */
    private suspend fun ProducerScope<SpeechEvent>.transcribeUtterance(
        utterance: Utterance,
        joiner: SentenceJoiner?,
        previews: PreviewTracker?,
        streaming: StreamingTracker?,
        queue: UtteranceQueue,
        stop: AtomicBoolean,
        language: String?,
        languageSession: LanguageSession?,
    ): Boolean? {
        val early = utterance.early
        if (early != null && early.isDead) {       // retracted or stopped while it waited: the recogniser never ran
            utterance.samples.fill(0f)
            if (early.tracked) previews?.earlyEnded()
            return null
        }
        val result = try {
            var failure: Throwable? = null
            var decided = early == null
            var recognition: Recognition? = try {
                onWhisperThread<Recognition?> { token ->
                    var rest: FloatArray? = null        // with streaming: the audio the committed words do not cover
                    var plan: FinalPlan? = null
                    try {
                        if (early != null && !early.begin(token, System.nanoTime())) return@onWhisperThread null
                        plan = streaming?.finalPlan(utterance)
                        rest = plan?.samples()
                        val recognised = recognizeFinal(token, rest ?: utterance.samples, utterance, early, streaming, plan, { queue.waiting > 0 }, { stop.get() }, language, languageSession)
                        if (streaming != null && plan != null) {
                            decided = true
                            afterStreamingFinal(token, joiner, streaming, plan, recognised, languageSession)
                        } else if (early == null || early.isConfirmed) {
                            decided = true
                            afterRecognition(token, joiner, recognised, utterance)
                        } else {
                            recognised      // the sentence decision waits for the capture thread's
                        }
                    } finally {
                        utterance.samples.fill(0f)
                        rest?.fill(0f)
                        // the context of the piece before: zeroed here whatever happened (once; the run that used it did so already)
                        plan?.let { streaming?.releasePlanContext(it) }
                        early?.end(System.nanoTime())
                    }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
                if (early == null) throw e
                failure = e         // an error only if the utterance turns out to be one
                null
            }
            if (early != null) {
                if (early.tracked) previews?.earlyEnded()
                hooks.earlyStartAwaited()
                val outcome = early.awaitOutcome()
                if (outcome == EarlyRun.Outcome.PREVIEW) {
                    // not an utterance: its text, if it ran to its end, previews the utterance in progress
                    val text = recognition?.text.orEmpty()
                    val shown = !early.wasStopped && previews?.fromPause(utterance.startMs, text) == true
                    earlyStartDecided(early, outcome, shown)
                    if (failure != null) CsLog.w("early start for a preview failed", failure)
                    return null
                }
                earlyStartDecided(early, outcome, shown = false)
                if (outcome != EarlyRun.Outcome.USED) return null
                failure?.let { throw it }
                val undecided = recognition
                if (!decided && joiner != null && undecided != null) {
                    recognition = onWhisperThread { token -> afterRecognition(token, joiner, undecided, utterance) }
                }
            }
            recognition ?: Recognition("", 0)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Throwable) {
            if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
            // Whatever the utterance's own work threw, an Error included (a class that failed to initialise): it never leaves
            // this function, so that it cannot reach the collector's scope and end the host's process. Cancellation
            // does (above), and an OutOfMemoryError, which ends the session as it always did (the engine's precedent; the
            // flow throws it, the app ends dictation). The audio was zeroed by the whisper task; once more is free.
            utterance.samples.fill(0f)
            runCatching {
                joiner?.reset()         // its speech is lost: the next transcript does not continue the one before
                CsLog.w("utterance failed", e)
            }
            send(SpeechEvent.Error(e))
            if (utterance.speechMs >= EmptyFinalRetry.CLEAR_SPEECH_MS) send(SpeechEvent.UtteranceNotWritten(utterance.startMs))
            return false
        }
        if (result.text.isNotEmpty()) {
            val endMs = if (result.endMs >= 0) result.endMs else utterance.endMs
            val transcript = Transcript(result.text, utterance.startMs, endMs, result.processingMs, result.language, result.join)
            // a streamed piece reports its language at its end (afterStreamingFinal), also when its words were all written while speaking
            if (streaming == null) languageSession?.delivered(result.language, result.rescued)
            send(SpeechEvent.Transcript(transcript))
        } else if (result.dropped && utterance.speechMs >= EmptyFinalRetry.CLEAR_SPEECH_MS) {
            // clear speech that ends up without a letter or a digit, after the retries (or without them): the app is told
            send(SpeechEvent.UtteranceNotWritten(utterance.startMs))
        }
        return result.text.isNotEmpty()
    }

    /**
     * What follows the recognition of a session's utterance on the whisper thread: with [joiner], the decision against
     * the previous transcript.
     */
    private fun afterRecognition(
        token: AbortToken,
        joiner: SentenceJoiner?,
        recognition: Recognition,
        utterance: Utterance,
    ): Recognition = when {
        joiner == null -> recognition
        recognition.text.isEmpty() -> {
            // speech without words (or none at all): no transcript continues another across it, as
            // the measurement never paired transcripts across such a part; a cancelled one ends all
            if (!token.isCancelled) joiner.reset()
            recognition
        }
        recognition.language.isNotEmpty() && recognition.language != rule.primary -> {
            // the language model of the sentence decision is the first language's: another language's text stays as Whisper wrote it,
            // and no transcript continues another across it
            if (!token.isCancelled) joiner.reset()
            recognition
        }
        else -> {
            hooks.sentenceBoundaryComing()
            // after close() or a cancellation the language model has nothing to decide
            if (token.isCancelled) recognition else joinSentence(joiner, recognition, utterance)
        }
    }

    /**
     * What follows the final recognition of a streamed utterance on the whisper thread ([StreamingTracker]): the words
     * after the ones its passes committed are the last chunk. With [joiner], the first chunk of the utterance is
     * decided against the previous transcript, a later one only extends it. No chunk (everything was written while
     * speaking, or the rest has no words): no transcript, and the joiner forgets the previous one only when the
     * utterance had no chunk at all, as for any utterance without words.
     */
    private fun afterStreamingFinal(
        token: AbortToken,
        joiner: SentenceJoiner?,
        streaming: StreamingTracker,
        plan: FinalPlan,
        recognised: Recognition,
        languageSession: LanguageSession?,
    ): Recognition {
        if (token.isCancelled) return Recognition("", recognised.processingMs)
        val chunk = streaming.absorbFinal(
            plan, recognised.text, recognised.recogniserMs, recognised.language, recognised.runSamples, recognised.replacedHeld,
            rescued = recognised.rescued,
        )
        if (recognised.retryMs > 0) streaming.retryTime(recognised.retryMs)
        // The session goes on in the language of the last text the piece wrote (a rescue's text leaves it as it is); a piece
        // that wrote nothing changes nothing.
        languageSession?.let { session ->
            val last = chunk?.language ?: if (plan.hadChunks) streaming.lastChunkLanguage(plan.utterance.startMs) else null
            if (last != null) session.delivered(last, rescued = chunk != null && recognised.rescued)
        }
        if (chunk == null) {
            if (!plan.hadChunks) joiner?.reset() else joiner?.endedAt(plan.utterance.endMs)
            // a piece that wrote words while it was spoken lost nothing when what remained was empty
            return Recognition("", recognised.processingMs, dropped = recognised.dropped && !plan.hadChunks, retryCutByStop = recognised.retryCutByStop)
        }
        val step = joiner?.let { joinChunk(it, chunk, recognised.replacedHeld) }
        streaming.languageModelTime(step?.languageModelMs ?: 0)
        return Recognition(chunk.text, recognised.processingMs + (step?.languageModelMs ?: 0), step?.join, endMs = chunk.endMs, language = chunk.language)
    }

    /**
     * Sentence joining for a chunk of a streamed utterance (whisper thread): the first chunk of an utterance is decided
     * against the previous transcript, as [joinSentence] does, a later one only extends the previous text. Never fails:
     * a chunk is never held back for it. The language model of the decision is the first language's, so a chunk in another
     * language is not joined, whatever kind of chunk it is (the first of an utterance, after a cut, a later one): no
     * join and no language model, and no transcript continues another across it ([SentenceJoiner.reset]), as in
     * [afterRecognition].
     */
    private fun joinChunk(joiner: SentenceJoiner, chunk: Chunk, replacedHeld: Boolean = false): SentenceJoiner.Step? {
        if (chunk.language.isNotEmpty() && chunk.language != rule.primary) {
            joiner.reset()
            return null
        }
        return joinFirstLanguageChunk(joiner, chunk, replacedHeld)
    }

    private fun joinFirstLanguageChunk(joiner: SentenceJoiner, chunk: Chunk, replacedHeld: Boolean = false): SentenceJoiner.Step? = try {
        if (chunk.firstOfPiece && chunk.continuesCut) {
            // the speech goes on across a cut: no sentence decision; the language model only decides whether the first word
            // (a name) keeps its capital, once per piece
            if (replacedHeld) {
                // the text was recognised with the audio on both sides of the seam and starts with the new version of the word held
                // back: its case and punctuation there are Whisper's, not artefacts of the cut; no language model
                CsLog.i("sentence boundary: continued (cut, held word replaced)")
                joiner.afterContext(chunk.text, chunk.startMs, chunk.endMs)
            } else {
                val scorer = languageModelScorer()?.let { score -> { texts: List<String> -> hooks.sentenceBoundaryComing(); score(texts) } }
                joiner.afterCut(chunk.text, chunk.startMs, chunk.endMs, scorer)
            }
        } else if (chunk.firstOfPiece) {
            hooks.sentenceBoundaryComing()
            joiner.next(chunk.text, chunk.startMs, chunk.endMs, languageModelScorer(), quiet = chunk.quietCut)
        } else {
            joiner.continued(chunk.text, chunk.endMs)
            null
        }
    } catch (e: Throwable) {
        if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
        CsLog.w("sentence joining of a chunk failed", e)
        null
    }

    /**
     * Recognition loop: the capture thread has decided about an utterance handed out early. Counts it, and writes the
     * log line of the decision when the recognition's own line could not have it yet. Counts and times only.
     */
    private fun earlyStartDecided(early: EarlyRun, outcome: EarlyRun.Outcome, shown: Boolean) {
        val now = System.nanoTime()
        early.abortMs(now).let { if (it >= 0) abortStopped(it) }
        when {
            outcome == EarlyRun.Outcome.USED && early.hadHeadStart -> {
                val saved = early.savedMs(now)
                earlyStartsUsed++
                earlyStartSavedMs += saved
                if (!early.wasNoted) CsLog.i("utterance: early start used, $saved ms saved")
            }
            outcome == EarlyRun.Outcome.USED -> {
                // its recognition began after the utterance had ended: as without early start
                earlyStartsTooLate++
                if (!early.wasNoted) CsLog.i("utterance: early start too late")
            }
            !early.hasBegun -> Unit                 // nothing was computed for nothing
            outcome == EarlyRun.Outcome.DROPPED -> {
                val spent = early.recogniserMs(now)
                earlyStartsDiscarded++
                earlyStartDiscardedMs += spent
                if (!early.wasNoted) CsLog.i("utterance: early start discarded, $spent ms of recogniser time")
                hooks.earlyStartDiscarded()
            }
            shown -> {
                earlyStartsShown++
                CsLog.i("utterance: early start shown as a preview")
            }
            early.wasStopped -> {
                earlyStartsAborted++
                earlyStartDiscardedMs += early.recogniserMs(now)
            }
            else -> Unit                            // ran to its end with nothing to show
        }
    }

    /** An abort reached a running recognition, which stopped [ms] later (diagnostics: what an abort can cost). */
    private fun abortStopped(ms: Long) {
        abortCount.incrementAndGet()
        abortMax.accumulateAndGet(ms) { a, b -> maxOf(a, b) }
        sessionAbortMax.accumulateAndGet(ms) { a, b -> maxOf(a, b) }
    }

    /**
     * Whether the session can still give an utterance after the one just recognised: one waits in the queue, or capture
     * goes on. After stop() capture ends once the word in progress has (at most [STOP_TAIL_MAX_MS]), so this waits for
     * that first; then only what is queued counts. Recognition loop only.
     */
    private suspend fun anotherUtteranceMayCome(utterances: UtteranceQueue, stop: AtomicBoolean): Boolean {
        if (stop.get()) utterances.awaitClosed()
        return !utterances.isFinished
    }

    /**
     * Sentence joining's language model, after the session's first utterance has been recognised (its transcript, if
     * any, sent), so that it never delays that one, and only when another utterance may still come: a session of one
     * sentence ended by stop() never loads it. A task of its own on the whisper thread, so never under a running
     * preview. A load that failed (the file not there yet, say) is tried again in the next session, also when the
     * model stays loaded. Variant selection loads the language model with the model instead ([ensureWhisper]): its
     * first utterance needs it.
     */
    private suspend fun loadLanguageModelForJoining(lease: Lease) {
        onWhisperThread { token ->
            if (!token.isCancelled && lease.held && whisperHandle != 0L && languageModel == null) loadLanguageModel()
        }
    }

    /**
     * Sentence joining (whisper thread): [recognition] against the session's previous transcript, with the language
     * model when one is loaded. Never fails: without a decision the transcript stays as it is.
     */
    private fun joinSentence(joiner: SentenceJoiner, recognition: Recognition, utterance: Utterance): Recognition {
        val step = joiner.next(recognition.text, utterance.startMs, utterance.endMs, languageModelScorer())
        return Recognition(recognition.text, recognition.processingMs + step.languageModelMs, step.join, language = recognition.language, decided = recognition.decided, shares = recognition.shares, rescued = recognition.rescued)
    }

    /** The language model's scoring for [SentenceJoiner.next] (whisper thread), null while none is loaded. */
    private fun languageModelScorer(): ((List<String>) -> DoubleArray?)? = languageModel?.let { model ->
        { texts -> placeThreads().use { placement -> model.score(texts, placement.threads).also { placement.computed() } } }
    }

    private suspend fun ProducerScope<SpeechEvent>.capture(
        source: AudioSource,
        utterances: UtteranceQueue,
        stop: AtomicBoolean,
        previews: PreviewTracker?,
        streaming: StreamingTracker?,
        joiner: SentenceJoiner?,
    ) {
        val vad = try {
            native.vadInit(vadBytes, 1)
        } catch (e: Throwable) {
            closeQuietly(source)
            throw e
        }
        if (vad == 0L) {
            closeQuietly(source)
            throw CzechSpeechException("Could not load the Silero VAD model")
        }
        // whisper_vad_init allocates the LSTM state without zeroing it; the streaming calls do not reset it, so
        // without this the first seconds would depend on whatever the heap held (e.g. a previous session's state).
        native.vadReset(vad)
        val frame = FloatArray(FRAME_SAMPLES)
        val prob = FloatArray(1)
        // The utterance handed out early in the pause in progress, not decided yet, and whether the queue took it.
        var early: Utterance? = null
        var earlyQueued = false
        // Early recognitions the speech ran over, kept for a preview: stopped when the next pause is handed out or the
        // utterance in progress ends, whichever comes first.
        val forPreview = ArrayList<Utterance>(2)
        val stopForPreview = {
            for (kept in forPreview) if (kept.early?.stop(System.nanoTime()) == true) discard(kept)
            forPreview.clear()
        }
        val keepForPreview = previews != null && previews.fromPauses
        val endpointer = Endpointer(if (streaming != null) config.copy(earlyStart = null) else config) { event ->
            when (event) {
                is EndpointerEvent.UtteranceEarly -> {
                    early = event.utterance
                    // Final text first: this pause may be the end, so the recogniser is freed for its recognition. A
                    // timed preview is cancelled, and so is the recognition of an earlier pause kept for a preview.
                    previews?.finalStartingEarly()
                    stopForPreview()
                    // counted before the offer: the recognition loop may take the utterance at once
                    if (previews != null) {
                        event.utterance.early?.tracked = true
                        previews.earlyStarted(event.utterance)
                    }
                    earlyQueued = utterances.offerEarly(event.utterance)
                    if (!earlyQueued && previews != null) {
                        event.utterance.early?.tracked = false
                        previews.earlyEnded()
                    }
                    hooks.earlyStartOffered(earlyQueued)
                }
                is EndpointerEvent.EarlyRetracted -> {
                    early = null
                    if (keepForPreview && earlyQueued) {
                        // the speech goes on: the recognition runs on, for a preview of it
                        utterances.forget(event.utterance)
                        event.utterance.early?.demote(System.nanoTime())
                        forPreview.removeAll { it.early?.isOver != false }      // done with: nothing left to stop
                        forPreview += event.utterance
                    } else {
                        retract(event.utterance, utterances)
                    }
                }
                is EndpointerEvent.SpeechStarted -> trySend(SpeechEvent.SpeechStarted(event.atMs))
                is EndpointerEvent.SpeechEnded -> {
                    stopForPreview()            // final text first
                    previews?.pieceEnded()      // no preview of this speech after its SpeechEnded
                    streaming?.pieceEnded()     // likewise, and its passes end
                    native.vadReset(vad)
                    if (event.confirmed) trySend(SpeechEvent.SpeechEnded(event.atMs))
                }
                is EndpointerEvent.UtteranceReady -> {
                    stopForPreview()            // also at the length cap, which ends the piece without a SpeechEnded
                    previews?.finalComing(event.utterance)      // before its Transcript can exist
                    streaming?.finalComing(event.utterance)
                    // One handed out early that the queue took is there already, or even recognised: it only waits
                    // for this. One the queue did not take then is offered now, as any utterance.
                    val offer = if (event.utterance === early && earlyQueued) {
                        UtteranceQueue.Offer.QUEUED
                    } else {
                        utterances.offer(event.utterance)
                    }
                    early = null
                    if (offer == UtteranceQueue.Offer.QUEUED) event.utterance.early?.confirm(System.nanoTime())
                    when (offer) {
                        UtteranceQueue.Offer.QUEUED -> Unit
                        UtteranceQueue.Offer.FULL -> skip(event.utterance, utterances.queuedMs, joiner)
                        UtteranceQueue.Offer.CLOSED -> discard(event.utterance)     // the session is ending
                    }
                    if (offer != UtteranceQueue.Offer.QUEUED) previews?.finalNotComing(event.utterance)
                    if (offer != UtteranceQueue.Offer.QUEUED) streaming?.finalNotComing(event.utterance)
                }
            }
        }
        if (streaming != null) endpointer.cutNotBefore = streaming::cutNotBeforeMs
        val snapshot = endpointer::snapshot
        val snapshotFrom = endpointer::snapshotFrom
        var frames = 0L
        var vadNanos = 0L
        var tail = -1           // stop(): frames read since it while the word in progress finishes; -1 before stop()
        var stopAtFrame = 0L
        try {
            source.open()
            while (true) {
                currentCoroutineContext().ensureActive()
                if (tail < 0 && stop.get()) {
                    stopAtFrame = frames
                    tail = 0
                    previews?.stop()        // no preview while the last word ends; one running is aborted now
                    streaming?.stop()       // likewise for passes
                    stopForPreview()
                    if (stopWaitOver(endpointer, tail, stopAtFrame)) break
                }
                if (source.read(frame) < 0) break
                val t0 = System.nanoTime()
                if (native.vadProbs(vad, frame, FRAME_SAMPLES, prob) != 1) {
                    throw CzechSpeechException("Voice activity detection failed")
                }
                vadNanos += System.nanoTime() - t0
                endpointer.accept(frame, prob[0])
                val previewRequested = previews?.onFrame(
                    endpointer.pieceMs, endpointer.positionMs, endpointer.capMs, endpointer.speaking, snapshot,
                )
                if (previewRequested != null) hooks.previewRequested(previewRequested.samples)
                if (streaming != null) {
                    when (val step = streaming.onFrame(endpointer.pieceMs, endpointer.audioStartMs, endpointer.positionMs, endpointer.quietFrames, snapshotFrom)) {
                        is StreamingTracker.Step.Pass -> hooks.streamingPassRequested(step.pass.samples)
                        StreamingTracker.Step.Cut -> {
                            // in this very frame: the closing recognition must start in time; the tracker asks once per piece
                            if (!endpointer.cutNow()) {
                                CsLog.i("streaming: the lag cut was refused (too little audio buffered)")
                                streaming.cutRefused()
                            }
                        }
                        StreamingTracker.Step.CutQuiet -> {
                            if (!endpointer.cutAtQuietStretch()) {
                                CsLog.i("streaming: the quiet-point cut was refused (too little audio buffered)")
                                streaming.cutRefused()
                            }
                        }
                        StreamingTracker.Step.None -> Unit
                    }
                }
                frames++
                if (tail >= 0) {
                    tail++
                    if (stopWaitOver(endpointer, tail, stopAtFrame)) break
                }
            }
            endpointer.finish()
        } finally {
            frame.fill(0f)
            // Capture ended without a decision (a failure, a cancellation): the utterance handed out early is none.
            early?.let { retract(it, utterances) }
            early = null
            stopForPreview()
            endpointer.clear()
            closeQuietly(source)       // the microphone closes as soon as capture ends, before the queue drains
            native.vadFree(vad)
            val perFrameUs = if (frames > 0) vadNanos / frames / 1000 else 0
            CsLog.i("capture: ${frames * FRAME_MS} ms of audio, VAD $perFrameUs us per 32 ms frame")
        }
    }

    /**
     * Capture thread: an utterance handed out early is none after all (the speech went on, or capture ended). It no
     * longer counts as waiting audio; a recognition of it that runs is aborted, and its audio is zeroed: here when the
     * recogniser never took it, else by the whisper thread as soon as the recognition has stopped.
     */
    private fun retract(utterance: Utterance, utterances: UtteranceQueue) {
        utterances.forget(utterance)
        if (utterance.early?.retract(System.nanoTime()) != false) discard(utterance)
    }

    /**
     * stop() lets the word in progress finish: with speech in progress, capture goes on until a pause of
     * [STOP_TAIL_SILENCE_MS] (one under way counts) or [STOP_TAIL_MAX_FRAMES] frames after stop(); without speech it
     * ends at once. An abrupt cut makes Whisper invent words. The web SDK's capture worker does the same.
     */
    private fun stopWaitOver(endpointer: Endpointer, tail: Int, stopAtFrame: Long): Boolean {
        val atPause = endpointer.atPause
        if (!atPause && tail < STOP_TAIL_MAX_FRAMES) return false
        val at = "capture: stop() at ${stopAtFrame * FRAME_MS} ms of audio"
        CsLog.i(
            if (tail == 0) "$at, no word in progress"
            else "$at, waited ${tail * FRAME_MS} ms ${if (atPause) "for a pause" else "(the limit)"}",
        )
        return true
    }

    /**
     * What a recognition's log line says about its language, ending in ", " (or nothing, with one language, where the
     * line has always been silent about it): the language it ran in and whether it was forced, or the shares the rule
     * decided from. Language codes and numbers only, never the text.
     */
    /**
     * The decision of a run that chose its language by the rule, for the line that explains a dropped final: the language,
     * the audio length the thresholds were made from, the current language, the shares and the thresholds (NaN: not a
     * candidate; all NaN: too short to decide). Language codes and numbers only.
     */
    private fun decisionNote(decided: String, current: String, audioMs: Long, shares: FloatArray?, thresholds: FloatArray?): String {
        fun f(x: Float) = String.format(Locale.ROOT, "%.3f", x)
        val sharesText = if (shares != null && shares.size == languages.size) {
            languages.indices.joinToString(" ") { "${languages[it]} ${f(shares[it])}" }
        } else {
            "none"
        }
        val thresholdsText = when {
            thresholds == null -> "none"
            thresholds.drop(1).all { it.isNaN() } -> "too short to decide, no candidate"
            else -> (1 until languages.size).filter { !thresholds[it].isNaN() }.joinToString(" ") { "${languages[it]} ${f(thresholds[it])}" }
        }
        return "chose ${decided.ifEmpty { "none" }}, current $current, audio $audioMs ms, shares $sharesText, thresholds $thresholdsText"
    }

    private fun languageNote(decided: String, forced: Boolean, shares: FloatArray?, rescue: Boolean): String = when {
        languages.size < 2 || decided.isEmpty() -> ""
        rescue -> "language $decided (rescue), "
        forced -> "language $decided (forced), "
        shares != null && shares.size == languages.size ->
            "language $decided (" + languages.indices.joinToString(", ") { "${languages[it]} ${String.format(Locale.ROOT, "%.3f", shares[it])}" } + "), "
        else -> "language $decided, "
    }

    /** Speech gate, Whisper, filter and repetition guard. Whisper thread only, with a lease held. */
    private fun recognize(
        token: AbortToken,
        samples: FloatArray,
        what: String,
        early: EarlyRun? = null,
        floor: Int = config.minEncoderWindow,
        deferDropped: Boolean = false,
        /** A language to force: asked for, or the language of a rescue or of a retry (no recognition, no thresholds). */
        language: String? = null,
        /** The language the session is in, which the thresholds of a recognition among several languages are made from. */
        current: String = rule.primary,
        /** The run is a rescue in another language than the one that came out empty (logged so; its text is marked [Recognition.rescued]). */
        rescue: Boolean = false,
    ): Recognition {
        val t0 = System.nanoTime()
        // What became of an early start, for the log line, at the time the line is written.
        val earlyNote = { early?.note(System.nanoTime()).orEmpty() }
        if (samples.isEmpty() || token.isCancelled) return Recognition("", 0)
        val audioMs = samples.size * 1000L / SAMPLE_RATE
        val window = EncoderWindow.frames(samples.size, floor)
        val segments = native.vadSpeechSegments(ensureGate(), samples, samples.size, config.threshold)
        val gateMs = msSince(t0)
        if (segments < 0) throw CzechSpeechException("Voice activity detection failed")
        if (segments == 0) {
            CsLog.i("$what: $audioMs ms of audio, no speech (gate $gateMs ms), recogniser not run${earlyNote()}")
            return Recognition("", msSince(t0))
        }
        if (token.isCancelled) return Recognition("", msSince(t0))   // cancelled during the gate: do not load the model
        val whisper = ensureWhisper()
        // One placement for Whisper and the language model of variant selection; put back before the text is filtered.
        val placement = placeThreads()
        val t1 = System.nanoTime()
        // One language (asked for, or the model's only one) is forced, as before: no detection pass. Several are
        // passed as a list and the recogniser picks among them by the thresholds of the rule, when there is one (else
        // the most probable); it reports the pick and the shares right after the call, on this thread, so nothing
        // else can touch the context in between.
        val forced = language ?: languages.singleOrNull()
        val allowed = forced ?: languages.joinToString(",")
        val thresholds = if (language == null) rule.thresholds(current, audioMs) else null
        var detected: String? = null
        var shares: FloatArray? = null
        var decided = ""
        val selection = config.variantSelection
        var variants = ""
        var text = ""
        var whisperMs = 0L
        try {
            if (!token.begin(whisper)) return Recognition("", msSince(t0))
            val bytes = try {
                if (selection == null) {
                    native.whisperTranscribe(whisper, token.generation, samples, samples.size, placement.threads, allowed, thresholds)
                } else {
                    native.whisperTranscribeNBest(
                        whisper, token.generation, samples, samples.size, placement.threads, selection.beamSize, allowed, thresholds,
                    )
                }
            } finally {
                token.end()
            }
            if (forced == null && bytes != null) {
                detected = native.whisperLanguage(whisper)
                if (thresholds != null) shares = native.whisperLanguageShares(whisper)
            }
            decided = forced ?: detected?.takeIf { it in languages }.orEmpty()
            whisperMs = msSince(t1)
            if (!token.isCancelled) placement.computed()        // the cores it ran on tell something
            if (token.isCancelled) {
                // null: whisper_full stopped at the abort callback; no text or no window: stopped before the encoder
                if (bytes == null || bytes.isEmpty() || (selection != null && !NBest.hasWindows(bytes))) {
                    abortedRecognitions++
                } else {
                    early?.abortStoppedNothing()        // whisper.cpp ran to its end all the same
                }
                bytes?.fill(0)
                CsLog.i(
                    "$what: $audioMs ms of audio, recogniser cancelled after $whisperMs ms, window $window frames, " +
                        placement.describe() + earlyNote(),
                )
                return Recognition("", msSince(t0))
            }
            if (bytes == null) throw CzechSpeechException("Recognition failed")
            text = try {
                if (selection == null) {
                    String(bytes, Charsets.UTF_8).trim()
                } else {
                    val choice = chooseVariant(bytes, selection, placement.threads, lmDecides = decided.isEmpty() || decided == rule.primary) { token.isCancelled }
                    variants = "${choice.variants} variant(s), language model ${choice.lmMs} ms in ${choice.lmRuns} run(s), " +
                        (if (choice.fallbackWindows > 0) "${choice.fallbackWindows} window(s) at a fallback temperature, " else "") +
                        if (choice.stopped) "language model stopped by a cancellation, " else ""
                    choice.text
                }
            } finally {
                bytes.fill(0)
            }
        } finally {
            placement.close()
        }
        // Silence and noise that slip through make Whisper emit punctuation only (",", "... ..."); drop it.
        val kept = text.any { it.isLetterOrDigit() }
        val decision = if (forced == null && rule.active) decisionNote(decided, current, audioMs, shares, thresholds) else ""
        val line = "$what: $audioMs ms of audio, $segments speech segment(s), gate $gateMs ms, recogniser $whisperMs ms, " +
            "window $window frames, " + languageNote(decided, forced != null, shares, rescue) +
            "${placement.describe()}, " + variants +
            (if (kept) "transcript kept" else "no letters or digits, dropped (${DroppedTranscript.describe(text, decided, forced != null)})") +
            earlyNote()
        if (kept || !deferDropped) CsLog.i(line)
        if (!kept) {
            return Recognition("", msSince(t0), recogniserMs = whisperMs, dropped = true, window = window, droppedLine = if (deferDropped) line else null, decided = decided, shares = shares, decision = decision)
        }
        val spoken = forced ?: detected?.takeIf { it in languages } ?: throw CzechSpeechException("Recognition failed")
        // Audio that ends mid-word (stop(), the length cap) can make greedy decoding loop; keep one copy (RepetitionGuard).
        val guarded = RepetitionGuard.clean(text)
        if (guarded.loops.isNotEmpty()) CsLog.i("$what: ${RepetitionGuard.describe(guarded.loops)}")
        return Recognition(guarded.text, msSince(t0), recogniserMs = whisperMs, window = window, language = spoken, decided = decided, shares = shares, rescued = rescue)
    }

    /**
     * The final recognition of [utterance] (whisper thread). A result without a letter or a digit is repeated on a longer
     * encoder window ([EmptyFinalRetry]) when [CzechSpeechConfig.retryEmptyFinals] is on, the endpointer saw clear speech
     * in the utterance, and it is not an early-start recognition; the first result with words is the final's, the audio is
     * the same array every time (the caller zeroes it after the last attempt). The returned [Recognition.recogniserMs] is the
     * first run's, which is what the guard's cost estimates are made of; the repeats are in [Recognition.retryMs].
     *
     * With a language rule ([LanguageRule.active], no language asked for) an empty result is first rescued: one run forced in
     * the language [LanguageRule.rescue] names, on the same window, whatever [CzechSpeechConfig.retryEmptyFinals] says and
     * without the second of clear speech the retries need. The retries on a longer window follow, in the language of the
     * last attempt (the rescue's, else the one the first run chose). Not for an early start, and not for the rest of a
     * streamed piece that wrote words.
     */
    private fun recognizeFinal(
        token: AbortToken,
        samples: FloatArray,
        utterance: Utterance,
        early: EarlyRun?,
        streaming: StreamingTracker?,
        plan: FinalPlan?,
        othersQueued: () -> Boolean,
        stopped: () -> Boolean,
        languageAsked: String?,
        languageSession: LanguageSession?,
    ): Recognition {
        val info = plan?.seam
        if (plan == null || info == null || plan.context == null || streaming == null || early != null) {
            return recognizeTail(token, samples, utterance, early, streaming, plan, othersQueued, stopped, languageAsked, languageSession, false)
        }
        // A piece that continues a cut inside speech: first one run over the audio of the end of the piece before and its own
        // ([SeamAnchor]); the words it repeats are left out. No anchor: the piece's own audio alone, as without the context.
        val contextMs = plan.context.size / StreamingTracker.SAMPLES_PER_MS
        val tailMs = samples.size / StreamingTracker.SAMPLES_PER_MS
        val joined = plan.contextRunSamples()
        val run = try {
            recognize(
                token, joined, "utterance (seam context)", null, config.minEncoderWindow, deferDropped = false,
                language = languageAsked ?: info.language.ifEmpty { null }, current = plan.current ?: languageSession?.current ?: rule.primary,
            )
        } finally {
            streaming.releaseContext(joined)
            streaming.releasePlanContext(plan)
        }
        if (token.isCancelled) return Recognition("", run.processingMs)
        val hyp = WordAgreement.words(run.text)
        val heldPending = info.heldWord != null
        // 1. a word was held back: the text should start with its new version, after the delivered words (anchor on those alone)
        val onDelivered = if (heldPending && run.text.isNotEmpty()) SeamAnchor.decide(hyp, info.delivered, info.expectedWithHeld) else null
        // The text after the replacement word must hold more when the piece has speech of its own: a run that stopped after the
        // seam writes the held word and nothing else (the piece's words would be lost); path 2 then falls back.
        val textAfterReplacement = onDelivered != null && (
            SeamAnchor.hasText(hyp.drop(onDelivered.prefix + 1)) || utterance.speechMs < EmptyFinalRetry.CLEAR_SPEECH_MS
            )
        if (onDelivered != null && onDelivered.ok && SeamAnchor.hasText(hyp.drop(onDelivered.prefix)) && textAfterReplacement) {
            CsLog.i(
                "utterance: seam context $contextMs ms + tail $tailMs ms, window ${run.window} frames, anchor ${onDelivered.matched} of " +
                    "${onDelivered.inBuffer} word(s) matched, new text after ${onDelivered.prefix} word(s), held word replaced",
            )
            streaming.seamRun(anchored = true, fallbackMs = 0, window = run.window)
            return Recognition(
                hyp.drop(onDelivered.prefix).joinToString(" "), run.processingMs, recogniserMs = run.recogniserMs, window = run.window,
                language = run.language, decided = run.decided, shares = run.shares, replacedHeld = true,
            )
        }
        // 2. no anchor on the delivered words (or nothing held): the anchor on all the words, the held one as it was; no new recognition
        val decision = if (run.text.isEmpty()) null else SeamAnchor.decide(hyp, info.allWords, info.expected)
        val reason = when {
            run.text.isEmpty() -> if (run.dropped) "empty" else "gate"
            decision == null || !decision.ok -> "no anchor"
            !SeamAnchor.hasText(hyp.drop(decision.prefix)) -> "empty rest"
            else -> null
        }
        if (reason == null && decision != null) {
            // the held word goes as it was, before the text that follows it
            if (heldPending) streaming.flushHeld("no anchor on the held word")
            CsLog.i(
                "utterance: seam context $contextMs ms + tail $tailMs ms, window ${run.window} frames, anchor ${decision.matched} of " +
                    "${decision.inBuffer} word(s) matched, new text after ${decision.prefix} word(s)" +
                    if (heldPending) ", held word kept" else "",
            )
            streaming.seamRun(anchored = true, fallbackMs = 0, window = run.window)
            return Recognition(
                hyp.drop(decision.prefix).joinToString(" "), run.processingMs, recogniserMs = run.recogniserMs, window = run.window,
                language = run.language, decided = run.decided, shares = run.shares,
            )
        }
        // 3. nothing to anchor on: the held word goes as it was, then the piece's own audio alone
        if (heldPending) streaming.flushHeld(reason ?: "no anchor")
        CsLog.i("utterance: seam fallback ($reason), tail-only run follows")
        val tail = recognizeTail(token, samples, utterance, null, streaming, plan, othersQueued, stopped, languageAsked, languageSession, true)
        // the context run is the final's own run (the guard's cost is of it); the run on the piece's audio alone is booked as a repeat
        val contextRan = run.recogniserMs > 0
        streaming.seamRun(anchored = false, fallbackMs = tail.recogniserMs + tail.retryMs, window = run.window, reason = reason)
        return Recognition(
            tail.text, run.processingMs + tail.processingMs, tail.join,
            recogniserMs = if (contextRan) run.recogniserMs else tail.recogniserMs,
            endMs = tail.endMs, dropped = tail.dropped, window = tail.window, droppedLine = tail.droppedLine,
            retryMs = if (contextRan) tail.recogniserMs + tail.retryMs else tail.retryMs, retryCutByStop = tail.retryCutByStop,
            language = tail.language, decided = tail.decided, shares = tail.shares, decision = tail.decision, rescued = tail.rescued,
            runSamples = if (contextRan) 0 else samples.size,
        )
    }

    /**
     * The final recognition of the piece's own audio [samples] and the retries of an empty result (see [recognizeFinal]).
     * [continuation]: the piece's first run was over the audio of the piece before ([SeamAnchor]) and found no anchor, so this
     * run is the second and the ladder of retries is the continuation's: no step on 512 frames, the full window only if it fits the lag.
     */
    private fun recognizeTail(
        token: AbortToken,
        samples: FloatArray,
        utterance: Utterance,
        early: EarlyRun?,
        streaming: StreamingTracker?,
        plan: FinalPlan?,
        othersQueued: () -> Boolean,
        stopped: () -> Boolean,
        languageAsked: String?,
        languageSession: LanguageSession?,
        continuation: Boolean,
    ): Recognition {
        // A piece of a streamed utterance that committed words has them in its language, and its final is forced in it
        // (no recognition, no thresholds, no rescue: the audio is only the unconfirmed end of the piece, often short).
        val language = plan?.language ?: languageAsked
        // a piece of a streamed utterance that already wrote words: only when words were still pending is the rest worth another run
        // (else it is the tail of the piece, a breath)
        val wroteWordsWithoutPending = plan != null && plan.hadChunks && !plan.hadPending
        val retryable = config.retryEmptyFinals && early == null && utterance.speechMs >= EmptyFinalRetry.CLEAR_SPEECH_MS && !wroteWordsWithoutPending
        // the language the session is in when this utterance starts: the one its thresholds are made from
        // (a streamed piece: the language the piece is in, which it took from the session at its first pass)
        val current = plan?.current ?: languageSession?.current ?: rule.primary
        val first = recognize(token, samples, "utterance", early, deferDropped = retryable, language = language, current = current)
        if (!first.dropped) return first
        // The rescue, before the retries: not for an early start (it may still be retracted) and not for the rest of a
        // piece that wrote words (its language is the piece's).
        val rescue = if (early == null && !(plan != null && plan.hadChunks)) {
            rescueRun(token, samples, "utterance", first, language, config.minEncoderWindow, deferDropped = retryable)
        } else {
            null
        }
        if (language == null && first.decision.isNotEmpty()) {
            val outcome = when {
                rescue != null -> "rescue in ${rescue.language.ifEmpty { rescue.decided }.ifEmpty { "?" }}"
                early != null -> "no rescue: early start"
                plan != null && plan.hadChunks -> "no rescue: piece wrote words"
                else -> "no rescue: " + rule.noRescueReason(first.decided, first.shares)
            }
            CsLog.i("utterance: language decision of the first run: ${first.decision}, $outcome")
        }
        if (rescue == null && !retryable) return first
        var last = first
        var retries = 0
        var processingMs = first.processingMs
        var retryMs = 0L
        var cutByStop = false
        // The retries change the window, not the language: they run in the language of the last attempt (with a rule,
        // where the first run chose one itself); without a rule they ask as the first run did.
        var retryLanguage = language
        var nothingMoreToTry = false
        if (rescue != null) {
            processingMs += rescue.processingMs
            retryMs += rescue.recogniserMs
            if (rescue.text.isNotEmpty()) {
                return Recognition(
                    rescue.text, processingMs, recogniserMs = first.recogniserMs, window = rescue.window, retryMs = retryMs,
                    language = rescue.language, decided = rescue.decided, shares = first.shares, rescued = true,
                )
            }
            if (rescue.dropped) last = rescue else nothingMoreToTry = true    // else cancelled, or the gate found no speech
            retryLanguage = rescue.decided.ifEmpty { null }
        } else if (rule.active && language == null && first.decided.isNotEmpty()) {
            retryLanguage = first.decided
        }
        while (retryable && !nothingMoreToTry && !token.isCancelled) {
            // asked again before every retry: what waits behind this recognition now, not what waited before the first run
            val floor = EmptyFinalRetry.nextFloor(
                first.window, last.window, samples.size, retries, streaming?.pendingFinals() ?: 0, othersQueued(),
                continuation = continuation, fullFitsLag = !continuation || streaming?.fullFitsLag(utterance.startMs) != false,
            ) ?: break
            if (stopped()) {
                cutByStop = true         // after stop() no retry of any kind starts
                break
            }
            retries++
            val handle = ensureWhisper()
            val run = try {
                native.whisperSetMinAudioCtx(handle, floor)
                recognize(token, samples, "utterance", null, floor = floor, deferDropped = true, language = retryLanguage, current = current)
            } finally {
                native.whisperSetMinAudioCtx(handle, config.minEncoderWindow)
            }
            processingMs += run.processingMs
            retryMs += run.recogniserMs
            if (run.text.isNotEmpty()) {
                CsLog.i("utterance: empty result, retry with window ${run.window}: rescued")
                return Recognition(
                    run.text, processingMs, recogniserMs = first.recogniserMs, window = run.window, retryMs = retryMs,
                    language = run.language, decided = run.decided, shares = run.shares, rescued = rescue != null,
                )
            }
            if (!run.dropped) break         // cancelled, or the gate found no speech: nothing more to try
            CsLog.i("utterance: empty result, retry with window ${run.window}: still empty")
            last = run
        }
        last.droppedLine?.let { CsLog.i(it) }
        return Recognition(
            "", processingMs, recogniserMs = first.recogniserMs, dropped = last.dropped, window = last.window, retryMs = retryMs,
            retryCutByStop = cutByStop, decided = last.decided, shares = last.shares,
        )
    }

    /**
     * The rescue after [first], a run of [language] (null: chosen by the rule) that came out empty (whisper thread): one
     * run forced in the language [LanguageRule.rescue] names, on the encoder window [floor] ([first]'s). Null when none
     * is to be tried: a language was asked for, there is no rule, no language other than the one that ran has at least half of the shares.
     * A run that gave text is marked [Recognition.rescued]. Logs the languages and the outcome, never the text.
     */
    private fun rescueRun(
        token: AbortToken,
        samples: FloatArray,
        what: String,
        first: Recognition,
        language: String?,
        floor: Int,
        deferDropped: Boolean,
    ): Recognition? {
        if (language != null || token.isCancelled || !first.dropped) return null
        val to = rule.rescue(first.decided, first.shares) ?: return null
        val run = recognize(token, samples, what, null, floor = floor, deferDropped = deferDropped, language = to, rescue = true)
        CsLog.i("$what: empty result in ${first.decided}, rescue in $to: ${if (run.text.isNotEmpty()) "rescued" else "still empty"}")
        return run
    }

    /**
     * The threads and cores of one recognition or run of the language model ([ThreadPlacement]), on the whisper thread,
     * which it pins until [ThreadPlacement.close]. The cores and their capacities go to the log with the first one of
     * each session (and of the engine, for [transcribe] alone).
     */
    private fun placeThreads(): ThreadPlacement {
        if (!cpusLogged) {
            cpusLogged = true
            val limit = if (config.autoThreads) "at most $threadCount threads, placed per recognition"
            else "$threadCount threads, automatic placement off"
            CsLog.i("cpus: ${topology.describe()}, $limit")
        }
        return ThreadPlacement.place(native, config.autoThreads, topology, threadCount, pinGuard)
    }

    private class Choice(
        val text: String,
        val variants: Int,
        val lmMs: Long,
        val lmRuns: Int,
        val fallbackWindows: Int,
        val stopped: Boolean,
    )

    /**
     * Variant selection (whisper thread): in each window, the variant with the best combined score of Whisper and the
     * language model ([VariantChooser]). Whisper's own choice (the first candidate) stays in a window decoded at a
     * fallback temperature, when it has no letter or digit (no speech), where the variants' words differ only in case
     * and punctuation, and when the language model fails. The windows' choices are joined as Whisper joins its segments.
     * The language model runs on [threads] threads, in the recognition's placement. A run of it (one window's variants)
     * cannot be interrupted; once [cancelled] (the recognition's result is no longer wanted: an early recognition the
     * speech ran over, a closed engine) no further window is scored, and the text is cut there, for nobody.
     */
    private fun chooseVariant(
        bytes: ByteArray,
        selection: VariantSelection,
        threads: Int,
        /** The text is in the first language, the only one the language model knows; else Whisper's own choice stays. */
        lmDecides: Boolean,
        cancelled: () -> Boolean,
    ): Choice {
        val lambda = selection.languageModelWeight.toString().toDouble()      // 0.3f as 0.3, as measured
        val beta = selection.lengthBonus.toString().toDouble()
        val lm = if (lambda > 0) checkNotNull(languageModel) { "language model not loaded" } else null
        val out = StringBuilder()
        var variants = 0
        var lmNanos = 0L
        var lmRuns = 0
        var fallbackWindows = 0
        var stopped = false
        for (window in NBest.parse(bytes)) {
            if (cancelled()) {
                stopped = true
                break
            }
            val candidates = VariantChooser.candidates(window)
            if (candidates.isEmpty()) continue
            variants += candidates.size
            var chosen = 0
            if (VariantChooser.atFallbackTemperature(window)) {
                fallbackWindows++
            } else if (!lmDecides) {
                // another language than the first: the language model and the length bonus are made for the first
            } else if (VariantChooser.whisperChoseNoText(candidates)) {
                // no speech, says Whisper: a beam it turned down must not override that
            } else if (!VariantChooser.needsLanguageModel(candidates)) {
                // the same words in every variant: Whisper's choice
            } else if (lm == null) {
                chosen = VariantChooser.choose(candidates, DoubleArray(candidates.size), 0.0, beta)
            } else {
                val t0 = System.nanoTime()
                val scores = lm.score(candidates.map { it.text }, threads)
                lmNanos += System.nanoTime() - t0
                lmRuns++
                if (scores != null) {
                    chosen = VariantChooser.choose(candidates, scores, lambda, beta)
                } else {
                    CsLog.w("language model failed; Whisper's own choice is kept")
                }
            }
            out.append(candidates[chosen].raw)
        }
        return Choice(out.toString().trim(), variants, lmNanos / 1_000_000, lmRuns, fallbackWindows, stopped)
    }

    /**
     * Whisper and the filters for a preview: one greedy pass without the temperature fallback
     * ([NativeApi.whisperPreview]), and without the speech gate, since Silero already found speech in the utterance
     * and its final text still passes the gate. Whisper thread only, with the session's lease held. It never loads
     * the model: without one (the session's load failed) there is nothing to show. The text is "" when there is
     * nothing to show; the time is the recogniser's, 0 when it did not run to its end.
     */
    private fun recognizePreview(token: AbortToken, run: PreviewRun, language: String?, languageSession: LanguageSession?): Recognition {
        val samples = run.samples
        val audioMs = samples.size * 1000L / SAMPLE_RATE
        val window = EncoderWindow.frames(samples.size, config.minEncoderWindow)
        val whisper = whisperHandle
        if (whisper == 0L) return Recognition("", 0)
        val forced = language ?: languages.singleOrNull()
        val allowed = forced ?: languages.joinToString(",")
        // the session's language as it is now; a preview never changes it
        val thresholds = if (language == null) rule.thresholds(languageSession?.current ?: rule.primary, audioMs) else null
        val placement = placeThreads()
        val t0 = System.nanoTime()
        val bytes = try {
            if (!token.begin(whisper)) return Recognition("", 0)
            previewRecognitions++
            previewAudioMs += audioMs
            try {
                native.whisperPreview(whisper, token.generation, samples, samples.size, placement.threads, allowed, thresholds)
                    .also { if (!token.isCancelled) placement.computed() }
            } finally {
                token.end()
            }
        } finally {
            placement.close()
        }
        val whisperMs = msSince(t0)
        if (token.isCancelled) {
            if (bytes == null || bytes.isEmpty()) abortedPreviews++
            bytes?.fill(0)
            // counted when the abort stopped whisper.cpp at work, not when it ran to its end all the same
            val stopped = if (bytes == null || bytes.isEmpty()) run.msSinceCancel(System.nanoTime()) else -1
            if (stopped >= 0) abortStopped(stopped)
            CsLog.i(
                "preview: $audioMs ms of audio, recogniser cancelled after $whisperMs ms, window $window frames, " +
                    placement.describe() +
                    if (stopped >= 0) ", stopped $stopped ms after the abort" else "",
            )
            return Recognition("", 0)
        }
        if (bytes == null) throw CzechSpeechException("Preview recognition failed")
        val text = try {
            String(bytes, Charsets.UTF_8).trim()
        } finally {
            bytes.fill(0)
        }
        CsLog.i("preview: $audioMs ms of audio, recogniser $whisperMs ms, window $window frames, ${placement.describe()}")
        if (text.none { it.isLetterOrDigit() }) return Recognition("", whisperMs)
        return Recognition(RepetitionGuard.clean(text).text, whisperMs)
    }

    /**
     * One pass of streaming transcription over [pass]'s audio (whisper thread only, with the session's lease held): the
     * same greedy decoding as a preview ([recognizePreview]), without the speech gate and without loading the model.
     * Null when it did not run to its end (cancelled, or there is no model); the text is cleaned ([PassText]) and its
     * time is at least 1 ms, so that a pass that ran is told from one that did not.
     */
    private fun recognizePass(
        token: AbortToken,
        pass: StreamingPass,
        head: StringBuilder,
        language: String?,
        streaming: StreamingTracker,
        languageSession: LanguageSession?,
    ): PassResult? {
        val samples = pass.samples
        val window = EncoderWindow.frames(samples.size, config.minEncoderWindow)
        val audioMs = samples.size * 1000L / SAMPLE_RATE
        val whisper = whisperHandle
        if (whisper == 0L) return null
        val forced = language ?: languages.singleOrNull()
        val allowed = forced ?: languages.joinToString(",")
        // With a language rule the thresholds come from the language of the piece (the session's at its first pass) and the
        // length of this pass; a pass too short to change the language of its piece does not decide ([PassResult.decisive]).
        val pieceLanguage = if (language == null && languageSession != null) streaming.languageFor(pass) else null
        val thresholds = pieceLanguage?.let { rule.thresholds(it, audioMs) }
        val decisive = pieceLanguage?.let { rule.decisive(it, audioMs) } ?: true
        val placement = placeThreads()
        val t0 = System.nanoTime()
        val bytes = try {
            if (!token.begin(whisper)) return null
            previewRecognitions++
            previewAudioMs += audioMs
            try {
                native.whisperPreview(whisper, token.generation, samples, samples.size, placement.threads, allowed, thresholds)
                    .also { if (!token.isCancelled) placement.computed() }
            } finally {
                token.end()
            }
        } finally {
            placement.close()
        }
        val detected = if (forced == null && bytes != null) native.whisperLanguage(whisper) else null
        val whisperMs = msSince(t0)
        if (token.isCancelled) {
            if (bytes == null || bytes.isEmpty()) abortedPreviews++
            bytes?.fill(0)
            val stopped = if (bytes == null || bytes.isEmpty()) pass.msSinceCancel(System.nanoTime()) else -1
            if (stopped >= 0) abortStopped(stopped)
            CsLog.i(
                "pass: $audioMs ms of audio from ${pass.startMs} ms, recogniser cancelled after $whisperMs ms, window $window frames, " +
                    placement.describe() + if (stopped >= 0) ", stopped $stopped ms after the abort" else "",
            )
            // one that ran to its end before the cancel still tells how fast the recogniser is (its words are not wanted)
            return if (bytes != null && bytes.isNotEmpty()) PassResult("", maxOf(1L, whisperMs), forced ?: detected.orEmpty(), decisive) else null
        }
        if (bytes == null) throw CzechSpeechException("Streaming pass recognition failed")
        val text = try {
            String(bytes, Charsets.UTF_8)
        } finally {
            bytes.fill(0)
        }
        val spoken = forced ?: detected?.takeIf { it in languages } ?: throw CzechSpeechException("Streaming pass recognition failed")
        head.append("pass: $audioMs ms of audio from ${pass.startMs} ms, recogniser $whisperMs ms, window $window frames, ")
        // only with a rule: the line of a model without one is as it was
        if (pieceLanguage != null) head.append("language $spoken${if (decisive) "" else " (too short to decide)"}, ")
        head.append(placement.describe())
        return PassResult(PassText.clean(text), maxOf(1L, whisperMs), spoken, decisive)
    }

    /** The end of a pass's log line: what the tracker made of it (counts only). */
    private fun passOutcome(absorbed: Absorbed?, pass: StreamingPass): String {
        if (absorbed == null) return "; dropped, ${pass.dropReason ?: "its result is not wanted"}"
        val o = absorbed.outcome
        val cut = o.cut?.let { "; buffer cut at ${it.atMs} ms (${it.bufferBeforeMs} -> ${it.bufferAfterMs} ms)" }.orEmpty()
        return "; ${o.committed.size} word(s) committed, ${o.pending.size} pending$cut"
    }

    /** The log line of a streamed piece, after its final recognition: counts only. */
    private fun logPiece(utterance: Utterance, s: PieceSummary) {
        CsLog.i(
            "streaming: utterance of ${utterance.endMs - utterance.startMs} ms: ${s.passes} passes (${s.abortedPasses} aborted), " +
                "${s.wordsWhileSpeaking} words while speaking, ${s.wordsAfter} after it ended, ${s.cuts} cuts" +
                if (s.forcedCut) ", forced cut" else "",
        )
    }

    /** The log line of a streaming session and the hook, once at its teardown: counts and times, never text. */
    private fun logStreamingEnded(streaming: StreamingTracker) {
        val s = streaming.stats()
        val delays = s.chunkDelaysMs.sorted()
        val median = if (delays.isEmpty()) 0 else delays[delays.size / 2]
        CsLog.i(
            "streaming: ${s.passes} passes (${s.abortedPasses} aborted, ${s.passRecogniserMs} ms of recogniser time), " +
                "${s.transcripts} transcripts, ${s.wordsWhileSpeaking} + ${s.wordsAfter} words, ${s.cuts} cuts, " +
                "${s.forcedCuts} forced cuts, passes paused ${s.pauses} times, buffer up to ${s.maxBufferMs} ms, " +
                "median piece ${s.pieceMedianMs} ms, " +
                "text after its audio median $median ms, max ${delays.lastOrNull() ?: 0} ms, " +
                "${s.finalSeams} seam(s) decided by a final, ${s.emptyFinals} empty final(s) " +
                "(${s.lostPendingWords} pending word(s) lost), ${s.stalePasses} stale pass(es), ${s.errors} error(s)" +
                if (config.streaming?.cutSeam == CutSeam.NONE) "" else {
                    ", seams: ${s.seamsWithContext} with context (${s.seamsAnchored} anchored, ${s.seamsFallback} fallback${if (s.seamFallbackBy.isEmpty()) "" else " (" + s.seamFallbackBy.entries.joinToString(", ") { "${it.key} ${it.value}" } + ")"}, " +
                        "${s.seamFallbackMs} ms of fallback runs, windows above ${config.minEncoderWindow}: ${s.seamWindowsOverFloor}), " +
                        "${s.seamsUnused} context(s) never used" +
                        if (config.streaming?.cutSeam != CutSeam.CONTEXT_AND_HELD_WORD) "" else {
                            val delays = s.heldDelaysMs.sorted()
                            ", held ${s.heldReplaced + s.heldKept}: replaced ${s.heldReplaced}, kept ${s.heldKept}" +
                                (if (s.heldKept > 0) " (" + s.heldKeptBy.entries.joinToString(", ") { "${it.key} ${it.value}" } + ")" else "") +
                                ", held-word delay median ${if (delays.isEmpty()) 0 else delays[delays.size / 2]} ms max ${delays.lastOrNull() ?: 0} ms"
                        }
                },
        )
        hooks.streamingEnded(s)
    }

    private fun acquire(lease: Lease) {
        if (lease.held) return
        lease.held = true
        users++
    }

    private fun release(lease: Lease) {
        if (!lease.held) return
        lease.held = false
        users--
        if (users == 0) freeAll()
    }

    private fun ensureGate(): Long {
        if (gateHandle == 0L) {
            gateHandle = native.vadInit(vadBytes, 1)
            if (gateHandle == 0L) throw CzechSpeechException("Could not load the Silero VAD model")
        }
        return gateHandle
    }

    private fun ensureWhisper(): Long {
        if (whisperHandle != 0L) return whisperHandle
        if (!modelFile.isFile) throw CzechSpeechException("Model file not found: ${modelFile.path}")
        if (selectionNeedsLanguageModel && languageModel == null) loadLanguageModel()
        val t0 = System.nanoTime()
        val handle = native.whisperInit(modelFile.absolutePath)
        if (handle == 0L) throw CzechSpeechException("Could not load the model from ${modelFile.path}")
        whisperHandle = handle
        // The floor of the encoder window belongs to the model: set once on its context, for every recognition on it.
        native.whisperSetMinAudioCtx(handle, config.minEncoderWindow)
        isModelLoaded = true
        modelLoads++
        if (keepModelLoaded) acquire(kept)       // outlives the session or call that loaded it
        val threads = if (config.autoThreads) "at most $threadCount threads" else "$threadCount threads"
        CsLog.i(
            "model loaded in ${msSince(t0)} ms (${modelFile.length() / 1_000_000} MB, $threads, " +
                "encoder window at least ${config.minEncoderWindow} frames)",
        )
        if (!systemInfoLogged) {
            systemInfoLogged = true
            CsLog.i("native: ${NativeLoader.loadedLibrary ?: "-"}; ${native.systemInfo().trim()}")
        }
        return handle
    }

    /** Variant selection weighs the language model in: then it loads with the model, before the first utterance. */
    private val selectionNeedsLanguageModel = config.variantSelection.let { it != null && it.languageModelWeight > 0f }

    /**
     * Loads the language model of variant selection (with the model, [ensureWhisper]) or of sentence joining (after a
     * session's first utterance, [loadLanguageModelForJoining]): one model, the same file for both (CzechSpeechConfig
     * checks it). Variant selection cannot do without it, so a failed load fails there; sentence joining goes on
     * without it, and transcripts come as Whisper wrote them.
     */
    private fun loadLanguageModel() {
        val selection = config.variantSelection?.takeIf { it.languageModelWeight > 0f }
        val joining = config.sentenceJoining
        val file = selection?.let { checkNotNull(it.languageModelFile) { "no language model file" } }   // see VariantSelection
            ?: checkNotNull(joining?.languageModelFile) { "no language model to load" }
        val t = System.nanoTime()
        try {
            languageModel = LanguageModel.load(native, file)
        } catch (e: Throwable) {
            if (e is OutOfMemoryError) throw e      // an OutOfMemoryError ends the session as it always did (see StreamingSessionTest)
            if (selection != null) throw e
            CsLog.w("language model not loaded, sentences are not joined", e)
            return
        }
        val uses = listOfNotNull(
            selection?.let { "variant selection, beam size ${it.beamSize}" },
            joining?.let { "sentence joining" },
        )
        CsLog.i("language model loaded in ${msSince(t)} ms (${file.length() / 1_000_000} MB) for ${uses.joinToString(" and ")}")
    }

    private fun freeAll() {
        if (whisperHandle != 0L) {
            native.whisperFree(whisperHandle)
            whisperHandle = 0L
            isModelLoaded = false
            CsLog.i("model freed")
        }
        languageModel?.let {
            languageModel = null
            it.free()
            CsLog.i("language model freed")
        }
        if (gateHandle != 0L) {
            native.vadFree(gateHandle)
            gateHandle = 0L
        }
    }

    /**
     * Runs [block] on the whisper thread. Cancellation of the caller aborts a running transcription. With
     * [failOnClose], the call is registered so that [close] fails it the same way, and it fails at once when the
     * engine is already closed. Either way the caller resumes at once; [block] sees [AbortToken.isCancelled] and the
     * whisper thread finishes it (releasing what it holds) in the background.
     */
    private suspend fun <T> onWhisperThread(failOnClose: Boolean = false, block: (AbortToken) -> T): T =
        suspendCancellableCoroutine { cont ->
            val token = AbortToken(native)
            cont.invokeOnCancellation {
                token.cancel()
                if (failOnClose) callEnded(cont, onTheWhisperThread = false)
            }
            if (failOnClose) {
                hooks.callStarting()
                val registered = synchronized(lock) {
                    if (closed) return@synchronized false
                    calls.add(cont)
                    releaseWhenIdle = false     // a release still waiting does not apply to a call started after it
                    true
                }
                if (!registered) {
                    cont.cancel(IllegalStateException(CLOSED))
                    return@suspendCancellableCoroutine
                }
                hooks.callRegistered()
            }
            try {
                whisperExecutor.execute {
                    val result = runCatching { block(token) }
                    try {
                        if (failOnClose) callEnded(cont, onTheWhisperThread = true)
                    } finally {
                        cont.resumeWith(result)
                    }
                }
            } catch (e: RejectedExecutionException) {
                if (failOnClose) callEnded(cont, onTheWhisperThread = false)
                cont.resumeWithException(CzechSpeechException("CzechSpeech worker thread unavailable", e))
            }
        }

    /**
     * Unregisters a [transcribe] call that has ended or was cancelled. When it was the last thing a [releaseModel]
     * waited for, drops [kept]: at once [onTheWhisperThread], where nothing else can run first, so the call returns
     * with the model freed; otherwise queued there in the same step as the unregistration.
     */
    private fun callEnded(call: CancellableContinuation<*>, onTheWhisperThread: Boolean) {
        val due = synchronized(lock) {
            calls.remove(call)
            takeDueRelease().also { if (it && !onTheWhisperThread) releaseLater(kept) }
        }
        if (due && onTheWhisperThread) release(kept)
    }

    /** Releases [lease] on the whisper thread after whatever it is doing now, without waiting. */
    private fun releaseLater(lease: Lease) {
        try {
            whisperExecutor.execute { release(lease) }
        } catch (e: RejectedExecutionException) {
            CsLog.w("cannot free the model: worker thread unavailable", e)
        }
    }

    /**
     * A hold on the model ([held]; whisper thread only) and, for a [listen] session, how far its model load has got,
     * so that a session that ends early neither loads the model nor waits for a load (the other members: any thread).
     */
    private class Lease {
        var held = false
        private var load = LOAD_NOT_STARTED

        /** On the whisper thread, before the session's load; false once the session has ended. */
        @Synchronized
        fun startLoad(): Boolean {
            if (load != LOAD_NOT_STARTED) return false
            load = LOAD_RUNNING
            return true
        }

        @Synchronized
        fun loadDone() {
            if (load == LOAD_RUNNING) load = LOAD_DONE
        }

        /** The session has ended: returns how far its load had got. A load that had not started never will. */
        @Synchronized
        fun endSession(): Int = load.also { load = SESSION_ENDED }

        companion object {
            const val LOAD_NOT_STARTED = 0
            const val LOAD_RUNNING = 1
            const val LOAD_DONE = 2
            const val SESSION_ENDED = 3
        }
    }

    /**
     * [recogniserMs]: the time whisper.cpp took when it ran to its end, else 0. [endMs]: where the transcript ends
     * when that is not the utterance's end (the last chunk of a streamed utterance), else -1.
     */
    private class Recognition(
        val text: String,
        val processingMs: Long,
        val join: SentenceJoin? = null,
        val recogniserMs: Long = 0,
        val endMs: Long = -1,
        /** The model ran to its end and returned nothing with a letter or a digit (the text is dropped). */
        val dropped: Boolean = false,
        /** The encoder window of the run that gave this result, in frames. */
        val window: Int = 0,
        /** The log line of a dropped run, kept back for the caller while a retry may still follow (no text in it). */
        val droppedLine: String? = null,
        /** The recogniser ms of the repeats on a longer window ([EmptyFinalRetry]); [recogniserMs] is the first run's. */
        val retryMs: Long = 0,
        /** A repeat that the rules allowed was not started because stop() had been called: the stretch was not tried to the end. */
        val retryCutByStop: Boolean = false,
        /** The language of the text: the one asked for, the model's only one, or the one the recogniser chose among several ("" when there is no text). */
        val language: String = "",
        /** The language the recogniser decoded in, also when its text was dropped ("" when it did not run to its end or is not known). */
        val decided: String = "",
        /** The shares of the languages of the list the decision was made from, when the rule decided (else null). */
        val shares: FloatArray? = null,
        /** What the rule decided from, for the log (see [decisionNote]); "" when the run was forced or had no rule. */
        val decision: String = "",
        /** The text is the result of a rescue in another language: it does not change the language of the session. */
        val rescued: Boolean = false,
        /** With a context run that did not run to its end: the size of the run [recogniserMs] is of, in samples (0: the plan's first run). */
        val runSamples: Int = 0,
        /** The text starts with the new version of the word held back from the piece before: it replaces that word ([StreamingTracker.absorbFinal]). */
        val replacedHeld: Boolean = false,
    )

    private companion object {
        const val CLOSED = "CzechSpeech is closed"

        fun msSince(t0: Long): Long = (System.nanoTime() - t0) / 1_000_000

        /**
         * [transcribe] input: at most [MAX_TRANSCRIBE_SAMPLES], every sample finite and of magnitude at most
         * [MAX_SAMPLE_MAGNITUDE]. Rejected rather than sanitised: a single NaN can make the VAD report speech in the
         * next frame, and 16-bit PCM passed as floats gives the VAD no speech at all, so the call would return ""
         * with no hint. Both are bugs in the caller's audio path, which an exception points at.
         */
        fun requireTranscribable(samples: FloatArray) {
            require(samples.size <= MAX_TRANSCRIBE_SAMPLES) {
                "samples hold ${samples.size / SAMPLE_RATE} s of audio; transcribe() takes at most " +
                    "${MAX_TRANSCRIBE_MS / 1000} s ($MAX_TRANSCRIBE_SAMPLES samples at 16 kHz), so split longer audio"
            }
            for (i in samples.indices) {
                val x = samples[i]
                if (abs(x) <= MAX_SAMPLE_MAGNITUDE) continue     // false for NaN too
                throw IllegalArgumentException(
                    if (!x.isFinite()) {
                        "samples[$i] is $x; audio samples must be finite numbers"
                    } else if (abs(x) >= PCM16_HINT_MAGNITUDE) {
                        "samples[$i] is $x; transcribe() takes floats in [-1, 1], so 16-bit PCM must be scaled " +
                            "first (divide by 32768)"
                    } else {
                        "samples[$i] is $x; transcribe() takes floats in [-1, 1] (up to ±2 is tolerated): clip or " +
                            "normalise the audio"
                    },
                )
            }
        }
    }
}

/**
 * Cancellation of one recognition. [cancel] may run on any thread; the lock orders it against [begin]/[end], so an
 * abort reaches exactly the native call it belongs to and never a freed context.
 */
internal class AbortToken(private val native: NativeApi) {
    val generation: Int = generations.incrementAndGet()
    private var cancelled = false
    private var running = 0L

    val isCancelled: Boolean
        @Synchronized get() = cancelled

    @Synchronized
    fun begin(whisperHandle: Long): Boolean {
        if (cancelled) return false
        running = whisperHandle
        return true
    }

    @Synchronized
    fun end() {
        running = 0L
    }

    /** True when a native call was running and got the abort; false when there was nothing to stop. */
    @Synchronized
    fun cancel(): Boolean {
        cancelled = true
        if (running == 0L) return false
        native.whisperAbort(running, generation)
        return true
    }

    private companion object {
        val generations = AtomicInteger()
    }
}

/**
 * The utterances of a session waiting for the recogniser, bounded by the audio they hold: at most [maxSamples], or a
 * single utterance of any length. One producer (capture) and one consumer (recognition). [offer] never waits for
 * room: a live microphone cannot be paused.
 */
internal class UtteranceQueue(private val maxSamples: Long) {
    enum class Offer { QUEUED, FULL, CLOSED }

    // An utterance handed to a consumer that is cancelled before it runs is neither processed nor left for the
    // drain: the channel zeroes it here. A plain fill, so zeroing it twice does no harm.
    private val channel = Channel<Utterance>(Channel.UNLIMITED, onUndeliveredElement = { it.samples.fill(0f) })
    private val queuedSamples = AtomicLong()

    /** Utterances in the channel, those no longer counted as waiting audio ([forget]) included. */
    private val items = AtomicInteger()

    /** Of those, the ones that are none any more ([forget]): the consumer drops them without recognising anything. */
    private val forgotten = AtomicInteger()
    private val closed = CompletableDeferred<Unit>()

    /** Utterances waiting for the recogniser, those that are none any more ([forget]) not counted. */
    val waiting: Int get() = items.get() - forgotten.get()

    /** Audio waiting now, in ms. */
    val queuedMs: Long get() = queuedSamples.get() * 1000 / SAMPLE_RATE

    /** The most audio that ever waited, in ms. */
    @Volatile
    var peakMs: Long = 0
        private set

    /** Producer only. Unless the result is [Offer.QUEUED], the caller still owns [utterance] (and zeroes it). */
    fun offer(utterance: Utterance): Offer {
        val n = utterance.samples.size.toLong()
        // Only the producer adds, so nothing can fill the queue between this check and the add below. An empty queue
        // takes any utterance: the bound limits a backlog, and skipping speech while nothing waits would gain nothing.
        val waiting = queuedSamples.get()
        if (waiting > 0 && waiting + n > maxSamples) return Offer.FULL
        utterance.queued.set(true)      // before the consumer can take it
        val queued = queuedSamples.addAndGet(n)
        items.incrementAndGet()
        if (!channel.trySend(utterance).isSuccess) {
            items.decrementAndGet()
            uncount(utterance)
            return Offer.CLOSED
        }
        val ms = queued * 1000 / SAMPLE_RATE
        if (ms > peakMs) peakMs = ms
        return Offer.QUEUED
    }

    /**
     * Producer only: offers an utterance whose end is not certain yet ([EndpointerEvent.UtteranceEarly]), only when
     * nothing waits in the queue to be recognised: behind a backlog an early start gains nothing. Utterances that are
     * none any more ([forget]) do not count: the consumer drops them at once, so the recognition of the next pause
     * starts right behind the one that gave way to it. False: not taken, the caller still owns it.
     */
    fun offerEarly(utterance: Utterance): Boolean =
        items.get() - forgotten.get() <= 0 && offer(utterance) == Offer.QUEUED

    /**
     * Producer: [utterance], offered early, is none after all. It no longer counts as waiting audio, so it never makes
     * the queue refuse real speech; the consumer still gets it, and drops it.
     */
    fun forget(utterance: Utterance) {
        if (uncount(utterance)) forgotten.incrementAndGet()
    }

    /** True when [utterance] was counted as waiting audio until now. */
    private fun uncount(utterance: Utterance): Boolean {
        val counted = utterance.queued.compareAndSet(true, false)
        if (counted) queuedSamples.addAndGet(-utterance.samples.size.toLong())
        return counted
    }

    /** Consumer: [utterance] has left the channel. One that was forgotten meanwhile is no longer waiting to be dropped. */
    private fun taken(utterance: Utterance) {
        // forgotten first: between the two steps the queue looks fuller, never emptier, than it is
        if (!uncount(utterance)) forgotten.decrementAndGet()
        items.decrementAndGet()
    }

    /** Consumer: hands each utterance to [action], in order, until the queue is closed and empty. */
    suspend fun forEach(action: suspend (Utterance) -> Unit) {
        for (utterance in channel) {
            taken(utterance)
            action(utterance)
        }
    }

    /** An utterance still waiting, or null; drains the queue after [close]. */
    fun poll(): Utterance? = channel.tryReceive().getOrNull()?.also { taken(it) }

    fun close() {
        channel.close()
        closed.complete(Unit)
    }

    /** Suspends until [close]: the producer has offered its last utterance. */
    suspend fun awaitClosed() = closed.await()

    /** Closed and nothing waiting: no utterance will come any more. */
    val isFinished: Boolean get() = closed.isCompleted && queuedSamples.get() == 0L
}

internal class NamedThreadFactory(private val name: String) : ThreadFactory {
    override fun newThread(r: Runnable): Thread = Thread(r, name).apply { isDaemon = true }
}
