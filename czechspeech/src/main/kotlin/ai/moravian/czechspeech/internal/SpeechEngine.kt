package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.Transcript
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
 * [Endpointer] -> queue -> speech gate -> Whisper -> filter -> [RepetitionGuard].
 *
 * Threads:
 * - capture: one thread per [listen] session reads audio, runs the streaming VAD and the endpointer;
 * - whisper: one thread per engine owns the whisper context and the gate VAD (neither is thread-safe) and runs
 *   every recognition, including [transcribe]. It is the only thread that touches [users], [kept], [whisperHandle]
 *   and [gateHandle], which makes it the lock for the model lifecycle; the model is only ever freed there. It exits
 *   when idle.
 *
 * Model lifecycle: the model is loaded when a [listen] session starts (in parallel with opening the microphone)
 * and freed when the session ends; [transcribe] loads it only after the speech gate passes and frees it afterwards,
 * unless a session holds it. An idle ~0.5 GB model would be pushed to zram by Android, making the next
 * transcription take up to a minute. A session or call cancelled before its load starts never loads the model. A
 * cancelled session completes at once and never waits for the whisper thread: work running there that whisper.cpp
 * cannot interrupt (a load, an encoder pass, another caller's transcription) finishes, and the model is freed right
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
    /** Most audio that may wait for the recogniser in a session's queue; smaller in tests. */
    private val maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS,
    private val hooks: Hooks = Hooks.NONE,
) : Closeable {
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

        companion object {
            val NONE: Hooks = object : Hooks {}
        }
    }

    private val vadBytes: ByteArray by lazy(vadModel)
    private val threadCount: Int by lazy(threads)

    private val whisperExecutor = ThreadPoolExecutor(
        1, 1, 5L, TimeUnit.SECONDS, LinkedBlockingQueue(), NamedThreadFactory("czechspeech-whisper"),
    ).apply { allowCoreThreadTimeOut(true) }

    // Owned by the whisper thread.
    private var users = 0
    private var whisperHandle = 0L
    private var gateHandle = 0L
    private var systemInfoLogged = false

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
    }

    /** [language] must be one of the model's languages; null is allowed. */
    private fun checkLanguage(language: String?) {
        if (language != null) requireLanguageCode(language)
        require(language == null || language in languages) {
            "language \"$language\" is not one of this model's languages: ${languages.joinToString()}"
        }
    }

    /** Throws IllegalArgumentException at once for a [language] outside the model's languages. */
    fun listen(language: String? = null): Flow<SpeechEvent> {
        checkLanguage(language)
        return listenFlow(language)
    }

    private fun listenFlow(language: String?): Flow<SpeechEvent> = channelFlow {
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
        val captureExecutor = Executors.newSingleThreadExecutor(NamedThreadFactory("czechspeech-capture"))
        val captureDispatcher = captureExecutor.asCoroutineDispatcher()
        CsLog.i("listen: session started")
        try {
            val source = audioSourceFactory()
            try {
                runSession(source, lease, utterances, captureDispatcher, stop, language)
            } finally {
                // Capture closes the source as soon as it ends; this covers a session that ended before capture
                // started (a load that failed at once, say). AudioSource.close() is idempotent, like any Closeable.
                closeQuietly(source)
            }
        } finally {
            try {
                utterances.close()
                while (true) discard(utterances.poll() ?: break)
                if (utterances.peakMs > peakQueuedAudioMs) peakQueuedAudioMs = utterances.peakMs
                // A cancelled session never waits for the whisper thread: it may be busy with work that cannot be
                // interrupted (a load, an encoder pass, another caller's transcribe()), and the model is freed right
                // after it. A session that ends normally (stop(), the end of the audio) has freed it when it completes.
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
        checkLanguage(language)
        requireTranscribable(samples)
        if (samples.isEmpty()) return ""
        val lease = Lease()
        return onWhisperThread(failOnClose = true) { token ->
            if (token.isCancelled) return@onWhisperThread ""
            acquire(lease)
            try {
                recognize(token, samples, "transcribe", language).text
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
     * gate, an encoder pass), and the freeing of the model. Returns at once when nothing was running.
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
     * end of the frame being read. Native work already running (a model load, the speech gate, an encoder pass)
     * cannot be interrupted: it finishes in the background and the model is freed right after, a model kept by
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

    /** The two halves of a session: model load then recognition, and capture. Returns when both have ended. */
    private suspend fun ProducerScope<SpeechEvent>.runSession(
        source: AudioSource,
        lease: Lease,
        utterances: UtteranceQueue,
        captureDispatcher: CoroutineDispatcher,
        stop: AtomicBoolean,
        language: String?,
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
            utterances.forEach { transcribeUtterance(it, language) }
        }
        launch(captureDispatcher) {
            try {
                capture(source, utterances, stop)
            } finally {
                utterances.close()
            }
        }
    }

    private fun closeQuietly(source: AudioSource) {
        try {
            source.close()
        } catch (e: Exception) {
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
    private fun ProducerScope<SpeechEvent>.skip(utterance: Utterance, queuedMs: Long) {
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

    private suspend fun ProducerScope<SpeechEvent>.transcribeUtterance(utterance: Utterance, language: String?) {
        val result = try {
            onWhisperThread { token ->
                try {
                    recognize(token, utterance.samples, "utterance", language)
                } finally {
                    utterance.samples.fill(0f)
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            CsLog.w("utterance failed", e)
            send(SpeechEvent.Error(e))
            return
        }
        if (result.text.isNotEmpty()) {
            send(SpeechEvent.Transcript(Transcript(result.text, utterance.startMs, utterance.endMs, result.processingMs, result.language)))
        }
    }

    private suspend fun ProducerScope<SpeechEvent>.capture(
        source: AudioSource,
        utterances: UtteranceQueue,
        stop: AtomicBoolean,
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
        val endpointer = Endpointer(config) { event ->
            when (event) {
                is EndpointerEvent.SpeechStarted -> trySend(SpeechEvent.SpeechStarted(event.atMs))
                is EndpointerEvent.SpeechEnded -> {
                    native.vadReset(vad)
                    if (event.confirmed) trySend(SpeechEvent.SpeechEnded(event.atMs))
                }
                is EndpointerEvent.UtteranceReady -> when (utterances.offer(event.utterance)) {
                    UtteranceQueue.Offer.QUEUED -> Unit
                    UtteranceQueue.Offer.FULL -> skip(event.utterance, utterances.queuedMs)
                    UtteranceQueue.Offer.CLOSED -> discard(event.utterance)     // the session is ending
                }
            }
        }
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
                    if (stopWaitOver(endpointer, tail, stopAtFrame)) break
                }
                if (source.read(frame) < 0) break
                val t0 = System.nanoTime()
                if (native.vadProbs(vad, frame, FRAME_SAMPLES, prob) != 1) {
                    throw CzechSpeechException("Voice activity detection failed")
                }
                vadNanos += System.nanoTime() - t0
                endpointer.accept(frame, prob[0])
                frames++
                if (tail >= 0) {
                    tail++
                    if (stopWaitOver(endpointer, tail, stopAtFrame)) break
                }
            }
            endpointer.finish()
        } finally {
            frame.fill(0f)
            endpointer.clear()
            closeQuietly(source)       // the microphone closes as soon as capture ends, before the queue drains
            native.vadFree(vad)
            val perFrameUs = if (frames > 0) vadNanos / frames / 1000 else 0
            CsLog.i("capture: ${frames * FRAME_MS} ms of audio, VAD $perFrameUs us per 32 ms frame")
        }
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

    /** Speech gate, Whisper, filter and repetition guard. Whisper thread only, with a lease held. */
    private fun recognize(token: AbortToken, samples: FloatArray, what: String, language: String?): Recognition {
        val t0 = System.nanoTime()
        if (samples.isEmpty() || token.isCancelled) return Recognition("", 0)
        val audioMs = samples.size * 1000L / SAMPLE_RATE
        val segments = native.vadSpeechSegments(ensureGate(), samples, samples.size, config.threshold)
        val gateMs = msSince(t0)
        if (segments < 0) throw CzechSpeechException("Voice activity detection failed")
        if (segments == 0) {
            CsLog.i("$what: $audioMs ms of audio, no speech (gate $gateMs ms), recogniser not run")
            return Recognition("", msSince(t0))
        }
        if (token.isCancelled) return Recognition("", msSince(t0))   // cancelled during the gate: do not load the model
        val whisper = ensureWhisper()
        if (!token.begin(whisper)) return Recognition("", msSince(t0))
        val t1 = System.nanoTime()
        // One language (asked for, or the model's only one) is forced, as before: no detection pass. Several are
        // passed as a list and the recogniser picks among them; it reports the pick right after the call, on this
        // thread, so nothing else can touch the context in between.
        val forced = language ?: languages.singleOrNull()
        val allowed = forced ?: languages.joinToString(",")
        var detected: String? = null
        val bytes = try {
            native.whisperTranscribe(whisper, token.generation, samples, samples.size, threadCount, allowed).also {
                if (forced == null && it != null) detected = native.whisperLanguage(whisper)
            }
        } finally {
            token.end()
        }
        val whisperMs = msSince(t1)
        if (token.isCancelled) {
            // null: whisper_full stopped at the abort callback; empty: stopped before the encoder
            if (bytes == null || bytes.isEmpty()) abortedRecognitions++
            bytes?.fill(0)
            CsLog.i("$what: $audioMs ms of audio, recogniser cancelled after $whisperMs ms")
            return Recognition("", msSince(t0))
        }
        if (bytes == null) throw CzechSpeechException("Recognition failed")
        val text = try {
            String(bytes, Charsets.UTF_8).trim()
        } finally {
            bytes.fill(0)
        }
        // Silence and noise that slip through make Whisper emit punctuation only (",", "... ..."); drop it.
        val kept = text.any { it.isLetterOrDigit() }
        CsLog.i(
            "$what: $audioMs ms of audio, $segments speech segment(s), gate $gateMs ms, recogniser $whisperMs ms, " +
                if (kept) "transcript kept" else "no letters or digits, dropped",
        )
        if (!kept) return Recognition("", msSince(t0))
        val spoken = forced ?: detected?.takeIf { it in languages } ?: throw CzechSpeechException("Recognition failed")
        // Audio that ends mid-word (stop(), the length cap) can make greedy decoding loop; keep one copy (RepetitionGuard).
        val guarded = RepetitionGuard.clean(text)
        if (guarded.loops.isNotEmpty()) CsLog.i("$what: ${RepetitionGuard.describe(guarded.loops)}")
        return Recognition(guarded.text, msSince(t0), spoken)
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
        val t0 = System.nanoTime()
        val handle = native.whisperInit(modelFile.absolutePath)
        if (handle == 0L) throw CzechSpeechException("Could not load the model from ${modelFile.path}")
        whisperHandle = handle
        isModelLoaded = true
        modelLoads++
        if (keepModelLoaded) acquire(kept)       // outlives the session or call that loaded it
        CsLog.i("model loaded in ${msSince(t0)} ms (${modelFile.length() / 1_000_000} MB, $threadCount threads)")
        if (!systemInfoLogged) {
            systemInfoLogged = true
            CsLog.i("native: ${NativeLoader.loadedLibrary ?: "-"}; ${native.systemInfo().trim()}")
        }
        return handle
    }

    private fun freeAll() {
        if (whisperHandle != 0L) {
            native.whisperFree(whisperHandle)
            whisperHandle = 0L
            isModelLoaded = false
            CsLog.i("model freed")
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

    private class Recognition(val text: String, val processingMs: Long, val language: String = "")

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

    @Synchronized
    fun cancel() {
        cancelled = true
        if (running != 0L) native.whisperAbort(running, generation)
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
        val queued = queuedSamples.addAndGet(n)
        if (!channel.trySend(utterance).isSuccess) {
            queuedSamples.addAndGet(-n)
            return Offer.CLOSED
        }
        val ms = queued * 1000 / SAMPLE_RATE
        if (ms > peakMs) peakMs = ms
        return Offer.QUEUED
    }

    /** Consumer: hands each utterance to [action], in order, until the queue is closed and empty. */
    suspend fun forEach(action: suspend (Utterance) -> Unit) {
        for (utterance in channel) {
            queuedSamples.addAndGet(-utterance.samples.size.toLong())
            action(utterance)
        }
    }

    /** An utterance still waiting, or null; drains the queue after [close]. */
    fun poll(): Utterance? =
        channel.tryReceive().getOrNull()?.also { queuedSamples.addAndGet(-it.samples.size.toLong()) }

    fun close() {
        channel.close()
    }
}

internal class NamedThreadFactory(private val name: String) : ThreadFactory {
    override fun newThread(r: Runnable): Thread = Thread(r, name).apply { isDaemon = true }
}
