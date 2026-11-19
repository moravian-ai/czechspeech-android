package ai.moravian.czechspeech

import ai.moravian.czechspeech.internal.AndroidLog
import ai.moravian.czechspeech.internal.AndroidMicSource
import ai.moravian.czechspeech.internal.CpuThreads
import ai.moravian.czechspeech.internal.CpuTopology
import ai.moravian.czechspeech.internal.JniNativeApi
import ai.moravian.czechspeech.internal.SpeechEngine
import android.content.Context
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.filterIsInstance
import kotlinx.coroutines.flow.map
import java.io.Closeable
import java.io.File

/**
 * Offline Czech speech-to-text.
 *
 * ```
 * val job = scope.launch {
 *     val speech = CzechSpeech(context, CzechSpeechModel.download(context, "cs"), CzechSpeechModel.entry("cs"))
 *     speech.transcripts().collect { show(it.text) }   // opens the microphone
 * }
 * job.cancel()   // to stop: the microphone closes and the model is freed, unless keepModelLoaded
 * ```
 *
 * Creating an instance is cheap: nothing is loaded until [listen] is collected or [transcribe] is called.
 *
 * Privacy: audio stays in memory on the device. It is never written to disk or sent anywhere, every buffer holding
 * it is zeroed when it is no longer needed, and transcript text is never logged.
 *
 * @param context any context; the application context is kept.
 * @param modelFile the CzechSpeech ggml model, from [CzechSpeechModel.download] or your own copy.
 * @param keepModelLoaded keep the model loaded from the first [listen] session or [transcribe] call that loads it
 *   until [releaseModel] or [close], instead of loading and freeing it in every session and call. It saves the load of
 *   each session (about 0.5 to 3 s) at the price of holding the model (474 MB) and whisper.cpp's buffers in memory
 *   between sessions; see [releaseModel]. Use one instance per process: each keeps a model of its own. Default
 *   false.
 * @param languages the languages of the model, two- or three-letter lowercase whisper language codes; the default is
 *   the default model's, `["cs"]` (see [CzechSpeechModel.DEFAULT_ID] and [CzechSpeechModel.Entry.languages]). A repeated code counts once. One language is
 *   forced; with several, [listen], [listenWithPreviews] and [transcribe] take a `language`, or the recogniser decides it among
 *   them by the rule of [languageSwitch] (the first language preferred), inside the recognition itself. Must
 *   not be empty. For a model of the catalog use the constructor that takes the [CzechSpeechModel.Entry] instead.
 * @param languageSwitch the rule for leaving the first language automatically (see [CzechSpeechModel.LanguageSwitch]),
 *   or null; then several [languages] take the most probable of them for every recognition, with no state and no rescue.
 *   With one, a `listen` collection keeps a language between utterances, a language that came out empty is rescued, and a
 *   language without a threshold is chosen only when asked for. Needs at least two languages and names only
 *   languages after the first, otherwise IllegalArgumentException at construction.
 */
public class CzechSpeech @JvmOverloads public constructor(
    context: Context,
    public val modelFile: File,
    public val config: CzechSpeechConfig = CzechSpeechConfig(),
    public val keepModelLoaded: Boolean = false,
    languages: List<String> = CzechSpeechModel.entry(CzechSpeechModel.DEFAULT_ID).languages,
    public val languageSwitch: CzechSpeechModel.LanguageSwitch? = null,
) : Closeable {
    /** A copy of the list passed in, without repeats; the engine rejects an empty list and bad codes at construction. */
    public val languages: List<String> = languages.distinct()

    private val appContext: Context = context.applicationContext ?: context

    private val engine = SpeechEngine(
        modelFile = modelFile,
        config = config,
        native = JniNativeApi,
        vadModel = { appContext.assets.open(VAD_ASSET).use { it.readBytes() } },
        audioSourceFactory = { AndroidMicSource.create(appContext) },
        threads = { if (config.threads > 0) config.threads else CpuThreads.defaultThreadCount() },
        keepModelLoaded = keepModelLoaded,
        cpuTopology = { CpuTopology.read() },
        languages = this.languages,
        languageSwitch = languageSwitch,
    )

    /**
     * A model of the catalog: the languages follow from the [model] entry, so they are never given separately and a
     * model cannot be forced into the wrong language.
     *
     * ```
     * CzechSpeech(context, CzechSpeechModel.download(context, "cs"), CzechSpeechModel.entry("cs"))
     * ```
     *
     * Use the constructor with `languages` for a model file outside the catalog. The other parameters are as there.
     */
    @JvmOverloads
    public constructor(
        context: Context,
        modelFile: File,
        model: CzechSpeechModel.Entry,
        config: CzechSpeechConfig = CzechSpeechConfig(),
        keepModelLoaded: Boolean = false,
    ) : this(context, modelFile, config, keepModelLoaded, model.languages, model.languageSwitch)

    init {
        AndroidLog.install()
    }

    /**
     * Listens to the microphone and emits speech events and transcripts, one per utterance.
     *
     * The flow is cold. Collecting it opens the microphone (the app needs the RECORD_AUDIO permission) and loads
     * the model in parallel. To finish, call [stop]: the last utterance is still transcribed and the flow completes.
     * Cancelling the collection instead discards speech that is still in progress or waiting for recognition: the
     * microphone closes at the end of the frame being read and the collection completes at once. Native work that
     * whisper.cpp cannot interrupt finishes in the background, and the model is freed right after it (with
     * [keepModelLoaded] it stays loaded): a recognition running at that moment stops at whisper.cpp's next check (it
     * checks between stages only, so up to a few seconds for a long utterance), and a model load runs to its end. A
     * session cancelled before its model load started never loads it. One collection at a time per instance.
     *
     * Results are whole utterances, not words as they are spoken (see the README).
     *
     * Utterances wait for the recogniser in a queue that holds at most 60 s of audio. When recognition is slower
     * than the speech for that long, an utterance that does not fit is skipped: its audio is zeroed at once and a
     * [SpeechEvent.Error] says which speech was skipped. Speech is never dropped without that event; the microphone
     * is never paused for the recogniser, since audio that is not read is lost without notice.
     *
     * @throws SecurityException (from the flow) without RECORD_AUDIO.
     * @throws CzechSpeechException (from the flow) when the microphone, the model or the native library fails.
     * @throws IllegalStateException (from the flow) when another collection of this instance is active ("listen() is
     *   already being collected; one session at a time"), and after [close] ("CzechSpeech is closed"), also for a
     *   collection that starts while close() runs.
     * @param language force this language, one of [languages]; null means the model's only language, or detection
     *   among [languages] when there are several.
     * @throws IllegalArgumentException (at once, from this call) for a [language] outside [languages].
     */
    @JvmOverloads
    public fun listen(language: String? = null): Flow<SpeechEvent> = engine.listen(language)

    /**
     * [listen] with previews: while an utterance is still being spoken, its text so far comes as [SpeechPreview]s,
     * between the same [SpeechEvent]s as from [listen], in the same order. Without this call nothing of it runs.
     *
     * A preview is unstable text: show it as provisional (in grey, as composing text) and never insert it as finished
     * text. The [SpeechEvent.Transcript] with the same `startMs` replaces it; a preview with empty text withdraws it
     * when its utterance gives no transcript (see [SpeechPreview]).
     *
     * How previews run ([previews] sets the times):
     * - The first preview of an utterance is recognised once it has lasted [PreviewConfig.firstAfterMs] (5 s), from
     *   the audio captured so far, which stays in memory and is zeroed after use. The next ones come at the earliest
     *   [PreviewConfig.intervalMs] (5 s) of audio after the previous one started, one at a time.
     * - Final text first: no preview starts while an utterance waits for or is in its final recognition. A preview
     *   that runs when its utterance ends is aborted (whisper.cpp stops within one step of its computation, also
     *   inside the encoder) and its result is dropped, so it delays the final recognition by the time the abort
     *   takes, which logcat reports. No preview starts in a pause (which may be the end), nor when, judged by the
     *   speed of the last preview, it would still run when the utterance reaches [CzechSpeechConfig.maxUtteranceMs].
     *   A preview never comes after the end of its speech nor after the Transcript of its utterance.
     * - A preview decodes once, greedily, without whisper.cpp's temperature fallback and without the speech check,
     *   both of which the final text keeps. Each one runs the recogniser over the whole utterance so far: with the
     *   defaults a 28 s utterance gets up to 5 previews over about 75 s of audio, 2.7 times its own, on top of its
     *   final recognition. On a slower phone previews come less often (one at a time), never at the expense of the
     *   final text.
     * - After [stop] no preview starts, and one that is running is aborted at once; cancelling the collection or
     *   [close] aborts a preview that is running too.
     *
     * Everything else is as in [listen]: the same session, errors and rules, one collection at a time per instance.
     *
     * @param language as for [listen]: force this language, one of [languages]; null means the model's only language,
     *   or the rule for several.
     * @throws IllegalArgumentException (at once, from this call) for a [language] outside [languages].
     */
    @JvmOverloads
    public fun listenWithPreviews(previews: PreviewConfig = PreviewConfig(), language: String? = null): Flow<ListenEvent> =
        engine.listenWithPreviews(previews, language)

    /**
     * Finishes the active [listen] session. It finishes the word in progress: with speech in progress the microphone
     * stays on until a short pause (256 ms), at most 1 s; without speech it closes at once. The utterance in progress
     * then ends as if silence followed, everything still waiting is transcribed and emitted, the flow completes
     * normally and the model is freed, unless [keepModelLoaded] keeps it. Returns immediately; the collector sees the
     * remaining events. Does nothing without an active session (it never stops a later one). To discard instead,
     * cancel the collection.
     */
    public fun stop(): Unit = engine.stop()

    /**
     * The transcripts of [listen]: `listen(language).filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }`.
     *
     * @param language as for [listen]: one of [languages], or null.
     * @throws IllegalArgumentException (at once, from this call) for a [language] outside [languages].
     */
    @JvmOverloads
    public fun transcripts(language: String? = null): Flow<Transcript> =
        listen(language).filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }

    /**
     * Transcribes audio the app already has: 16 kHz mono float samples in [-1, 1].
     *
     * The same checks as [listen] apply: when Silero VAD finds no speech the recogniser is not run (and the model is
     * not even loaded), and a result without any letter or digit is dropped; both give "". A phrase that Whisper
     * repeats in a loop is kept once. The model is loaded for the call and freed afterwards, unless a [listen]
     * session holds it or [keepModelLoaded] keeps it.
     *
     * The caller owns [samples]: they are read, never modified, so zero them yourself when you are done.
     *
     * Input is checked before any work: at most 10 minutes of audio (9,600,000 samples), every sample finite and of
     * magnitude at most 2. Invalid input is rejected, not repaired: NaN or infinity, and 16-bit PCM passed without
     * scaling (it would silently give ""), are bugs in the audio path.
     *
     * @param language as for [listen]: one of [languages], or null.
     * @throws IllegalArgumentException for a [language] outside [languages], audio longer than 10 minutes, a NaN or
     *   infinite sample, or a sample of magnitude above 2; the message names the first offending sample and, from a
     *   magnitude of 256 up, points at unscaled 16-bit PCM (divide by 32768), below it at clipping (clip or normalise
     *   the audio).
     * @throws IllegalStateException ("CzechSpeech is closed") after [close], and when [close] is called during the
     *   call: the call then ends at once and a running recognition is aborted.
     */
    @JvmOverloads
    public suspend fun transcribe(samples: FloatArray, language: String? = null): String =
        engine.transcribe(samples, language)

    /**
     * Closes this instance. Returns at once, without blocking (safe on the main thread). Once it has returned:
     * - no new session or call starts: [listen] and [transcribe] throw IllegalStateException("CzechSpeech is
     *   closed"), also a collection or call that starts while close() runs;
     * - the active [listen] collection is cancelled (the collector sees a CancellationException), and every
     *   [transcribe] call in progress resumes at once with that IllegalStateException;
     * - the microphone closes at the end of the frame being read (32 ms of audio);
     * - a recognition that is running is aborted and stops within one step of its computation; native work that
     *   cannot be interrupted (a model load, the speech check, a run of the language model) finishes in the
     *   background, and the model is freed right after, also a model kept by [keepModelLoaded].
     *   [awaitClosed] waits for that.
     *
     * Calling it again does nothing.
     */
    override fun close(): Unit = engine.close()

    /**
     * Frees the model that [keepModelLoaded] keeps loaded, once the [listen] session and the [transcribe] calls
     * running now have ended: at once when none is running, otherwise when the last of them ends. A session that ends
     * normally ([stop], the end of the audio) frees it before its flow completes, a call before it returns, and a
     * cancelled session or call right after, in the background. Unlike [close], it does not end the instance.
     *
     * A session or call that starts afterwards keeps the model again: it loads the model if it has been freed, and a
     * release still waiting for the work that was running is dropped.
     *
     * Returns at once, without blocking (safe on the main thread). The model is freed on the SDK's recognition
     * thread, after native work already running there (the model load of a cancelled session, say).
     *
     * Call it when no speech is expected for a while (for example two minutes after the last session) and from
     * onTrimMemory(): Android moves memory that nobody touches to zram, and a model swapped out there makes the next
     * recognition take up to a minute. It also frees whisper.cpp's buffers derived from the last audio (the mel
     * spectrogram, the encoder activations), which stay in RAM while the model is loaded.
     *
     * Does nothing without [keepModelLoaded], when no model is loaded and nothing is running, and after [close]
     * (which frees it).
     */
    public fun releaseModel(): Unit = engine.releaseModel()

    /**
     * Suspends until the work [close] could not stop at once has finished: the microphone is closed, native work
     * that was running (a model load, the speech check, a recognition being aborted) has ended, and the model is freed. Use it,
     * for example, before creating another instance, so that two models are never held at once. Returns at once
     * when nothing was running. Do not block the thread the [listen] collector runs on while waiting.
     *
     * @throws IllegalStateException when [close] has not been called.
     */
    public suspend fun awaitClosed(): Unit = engine.awaitClosed()

    private companion object {
        /** Silero VAD v6.2.0 in ggml format (MIT), sha256 2aa269b7…6987. */
        const val VAD_ASSET = "czechspeech/ggml-silero-v6.2.0.bin"
    }
}
