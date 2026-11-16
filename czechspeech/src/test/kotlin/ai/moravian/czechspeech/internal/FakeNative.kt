package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement
import java.io.File
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs

/**
 * The native layer for pipeline tests: no model needed. "Speech" is any frame with a sample above 0.05 in magnitude.
 * It counts handles, checks for double frees, use after free and aborts on freed contexts, and counts the contexts
 * freed off their own thread (checked after every test by [FreedOnOwnThreads]); latches hold a model load, the speech
 * gate, a recognition or a free where a test needs the whisper thread busy.
 *
 * Threads are compared by name: the whisper executor's thread exits after 5 s idle and the next task runs on a new
 * one, also named czechspeech-whisper. That is harmless, since a whisper.cpp context is not tied to an OS thread, only
 * to one thread at a time.
 */
internal class FakeNative : NativeApi {
    init {
        created += this
    }

    @Volatile var text = "dobrý den"
    @Volatile var failNext = 0
    @Volatile var blockUntilAborted = false
    /** When set, each recognition waits for it (to keep later utterances waiting in the queue). */
    @Volatile var holdRecognition: CountDownLatch? = null
    /** When set, loading the model waits for it (a load in progress). */
    @Volatile var holdInit: CountDownLatch? = null
    /** When set, the speech gate waits for it (the whisper thread busy, no model loaded). */
    @Volatile var holdGate: CountDownLatch? = null
    /** When set, freeing the model waits for it (a free in progress). */
    @Volatile var holdFree: CountDownLatch? = null
    val freeStarted = CountDownLatch(1)
    /** Order of events: "whisperFree" when a model free has ended; tests add their own. */
    val events: MutableList<String> = Collections.synchronizedList(mutableListOf())
    /** Wall-clock time of each recognition (a recogniser slower than the audio). */
    @Volatile var recognitionMs = 0L
    val initStarted = CountDownLatch(1)
    val gateStarted = CountDownLatch(1)
    val modelFreed = CountDownLatch(1)
    val whisperInits = AtomicInteger()
    val whisperFrees = AtomicInteger()
    val vadInits = AtomicInteger()
    val vadFrees = AtomicInteger()
    val vadResets = AtomicInteger()
    val gateCalls = AtomicInteger()
    val transcribeStarted = CountDownLatch(1)
    /** The language argument of each recognition ("cs", "en" or a list such as "pl,en"). */
    val languagesPassed: MutableList<String> = Collections.synchronizedList(mutableListOf())
    /** What whisperLanguage reports after a recognition (the detected language when several were allowed). */
    @Volatile var detectedLanguage: String? = null
    /** How often whisperLanguage was called: it must stay 0 when a language is forced. */
    val whisperLanguageCalls = AtomicInteger()
    val passedToWhisper: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
    val aborts: MutableList<Int> = Collections.synchronizedList(mutableListOf())
    /** The thread of each whisperFree, by name. */
    val whisperFreeThreads: MutableList<String> = Collections.synchronizedList(mutableListOf())
    /** Models freed off the whisper thread. */
    val whisperFreesOffThread = AtomicInteger()
    /** VAD contexts freed on a thread other than the one that created them, or freed twice. */
    val vadFreesOffThread = AtomicInteger()
    private val vadThreads = ConcurrentHashMap<Long, String>()
    private val liveHandles = Collections.synchronizedSet(mutableSetOf<Long>())
    private val nextHandle = AtomicInteger(100)

    override fun systemInfo() = "fake"
    override fun whisperInit(modelPath: String): Long {
        whisperInits.incrementAndGet()
        initStarted.countDown()
        holdInit?.let { check(it.await(10, TimeUnit.SECONDS)) { "model load never released" } }
        return nextHandle.incrementAndGet().toLong().also { liveHandles += it }
    }

    override fun whisperFree(handle: Long) {
        val thread = Thread.currentThread().name
        whisperFreeThreads += thread
        if (thread != WHISPER_THREAD) whisperFreesOffThread.incrementAndGet()
        freeStarted.countDown()
        holdFree?.let { check(it.await(10, TimeUnit.SECONDS)) { "model free never released" } }
        whisperFrees.incrementAndGet()
        check(liveHandles.remove(handle)) { "double free" }
        events += "whisperFree"
        modelFreed.countDown()
    }

    override fun whisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
    ): ByteArray? {
        check(handle in liveHandles) { "use of a freed context" }
        languagesPassed += languages
        passedToWhisper += samples
        transcribeStarted.countDown()
        holdRecognition?.let { check(it.await(10, TimeUnit.SECONDS)) { "recognition never released" } }
        if (recognitionMs > 0) Thread.sleep(recognitionMs)
        if (blockUntilAborted) {
            val deadline = System.currentTimeMillis() + 10_000
            while (generation !in aborts) {
                if (System.currentTimeMillis() > deadline) error("never aborted")
                Thread.sleep(2)
            }
            return null
        }
        if (failNext > 0) {
            failNext--
            return null
        }
        return text.toByteArray()
    }

    override fun whisperLanguage(handle: Long): String? {
        check(handle in liveHandles) { "language of a freed context" }
        whisperLanguageCalls.incrementAndGet()
        return detectedLanguage
    }

    override fun whisperAbort(handle: Long, generation: Int) {
        check(handle in liveHandles) { "abort on a freed context" }
        aborts += generation
    }

    override fun vadInit(model: ByteArray, threads: Int): Long {
        vadInits.incrementAndGet()
        return nextHandle.incrementAndGet().toLong().also { vadThreads[it] = Thread.currentThread().name }
    }

    override fun vadFree(handle: Long) {
        vadFrees.incrementAndGet()
        if (vadThreads.remove(handle) != Thread.currentThread().name) vadFreesOffThread.incrementAndGet()
    }

    override fun vadReset(handle: Long) {
        vadResets.incrementAndGet()
        resetHandles += handle
    }

    /** Streaming VAD calls on a context whose (uninitialised) LSTM state was never reset. */
    val unresetStreamingCalls = AtomicInteger()
    private val resetHandles = Collections.synchronizedSet(mutableSetOf<Long>())

    override fun vadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int {
        if (handle !in resetHandles) unresetStreamingCalls.incrementAndGet()
        val windows = (n + FRAME_SAMPLES - 1) / FRAME_SAMPLES
        for (w in 0 until minOf(windows, out.size)) {
            var peak = 0f
            for (i in w * FRAME_SAMPLES until minOf(n, (w + 1) * FRAME_SAMPLES)) peak = maxOf(peak, abs(samples[i]))
            out[w] = if (peak > 0.05f) 0.9f else 0.02f
        }
        return windows
    }

    override fun vadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int {
        gateCalls.incrementAndGet()
        gateStarted.countDown()
        holdGate?.let { check(it.await(10, TimeUnit.SECONDS)) { "speech gate never released" } }
        return if ((0 until n).any { abs(samples[it]) > 0.05f }) 1 else 0
    }

    companion object {
        const val WHISPER_THREAD = "czechspeech-whisper"

        /** Every instance since [FreedOnOwnThreads] last looked. */
        val created: MutableList<FakeNative> = Collections.synchronizedList(mutableListOf())
    }
}

/**
 * Fails a test that passed when a [FakeNative] it created freed a context off its own thread: the model off the
 * whisper thread, a VAD context off the thread that created it. Declared in every test class that uses FakeNative.
 */
internal class FreedOnOwnThreads : TestRule {
    override fun apply(base: Statement, description: Description): Statement = object : Statement() {
        override fun evaluate() {
            val failure = runCatching { base.evaluate() }.exceptionOrNull()
            val natives = synchronized(FakeNative.created) {
                FakeNative.created.toList().also { FakeNative.created.clear() }
            }
            if (failure != null) throw failure          // the test's own failure comes first
            val models = natives.sumOf { it.whisperFreesOffThread.get() }
            val vads = natives.sumOf { it.vadFreesOffThread.get() }
            if (models + vads > 0) {
                throw AssertionError("freed off their own thread: $models model(s), $vads VAD context(s)")
            }
        }
    }
}

private val fakeModelFile: File by lazy { File.createTempFile("fake-model", ".bin").apply { deleteOnExit() } }

/** A [SpeechEngine] over [native] with the default config, 2 threads and an existing (empty) model file. */
internal fun fakeEngine(
    native: FakeNative,
    hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE,
    maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS,
    keepModelLoaded: Boolean = false,
    languages: List<String> = listOf("cs"),
    source: () -> AudioSource,
) = SpeechEngine(
    modelFile = fakeModelFile,
    config = CzechSpeechConfig(),
    native = native,
    vadModel = { ByteArray(16) },
    audioSourceFactory = source,
    threads = { 2 },
    keepModelLoaded = keepModelLoaded,
    languages = languages,
    maxQueuedAudioMs = maxQueuedAudioMs,
    hooks = hooks,
)
