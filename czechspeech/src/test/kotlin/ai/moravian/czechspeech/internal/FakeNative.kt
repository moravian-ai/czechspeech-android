package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import org.junit.rules.TestRule
import org.junit.runner.Description
import org.junit.runners.model.Statement
import java.io.File
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
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
    /** Texts of successive recognitions, taken before [text] (a preview and the final text of one utterance). */
    val texts: ConcurrentLinkedQueue<String> = ConcurrentLinkedQueue()
    @Volatile var failNext = 0
    @Volatile var blockUntilAborted = false
    /** With [blockUntilAborted]: the abort comes before the encoder, so whisper_full returns no text, not a failure. */
    @Volatile var abortedBeforeEncoder = false
    /** Recognitions, by index from 0, that wait for their abort and return null like [blockUntilAborted]. */
    @Volatile var abortableCalls: Set<Int> = emptySet()
    /** Runs at the start of each recognition with its index from 0, on the whisper thread (to hold one). */
    @Volatile var onTranscribe: ((index: Int) -> Unit)? = null
    val transcribeCalls = AtomicInteger()
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
    /** The thresholds argument of each recognition (null: no rule), a copy. */
    val thresholdsPassed: MutableList<FloatArray?> = Collections.synchronizedList(mutableListOf())
    /**
     * The shares of the languages of a recognition with a list: index of the recognition (from 0, as in
     * [transcribeCalls]) and the session time its audio covers (from [clockWords]' samples, else from 0 to its length).
     * Null: the first language 1.0, the others 0. With a function set, the language of such a recognition is the one
     * [pickLanguage] picks from these shares and the thresholds passed (unless [detectedLanguage] is set).
     */
    @Volatile var sharesOf: ((index: Int, startMs: Long, endMs: Long) -> FloatArray)? = null
    /** Rewrites the text of a recognition by the language it ran in (a recognition's `language` and its text). */
    @Volatile var textIn: ((language: String, text: String) -> String)? = null
    /** How often whisperLanguageShares was called: it must stay 0 when a language is forced. */
    val sharesCalls = AtomicInteger()
    @Volatile private var lastShares: FloatArray? = null
    @Volatile private var lastLanguage: String? = null
    /** What whisperLanguage reports after a recognition (the detected language when several were allowed). */
    @Volatile var detectedLanguage: String? = null
    /** How often whisperLanguage was called: it must stay 0 when a language is forced. */
    val whisperLanguageCalls = AtomicInteger()
    val passedToWhisper: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
    /** With [keepCopies]: a copy of each array as the recogniser got it (the engine zeroes the arrays afterwards). */
    @Volatile var keepCopies = false
    val copiesPassed: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())
    /** The most recognitions that ever ran at once: one context, so never more than 1. */
    val maxConcurrentRecognitions = AtomicInteger()
    private val recognising = AtomicInteger()
    val aborts: MutableList<Int> = Collections.synchronizedList(mutableListOf())
    /** The thread of each whisperFree, by name. */
    val whisperFreeThreads: MutableList<String> = Collections.synchronizedList(mutableListOf())
    /** Models freed off the whisper thread. */
    val whisperFreesOffThread = AtomicInteger()
    /** Greedy recognitions (whisperTranscribe) and the beam size of each beam search (whisperTranscribeNBest). */
    val greedyCalls = AtomicInteger()
    val beamSizes: MutableList<Int> = Collections.synchronizedList(mutableListOf())
    /** Every array a recognition returned, to check that the engine zeroes them. */
    val returned: MutableList<ByteArray> = Collections.synchronizedList(mutableListOf())
    /** What the beam search finds, per window. */
    @Volatile var hypotheses: List<List<NBestBytes.Hyp>> = listOf(listOf(NBestBytes.Hyp(" dobrý den", -1.0, 4, best = true)))
    /** What the next beam-search calls find, one entry per call, before [hypotheses] answers every call again. */
    val hypothesesQueue: ConcurrentLinkedQueue<List<List<NBestBytes.Hyp>>> = ConcurrentLinkedQueue()

    /** Language model: loads fail when set; [lmScore] gives this per sequence, or fails when it is null. */
    @Volatile var failLmInit = false
    @Volatile var lmVocab = 420
    @Volatile var lmScoreOf: ((IntArray) -> Double)? = { -it.size.toDouble() }
    val lmInits = AtomicInteger()
    val lmFrees = AtomicInteger()
    /** The sequences of each lmScore call, and the thread of each LM load and free. */
    val lmScoreCalls: MutableList<List<IntArray>> = Collections.synchronizedList(mutableListOf())
    val lmThreads: MutableList<String> = Collections.synchronizedList(mutableListOf())
    private val liveLms = Collections.synchronizedSet(mutableSetOf<Long>())
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
        if (initFails) return 0
        return nextHandle.incrementAndGet().toLong().also { liveHandles += it }
    }

    /** Loading the model fails (whisper_init returns null). */
    @Volatile var initFails = false

    /**
     * Recognitions made through [whisperPreview]; they count in [transcribeCalls] too, not in [greedyCalls] (the final
     * text's greedy decoding) nor in [beamSizes].
     */
    val previewCalls = AtomicInteger()

    /**
     * When set, every [whisperPreview] and [whisperTranscribe] returns the words it gives for the range of session time
     * the audio passed covers, read back from the samples of [Signals.clock] (see [clockRange]); it comes before [texts].
     */
    @Volatile var clockWords: ((startMs: Long, endMs: Long) -> String)? = null

    /** With [clockWords]: [whisperPreview] number `index` (from 0) returns this for the words [clockWords] gave. */
    @Volatile var previewOverride: ((index: Int, words: String) -> String)? = null

    /** The sample count `n` of every [whisperPreview], in order. */
    val previewSampleCounts: MutableList<Int> = Collections.synchronizedList(mutableListOf())

    override fun whisperPreview(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray? {
        val index = previewCalls.getAndIncrement()
        previewSampleCounts += n
        compute("preview", threads)
        return recognise(handle, generation, samples, n, languages, thresholds, ByteArray(0)) { language ->
            val words = clockWords?.let { words -> clockRange(samples, n)?.let { (start, end) -> words(start, end) } }
            when {
                words == null -> (texts.poll() ?: text)
                else -> (previewOverride?.invoke(index, words) ?: words)
            }.let { textIn?.invoke(language, it) ?: it }.toByteArray()
        }
    }

    /**
     * The range of session time `[startMs, endMs)` that the first [n] samples of a [Signals.clock] signal cover: the
     * first sample above 0.05 (index j0, value v) says its time t0 (its level, 0.1 in a word and 0.06 in a gap, plus
     * t * 1e-7), the start is t0 - j0 / 16 and the end start + n / 16. Null for audio without such a sample.
     */
    private fun clockRange(samples: FloatArray, n: Int): Pair<Long, Long>? {
        val j0 = (0 until n).firstOrNull { samples[it] > 0.05f } ?: return null
        val v = samples[j0]
        val t0 = Math.round((v - (if (v >= 0.08f) 0.1 else 0.06)) / 1e-7)
        val start = t0 - j0 / 16
        return start to start + n / 16
    }

    // A model of the kernel's CPU affinity of the whisper thread (the only thread that computes), as Linux 6.2+ keeps
    // it: sched_setaffinity's mask is kept as the thread's own request (user_cpus_ptr), every change of the cpuset is
    // intersected with it, and threads the whisper thread starts copy it. Older kernels reset the request when the
    // cpuset changes ([cpusetChangeDropsRequest]).

    /** The process's cpuset (Android's group: 0xff top-app, 0x7f foreground on a Pixel 9a); 0 = the thread's CPUs unknown. */
    @Volatile var cpuset = 0L
        set(value) {
            field = value
            if (cpusetChangeDropsRequest) request = ALL_CPUS
        }

    /** Linux before 6.2: a change of the cpuset sets the thread's mask to the cpuset, dropping its own request. */
    @Volatile var cpusetChangeDropsRequest = false

    /**
     * Cores parked while the device idles (Qualcomm's core_ctl, say): still in the thread's mask, not in
     * sched_getaffinity, until a computation wakes them; they park again once the thread is unpinned.
     */
    @Volatile var parked = 0L

    /** Cores isolated (thermal isolation, say): in the thread's mask, never in sched_getaffinity, also under load. */
    @Volatile var isolated = 0L

    /** A computation has woken the parked cores. */
    @Volatile private var awake = false

    /** Cores that do not run now. */
    private val sleeping: Long get() = isolated or if (awake) 0L else parked

    /** The thread's own request (sched_setaffinity); [ALL_CPUS] = none. Threads it starts copy it. */
    @Volatile var request = ALL_CPUS
        private set

    /** The thread's mask (Cpus_allowed): its request within the cpuset, the whole cpuset when they do not meet. */
    val cpus: Long get() = (cpuset and request).takeIf { it != 0L } ?: cpuset

    /** One affinity call: "set 0x70" or "unpin", and the thread it was made on. */
    class CpuCall(val what: String, val mask: Long, val thread: String) {
        override fun toString() = if (what == "set") "set ${CpuMask.hex(mask)}" else what
    }

    /** Every call of [setThreadCpus] and [unpinThread], in order. */
    val cpuCalls: MutableList<CpuCall> = Collections.synchronizedList(mutableListOf())
    val cpuCallNames: List<String> get() = synchronized(cpuCalls) { cpuCalls.map { it.toString() } }

    /** These calls (by index from 0 in [cpuCalls]) are refused: false, nothing changes. */
    @Volatile var refusedCpuCalls: Set<Int> = emptySet()

    /** When set, [setThreadCpus] and [unpinThread] throw it (after recording the call); [threadCpusError] makes the reads throw. */
    @Volatile var cpusError: Throwable? = null
    @Volatile var threadCpusError: Throwable? = null

    /** One native computation: what ("greedy", "preview", "beam", "lm"), its threads, and [cpus] while it ran. */
    class Compute(val what: String, val threads: Int, val cpus: Long) {
        override fun toString() = "$what/$threads/${CpuMask.hex(cpus)}"
    }

    val computes: MutableList<Compute> = Collections.synchronizedList(mutableListOf())

    /** A computation starts: it wakes the parked cores, and is recorded with the thread's mask (tests call it too). */
    fun compute(what: String, threads: Int) {
        awake = true
        computes += Compute(what, threads, cpus)
    }

    override fun threadCpus(): Long {
        threadCpusError?.let { throw it }
        return cpus
    }

    override fun activeThreadCpus(): Long {
        threadCpusError?.let { throw it }
        return cpus and sleeping.inv()
    }

    private fun cpuCall(what: String, mask: Long): Boolean {
        val index = synchronized(cpuCalls) { cpuCalls.size.also { cpuCalls += CpuCall(what, mask, Thread.currentThread().name) } }
        cpusError?.let { throw it }
        return index !in refusedCpuCalls
    }

    override fun setThreadCpus(mask: Long): Boolean {
        // the kernel refuses a mask without a core of the cpuset that is active now (EINVAL)
        if (!cpuCall("set", mask) || mask and cpuset and sleeping.inv() == 0L) return false
        request = mask
        return true
    }

    override fun unpinThread(): Boolean {
        if (!cpuCall("unpin", ALL_CPUS)) return false
        request = ALL_CPUS
        awake = false               // the call is over: the device idles again
        return true
    }

    /** Every floor of the encoder window the engine set, in order. */
    val minAudioCtxSet: MutableList<Int> = Collections.synchronizedList(mutableListOf())

    override fun whisperSetMinAudioCtx(handle: Long, frames: Int) {
        check(handle in liveHandles) { "floor set on a freed context" }
        check(Thread.currentThread().name == WHISPER_THREAD) { "floor set off the whisper thread" }
        minAudioCtxSet += frames
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
        thresholds: FloatArray?,
    ): ByteArray? {
        greedyCalls.incrementAndGet()
        compute("greedy", threads)
        return recognise(handle, generation, samples, n, languages, thresholds, ByteArray(0)) { language ->
            val words = clockWords?.let { words -> clockRange(samples, n)?.let { (start, end) -> words(start, end) } }
            (words ?: texts.poll() ?: text).let { textIn?.invoke(language, it) ?: it }.toByteArray()
        }
    }

    override fun whisperTranscribeNBest(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        beamSize: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray? {
        beamSizes += beamSize
        compute("beam", threads)
        return recognise(handle, generation, samples, n, languages, thresholds, NBestBytes.encode(emptyList())) { language ->
            val rewrite = textIn
            val found = hypothesesQueue.poll() ?: hypotheses
            NBestBytes.encode(
                if (rewrite == null) found
                else found.map { window ->
                    window.map { NBestBytes.Hyp(rewrite(language, it.text), it.sumLogprob, it.length, it.best, it.score, it.temperature) }
                },
            )
        }
    }

    /** What `cs_thresholds` in the real library refuses: thresholds with one language, of another length or out of 0..1 (NaN is fine). */
    private fun requireThresholdsFit(languages: String, thresholds: FloatArray?) {
        if (thresholds == null) return
        val n = languages.split(',').size
        require(n >= 2 && thresholds.size == n) { "thresholds do not fit the language list" }
        require(thresholds.all { it.isNaN() || it in 0f..1f }) { "threshold out of range" }
    }

    private fun recognise(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        languages: String,
        thresholds: FloatArray?,
        nothing: ByteArray,
        result: (language: String) -> ByteArray,
    ): ByteArray? {
        check(handle in liveHandles) { "use of a freed context" }
        requireThresholdsFit(languages, thresholds)
        languagesPassed += languages
        thresholdsPassed += thresholds?.copyOf()
        lastShares = null
        lastLanguage = null
        passedToWhisper += samples
        if (keepCopies) copiesPassed += samples.copyOf()
        val running = recognising.incrementAndGet()
        maxConcurrentRecognitions.accumulateAndGet(running) { x, y -> maxOf(x, y) }
        try {
            val index = transcribeCalls.getAndIncrement()
            val list = languages.split(',')
            val shares = if (list.size > 1) {
                val (start, end) = clockRange(samples, n) ?: (0L to n / 16L)
                sharesOf?.invoke(index, start, end) ?: FloatArray(list.size) { if (it == 0) 1f else 0f }
            } else {
                null
            }
            if (shares != null) check(shares.size == list.size) { "${shares.size} shares for $languages" }
            val picked = when {
                list.size == 1 -> list[0]
                sharesOf != null -> list[pickLanguage(shares!!, thresholds)]
                else -> null
            }
            transcribeStarted.countDown()
            onTranscribe?.invoke(index)
            holdRecognition?.let { check(it.await(10, TimeUnit.SECONDS)) { "recognition never released" } }
            if (recognitionMs > 0) Thread.sleep(recognitionMs)
            if (blockUntilAborted || index in abortableCalls) {
                val deadline = System.currentTimeMillis() + 10_000
                while (generation !in aborts) {
                    if (System.currentTimeMillis() > deadline) error("never aborted")
                    Thread.sleep(2)
                }
                return if (abortedBeforeEncoder) nothing.also { returned += it } else null
            }
            if (failNext > 0) {
                failNext--
                return null
            }
            return result(detectedLanguage ?: picked ?: list[0]).also {
                returned += it
                lastShares = shares
                lastLanguage = picked
            }
        } finally {
            recognising.decrementAndGet()
        }
    }

    /** When set, loading the language model waits for it (a load in progress). */
    @Volatile var holdLmInit: CountDownLatch? = null

    /** When set, loading the language model throws it (a native failure other than a refused file). */
    @Volatile var lmInitError: RuntimeException? = null

    override fun lmInit(modelPath: String): Long {
        lmThreads += Thread.currentThread().name
        lmInitError?.let { throw it }
        holdLmInit?.let { check(it.await(10, TimeUnit.SECONDS)) { "language model load never released" } }
        if (failLmInit) return 0
        lmInits.incrementAndGet()
        return nextHandle.incrementAndGet().toLong().also { liveLms += it }
    }

    /** When set, each lmScore waits for it (a decision in progress); [lmScoreStarted] opens when one has started. */
    @Volatile var holdLmScore: CountDownLatch? = null
    val lmScoreStarted = CountDownLatch(1)
    private val scoring = AtomicInteger()
    /** Language models freed while an lmScore call was running on them. */
    val lmFreedWhileScoring = AtomicInteger()

    override fun lmFree(handle: Long) {
        lmThreads += Thread.currentThread().name
        if (scoring.get() > 0) lmFreedWhileScoring.incrementAndGet()
        check(liveLms.remove(handle)) { "language model freed twice" }
        lmFrees.incrementAndGet()
        events += "lmFree"
    }

    override fun lmVocabSize(handle: Long): Int {
        check(handle in liveLms) { "use of a freed language model" }
        return lmVocab
    }

    @Volatile var lmContext = 128

    override fun lmContextLength(handle: Long): Int = lmContext

    override fun lmScore(handle: Long, tokens: IntArray, offsets: IntArray, threads: Int): DoubleArray? {
        check(handle in liveLms) { "use of a freed language model" }
        compute("lm", threads)
        scoring.incrementAndGet()
        try {
            val seqs = (0 until offsets.size - 1).map { tokens.copyOfRange(offsets[it], offsets[it + 1]) }
            lmScoreCalls += seqs
            lmScoreStarted.countDown()
            holdLmScore?.let { check(it.await(10, TimeUnit.SECONDS)) { "language model score never released" } }
            val of = lmScoreOf ?: return null
            return DoubleArray(seqs.size) { of(seqs[it]) }
        } finally {
            scoring.decrementAndGet()
        }
    }

    /** Language models loaded and not freed. */
    val liveLanguageModels: Int get() = liveLms.size

    override fun whisperLanguage(handle: Long): String? {
        check(handle in liveHandles) { "language of a freed context" }
        whisperLanguageCalls.incrementAndGet()
        return detectedLanguage ?: lastLanguage
    }

    override fun whisperLanguageShares(handle: Long): FloatArray? {
        check(handle in liveHandles) { "shares of a freed context" }
        sharesCalls.incrementAndGet()
        return lastShares?.copyOf()
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

    /** The streaming VAD call with this index (from 0, a frame of a session) fails; -1: none. */
    @Volatile var failVadProbsAt = -1
    private val vadProbsCalls = AtomicInteger()

    override fun vadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int {
        if (vadProbsCalls.getAndIncrement() == failVadProbsAt) return -1
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
        const val ALL_CPUS = -1L

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

/**
 * A [SpeechEngine] over [native] with [config] (the default), a limit of [threads] threads (2), the CPU layout [cpus]
 * (unknown: today's threads, never pinned), the pinning pause of [pinGuard] and an existing (empty) model file.
 */
internal fun fakeEngine(
    native: FakeNative,
    hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE,
    maxQueuedAudioMs: Long = MAX_QUEUED_AUDIO_MS,
    keepModelLoaded: Boolean = false,
    config: CzechSpeechConfig = CzechSpeechConfig(),
    threads: Int = 2,
    cpus: CpuTopology = CpuTopology.UNKNOWN,
    pinGuard: PinGuard = PinGuard(),
    patternCheck: () -> Int = SdkPatterns::verify,
    languages: List<String> = listOf("cs"),
    languageSwitch: ai.moravian.czechspeech.CzechSpeechModel.LanguageSwitch? = null,
    source: () -> AudioSource,
) = SpeechEngine(
    modelFile = fakeModelFile,
    config = config,
    native = native,
    vadModel = { ByteArray(16) },
    audioSourceFactory = source,
    threads = { threads },
    keepModelLoaded = keepModelLoaded,
    languages = languages,
    languageSwitch = languageSwitch,
    maxQueuedAudioMs = maxQueuedAudioMs,
    hooks = hooks,
    cpuTopology = { cpus },
    pinGuard = pinGuard,
    patternCheck = patternCheck,
)
