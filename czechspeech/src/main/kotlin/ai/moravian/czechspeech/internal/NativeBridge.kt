package ai.moravian.czechspeech.internal

import java.io.File

/**
 * The calls the pipeline makes into native code. [JniNativeApi] is the real implementation; tests can supply a fake.
 *
 * Audio arrays are copied by the native side into its own buffer, which it zeroes and frees before returning;
 * the arrays passed in are never modified.
 */
internal interface NativeApi {
    fun systemInfo(): String

    /** Loads the whisper model; 0 on failure. */
    fun whisperInit(modelPath: String): Long
    fun whisperFree(handle: Long)

    /**
     * Sets the floor of the encoder window, in frames, for every recognition on this context: final text, preview and
     * n-best alike. A loaded context has 512 until this is called.
     */
    fun whisperSetMinAudioCtx(handle: Long, frames: Int)

    /**
     * Transcript as UTF-8 bytes, or null when whisper failed or was aborted with [whisperAbort] for [generation].
     * [languages] is one code ("cs": forced, no detection) or a comma-separated list ("pl,en": whisper_full detects the
     * language on the window of the call and the rule of [pickLanguage] picks one of them). [thresholds] is that
     * rule's input, one per language of the list (the first is not read; NaN = not a candidate, otherwise 0..1), or
     * null: the most probable of the list. A length other than the list's, a value out of range, or thresholds with
     * a single code throw IllegalArgumentException before any work.
     */
    fun whisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    /** The language the last [whisperTranscribe] on [handle] decided on (the detected one), or null when unknown. */
    fun whisperLanguage(handle: Long): String?

    /**
     * The shares of the languages of the list of the last recognition on [handle], in the list's order and rescaled
     * to sum 1: what the language was picked from. Null when the language was forced, or the call failed or was
     * cancelled.
     */
    fun whisperLanguageShares(handle: Long): FloatArray?

    /**
     * [whisperTranscribe] for a preview: one greedy pass, without whisper.cpp's temperature fallback, which audio cut
     * mid-word triggers often for text that is replaced anyway. [whisperAbort] stops it the same way; [languages] and
     * [thresholds] as in [whisperTranscribe] (the same rule, the same checks).
     */
    fun whisperPreview(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    /** Makes a running [whisperTranscribe] with the same [generation] stop early. Callable from any thread. */
    fun whisperAbort(handle: Long, generation: Int)

    /**
     * Beam search with [beamSize] beams; every beam's hypothesis per window, serialised as [NBest.parse] reads it, or
     * null when whisper failed or was aborted with [whisperAbort] for [generation]. [languages] and [thresholds] as in
     * [whisperTranscribe] (the same rule, the same checks); [whisperLanguage] tells the language of the last one.
     */
    fun whisperTranscribeNBest(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        beamSize: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    /** Loads the GPT-2 language model (GGUF) of variant selection; 0 on failure. */
    fun lmInit(modelPath: String): Long
    fun lmFree(handle: Long)
    fun lmVocabSize(handle: Long): Int
    fun lmContextLength(handle: Long): Int

    /**
     * Sum of token log-probabilities of each sequence tokens[offsets[i] until offsets[i + 1]] (each starting with the
     * start token), or null on invalid input or a failure.
     */
    fun lmScore(handle: Long, tokens: IntArray, offsets: IntArray, threads: Int): DoubleArray?

    /** Loads Silero VAD from the ggml file bytes; 0 on failure. */
    fun vadInit(model: ByteArray, threads: Int): Long
    fun vadFree(handle: Long)
    fun vadReset(handle: Long)

    /** Speech probabilities of consecutive 512-sample windows, continuing the LSTM state; returns the window count. */
    fun vadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int

    /** Speech segments found by whisper_vad_segments_from_samples at [threshold] (fresh state), -1 on error. */
    fun vadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int

    /**
     * The CPUs the calling thread may run on, as a bit mask (bit i = CPU i): the affinity mask the kernel keeps for it,
     * which Android's cpuset of the process (top-app, foreground, background) bounds. 0 when it cannot be read, and
     * when a CPU above 63 is in it.
     */
    fun threadCpus(): Long

    /**
     * The CPUs of [threadCpus] that run now (sched_getaffinity: without the offline and parked ones), as a bit mask; 0
     * when it cannot be read.
     */
    fun activeThreadCpus(): Long

    /**
     * Sets the calling thread's CPU affinity to [mask] (sched_setaffinity); threads it starts afterwards inherit it.
     * False when the kernel refuses it (no allowed CPU in it, say); the affinity then stays as it was.
     */
    fun setThreadCpus(mask: Long): Boolean

    /**
     * Lets the calling thread run on every CPU (sched_setaffinity with all of them), which the kernel bounds by the
     * cpuset of the moment. On Linux 6.2+ this also drops the thread's earlier request, which the kernel otherwise
     * keeps, intersects every later cpuset with and gives to every thread the thread starts. False when refused.
     */
    fun unpinThread(): Boolean
}

/** [NativeApi] over the JNI library; loads the right CPU variant on first use. */
internal object JniNativeApi : NativeApi {
    // Every handle comes from an *Init call, so the library is loaded before any call that takes a handle.
    override fun systemInfo(): String {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeSystemInfo()
    }

    override fun whisperInit(modelPath: String): Long {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeWhisperInit(modelPath)
    }

    override fun whisperFree(handle: Long) = NativeBridge.nativeWhisperFree(handle)

    override fun whisperSetMinAudioCtx(handle: Long, frames: Int) = NativeBridge.nativeWhisperSetMinAudioCtx(handle, frames)

    override fun whisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray? = NativeBridge.nativeWhisperTranscribe(handle, generation, samples, n, threads, languages, thresholds)

    override fun whisperLanguage(handle: Long): String? = NativeBridge.nativeWhisperLanguage(handle)

    override fun whisperLanguageShares(handle: Long): FloatArray? = NativeBridge.nativeWhisperLanguageShares(handle)

    override fun whisperPreview(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray? = NativeBridge.nativeWhisperPreview(handle, generation, samples, n, threads, languages, thresholds)

    override fun whisperAbort(handle: Long, generation: Int) = NativeBridge.nativeWhisperAbort(handle, generation)

    override fun whisperTranscribeNBest(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        beamSize: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray? =
        NativeBridge.nativeWhisperTranscribeNBest(handle, generation, samples, n, threads, beamSize, languages, thresholds)

    override fun lmInit(modelPath: String): Long {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeLmInit(modelPath)
    }

    override fun lmFree(handle: Long) = NativeBridge.nativeLmFree(handle)
    override fun lmVocabSize(handle: Long): Int = NativeBridge.nativeLmVocabSize(handle)
    override fun lmContextLength(handle: Long): Int = NativeBridge.nativeLmContextLength(handle)

    override fun lmScore(handle: Long, tokens: IntArray, offsets: IntArray, threads: Int): DoubleArray? =
        NativeBridge.nativeLmScore(handle, tokens, offsets, threads)

    override fun vadInit(model: ByteArray, threads: Int): Long {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeVadInit(model, threads)
    }

    override fun vadFree(handle: Long) = NativeBridge.nativeVadFree(handle)
    override fun vadReset(handle: Long) = NativeBridge.nativeVadReset(handle)

    override fun vadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int =
        NativeBridge.nativeVadProbs(handle, samples, n, out)

    override fun vadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int =
        NativeBridge.nativeVadSpeechSegments(handle, samples, n, threshold)

    /**
     * Cpus_allowed of /proc/thread-self/status: the thread's affinity mask as the kernel keeps it. sched_getaffinity
     * (the fallback) leaves out the cores that are offline or parked at that moment (Qualcomm's core_ctl parks idle
     * big cores), which would pin the recogniser to fewer cores than its cpuset allows for the whole call.
     */
    override fun threadCpus(): Long = threadCpus(File("/proc/thread-self/status"))

    /** [threadCpus] from [status], the calling thread's status file, else from sched_getaffinity. */
    fun threadCpus(status: File): Long {
        val text = try {
            status.readText()
        } catch (e: Exception) {
            null
        }
        text?.let(CpuMask::parseCpusAllowed)?.let { return it }
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeThreadCpus()
    }

    override fun activeThreadCpus(): Long {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeThreadCpus()
    }

    override fun setThreadCpus(mask: Long): Boolean {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeSetThreadCpus(mask)
    }

    override fun unpinThread(): Boolean {
        NativeLoader.ensureLoaded()
        return NativeBridge.nativeUnpinThread()
    }
}

/**
 * JNI declarations, implemented in native/src/czechspeech_jni.c. Names are bound by the consumer ProGuard rules.
 * The methods are public (not internal) so that Kotlin does not mangle their JVM names.
 */
internal object NativeBridge {
    @JvmStatic external fun nativeSystemInfo(): String

    /**
     * whisper.cpp audio_ctx (encoder window in frames, 0 = full) for an utterance of [nSamples] samples with the floor
     * [minFrames]: the formula alone, for tests.
     */
    @JvmStatic external fun nativeAudioCtx(nSamples: Int, minFrames: Int): Int

    /** Sets the floor of the encoder window of every recognition on this context; refuses one out of range. */
    @JvmStatic external fun nativeWhisperSetMinAudioCtx(handle: Long, minFrames: Int)
    @JvmStatic external fun nativeWhisperInit(modelPath: String): Long
    @JvmStatic external fun nativeWhisperFree(handle: Long)

    @JvmStatic external fun nativeWhisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    @JvmStatic external fun nativeWhisperPreview(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    @JvmStatic external fun nativeWhisperLanguage(handle: Long): String?

    /** The shares of the languages of the last list recognition (see [NativeApi.whisperLanguageShares]). */
    @JvmStatic external fun nativeWhisperLanguageShares(handle: Long): FloatArray?

    /** The index `cs_pick_language` gives for [shares] and [thresholds] (tests: compared with [pickLanguage]). */
    @JvmStatic external fun nativePickLanguage(shares: FloatArray, thresholds: FloatArray?): Int

    /** Encoder runs of this context so far (tests and logs). */
    @JvmStatic external fun nativeWhisperEncodeRuns(handle: Long): Int

    @JvmStatic external fun nativeWhisperAbort(handle: Long, generation: Int)

    @JvmStatic external fun nativeWhisperTranscribeNBest(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        beamSize: Int,
        languages: String,
        thresholds: FloatArray?,
    ): ByteArray?

    @JvmStatic external fun nativeLmInit(modelPath: String): Long
    @JvmStatic external fun nativeLmFree(handle: Long)
    @JvmStatic external fun nativeLmVocabSize(handle: Long): Int
    @JvmStatic external fun nativeLmContextLength(handle: Long): Int

    /** Bytes the language model holds: {weights, compute buffer kept between calls} (tests and diagnostics). */
    @JvmStatic external fun nativeLmBufferBytes(handle: Long): LongArray
    @JvmStatic external fun nativeLmScore(handle: Long, tokens: IntArray, offsets: IntArray, threads: Int): DoubleArray?
    @JvmStatic external fun nativeVadInit(model: ByteArray, threads: Int): Long
    @JvmStatic external fun nativeVadFree(handle: Long)
    @JvmStatic external fun nativeVadReset(handle: Long)
    @JvmStatic external fun nativeVadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int
    @JvmStatic external fun nativeVadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int

    /** sched_getaffinity of the calling thread as a bit mask (allowed and active CPUs); 0 on failure or above CPU 63. */
    @JvmStatic external fun nativeThreadCpus(): Long

    /** sched_setaffinity of the calling thread to the CPUs of [mask]; false when refused. */
    @JvmStatic external fun nativeSetThreadCpus(mask: Long): Boolean

    /** sched_setaffinity of the calling thread to every CPU (the kernel bounds it by the cpuset); false when refused. */
    @JvmStatic external fun nativeUnpinThread(): Boolean
}
