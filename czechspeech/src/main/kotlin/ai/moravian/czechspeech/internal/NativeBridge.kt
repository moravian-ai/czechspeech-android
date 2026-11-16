package ai.moravian.czechspeech.internal

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
     * Transcript as UTF-8 bytes, or null when whisper failed or was aborted with [whisperAbort] for [generation].
     * [languages] is one code ("cs": forced, no detection pass) or a comma-separated list ("pl,en": the language is
     * detected among them).
     */
    fun whisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
    ): ByteArray?

    /** The language the last [whisperTranscribe] on [handle] decided on (the detected one), or null when unknown. */
    fun whisperLanguage(handle: Long): String?

    /** Makes a running [whisperTranscribe] with the same [generation] stop early. Callable from any thread. */
    fun whisperAbort(handle: Long, generation: Int)

    /** Loads Silero VAD from the ggml file bytes; 0 on failure. */
    fun vadInit(model: ByteArray, threads: Int): Long
    fun vadFree(handle: Long)
    fun vadReset(handle: Long)

    /** Speech probabilities of consecutive 512-sample windows, continuing the LSTM state; returns the window count. */
    fun vadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int

    /** Speech segments found by whisper_vad_segments_from_samples at [threshold] (fresh state), -1 on error. */
    fun vadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int
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

    override fun whisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
    ): ByteArray? = NativeBridge.nativeWhisperTranscribe(handle, generation, samples, n, threads, languages)

    override fun whisperLanguage(handle: Long): String? = NativeBridge.nativeWhisperLanguage(handle)

    override fun whisperAbort(handle: Long, generation: Int) = NativeBridge.nativeWhisperAbort(handle, generation)

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
}

/**
 * JNI declarations, implemented in native/src/czechspeech_jni.c. Names are bound by the consumer ProGuard rules.
 * The methods are public (not internal) so that Kotlin does not mangle their JVM names.
 */
internal object NativeBridge {
    @JvmStatic external fun nativeSystemInfo(): String

    /** whisper.cpp audio_ctx (encoder window in frames, 0 = full) used for an utterance of [nSamples] samples. */
    @JvmStatic external fun nativeAudioCtx(nSamples: Int): Int
    @JvmStatic external fun nativeWhisperInit(modelPath: String): Long
    @JvmStatic external fun nativeWhisperFree(handle: Long)

    @JvmStatic external fun nativeWhisperTranscribe(
        handle: Long,
        generation: Int,
        samples: FloatArray,
        n: Int,
        threads: Int,
        languages: String,
    ): ByteArray?

    @JvmStatic external fun nativeWhisperLanguage(handle: Long): String?
    @JvmStatic external fun nativeWhisperAbort(handle: Long, generation: Int)
    @JvmStatic external fun nativeVadInit(model: ByteArray, threads: Int): Long
    @JvmStatic external fun nativeVadFree(handle: Long)
    @JvmStatic external fun nativeVadReset(handle: Long)
    @JvmStatic external fun nativeVadProbs(handle: Long, samples: FloatArray, n: Int, out: FloatArray): Int
    @JvmStatic external fun nativeVadSpeechSegments(handle: Long, samples: FloatArray, n: Int, threshold: Float): Int
}
