package ai.moravian.czechspeech.internal

/**
 * The encoder window of a recognition (whisper.cpp's audio_ctx), as the native library computes it (cs_audio_ctx in
 * czechspeech_jni.c; a test compares the two): the audio in 20 ms frames plus 32, rounded up to a multiple of 16, never
 * under the floor of the model ([ai.moravian.czechspeech.CzechSpeechConfig.minEncoderWindow]), and the whole window
 * from 1500 frames up. Kotlin needs it for the log only; the library uses its own.
 */
internal object EncoderWindow {
    const val DEFAULT_MIN_FRAMES = 512
    const val LOWEST_MIN_FRAMES = 256
    const val FULL_FRAMES = 1500

    /** A floor the library takes: a multiple of 16 from 256 up to the whole window. */
    fun isValidFloor(frames: Int): Boolean = frames in LOWEST_MIN_FRAMES..FULL_FRAMES && frames % 16 == 0

    /** whisper.cpp's audio_ctx for [samples] samples of audio: the window in frames, 0 for the whole window. */
    fun audioCtx(samples: Int, minFrames: Int): Int {
        var frames = (samples + 319) / 320 + 32
        frames = (frames + 15) / 16 * 16
        if (frames < minFrames) frames = minFrames
        return if (frames < FULL_FRAMES) frames else 0
    }

    /** The window in frames, the whole one as 1500: what the encoder works on. */
    fun frames(samples: Int, minFrames: Int): Int = audioCtx(samples, minFrames).let { if (it == 0) FULL_FRAMES else it }
}
