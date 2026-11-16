package ai.moravian.czechspeech.internal

import java.io.Closeable

/**
 * 16 kHz mono float audio in [-1, 1], read frame by frame on the capture thread. The microphone implementation is
 * [AndroidMicSource]; tests feed WAV files or synthetic signals.
 */
internal interface AudioSource : Closeable {
    /** Starts capture. Called once, on the capture thread. */
    fun open()

    /**
     * Fills [frame] (blocking). Returns the number of real samples, with the rest of the frame zeroed, or -1 at the
     * end of the stream (a microphone never ends).
     */
    fun read(frame: FloatArray): Int
}
