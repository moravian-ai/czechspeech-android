package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process

/**
 * The microphone: [AudioRecord] with the VOICE_RECOGNITION source (tuned for recognisers, little or no processing),
 * 16 kHz mono float. Samples go straight into the caller's frame; nothing is buffered here beyond AudioRecord's own
 * ring buffer, and nothing touches the disk.
 *
 * Every platform failure reaches the app as a [CzechSpeechException] whose message says what happened and what can
 * be done, with the platform exception as its cause. The one exception: a [SecurityException] (RECORD_AUDIO revoked
 * after the check in [create]) stays a SecurityException, as listen() documents.
 */
internal class AndroidMicSource internal constructor(private val platform: Platform) : AudioSource {
    /** The platform calls, behind a seam for JVM tests (android.jar has no implementations there). */
    internal interface Platform {
        fun raiseThreadPriority()
        fun minBufferBytes(): Int
        fun newRecorder(bufferBytes: Int): Recorder
    }

    /** The calls made on one [AudioRecord]. */
    internal interface Recorder {
        val isInitialized: Boolean
        val isRecording: Boolean
        fun start()
        fun read(frame: FloatArray, offset: Int, count: Int): Int
        fun stop()
        fun release()
    }

    private var recorder: Recorder? = null

    override fun open() {
        try {
            platform.raiseThreadPriority()
        } catch (e: RuntimeException) {
            // SecurityException or IllegalArgumentException on some builds. Capture works at normal priority too, and
            // a SecurityException from here must not read as a missing RECORD_AUDIO permission.
            CsLog.w("capture runs at normal priority", e)
        }
        val minBytes = platform.minBufferBytes()
        if (minBytes <= 0) throw unsupported("getMinBufferSize returned $minBytes", null)
        // At least 0.5 s of buffer so a busy moment on the capture thread does not drop audio.
        val bytes = maxOf(minBytes * 2, SAMPLE_RATE * 4 / 2)
        val r = try {
            platform.newRecorder(bytes)
        } catch (e: SecurityException) {
            throw SecurityException(PERMISSION_MESSAGE, e)
        } catch (e: IllegalArgumentException) {
            throw unsupported(e.message, e)
        } catch (e: UnsupportedOperationException) {
            throw unsupported(e.message, e)
        }
        if (!r.isInitialized) {
            releaseQuietly(r)
            throw CzechSpeechException(
                "The microphone could not be opened: it is unavailable or in use by another app; try again later",
            )
        }
        try {
            r.start()
        } catch (e: IllegalStateException) {
            releaseQuietly(r)
            throw CzechSpeechException(
                "The microphone could not be started: it may be in use by another app; try again later", e,
            )
        }
        if (!r.isRecording) {
            releaseQuietly(r)
            throw CzechSpeechException("The microphone is in use by another app (a call, a recorder); try again later")
        }
        recorder = r
    }

    override fun read(frame: FloatArray): Int {
        val r = recorder ?: throw IllegalStateException("microphone not open")
        var offset = 0
        var emptyReads = 0
        while (offset < frame.size) {
            val n = r.read(frame, offset, frame.size - offset)
            when {
                n > 0 -> {
                    offset += n
                    emptyReads = 0
                }
                n == 0 -> if (++emptyReads > MAX_EMPTY_READS) {
                    throw CzechSpeechException("The microphone stopped delivering audio; start listening again")
                }
                n == AudioRecord.ERROR_DEAD_OBJECT -> throw CzechSpeechException(
                    "The microphone was lost (the audio service restarted, or another app took the microphone); " +
                        "start listening again",
                )
                else -> throw CzechSpeechException("Microphone read failed (AudioRecord error $n)")
            }
        }
        return frame.size
    }

    override fun close() {
        val r = recorder ?: return
        recorder = null
        try {
            r.stop()
        } catch (e: IllegalStateException) {
            // not recording
        }
        r.release()
    }

    /** The real platform: [AudioRecord] and [Process]. */
    private object AndroidPlatform : Platform {
        override fun raiseThreadPriority() = Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

        override fun minBufferBytes(): Int = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)

        @SuppressLint("MissingPermission") // checked in create()
        override fun newRecorder(bufferBytes: Int): Recorder = AudioRecordRecorder(
            AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE, CHANNEL, ENCODING, bufferBytes),
        )
    }

    private class AudioRecordRecorder(private val record: AudioRecord) : Recorder {
        override val isInitialized: Boolean get() = record.state == AudioRecord.STATE_INITIALIZED
        override val isRecording: Boolean get() = record.recordingState == AudioRecord.RECORDSTATE_RECORDING
        override fun start() = record.startRecording()
        override fun read(frame: FloatArray, offset: Int, count: Int): Int =
            record.read(frame, offset, count, AudioRecord.READ_BLOCKING)
        override fun stop() = record.stop()
        override fun release() = record.release()
    }

    companion object {
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_FLOAT
        private const val MAX_EMPTY_READS = 100
        private const val PERMISSION_MESSAGE = "CzechSpeech.listen() needs the RECORD_AUDIO permission"

        fun create(context: Context): AndroidMicSource {
            if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                throw SecurityException(PERMISSION_MESSAGE)
            }
            return AndroidMicSource(AndroidPlatform)
        }

        private fun unsupported(detail: String?, cause: Throwable?) = CzechSpeechException(
            "This device cannot record 16 kHz mono float audio from the microphone" +
                (if (detail.isNullOrBlank()) "" else " ($detail)"),
            cause,
        )

        private fun releaseQuietly(recorder: Recorder) {
            try {
                recorder.release()
            } catch (e: RuntimeException) {
                CsLog.w("releasing the AudioRecord failed", e)
            }
        }
    }
}
