package ai.moravian.czechspeech.internal

import android.util.Log

/** Sends [CsLog] to logcat under the tag "CzechSpeech". */
internal object AndroidLog {
    private const val TAG = "CzechSpeech"

    @Volatile
    private var installed = false

    fun install() {
        if (installed || !NativeLoader.isAndroid()) return
        installed = true
        CsLog.sink = { level, message ->
            if (level >= CsLog.WARN) Log.w(TAG, message) else Log.i(TAG, message)
        }
    }
}
