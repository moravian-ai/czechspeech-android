package ai.moravian.czechspeech.internal

/**
 * Logging for the SDK. Only timings, counts and states are logged, never transcript text or audio.
 *
 * On Android [sink] is replaced by logcat ([AndroidLog]); on the JVM (tests) it prints to stderr.
 */
internal object CsLog {
    const val INFO = 4
    const val WARN = 5

    @Volatile
    var sink: (level: Int, message: String) -> Unit = { level, message ->
        System.err.println("CzechSpeech ${if (level >= WARN) "W" else "I"}: $message")
    }

    fun i(message: String) = sink(INFO, message)

    fun w(message: String, error: Throwable? = null) =
        sink(WARN, if (error == null) message else "$message: ${error.javaClass.simpleName}: ${error.message}")
}
