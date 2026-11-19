package ai.moravian.czechspeech.internal

/**
 * The language a `listen` collection is in: the language the next utterance is expected in, which gives every
 * recognition its thresholds ([LanguageRule.thresholds]). It starts as the first language of the list at the start of
 * each collection and changes only when a transcript is sent to the app ([delivered]); an empty, dropped, failed or
 * skipped result, a preview and a rescue change nothing. Written on the recognition loop, read on the whisper thread.
 */
internal class LanguageSession(private val rule: LanguageRule) {
    @Volatile
    var current: String = rule.primary
        private set

    /** How often the session changed its language. */
    var switches = 0
        private set

    /** How many delivered transcripts came out of a rescue. */
    var rescues = 0
        private set

    /**
     * A transcript in [language] was sent. A transcript from a rescue ([rescued]) leaves the language of the session as
     * it is, as it would otherwise get around the minimum lengths.
     */
    fun delivered(language: String, rescued: Boolean) {
        if (rescued) {
            rescues++
        } else if (language.isNotEmpty() && language != current) {
            current = language
            switches++
        }
    }
}
