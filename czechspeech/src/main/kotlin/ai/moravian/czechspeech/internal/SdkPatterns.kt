package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import java.util.regex.Pattern

/**
 * Every regular expression of the SDK, in one place. No other main source of the module may build one (a test fails
 * for a stray `Regex(`, `toRegex` or `Pattern.compile`): the unit tests run on the desktop JVM, whose regex engine
 * accepts what Android's (ICU) rejects, an unescaped `}` or `]` outside a character class, say, so a pattern that
 * passed every test crashed the app on the phone. The tests read [ALL] and refuse the syntax ICU does not take, and the
 * engine compiles [ALL] when it is created ([verify]), on the engine of the device it runs on.
 *
 * Rules: escape `{`, `}`, `[`, `]` wherever they are literal; no possessive quantifiers, atomic groups, named groups,
 * `\R`, `\h`, or look-behind of unbounded length; spell out Unicode classes (`\p{L}`, `\p{N}`) instead of
 * UNICODE_CHARACTER_CLASS. Prefer a hand-written scan to a pattern when it is as short.
 */
internal object SdkPatterns {
    /** A pattern's name (what an error says) and its source. */
    class Entry(val name: String, val source: String)

    /** `Content-Range: bytes 1000-473992234/473992235` of a resumed download. */
    const val CONTENT_RANGE = """bytes\s+(\d+)-(\d+)/(\d+|\*)"""

    /** A language code: two or three lowercase ASCII letters (`cs`, `yue`), as in the web SDK. */
    const val LANGUAGE_CODE = "[a-z]{2,3}"

    /** A word that is an ordinal number or a number with a point (`3.`, `(1.`, `2,5.`), not the end of a sentence. */
    const val ORDINAL = """^\(?\p{Nd}+[.,]?\p{Nd}*\.$"""

    /** Whatever is not a letter or a digit: what separates words. */
    const val NON_WORD = "[^\\p{L}\\p{N}]+"

    /**
     * Unicode White_Space, as `\s` matches it in the Hugging Face tokenizers (Oniguruma) and Python's regex. Spelled
     * out so that the JVM (ASCII `\s`) and Android (ICU) agree.
     */
    const val WS = "\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

    /** GPT-2's split: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ */
    const val GPT2_SPLIT =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^$WS\\p{L}\\p{N}]+|[$WS]+(?![^$WS])|[$WS]+"

    val ALL: List<Entry> = listOf(
        Entry("CONTENT_RANGE", CONTENT_RANGE),
        Entry("ORDINAL", ORDINAL),
        Entry("NON_WORD", NON_WORD),
        Entry("GPT2_SPLIT", GPT2_SPLIT),
        Entry("LANGUAGE_CODE", LANGUAGE_CODE),
    )

    /**
     * Compiles every pattern of [ALL] on the regex engine of this runtime (Android's on a phone). Throws a
     * [CzechSpeechException] that names the pattern and the cause, so that a defect shows at start-up and not on a rare
     * path weeks later. Returns the number of patterns.
     */
    @JvmStatic
    fun verify(): Int {
        for (entry in ALL) {
            try {
                Pattern.compile(entry.source)
            } catch (e: Throwable) {
                throw CzechSpeechException(
                    "Regular expression ${entry.name} is not valid on this runtime's regex engine: " +
                        "${e.javaClass.simpleName}: ${e.message}",
                    e,
                )
            }
        }
        return ALL.size
    }
}
