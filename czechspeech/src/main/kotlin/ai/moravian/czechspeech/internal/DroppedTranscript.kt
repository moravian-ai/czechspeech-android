package ai.moravian.czechspeech.internal

/**
 * What the log says about a recognition whose text has no letter or digit and is dropped: which kind of nothing the model
 * returned and in how many parts. Never the text, nor its length in characters (the text of a transcript stays out of
 * the log). Diagnostics only: it changes nothing about what is dropped.
 *
 * What cannot be said without a change of the native library: whisper.cpp's probability of no speech, the average
 * log-probability and the number of tokens of the output (the JNI hands over the text only). The line does say which
 * language the recogniser ran in and whether it was forced or chosen.
 *
 * No regular expression here: a pattern with an unescaped `}` passed every test on the desktop JVM and crashed on the
 * phone (Android's regex engine rejects it), and the hand-written scans below are no longer than the patterns were.
 */
internal object DroppedTranscript {
    enum class Category(val label: String) {
        /** The model returned no text (or only spaces). */
        EMPTY("empty"),

        /** Only brackets with nothing but punctuation, spaces and symbols in them: `[ ... ]`, `( )`, `[♪]`. */
        BRACKETED("bracketed"),

        /** Only punctuation and spaces: `,`, `... ...`, `?!`. */
        PUNCTUATION("punctuation"),

        /** Other characters without a letter or digit: `♪`, `©`, `+`. */
        SYMBOLS("symbols"),
    }

    private fun isPunctuation(c: Char): Boolean = when (Character.getType(c).toByte()) {
        Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION, Character.END_PUNCTUATION,
        Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION, Character.OTHER_PUNCTUATION,
        -> true
        else -> false
    }

    /**
     * [text] without its bracket pairs: `[` up to the first `]`, `(` up to the first `)`, `{` up to the first `}`,
     * left to right; an opening bracket without its closing one stays.
     */
    private fun withoutBracketPairs(text: String): String {
        val out = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val close = when (text[i]) {
                '[' -> ']'
                '(' -> ')'
                '{' -> '}'
                else -> null
            }
            val end = if (close == null) -1 else text.indexOf(close, i + 1)
            if (end < 0) {
                out.append(text[i])
                i++
            } else {
                i = end + 1
            }
        }
        return out.toString()
    }

    fun categoryOf(text: String): Category {
        if (text.isBlank()) return Category.EMPTY
        if (withoutBracketPairs(text).let { it.length < text.length && it.isBlank() }) return Category.BRACKETED
        return if (text.all { it.isWhitespace() || isPunctuation(it) }) Category.PUNCTUATION else Category.SYMBOLS
    }

    /** The parts of [text] between runs of spaces: what Whisper's tokens come to without the tokenizer. */
    fun partsOf(text: String): Int {
        var parts = 0
        var inPart = false
        for (c in text) {
            val space = c.isWhitespace()
            if (!space && !inPart) parts++
            inPart = !space
        }
        return parts
    }

    /**
     * The diagnosis for the log line of a dropped recognition of [text]: the language the recogniser ran in (empty when
     * it is not known) and whether it was forced or chosen among several.
     */
    fun describe(text: String, language: String, forced: Boolean): String =
        "model returned: ${categoryOf(text).label}, ${partsOf(text)} part(s); language ${language.ifEmpty { "unknown" }} (${if (forced) "forced" else "chosen"})"
}
