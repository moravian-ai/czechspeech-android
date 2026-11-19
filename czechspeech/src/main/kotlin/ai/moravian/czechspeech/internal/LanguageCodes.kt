package ai.moravian.czechspeech.internal

private val LANGUAGE_CODE = Regex(SdkPatterns.LANGUAGE_CODE)

/** A language code is two or three lowercase ASCII letters (whisper codes such as `cs`, `yue`, `haw`), as on the web. */
internal fun requireLanguageCode(code: String) {
    require(LANGUAGE_CODE.matches(code)) { "language code must be two or three lowercase letters: \"$code\"" }
}

/**
 * A [switch] belongs to a model of at least two languages and names only languages other than the first (the first is
 * what the rule falls back to, it has no threshold). Null always fits.
 */
internal fun requireSwitchFits(languages: List<String>, switch: ai.moravian.czechspeech.CzechSpeechModel.LanguageSwitch?) {
    if (switch == null) return
    val distinct = languages.distinct()
    require(distinct.size >= 2) { "languageSwitch needs a model of at least two languages" }
    val others = distinct.drop(1).toSet()
    val misplaced = switch.thresholds.keys.filter { it !in others }
    require(misplaced.isEmpty()) {
        "languageSwitch names ${misplaced.joinToString { "\"$it\"" }}, which is not one of the languages after the first (${others.joinToString()})"
    }
}
