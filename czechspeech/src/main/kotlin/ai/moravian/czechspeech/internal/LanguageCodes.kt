package ai.moravian.czechspeech.internal

private val LANGUAGE_CODE = Regex("[a-z]{2,3}")

/** A language code is two or three lowercase ASCII letters (whisper codes such as `cs`, `yue`, `haw`), as on the web. */
internal fun requireLanguageCode(code: String) {
    require(LANGUAGE_CODE.matches(code)) { "language code must be two or three lowercase letters: \"$code\"" }
}
