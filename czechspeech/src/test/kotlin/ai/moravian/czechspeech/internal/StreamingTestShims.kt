package ai.moravian.czechspeech.internal

/** The language the streaming tests recognise in, unless a test is about the language (the model's only one). */
internal const val TEST_LANGUAGE = "cs"

/** A pass result in [TEST_LANGUAGE]: most streaming tests are not about the language. */
internal fun PassResult(text: String, recogniserMs: Long) = PassResult(text, recogniserMs, TEST_LANGUAGE)

/** A chunk in [TEST_LANGUAGE]. */
internal fun Chunk(text: String, startMs: Long, endMs: Long, processingMs: Long, firstOfPiece: Boolean) =
    Chunk(text, startMs, endMs, processingMs, TEST_LANGUAGE, firstOfPiece)

/** [StreamingTracker.absorbFinal] for a final recognised in [TEST_LANGUAGE]. */
internal fun StreamingTracker.absorbFinal(plan: FinalPlan, text: String, recogniserMs: Long): Chunk? =
    absorbFinal(plan, text, recogniserMs, TEST_LANGUAGE)
