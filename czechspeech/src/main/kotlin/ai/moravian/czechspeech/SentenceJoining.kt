package ai.moravian.czechspeech

import ai.moravian.czechspeech.internal.SentenceBoundary
import java.io.File

/**
 * Sentence joining: whether a transcript continues the sentence of the previous one, although a pause split them. Off
 * unless set in [CzechSpeechConfig.sentenceJoining].
 *
 * Each utterance ends at a pause (~0.6 s) and is transcribed on its own, so a pause in the middle of a sentence often
 * comes out as two sentences: Whisper ends the first part with a period and starts the second with a capital
 * ("…můžeš nějak. S tebou spojit…", also "…nějak. s tebou…"), or writes a capital without a period ("A B", "A, B").
 * With sentence joining, the Czech language model of variant selection ([CzechLanguageModel]) decides at each such
 * boundary whether the new transcript continues the previous one, and the [Transcript] says so in
 * [Transcript.sentenceJoin]. [Transcript.text] itself never changes: a client that does not read the new field gets
 * exactly the text it got without sentence joining.
 *
 * The rule is the one measured on parliament speech and read sentences (12 Nov 2026): it fixed 42.5 % of the false
 * periods (17 of 40) and took the period of 0.9 % of the true sentence ends (1 of 117; 0.4 % of 770 in a simulation),
 * and it lowered none of the 6 names at such a boundary. A period is never added; a capital is lowered only when the
 * language model clearly prefers the small letter: names of people kept theirs in the measurement, multi-word names
 * and institutions lost it in 2 to 6 % of the cases ("Ministerstva zemědělství"); nothing is joined across two
 * [CzechSpeech.listen] sessions, across a pause longer than 3 s, or across speech that was lost (a failed or skipped
 * utterance, or one without words). Previews ([CzechSpeech.listenWithPreviews]) are never joined.
 *
 * Cost: only the transcript after a boundary is delayed, by the language model's time on it (a few short texts; the
 * target is under 0.3 s on a phone); the first transcript comes as soon as without sentence joining. The language
 * model (about 135 MB of weights, 11 MB of tokenizer tables on the Java heap) loads right after the first utterance
 * of a session, on the recogniser's thread (an utterance that ends meanwhile waits for it, once), and only when
 * another utterance may still come (not for one sentence ended by [CzechSpeech.stop]); it is freed with the CzechSpeech
 * model, and [CzechSpeech.transcribe] never loads it for sentence joining. With [VariantSelection] on too, both use the
 * same file, loaded once, with the model, before the first utterance. If it cannot be loaded,
 * transcripts come without [Transcript.sentenceJoin], the log says why, and the next session tries again; if it fails
 * on a boundary, that transcript comes without it.
 *
 * @property languageModelFile the language model, for example from
 *   `CzechSpeechModel.download(context, CzechLanguageModel.source(url))`; the same file as
 *   [VariantSelection.languageModelFile] when both are set.
 */
public data class SentenceJoining(public val languageModelFile: File)

/**
 * How a [Transcript] relates to the previous transcript of the same session, decided by [SentenceJoining]; null in
 * [Transcript.sentenceJoin] when the transcript stays as Whisper wrote it (the first one of a session, a boundary that
 * looks like no false sentence end, a true sentence end already written as one, or no decision).
 *
 * To use it, a client that inserted the previous transcript checks that its text still ends with what it inserted
 * (nothing typed or moved in between). If so, and [continuesPrevious] is true, it changes the end of the previous text
 * as [previousEnd] says and appends [continuingText]. Otherwise it appends [standaloneText] (a capital first letter, as
 * after the end of a sentence). When [continuesPrevious] is false, both texts are the same: the end of a sentence,
 * with the capital Whisper left out ("A. b" → "A. B").
 *
 * [toString] does not print the texts.
 *
 * @property continuesPrevious this transcript continues the sentence of the previous one.
 * @property previousEnd what to do with the end of the previous transcript's text before [continuingText] follows it.
 * @property continuingText this transcript's text after the previous one, its first letter as decided: small ("…nějak
 *   s tebou spojit…"), or a capital kept for a name ("Pozdravuj Honzu…").
 * @property standaloneText this transcript's text on its own, with a capital first letter.
 */
public data class SentenceJoin(
    public val continuesPrevious: Boolean,
    public val previousEnd: PreviousEnd,
    public val continuingText: String,
    public val standaloneText: String,
) {
    override fun toString(): String =
        "SentenceJoin(continuesPrevious=$continuesPrevious, previousEnd=$previousEnd, " +
            "continuingText=<${continuingText.length} chars>, standaloneText=<${standaloneText.length} chars>)"
}

/** What to do with the end of the previous transcript when the next one continues it ([SentenceJoin.previousEnd]). */
public enum class PreviousEnd {
    /** Leave it: it ends without a period ("A B", "A, B" written by Whisper, or a sentence end that stays). */
    KEEP,

    /** Delete its final period ("…nějak." → "…nějak"). */
    REMOVE_PERIOD,

    /** Replace its final period with a comma ("…nějak." → "…nějak,"). */
    PERIOD_TO_COMMA,

    /**
     * Delete its final ellipsis ("…nějak..." or "…nějak…" → "…nějak"): two or more dots, or "…", after a letter or digit.
     * The recogniser writes an ellipsis at a pause as an artefact more often than as a trailing off.
     * (Added after the first three values; clients that handle them with `when` need a branch.)
     */
    REMOVE_ELLIPSIS,

    /** Replace its final ellipsis with a comma ("…nějak..." → "…nějak,"). (Added with [REMOVE_ELLIPSIS].) */
    ELLIPSIS_TO_COMMA,

    /** Delete its final comma ("…nějak," → "…nějak"): a pause in the wrong place. (Added with [REMOVE_ELLIPSIS].) */
    REMOVE_COMMA,
    ;

    /**
     * [previousText] with its end changed: the period, the ellipsis ("..." and "…" alike) or the comma at its end (before
     * closing quotes or brackets, which stay) deleted or replaced by a comma, as this value says. Whitespace after it is
     * kept. Text without what the value works on comes back unchanged (a period is not an ellipsis, nor an ellipsis a
     * period).
     */
    public fun applyTo(previousText: String): String = SentenceBoundary.editPreviousEnd(previousText, this)
}
