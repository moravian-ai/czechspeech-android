package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Test

/** The stable part of a preview: the words two passes in a row start with (LocalAgreement-2). */
class PreviewAgreementTest {
    private fun stable(previous: String, current: String) =
        current.substring(0, PreviewAgreement.stableLength(previous, current))

    @Test
    fun theWordsBothPassesStartWithAreStable() {
        assertEquals("Dobrý den, jak se", stable("Dobrý den, jak se má", "Dobrý den, jak se máte dnes"))
        assertEquals("Dobrý den, jak se", stable("Dobrý den, jak se máte dnes", "Dobrý den, jak se vám daří"))
    }

    @Test
    fun aWordCutByTheEndOfTheAudioIsNotStableUntilTheNextPassRepeatsIt() {
        // the first pass ended inside "zákazníkovi"
        assertEquals("Pošlu to", stable("Pošlu to zákaz", "Pošlu to zákazníkovi zítra"))
        assertEquals("Pošlu to zákazníkovi zítra", stable("Pošlu to zákazníkovi zítra", "Pošlu to zákazníkovi zítra ráno."))
    }

    @Test
    fun nothingIsStableWhenTheFirstWordsDifferOrThereIsNoPreviousPass() {
        assertEquals("", stable("Ano", "A no tak"))
        assertEquals("", stable("", "Dobrý den"))
        assertEquals("", stable("Dobrý den", ""))
        assertEquals("", stable("   ", "Dobrý den"))
    }

    @Test
    fun theSameTextIsStableToItsEnd() {
        assertEquals("Dobrý den.", stable("Dobrý den.", "Dobrý den."))
    }

    @Test
    fun caseAndPunctuationAroundAWordDoNotMakeItAnotherWord() {
        // the sentence went on: the period became a comma, the capital a small letter
        assertEquals("Jasně, ale můžeš nějak, s tebou", stable("Jasně, ale můžeš nějak. S tebou", "Jasně, ale můžeš nějak, s tebou spojit."))
        assertEquals("„Dobrý“ den", stable("dobrý den...", "„Dobrý“ den přeji"))
    }

    @Test
    fun theStablePartEndsAtTheEndOfAWordWithItsPunctuation() {
        val current = "Dobrý den,  jak se máte"
        val n = PreviewAgreement.stableLength("Dobrý den jak to", current)
        assertEquals("Dobrý den,  jak", current.substring(0, n))
        assertEquals("  se máte".trimStart(), current.substring(n).trimStart())
    }

    @Test
    fun whitespaceAtTheStartAndBetweenWordsIsNotCompared() {
        assertEquals(" Dobrý\tden", stable("Dobrý  den a", " Dobrý\tden b"))
    }

    @Test
    fun wordsOfPunctuationAloneAreComparedAsTheyAre() {
        assertEquals("To je – to", stable("To je – to snad", "To je – to není"))
        assertEquals("To je", stable("To je – to", "To je ... to"))
    }

    @Test
    fun diacriticsMakeAnotherWord() {
        // endings and diacritics are what the passes disagree about: "byly" is not "bylý"
        assertEquals("Oni", stable("Oni byly doma", "Oni bylý doma"))
    }
}
