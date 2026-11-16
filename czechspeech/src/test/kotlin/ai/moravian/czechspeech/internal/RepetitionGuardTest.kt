package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.internal.RepetitionGuard.Loop
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The repetition-loop guard on recognition results; the web SDK's test/unit/repetition.test.mjs has the same cases.
 * The loop is a real one: stop() 3.5 s into ParCzech4Speech segment test_0000-0000000333 (CC BY 4.0), cut mid-word.
 */
class RepetitionGuardTest {
    private val phrase = "se státní podporou v roce 2012 až"
    private val loop = "pojišťování vývozu se státní podporou v roce 2012 až se státní podporou v roce 2012 až se státní " +
        "podporou v roce 2012 až se státní podporou v roce 2012 až se st"
    private val cleaned = "pojišťování vývozu se státní podporou v roce 2012 až"

    private fun unchanged(text: String) {
        val r = RepetitionGuard.clean(text)
        assertEquals(text, r.text)
        assertEquals(emptyList<Loop>(), r.loops)
    }

    @Test
    fun theLoopAfterStopMidWordKeepsTheFirstCopyAndDropsTheRepeatsAndTheCutOffCopy() {
        val r = RepetitionGuard.clean(loop)
        assertEquals(cleaned, r.text)
        assertEquals(listOf(Loop(7, 3, true)), r.loops)
        assertEquals("repetition loop removed (7 words, 3 repeats, partial copy at the end)", RepetitionGuard.describe(r.loops))
    }

    @Test
    fun theSameLoopRunOnToTheTokenLimit() {
        val long = "pojišťování vývozu ${List(9) { phrase }.joinToString(" ")} se státní podp"
        val r = RepetitionGuard.clean(long)
        assertEquals(cleaned, r.text)
        assertEquals(listOf(Loop(7, 8, true)), r.loops)
    }

    @Test
    fun normalTextStaysUntouched() {
        unchanged("pojišťování vývozu se státní podporou v roce 2012, sněmovní tisk 91. Děkuji.")
        unchanged("ne, ne, ne")
        unchanged("velmi děkuji, velmi děkuji")
        unchanged("1, 2, 3, 4, 5")
        unchanged("to je to")
        unchanged("Pane předsedající, já jsem říkal, já jsem říkal, že ten zákon nebude fungovat.")
    }

    @Test
    fun aSingleRepeatedWordStaysHoweverOftenItRepeats() {
        unchanged("Ne, ne, ne, ne, ne, ne, ne, ne.")
        unchanged("ne ne ne ne ne ne ne ne ne ne ne ne ne")
    }

    @Test
    fun aLoopInTheMiddleIsCollapsedAndTheTextAfterItIsKept() {
        val r = RepetitionGuard.clean(
            "Návrh zákona o podpoře bydlení o podpoře bydlení o podpoře bydlení projednal výbor pro sociální politiku.",
        )
        assertEquals("Návrh zákona o podpoře bydlení projednal výbor pro sociální politiku.", r.text)
        assertEquals(listOf(Loop(3, 2, false)), r.loops)
        assertEquals("repetition loop removed (3 words, 2 repeats)", RepetitionGuard.describe(r.loops))
    }

    @Test
    fun wordsAreComparedWithoutCaseAndPunctuationAndTheFirstCopyKeepsItsOwn() {
        val r = RepetitionGuard.clean("Děkuji vám. Děkuji vám, děkuji vám! DĚKUJI VÁM")
        assertEquals("Děkuji vám.", r.text)
        assertEquals(listOf(Loop(2, 3, false)), r.loops)
    }

    @Test
    fun theSmallestPhraseThatExplainsARunWins() {
        val r = RepetitionGuard.clean("ano ne ano ne ano ne ano ne ano ne ano ne")
        assertEquals("ano ne", r.text)
        assertEquals(listOf(Loop(2, 5, false)), r.loops)
    }

    @Test
    fun phrasesOf2To12WordsCountLongerOnesDoNot() {
        val twelve = "jedna dvě tři čtyři pět šest sedm osm devět deset jedenáct dvanáct"
        val r = RepetitionGuard.clean("$twelve $twelve $twelve")
        assertEquals(twelve, r.text)
        assertEquals(listOf(Loop(12, 2, false)), r.loops)
        val thirteen = "$twelve třináct"
        unchanged("$thirteen $thirteen $thirteen")
        assertEquals("to je", RepetitionGuard.clean("to je to je to je").text)
    }

    @Test
    fun aTrailingPartialCopyIsDroppedOnlyWhenItIsAPrefixOfThePhrase() {
        val cut = RepetitionGuard.clean("děkuji vám děkuji vám děkuji vám děk")
        assertEquals("děkuji vám", cut.text)
        assertEquals(listOf(Loop(2, 2, true)), cut.loops)
        val other = RepetitionGuard.clean("děkuji vám děkuji vám děkuji vám dobrou noc")
        assertEquals("děkuji vám dobrou noc", other.text)
        assertEquals(listOf(Loop(2, 2, false)), other.loops)
    }

    @Test
    fun aLoopOfLoopsIsRemovedTooTheRuleIsAppliedUntilNothingChanges() {
        val inner = "já jsem já jsem já jsem řekl"
        val r = RepetitionGuard.clean("$inner $inner $inner")
        assertEquals("já jsem řekl", r.text)
        assertEquals(listOf(2, 2, 2, 3), r.loops.map { it.words })
        assertEquals(
            "4 repetition loops removed (2 words, 2 repeats; 2 words, 2 repeats; 2 words, 2 repeats; 3 words, 2 repeats)",
            RepetitionGuard.describe(r.loops),
        )
    }

    @Test
    fun punctuationStandingAloneBelongsToTheWordBeforeIt() {
        val r = RepetitionGuard.clean("a tak dále – a tak dále – a tak dále – a pak konec")
        assertEquals("a tak dále – a pak konec", r.text)
        assertEquals(listOf(Loop(3, 2, false)), r.loops)
    }
}
