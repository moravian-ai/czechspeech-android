package ai.moravian.czechspeech.internal

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.Collections

/**
 * What the log says about a recognition that was dropped because its text has no letter or digit: the category of what
 * the model returned and the number of its parts, never the text nor its length in characters.
 */
class DroppedTranscriptTest {
    @Test
    fun noTextAtAllIsEmpty() {
        assertEquals(DroppedTranscript.Category.EMPTY, DroppedTranscript.categoryOf(""))
        assertEquals(DroppedTranscript.Category.EMPTY, DroppedTranscript.categoryOf("   "))
        assertEquals(0, DroppedTranscript.partsOf(""))
    }

    @Test
    fun punctuationAndSpacesOnlyArePunctuation() {
        for (text in listOf(",", "... ...", " . , ", "?!", "- -", "„ “", "…")) {
            assertEquals(text, DroppedTranscript.Category.PUNCTUATION, DroppedTranscript.categoryOf(text))
        }
        assertEquals(2, DroppedTranscript.partsOf("... ..."))
        assertEquals(1, DroppedTranscript.partsOf("  ,  "))
    }

    @Test
    fun textOnlyInBracketsIsBracketed() {
        for (text in listOf("[ ... ]", "( )", "[♪]", "[ ] (...)", "{}")) {
            assertEquals(text, DroppedTranscript.Category.BRACKETED, DroppedTranscript.categoryOf(text))
        }
        assertEquals(3, DroppedTranscript.partsOf("[ ... ]"))
    }

    @Test
    fun bracketPairsAreTakenOutLeftToRightAndAnUnpairedBracketStays() {
        assertEquals(DroppedTranscript.Category.BRACKETED, DroppedTranscript.categoryOf("{ ♪ } [ ] ( ... )"))
        assertEquals(DroppedTranscript.Category.PUNCTUATION, DroppedTranscript.categoryOf("[(])"))   // "[(]" goes, ")" stays
        assertEquals(DroppedTranscript.Category.PUNCTUATION, DroppedTranscript.categoryOf("[ ..."))
        assertEquals(DroppedTranscript.Category.PUNCTUATION, DroppedTranscript.categoryOf("} ] )"))
        assertEquals(DroppedTranscript.Category.SYMBOLS, DroppedTranscript.categoryOf("[♪ ♪"))
        assertEquals(DroppedTranscript.Category.SYMBOLS, DroppedTranscript.categoryOf("[♪] ♪"))
        assertEquals(3, DroppedTranscript.partsOf(" a\tb \n c "))
        assertEquals(2, DroppedTranscript.partsOf("a b"))
    }

    @Test
    fun otherCharactersWithoutLettersAreSymbols() {
        for (text in listOf("♪", "♪ ♪", "© ™", "+ =", "$ %")) {
            assertEquals(text, DroppedTranscript.Category.SYMBOLS, DroppedTranscript.categoryOf(text))
        }
    }

    @Test
    fun theDescriptionNamesTheCategoryAndThePartsAndNothingOfTheText() {
        assertEquals("model returned: punctuation, 2 part(s); language cs (forced)", DroppedTranscript.describe("... ...", "cs", forced = true))
        assertEquals("model returned: empty, 0 part(s); language en (chosen)", DroppedTranscript.describe("", "en", forced = false))
        assertEquals("model returned: bracketed, 1 part(s); language cs (forced)", DroppedTranscript.describe("[...]", "cs", forced = true))
        assertEquals("model returned: symbols, 2 part(s); language unknown (chosen)", DroppedTranscript.describe("♪ ♪", "", forced = false))
    }

    // ---- the line of the recognition ----

    private val logs = Collections.synchronizedList(mutableListOf<String>())
    private val sink = CsLog.sink

    @Before
    fun captureLogs() {
        CsLog.sink = { _, message -> logs += message }
    }

    @After
    fun restoreLogs() {
        CsLog.sink = sink
    }

    private fun lineOf(modelText: String): String = runBlocking {
        logs.clear()
        val native = FakeNative().apply { text = modelText }
        val engine = fakeEngine(native) { ArrayAudioSource(Signals.zeros(0.2)) }
        engine.transcribe(Signals.tone(1.0))
        engine.close()
        engine.awaitClosed()
        logs.single { it.contains("speech segment") }
    }

    @Test
    fun theLineOfADroppedRecognitionSaysWhatTheModelReturnedForEveryCategory() {
        val expected = mapOf(
            "" to "empty, 0 part(s)",
            " ... ..." to "punctuation, 2 part(s)",
            "[ ... ]" to "bracketed, 3 part(s)",
            "♪ ♪" to "symbols, 2 part(s)",
        )
        for ((text, what) in expected) {
            val line = lineOf(text)
            assertTrue(line, line.endsWith(", no letters or digits, dropped (model returned: $what; language cs (forced))"))
            assertFalse("no text in the line: $line", text.isNotBlank() && line.contains(text.trim()))
        }
    }

    @Test
    fun theLineOfAKeptRecognitionIsAsBefore() {
        val line = lineOf("dobrý den")
        assertTrue(line, line.endsWith(", transcript kept"))
    }
}
