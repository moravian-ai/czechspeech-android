package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The text anchor of a piece after a cut inside speech ([SeamAnchor]); the expected values are the measurement's. */
class SeamAnchorTest {
    private fun w(s: String) = s.split(" ").filter { it.isNotEmpty() }

    @Test
    fun theExpectedCountIsRoundedHalfToEvenLikePythonsRound() {
        assertEquals("2.5 gives 2", 2, SeamAnchor.expected(5, 5_000))
        assertEquals("3.5 gives 4", 4, SeamAnchor.expected(7, 5_000))
        assertEquals("a piece of 2.5 s or less is covered whole", 5, SeamAnchor.expected(5, 2_000))
        assertEquals(5, SeamAnchor.expected(5, 2_500))
        assertEquals("1.5 gives 2", 2, SeamAnchor.expected(3, 5_000))
        assertEquals("0.5 gives 0", 0, SeamAnchor.expected(1, 5_000))
    }

    @Test
    fun heldWordsAreTakenOffTheExpectedCountAndItNeverGoesBelowZero() {
        assertEquals(1, SeamAnchor.expected(3, 5_000, held = 1))
        assertEquals(0, SeamAnchor.expected(1, 10_000, held = 1))
        assertEquals(0, SeamAnchor.expected(0, 4_000))
    }

    @Test
    fun anAnchorOfTwoMatchingWordsIsAcceptedAndTheNewTextStartsAfterIt() {
        val delivered = w("a b c d e f g h")
        val hyp = w("g h i j k")
        val d = SeamAnchor.decide(hyp, delivered, expected = 2)
        assertEquals(2, d.inBuffer)
        assertEquals(2, d.prefix)
        assertEquals(2, d.matched)
        assertTrue(d.ok)
        assertEquals(2, SeamAnchor.continuation(hyp, delivered, 2))
    }

    @Test
    fun caseAndPunctuationDoNotMatterForTheAnchor() {
        val delivered = w("to je první věta a tady pokračuje druhá")
        assertEquals(3, SeamAnchor.continuation(w("Tady pokračuje, druhá! věta"), delivered, 3))
    }

    @Test
    fun oneMatchingWordIsNoAnchor() {
        val d = SeamAnchor.decide(w("d x y z"), w("a b c d"), expected = 2)
        assertEquals(1, d.prefix)
        assertEquals(1, d.matched)
        assertFalse(d.ok)
        assertNull(SeamAnchor.continuation(w("d x y z"), w("a b c d"), 2))
    }

    @Test
    fun aRunWithoutWordsHasNoAnchorAndAnEmptyDeliveredListNeither() {
        assertFalse(SeamAnchor.decide(emptyList(), w("a b c"), 2).ok)
        assertFalse(SeamAnchor.decide(w("a b c"), emptyList(), 0).ok)
        assertNull(SeamAnchor.continuation(emptyList(), emptyList(), 0))
    }

    @Test
    fun theRestNeedsALetterOrADigit() {
        assertTrue(SeamAnchor.hasText(w("... 2025")))
        assertTrue(SeamAnchor.hasText(w("Sb.")))
        assertFalse(SeamAnchor.hasText(w("... - ,")))
        assertFalse(SeamAnchor.hasText(emptyList()))
    }
}
