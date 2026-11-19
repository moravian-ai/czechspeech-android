package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Test

/** The expected values are those of the measurement's Python functions (the measurement's own code) on the same inputs. */
class WordAgreementTest {
    private fun w(s: String) = s.split(" ").filter { it.isNotEmpty() }

    @Test
    fun commonPrefixIsExactIncludingPunctuation() {
        assertEquals(2, WordAgreement.commonPrefix(w("to je konec."), w("to je konec věty")))
        assertEquals(0, WordAgreement.commonPrefix(emptyList(), w("a")))
    }

    @Test
    fun splitPrefixMatchesThePythonCases() {
        assertEquals(1, WordAgreement.splitPrefix(w("Další věta je"), w("Další"), w("věta")))
        assertEquals(2, WordAgreement.splitPrefix(w("konec. Další věta"), w("Další"), w("věta")))
        assertEquals(0, WordAgreement.splitPrefix(w("věta je tady"), w("Další"), w("věta je")))
        assertEquals(1, WordAgreement.splitPrefix(w("alší věta je"), w("Další"), w("věta je")))
        assertEquals(0, WordAgreement.splitPrefix(emptyList(), w("Další"), w("věta")))
        assertEquals(0, WordAgreement.splitPrefix(w("a b"), emptyList(), w("a")))
        assertEquals(1, WordAgreement.splitPrefix(w("dobrý den, jak se máte"), w("dobrý"), w("den, jak se")))
        assertEquals(4, WordAgreement.splitPrefix(w("Dobrý den, jak se máte?"), w("dobrý den, jak se"), w("máte")))
    }

    @Test
    fun splitPrefixMapsAShortHypothesisOntoCommittedWordsOnly() {
        assertEquals(2, WordAgreement.splitPrefix(w("w8 w9"), w("w5 w6 w7 w8 w9"), w("w10")))
    }

    @Test
    fun anchorOverlapMatchesThePythonCases() {
        val com = w("to je první věta a tady pokračuje druhá")
        assertEquals(WordAgreement.Anchor(3, 3), WordAgreement.anchorOverlap(w("Tady pokračuje druhá věta"), com, 3, w("věta")))
        assertEquals(WordAgreement.Anchor(3, 3), WordAgreement.anchorOverlap(w("Tady pokračuje druhá věta"), com, 2, w("věta")))
        assertEquals(WordAgreement.Anchor(3, 3), WordAgreement.anchorOverlap(w("Tady pokračuje druhá věta"), com, 4, w("věta")))
        assertEquals(WordAgreement.Anchor(3, 3), WordAgreement.anchorOverlap(w("dy pokračuje druhá věta"), com, 3, w("věta")))
        assertEquals(WordAgreement.Anchor(0, 0), WordAgreement.anchorOverlap(w("věta která"), com, 1, w("věta která")))
        assertEquals(WordAgreement.Anchor(3, 0), WordAgreement.anchorOverlap(emptyList(), com, 3, w("věta")))
        val eleven = (1..11).map { "w$it" }
        assertEquals(
            WordAgreement.Anchor(4, 4),
            WordAgreement.anchorOverlap((8..15).map { "w$it" }, eleven, 5, w("w12 w13")),
        )
        assertEquals(WordAgreement.Anchor(2, 2), WordAgreement.anchorOverlap(w("w10 w11"), eleven, 3, w("w12")))
    }

    /**
     * Known limits of the algorithm, inherited from the measurement's simulation and pinned so that a change of the
     * port is noticed: when the words after a cut are rewritten differently from the committed ones, the seam can
     * repeat a word or lose one. (The text already committed never changes; only what the seam adds to it can be wrong.)
     */
    @Test
    fun aSeamCanRepeatAWordWhenTheOverlapIsRewrittenDifferently() {
        // the committed end "kračuje drahá věta která" was heard as "to je první věta a tady pokračuje druhá": the best
        // anchor keeps nothing of the buffer, so the whole hypothesis counts as new
        assertEquals(
            WordAgreement.Anchor(0, 0),
            WordAgreement.anchorOverlap(
                w("kračuje drahá věta která"),
                w("to je první věta a tady pokračuje druhá"),
                2,
                w("věta"),
            ),
        )
    }

    @Test
    fun aSeamCanLoseARepeatedWord() {
        // committed "... no no"; if only the last "no" is still in the buffer, the pass "no no dobře je" starts with
        // that "no" and one new "no", yet the anchor takes both as the committed words: the new "no" is lost
        assertEquals(
            WordAgreement.Anchor(2, 2),
            WordAgreement.anchorOverlap(w("no no dobře je"), w("A B no no"), 2, w("no dobře")),
        )
    }

    @Test
    fun matchedWordsIsTheSizeOfDifflibsMatchingBlocks() { // SequenceMatcher(autojunk=False), not the LCS
        fun c(s: String) = s.map { it.toString() }
        assertEquals(4, WordAgreement.matchedWords(c("abxcd"), c("abcd")))
        assertEquals(1, WordAgreement.matchedWords(c("cbaca"), c("bbc"))) // LCS would be 2
        assertEquals(2, WordAgreement.matchedWords(c("aaaca"), c("abaa"))) // LCS 3
        assertEquals(2, WordAgreement.matchedWords(c("acbcc"), c("cccc"))) // LCS 3
        assertEquals(0, WordAgreement.matchedWords(emptyList(), c("a")))
    }

    @Test
    fun wordsAreSplitAtAsciiWhitespaceOnly() {
        assertEquals(listOf("a", "b", "c", "d"), WordAgreement.words(" a\tb\nc \u000B\u000C\r d "))
        assertEquals(listOf("a b"), WordAgreement.words("a b"))
        assertEquals(emptyList<String>(), WordAgreement.words(""))
    }

    @Test
    fun aKeyKeepsWhatPythonsIsalnumKeeps() {
        assertEquals("konec", WordAgreement.key("Konec."))
        assertEquals("ano", WordAgreement.key("„Ano“"))
        assertEquals("x²", WordAgreement.key("x²"))
        assertEquals("2½", WordAgreement.key("2½"))
        assertEquals("", WordAgreement.key("—"))
    }
}
