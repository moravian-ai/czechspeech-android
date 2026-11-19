package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class StreamCommitterTest {
    private val noCut: (Long, Long) -> Long = { _, _ -> error("no cut expected") }

    private fun words(n: Int) = (1..n).joinToString(" ") { "w$it" }

    /** The state after the sixth pass of the cut test: five passes without a cut, the sixth one cuts at 3 815 ms. */
    private fun cutCommitter(): StreamCommitter {
        val c = StreamCommitter(0)
        for (k in 1..5) c.absorbPass(0, (k + 1) * 1_000L, words(2 * k + 1), noCut)
        c.absorbPass(0, 7_000, words(13)) { _, _ -> 3_815 }
        return c
    }

    @Test
    fun wordsTwoPassesAgreeOnAreCommittedAndTheRestIsPending() {
        val c = StreamCommitter(audioStartMs = 0)
        val p1 = c.absorbPass(0, 2_000, "dobrý den jak", noCut)
        assertEquals(emptyList<String>(), p1.committed)
        assertEquals(listOf("dobrý", "den", "jak"), p1.pending)
        val p2 = c.absorbPass(0, 3_000, "dobrý den, jak se", noCut)
        assertEquals(listOf("dobrý"), p2.committed)
        assertEquals(listOf("den,", "jak", "se"), p2.pending)
        assertEquals(2_000, p2.agreedEndMs)
        val p3 = c.absorbPass(0, 4_000, "dobrý den, jak se máte", noCut)
        assertEquals(1, p3.prefix)
        assertEquals(listOf("den,", "jak", "se"), p3.committed)
        val f = c.absorbFinal("Dobrý den, jak se máte?")
        assertEquals(4, f.prefix)
        assertEquals(listOf("máte?"), f.words)
    }

    @Test
    fun aPassThatMayNotCommitLeavesEverythingPending() {
        val c = StreamCommitter(0)
        c.absorbPass(0, 2_000, "dobrý den jak", noCut, mayCommit = false)
        val second = c.absorbPass(0, 3_000, "dobrý den jak", noCut, mayCommit = false)
        assertEquals(emptyList<String>(), second.committed)
        assertEquals(listOf("dobrý", "den", "jak"), second.pending)
        assertEquals(0, c.committedCount)
        // the words are only pending, so the next pass that may commit finds them agreed on
        val third = c.absorbPass(0, 4_000, "dobrý den jak se", noCut, mayCommit = true)
        assertEquals(listOf("dobrý", "den", "jak"), third.committed)
        assertEquals(listOf("se"), third.pending)
    }

    @Test
    fun anEmptyHypothesisCommitsNothingAndClearsPending() {
        val c = StreamCommitter(0)
        c.absorbPass(0, 2_000, "a b c", noCut)
        assertEquals(emptyList<String>(), c.absorbPass(0, 3_000, "", noCut).pending)
        assertEquals(emptyList<String>(), c.absorbPass(0, 4_000, "a b c d", noCut).committed)
        assertEquals(listOf("a", "b", "c", "d"), c.absorbPass(0, 5_000, "a b c d e", noCut).committed)
    }

    @Test
    fun aCutMovesTheBufferToTheQuietPointBeforeTheNewestFullyCommittedPass() {
        // passes end every second; pass k (from 1) recognises w1..w(2k+1)
        val c = StreamCommitter(0)
        for (k in 1..5) assertNull(c.absorbPass(0, (k + 1) * 1_000L, words(2 * k + 1), noCut).cut)
        var asked: Pair<Long, Long>? = null
        val p6 = c.absorbPass(0, 7_000, words(13)) { lo, hi -> asked = lo to hi; 3_815 }
        assertEquals(3_600L to 4_100L, asked)
        assertEquals(Cut(atMs = 3_815, bufferBeforeMs = 7_000, bufferAfterMs = 3_185), p6.cut)
        assertEquals(3_815, c.bufferStartMs)
        assertEquals(11, c.committedCount)
    }

    @Test
    fun afterACutTheNextPassSkipsTheCommittedWordsLeftInTheBuffer() {
        val c = cutCommitter()
        val p7 = c.absorbPass(3_815, 8_000, (8..15).joinToString(" ") { "w$it" }, noCut)
        assertEquals(4, p7.prefix)
        assertEquals(listOf("w12", "w13"), p7.committed)
        assertEquals(listOf("w14", "w15"), p7.pending)
        assertEquals(7, p7.seamWord)
    }

    @Test
    fun noCutWithoutAFullyCommittedEarlierPass() {
        val c = StreamCommitter(0)
        for (k in 1..8) assertNull(c.absorbPass(0, (k + 1) * 1_000L, "x$k y$k", noCut).cut)
    }

    @Test
    fun aQuietPointNotAfterTheBufferStartCutsNothing() {
        val c = StreamCommitter(0)
        for (k in 1..5) c.absorbPass(0, (k + 1) * 1_000L, words(2 * k + 1), noCut)
        assertNull(c.absorbPass(0, 7_000, words(13)) { _, _ -> 0 }.cut)
        assertEquals(0, c.bufferStartMs)
    }

    @Test
    fun aBufferOfExactlySixSecondsIsNotCut() {
        val c = StreamCommitter(0)
        for (k in 1..5) assertNull(c.absorbPass(0, (k + 1) * 1_000L, words(2 * k + 1), noCut).cut)
    }

    @Test
    fun aPassFromAnotherBufferStartIsRefused() {
        val c = StreamCommitter(160)
        assertThrows(IllegalArgumentException::class.java) { c.absorbPass(0, 2_000, "a", noCut) }
    }

    @Test
    fun afterTheFinalTheCommitterIsClosed() {
        val c = StreamCommitter(0)
        c.absorbPass(0, 2_000, "a b", noCut)
        c.absorbFinal("a b c")
        assertThrows(IllegalStateException::class.java) { c.absorbPass(0, 3_000, "a b c d", noCut) }
        assertThrows(IllegalStateException::class.java) { c.absorbFinal("a b c d") }
    }
}
