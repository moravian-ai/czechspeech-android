package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** [SentenceJoiner.afterContext] and [SentenceJoiner.continuedLater] (the held word of a cut inside speech). */
class SentenceJoinerContextTest {
    @Test
    fun theTextAfterAReplacedHeldWordContinuesAsWhisperWroteIt() {
        val j = SentenceJoiner()
        j.next("Jedna dva tři", 1_000, 5_000, null)
        val step = j.afterContext("čtyři pět. Šest", 5_000, 9_000)
        assertEquals(SentenceJoin(true, PreviousEnd.KEEP, "čtyři pět. Šest", "Čtyři pět. Šest"), step.join)
        assertEquals(0L, step.languageModelMs)
    }

    @Test
    fun noJoinAfterAResetAcrossLostSpeechOrALongPause() {
        val reset = SentenceJoiner()
        reset.next("Jedna dva", 1_000, 5_000, null)
        reset.reset()
        assertNull(reset.afterContext("tři", 5_000, 6_000).join)
        val lost = SentenceJoiner()
        lost.next("Jedna dva", 1_000, 5_000, null)
        lost.lost(5_500)
        assertNull(lost.afterContext("tři", 6_000, 7_000).join)
        val pause = SentenceJoiner()
        pause.next("Jedna dva", 1_000, 5_000, null)
        assertNull(pause.afterContext("tři", 5_000 + SentenceBoundary.MAX_PAUSE_MS + 1, 20_000).join)
        assertNull(SentenceJoiner().afterContext("tři", 0, 1_000).join)
    }

    @Test
    fun afterContextBecomesThePreviousTextOfTheNextDecision() {
        val j = SentenceJoiner()
        j.next("Jedna dva tři", 1_000, 5_000, null)
        j.afterContext("čtyři pět", 5_000, 9_000)
        val step = j.afterCut("šest sedm", 9_000, 12_000)
        assertEquals(true, step.join!!.continuesPrevious)
    }

    @Test
    fun aHeldWordFlushedLaterExtendsThePreviousTextBeforeTheNextDecision() {
        val j = SentenceJoiner()
        j.next("Jedna dva tři", 1_000, 5_000, null)
        j.continuedLater("čtyři.", 5_000)                 // from another thread: queued
        val step = j.afterCut("pět šest", 5_000, 9_000)
        // the previous text ended with the period of the held word: it is the one the cut removes
        assertEquals(PreviousEnd.REMOVE_PERIOD, step.join!!.previousEnd)
    }

    @Test
    fun aFlushQueuedAfterAResetIsForgotten() {
        val j = SentenceJoiner()
        j.next("Jedna dva tři", 1_000, 5_000, null)
        j.reset()
        j.continuedLater("čtyři.", 5_000)
        assertNull(j.afterCut("pět", 5_000, 6_000).join)
    }
}
