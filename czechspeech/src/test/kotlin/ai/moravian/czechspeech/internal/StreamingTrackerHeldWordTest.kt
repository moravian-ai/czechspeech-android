package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The held word of [CutSeam.CONTEXT_AND_HELD_WORD]: a piece cut inside speech is delivered without its last word while the
 * passes are paused, its UtteranceDone waits, and the word is either replaced by its re-decoded copy at the start of the next
 * piece's text or flushed (once, before whatever comes next) on every path where it cannot be. Played by hand.
 */
class StreamingTrackerHeldWordTest {
    private val events = mutableListOf<ListenEvent>()
    private val released = mutableListOf<FloatArray>()
    private val flushed = mutableListOf<Pair<String, Long>>()

    private fun pausedGuard() = StreamingGuard(15_000, 384).apply { recorded(6_000, 3_000) }

    private fun tracker(cutSeam: CutSeam = CutSeam.CONTEXT_AND_HELD_WORD, guard: StreamingGuard = pausedGuard()): StreamingTracker {
        val t = StreamingTracker(StreamingConfig(cutSeam = cutSeam), 384, false, guard, null) { events += it }
        t.onContextReleased = { released += it }
        t.onHeldFlushed = { text, endMs -> flushed += text to endMs }
        return t
    }

    private fun frame(t: StreamingTracker, pieceMs: Long, audioStartMs: Long, positionMs: Long) =
        t.onFrame(pieceMs, audioStartMs, positionMs) { FloatArray(0) }

    private fun utt(startMs: Long, endMs: Long, quiet: Boolean = false) =
        Utterance(FloatArray(((endMs - startMs) * 16).toInt()) { 0.001f + (it % 997) / 1000f }, startMs, endMs, audioStartMs = startMs, quietCut = quiet)

    private fun words(n: Int, from: Int = 1) = (from until from + n).joinToString(" ") { "w$it" }

    /** What the engine does with a chunk: sends it as a Transcript (the tracker only sends what it holds back and the dones). */
    private fun absorb(t: StreamingTracker, plan: FinalPlan, text: String, replaced: Boolean = false): Chunk? {
        val chunk = t.absorbFinal(plan, text, 0, TEST_LANGUAGE, 0, replaced)
        if (chunk != null) events += SpeechEvent.Transcript(Transcript(chunk.text, chunk.startMs, chunk.endMs, 0, chunk.language, null))
        return chunk
    }

    /** The piece [startMs, endMs) ends and its final delivers [text]; the final is done (armed) unless [done] is false. */
    private fun piece(t: StreamingTracker, startMs: Long, endMs: Long, text: String, quiet: Boolean = false, bySilence: Boolean = false, done: Boolean = true): Pair<Utterance, Chunk?> {
        frame(t, startMs, startMs, endMs)
        if (bySilence) t.pieceEnded()
        val u = utt(startMs, endMs, quiet)
        t.finalComing(u)
        val plan = t.finalPlan(u)
        val chunk = absorb(t, plan, text)
        t.releasePlanContext(plan)
        if (done) t.finalDone(u, chunk != null)
        return u to chunk
    }

    private val transcripts get() = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }
    private val dones get() = events.filterIsInstance<SpeechEvent.UtteranceDone>().map { it.startMs }

    /** The continuation of a piece cut at 5 000 ms; its plan. */
    private fun continuation(t: StreamingTracker, endMs: Long = 9_000, bySilence: Boolean = false): Pair<Utterance, FinalPlan> {
        frame(t, 5_000, 5_000, 5_032)
        if (bySilence) t.pieceEnded()
        val u = utt(5_000, endMs)
        t.finalComing(u)
        return u to t.finalPlan(u)
    }

    @Test
    fun aPieceCutInsideSpeechIsDeliveredWithoutItsLastWordAndItsDoneWaits() {
        val t = tracker()
        val (_, chunk) = piece(t, 1_000, 5_000, words(6))
        assertEquals("w1 w2 w3 w4 w5", chunk!!.text)
        assertEquals(listOf("w1 w2 w3 w4 w5"), transcripts.map { it.text })
        assertEquals("the UtteranceDone waits for the held word", emptyList<Long>(), dones)
        assertTrue(t.heldWordPending())
        val (_, plan) = continuation(t)
        assertEquals("w6", plan.seam!!.heldWord)
        assertEquals(5, plan.seam!!.delivered.size)
        assertEquals("the pieces's words including the held one", 6, plan.seam!!.pieceWords)
        assertEquals(listOf("w1", "w2", "w3", "w4", "w5", "w6"), plan.seam!!.allWords)
    }

    @Test
    fun nothingIsHeldWithFewerThanThreeWordsWithoutPausedPassesAfterAQuietCutOrSilenceOrInContextMode() {
        // two words, three words
        run {
            val t = tracker()
            piece(t, 1_000, 5_000, words(2))
            assertEquals(listOf("w1 w2"), transcripts.map { it.text })
            assertEquals(listOf(1_000L), dones)
            assertFalse(t.heldWordPending())
        }
        run {
            events.clear()
            val t = tracker()
            piece(t, 1_000, 5_000, words(3))
            assertEquals(listOf("w1 w2"), transcripts.map { it.text })
            assertTrue(t.heldWordPending())
        }
        // passes running (the guard has not paused them)
        run {
            events.clear()
            val t = tracker(guard = StreamingGuard(15_000, 384))
            piece(t, 1_000, 5_000, words(6))
            assertEquals(listOf("w1 w2 w3 w4 w5 w6"), transcripts.map { it.text })
            assertEquals(listOf(1_000L), dones)
        }
        // a quiet cut, silence, the stage 1 mode
        for ((name, make) in listOf<Pair<String, (StreamingTracker) -> Unit>>(
            "quiet cut" to { t -> piece(t, 1_000, 5_000, words(6), quiet = true) },
            "silence" to { t -> piece(t, 1_000, 5_000, words(6), bySilence = true) },
        )) {
            events.clear()
            val t = tracker()
            make(t)
            assertEquals(name, listOf("w1 w2 w3 w4 w5 w6"), transcripts.map { it.text })
            assertEquals(name, listOf(1_000L), dones)
        }
        events.clear()
        val context = tracker(CutSeam.CONTEXT)
        piece(context, 1_000, 5_000, words(6))
        assertEquals(listOf("w1 w2 w3 w4 w5 w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
    }

    @Test
    fun anAnchoredNextTextReplacesTheHeldWordAndTheDoneComesBeforeIt() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val (u, plan) = continuation(t, bySilence = true)
        val chunk = absorb(t, plan, "w6 x1 x2", replaced = true)
        assertEquals("w6 x1 x2", chunk!!.text)
        assertEquals("the held word is not sent apart", listOf("w1 w2 w3 w4 w5", "w6 x1 x2"), transcripts.map { it.text })
        assertEquals("D(S1) is sent when the hold resolves, before the text that replaces it", listOf(1_000L), dones)
        assertFalse(t.heldWordPending())
        assertTrue(flushed.isEmpty())
        t.finalDone(u, true)
        assertEquals(listOf(1_000L, 5_000L), dones)
        assertEquals(1, t.stats().heldReplaced)
    }

    @Test
    fun aFlushSendsTheHeldWordOnceAsATranscriptOfItsPieceAndThenTheDone() {
        val t = tracker()
        val (first, _) = piece(t, 1_000, 5_000, words(6))
        t.flushHeld("no anchor")
        t.flushHeld("no anchor")
        val held = events.filterIsInstance<SpeechEvent.Transcript>().last().transcript
        assertEquals("w6", held.text)
        assertEquals(first.startMs, held.startMs)
        assertEquals(first.endMs, held.endMs)
        assertEquals(0L, held.processingMs)
        assertEquals(TEST_LANGUAGE, held.language)
        assertNull(held.sentenceJoin)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
        assertTrue("the joiner is told through the callback, once", flushed == listOf("w6" to 5_000L))
        assertTrue(events.indexOfLast { it is SpeechEvent.UtteranceDone } > events.indexOfLast { it is SpeechEvent.Transcript })
        assertEquals(1, t.stats().heldKept)
    }

    @Test
    fun aFlushAskedBeforeTheFinalIsDoneWaitsForItAndComesAfterTheFirstTranscript() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6), done = false)
        t.flushHeld("lag")
        assertEquals("nothing yet: the final of the piece has not ended", listOf("w1 w2 w3 w4 w5"), transcripts.map { it.text })
        assertTrue(dones.isEmpty())
        t.finalDone(Utterance(FloatArray(0), 1_000, 5_000, audioStartMs = 1_000), true)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
    }

    @Test
    fun aSkippedContinuationFlushesTheHeldWord() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        t.finalNotComing(second)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L, 5_000L), dones.sorted())
        assertEquals(1, t.stats().heldKept)
    }

    @Test
    fun aDroppedContinuationFlushesTheHeldWordAtTheNextFrame() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        assertEquals(listOf("w1 w2 w3 w4 w5"), transcripts.map { it.text })
        frame(t, -1, 0, 5_064)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
    }

    @Test
    fun aFinalThatDoesNotContinueTheCutFlushesTheHeldWordBeforeItsOwnEvents() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val other = utt(8_000, 10_000)
        frame(t, 8_000, 8_000, 8_032)
        t.finalComing(other)
        val plan = t.finalPlan(other)
        assertNull(plan.seam)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
        absorb(t, plan, "y1 y2")
        assertEquals("y1 y2", transcripts.last().text)
    }

    @Test
    fun aContinuationThatWouldBeDeliveredLaterThanTheLagLimitFlushesTheHeldWordAndAnchorsAgainstAllTheWords() {
        val guard = pausedGuard()               // cost 0.39 per ms of window: a closing of about 3 s
        val t = tracker(guard = guard)
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        frame(t, 5_000, 5_000, 5_000 + 12_500)  // the capture is far ahead of the cut
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan = t.finalPlan(second)
        assertNotNull("the context is still used", plan.context)
        assertNull("but nothing is held any more", plan.seam!!.heldWord)
        assertEquals(6, plan.seam!!.delivered.size)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
    }

    @Test
    fun theEndOfCaptureFlushesAHeldWordNobodyContinuesButNotOneWhoseFinalWaits() {
        val lonely = tracker()
        piece(lonely, 1_000, 5_000, words(6))
        lonely.end()
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)

        events.clear()
        val waiting = tracker()
        piece(waiting, 1_000, 5_000, words(6))
        frame(waiting, 5_000, 5_000, 5_032)
        val second = utt(5_000, 7_000)
        waiting.finalComing(second)
        waiting.end()
        assertEquals("still wanted", listOf("w1 w2 w3 w4 w5"), transcripts.map { it.text })
        val plan = waiting.finalPlan(second)
        assertEquals("w6", plan.seam!!.heldWord)
    }

    @Test
    fun theEndOfTheQueueFlushesWhatIsStillHeld() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)       // the continuation exists, its final never comes
        t.finalsOver()
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
        t.finalsOver()
        assertEquals("once", listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
    }

    @Test
    fun aFailedFinalOfTheContinuationStillFlushesTheHeldWord() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val (u, plan) = continuation(t)
        t.releasePlanContext(plan)
        t.finalDone(u, false)               // the engine failed before absorbing
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L, 5_000L), dones.sorted())
    }

    @Test
    fun anAnchoredRunWhoseTextTurnsOutEmptyKeepsTheHeldWordAfterAll() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val (u, plan) = continuation(t)
        val chunk = absorb(t, plan, "...", replaced = true)       // no letter: no words
        assertNull(chunk)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        t.finalDone(u, false)
        assertEquals(listOf(1_000L, 5_000L), dones.sorted())
    }

    @Test
    fun aTeardownSendsNothingAndZeroesTheAudio() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val before = events.size
        t.discardRequests()
        assertEquals("a cancelled session sends nothing", before, events.size)
        assertEquals(0, t.retainedContextArrays())
        assertTrue(released.isNotEmpty() && released.all { a -> a.all { it == 0f } })
    }

    @Test
    fun aChainHoldsTheLastWordOfEveryPieceAndEachContinuationReplacesIt() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan2 = t.finalPlan(second)
        absorb(t, plan2, "w6 w7 w8 w9", replaced = true)
        t.releasePlanContext(plan2)
        t.finalDone(second, true)
        assertEquals("the second piece holds its own last word", listOf("w1 w2 w3 w4 w5", "w6 w7 w8"), transcripts.map { it.text })
        assertEquals("the first piece's done came when its word was replaced", listOf(1_000L), dones)
        assertTrue(t.heldWordPending())
        frame(t, 9_000, 9_000, 9_032)
        val third = utt(9_000, 12_000)
        t.finalComing(third)
        val plan3 = t.finalPlan(third)
        assertEquals("w9", plan3.seam!!.heldWord)
        assertEquals(listOf("w6", "w7", "w8"), plan3.seam!!.delivered)
        assertEquals("the word carried in from the piece before is not of the piece's own audio", 3, plan3.seam!!.pieceWords)
        absorb(t, plan3, "w9 w10", replaced = true)
        t.releasePlanContext(plan3)
        t.finalDone(third, true)
        assertEquals(listOf(1_000L, 5_000L, 9_000L), dones)
    }

    @Test
    fun theHeldWordsAreCountedByHowTheyEnded() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        t.flushHeld("lag")
        piece(t, 20_000, 24_000, words(6))
        t.flushHeld("no anchor")
        val s = t.stats()
        assertEquals(0, s.heldReplaced)
        assertEquals(2, s.heldKept)
        assertEquals(mapOf("lag" to 1, "no anchor" to 1), s.heldKeptBy)
        assertEquals(2, s.heldDelaysMs.size)
    }

    @Test
    fun aReplacementWordCarriedInFromThePieceBeforeIsNotCountedAmongTheWordsOfThePiecesOwnAudio() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val (second, plan2) = continuation(t)
        absorb(t, plan2, "w6 x1 x2 x3 x4 x5", replaced = true)      // w6 is the replacement; x5 is held
        t.releasePlanContext(plan2)
        t.finalDone(second, true)
        frame(t, 9_000, 9_000, 9_032)
        val third = utt(9_000, 12_000)
        t.finalComing(third)
        val seam = t.finalPlan(third).seam!!
        assertEquals(listOf("w6", "x1", "x2", "x3", "x4"), seam.delivered)
        assertEquals("x1..x5 were spoken in the piece's own audio", 5, seam.pieceWords)
        assertEquals(listOf("w6", "x1", "x2", "x3", "x4", "x5"), seam.allWords)
    }

    @Test
    fun aPieceWhoseTextIsNotAReplacementCountsAllItsWordsAsItsOwn() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        val (second, plan2) = continuation(t)
        t.flushHeld("no anchor")
        absorb(t, plan2, "x1 x2 x3 x4 x5 x6")
        t.releasePlanContext(plan2)
        t.finalDone(second, true)
        frame(t, 9_000, 9_000, 9_032)
        val third = utt(9_000, 12_000)
        t.finalComing(third)
        assertEquals(6, t.finalPlan(third).seam!!.pieceWords)
    }

    @Test
    fun aContinuationThatWouldEndWithinTheLagMarginOfTheLimitFlushesTheHeldWord() {
        val guard = pausedGuard()
        val t = tracker(guard = guard)
        piece(t, 1_000, 5_000, words(6))
        val closing = guard.closingEstimateMs(4_000, SeamAnchor.CONTEXT_MS)
        // the text of the continuation would come 500 ms before the limit: inside the margin the cut-off rules keep (StreamingGuard.LAG_MARGIN_MS)
        frame(t, 5_000, 5_000, 5_000 + 15_000 - closing - 500)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        val plan = t.finalPlan(second)
        assertNotNull(plan.context)
        assertNull("flushed: it could not be replaced in time", plan.seam!!.heldWord)
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
    }

    @Test
    fun aContinuationSkippedWhileTheFinalOfTheHeldPieceIsStillRunningEndsFirstAndTheHeldWordComesWhenThatFinalEnds() {
        val t = tracker()
        val (first, _) = piece(t, 1_000, 5_000, words(6), done = false)
        frame(t, 5_000, 5_000, 5_032)
        val second = utt(5_000, 9_000)
        t.finalComing(second)
        t.finalNotComing(second)
        assertEquals("the skipped piece ends at once, the held word waits for its own piece's final", listOf(5_000L), dones)
        assertEquals(listOf("w1 w2 w3 w4 w5"), transcripts.map { it.text })
        t.finalDone(first, true)
        val order = events.mapNotNull {
            when (it) {
                is SpeechEvent.Transcript -> "T:" + it.transcript.text
                is SpeechEvent.UtteranceDone -> "D:" + it.startMs
                else -> null
            }
        }
        assertEquals(listOf("T:w1 w2 w3 w4 w5", "D:5000", "T:w6", "D:1000"), order)
        assertEquals(1, t.stats().heldKept)
    }

    @Test
    fun theEndOfTheQueueSendsAHeldWordWhoseFinalNeverReportedItsEnd() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6), done = false)
        t.finalsOver()
        assertEquals(listOf("w1 w2 w3 w4 w5", "w6"), transcripts.map { it.text })
        assertEquals(listOf(1_000L), dones)
    }

    @Test
    fun aHeldWordThatIsSentAsItWasCountsAsAWordAfterTheSpeechAndHasADelay() {
        val t = tracker()
        piece(t, 1_000, 5_000, words(6))
        assertEquals(5, t.stats().wordsAfter)
        assertEquals(1, t.stats().chunkDelaysMs.size)
        frame(t, 5_000, 5_000, 6_000)
        t.flushHeld("no anchor")
        assertEquals(6, t.stats().wordsAfter)
        assertEquals("a delay for the held word too", 2, t.stats().chunkDelaysMs.size)
    }
}
