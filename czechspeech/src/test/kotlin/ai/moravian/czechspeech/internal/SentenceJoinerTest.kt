package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.internal.SentenceBoundary.Choice
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.Collections

/** When a transcript of a session may continue the previous one at all ([SentenceJoiner]), with a fake language model. */
class SentenceJoinerTest {
    private val a = "Jasně, ale můžeš nějak."
    private val b = "s tebou spojit ve video hovoru."
    private val joined = SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, b, "S tebou spojit ve video hovoru.")

    /** The texts each language model call scored. */
    private val calls = mutableListOf<List<String>>()

    /** Likes small letters, few commas and no sentence end inside: joins any false boundary with a small letter. */
    private val joining: (List<String>) -> DoubleArray? = { texts ->
        calls += texts
        DoubleArray(texts.size) { i ->
            val t = texts[i]
            -40.0 - (if (". " in t) 12.6 else 0.0) - 10.0 * t.drop(1).count { it.isUpperCase() } - 3.5 * t.count { it == ',' }
        }
    }

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

    @Test
    fun theFirstTranscriptContinuesNothing() {
        val j = SentenceJoiner()
        assertNull(j.next(a, 0, 1_000, joining).join)
        assertTrue("no language model for it", calls.isEmpty())
    }

    @Test
    fun theNextTranscriptWithinThreeSecondsIsDecided() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        val step = j.next(b, 4_000, 5_000, joining)          // a pause of exactly 3 s
        assertEquals(joined, step.join)
        assertEquals(1, calls.size)
        assertEquals("the four candidates of an \"A. b\" boundary", 4, calls.single().size)
        assertTrue(step.languageModelMs >= 0)
    }

    @Test
    fun noJoinAcrossALongerPause() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        assertNull(j.next(b, 4_001, 5_000, joining).join)
        assertTrue("no language model across it", calls.isEmpty())
    }

    @Test
    fun aCutAtTheLengthCapIsAPauseOfZero() {
        val j = SentenceJoiner()
        j.next(a, 0, 28_000, joining)
        assertEquals(joined, j.next(b, 28_000, 30_000, joining).join)
    }

    @Test
    fun noJoinAcrossLostSpeech() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        j.lost(2_000)                                         // skipped between the two
        assertNull(j.next(b, 2_500, 3_500, joining).join)
        assertTrue(calls.isEmpty())
        // speech lost after a transcript does not stop the one before it from joining
        val k = SentenceJoiner()
        k.next(a, 0, 1_000, joining)
        k.lost(4_000)
        assertEquals(joined, k.next(b, 2_500, 3_500, joining).join)
        assertNull("but the next one", k.next("A pak.", 4_500, 5_000, joining).join)
        assertEquals(1, calls.size)
    }

    @Test
    fun noJoinAfterAFailedRecognition() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        j.reset()
        assertNull(j.next(b, 1_500, 2_500, joining).join)
        assertEquals("and then as before", true, j.next("A pak.", 3_000, 4_000, joining).join?.continuesPrevious)
    }

    @Test
    fun withoutALanguageModelOrWhenItFailsTheTranscriptStays() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, null)
        assertNull("not loaded", j.next(b, 1_500, 2_500, null).join)
        assertNull("failed", j.next(a, 3_000, 4_000) { null }.join)
        assertNull("threw", j.next(b, 4_500, 5_500) { throw IllegalStateException("boom: $it") }.join)
        assertTrue("logged without the text", logs.any { "IllegalStateException" in it })
        assertTrue(logs.none { "boom" in it || "tebou" in it || "nějak" in it })
    }

    @Test
    fun theNextBoundaryTakesWhispersTextOfTheTranscriptBefore() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        assertEquals(joined, j.next(b, 1_500, 2_500, joining).join)
        j.next("A pak.", 3_000, 4_000, joining)
        // the second boundary saw B as Whisper wrote it ("s tebou …"), not as the client may have inserted it
        assertTrue(calls[1].contains("s tebou spojit ve video hovoru. A pak."))
        assertEquals(2, calls.size)
    }

    @Test
    fun theLogSaysWhatWasDecidedAndHowLongButNoText() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        j.next(b, 1_500, 2_500, joining)
        assertTrue(logs.toString(), logs.any { it.startsWith("sentence boundary: ${Choice.JOIN.name.lowercase()}, language model ") })
        assertTrue(logs.none { "tebou" in it || "nějak" in it })
    }

    @Test
    fun aContinuedChunkExtendsThePreviousTextWithoutADecision() {
        val j = SentenceJoiner()
        assertNull(j.next("Jasně, ale", 0, 1_000, joining).join)
        j.continued("můžeš nějak.", 2_000)
        assertTrue("no language model for a chunk", calls.isEmpty())
        assertEquals(joined, j.next(b, 4_000, 5_000, joining).join)            // decided against "Jasně, ale můžeš nějak."
        assertTrue(calls.single().toString(), calls.single().any { "Jasně, ale můžeš nějak" in it })
    }

    @Test
    fun aContinuedChunkMovesTheEndOfThePrevious() {
        val j = SentenceJoiner()
        j.next("Jasně, ale", 0, 1_000, joining)
        j.continued("můžeš nějak.", 5_000)
        assertEquals(joined, j.next(b, 7_000, 8_000, joining).join)            // a pause of 2 s, not 6 s
    }

    @Test
    fun theUtteranceEndedWhereItsFinalSaysEvenWithoutMoreWords() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak.", 0, 8_000, joining)       // the last chunk end was an upper bound
        j.endedAt(1_000)
        assertEquals(joined, j.next(b, 3_000, 4_000, joining).join)     // a pause of 2 s from the real end
        j.reset()
        j.endedAt(9_000)                                            // nothing after a reset
        assertNull(j.next(b, 9_500, 10_000, joining).join)
    }

    @Test
    fun aContinuedChunkAfterAResetContinuesNothing() {
        val j = SentenceJoiner()
        j.next("Jasně, ale", 0, 1_000, joining)
        j.reset()
        j.continued("můžeš nějak.", 2_000)
        assertNull(j.next(b, 3_000, 4_000, joining).join)
    }

    @Test
    fun aContinuedChunkBeforeAnyTranscriptContinuesNothing() {
        val j = SentenceJoiner()
        j.continued("můžeš nějak.", 2_000)
        assertNull(j.next(b, 3_000, 4_000, joining).join)
        assertTrue(calls.isEmpty())
    }

    @Test
    fun severalContinuedChunksExtendInOrder() {
        val j = SentenceJoiner()
        j.next("Jasně, ale", 0, 1_000, joining)
        j.continued("můžeš", 2_000)
        j.continued("nějak.", 3_000)
        assertEquals(joined, j.next(b, 4_000, 5_000, joining).join)
        assertTrue(calls.single().toString(), calls.single().any { "Jasně, ale můžeš nějak" in it })
    }

    // ---- a piece cut inside speech (forced cut, length cap) continues the one before it: no language model ----

    @Test
    fun aPieceAfterACutContinuesThePreviousOneWithoutAskingTheLanguageModel() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak.", 0, 4_000, joining)
        val step = j.afterCut("Dnes s tebou mluvím.", 4_000, 8_000)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "dnes s tebou mluvím.", "Dnes s tebou mluvím."), step.join)
        assertEquals(0, step.languageModelMs)
        assertTrue("the language model was not asked", calls.isEmpty())
        assertEquals(1, logs.count { it == "sentence boundary: continued (cut)" })
    }

    // ---- a piece after a cut on a quiet frame of continuous speech: the decision leans to going on ----

    /** The end of the sentence is only a little likelier than the joins (by 1.0): a pause after a quiet cut, not a sentence end. */
    private val splitByAHair: (List<String>) -> DoubleArray? = { texts ->
        calls += texts
        DoubleArray(texts.size) { if (". " in texts[it]) -40.0 else -41.0 }
    }

    private fun lastBoundaryLog() = logs.last { it.startsWith("sentence boundary: ") }

    @Test
    fun aQuietCutJoinsWhereARealPauseSplitsAndSaysSoInTheLog() {
        val quiet = SentenceJoiner()
        quiet.next(a, 0, 1_000, joining)
        val join = quiet.next(b, 2_000, 3_000, splitByAHair, quiet = true)
        assertEquals(joined, join.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: join (quiet cut), language model "))
        // the same scores after real silence: the sentence ends (the measured decision, unchanged)
        val real = SentenceJoiner()
        real.next(a, 0, 1_000, joining)
        val split = real.next(b, 2_000, 3_000, splitByAHair)
        assertEquals(SentenceJoin(false, PreviousEnd.KEEP, "S tebou spojit ve video hovoru.", "S tebou spojit ve video hovoru."), split.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: split, language model "))
        assertTrue("no text in the log: $logs", logs.none { "tebou" in it || "nějak" in it })
    }

    @Test
    fun aQuietCutStillSplitsWhenTheEndOfTheSentenceIsClearlyLikelier() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        val clear: (List<String>) -> DoubleArray? = { texts -> DoubleArray(texts.size) { if (". " in texts[it]) -40.0 else -46.0 } }
        val step = j.next(b, 2_000, 3_000, clear, quiet = true)
        assertEquals(SentenceJoin(false, PreviousEnd.KEEP, "S tebou spojit ve video hovoru.", "S tebou spojit ve video hovoru."), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: split (quiet cut), language model "))
    }

    @Test
    fun aQuietCutCanBecomeACommaAndTheLogSaysComma() {
        val j = SentenceJoiner()
        j.next(a, 0, 1_000, joining)
        val comma: (List<String>) -> DoubleArray? = { texts ->
            DoubleArray(texts.size) { if (", s tebou" in texts[it]) -38.0 else if (". " in texts[it]) -45.0 else -40.0 }
        }
        val step = j.next(b, 2_000, 3_000, comma, quiet = true)
        assertEquals(SentenceJoin(true, PreviousEnd.PERIOD_TO_COMMA, b, "S tebou spojit ve video hovoru."), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: comma (quiet cut), language model "))
    }

    @Test
    fun aQuietCutKeepsTheCapitalOfAName() {
        val j = SentenceJoiner()
        j.next("Pozdravuj.", 0, 1_000, joining)
        val name: (List<String>) -> DoubleArray? = { texts ->
            DoubleArray(texts.size) { if (". " in texts[it]) -50.0 else if (" Honzu" in texts[it]) -40.0 else -45.0 }
        }
        val step = j.next("Honzu a Petru.", 2_000, 3_000, name, quiet = true)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "Honzu a Petru.", "Honzu a Petru."), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: join (quiet cut)"))
    }

    // ---- a piece after a cut on speech: continues; the language model only decides the capital of the first word ----

    @Test
    fun aPieceAfterACutLowersTheFirstWordWhenTheLanguageModelPrefersIt() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak.", 0, 4_000, joining)
        calls.clear()
        val lowerLikes: (List<String>) -> DoubleArray? = { texts ->
            calls += texts
            DoubleArray(texts.size) { if (" Dnes " in texts[it]) -45.0 else -40.0 }
        }
        val step = j.afterCut("Dnes s tebou mluvím.", 4_000, 8_000, lowerLikes)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "dnes s tebou mluvím.", "Dnes s tebou mluvím."), step.join)
        assertEquals("the two candidates, one call", 1, calls.size)
        assertEquals(2, calls.single().size)
        assertTrue(step.languageModelMs >= 0)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: continued (cut), lower-case, language model "))
    }

    @Test
    fun aPieceAfterACutKeepsTheCapitalOfAName() {
        val j = SentenceJoiner()
        j.next("Pozdravuj.", 0, 4_000, joining)
        val name: (List<String>) -> DoubleArray? = { texts -> DoubleArray(texts.size) { if (" Honzu " in texts[it]) -40.0 else -45.0 } }
        val step = j.afterCut("Honzu a Petru.", 4_000, 8_000, name)
        assertEquals("the period goes, the capital stays", SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "Honzu a Petru.", "Honzu a Petru."), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: continued (cut), capital kept, language model "))
        assertTrue("no text in the log: $logs", logs.none { "Honzu" in it || "Pozdravuj" in it })
    }

    @Test
    fun aPieceAfterACutDoesNotAskTheLanguageModelWhenThereIsNothingToLowerOrItFails() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak.", 0, 4_000, joining)
        calls.clear()
        val small = j.afterCut("dnes přijdu.", 4_000, 8_000, joining)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "dnes přijdu.", "Dnes přijdu."), small.join)
        assertTrue("a small letter already: no language model", calls.isEmpty())
        assertEquals(0, small.languageModelMs)
        // the language model fails: lowered, as without it, and the piece is not held back
        val failing: (List<String>) -> DoubleArray? = { null }
        assertEquals("dnes přijdu k tobě.", j.afterCut("Dnes přijdu k tobě.", 8_000, 12_000, failing).join!!.continuingText)
        val throwing: (List<String>) -> DoubleArray? = { error("boom") }
        assertEquals("dnes přijdu k tobě.", j.afterCut("Dnes přijdu k tobě.", 12_000, 16_000, throwing).join!!.continuingText)
        assertTrue("the exception's message is not logged: $logs", logs.none { "boom" in it })
    }

    // ---- an ellipsis or a comma at the end of the previous piece is decided by the language model ----

    private val ellipsisA = "Jasně, ale můžeš nějak..."
    private val bUpper = "S tebou spojit ve video hovoru."

    /** [split] for "A... B", [keep] for "A... b", the removals -40 (the capital -42), the comma variant -40. */
    private fun ellipsisModel(split: Double = -50.0, keep: Double = -50.0): (List<String>) -> DoubleArray? = { texts ->
        DoubleArray(texts.size) {
            val t = texts[it]
            when {
                "... S" in t -> split
                "... s" in t -> keep
                " S tebou" in t -> -42.0
                else -> -40.0
            }
        }
    }

    private fun ellipsisStep(model: (List<String>) -> DoubleArray?, quiet: Boolean = false): SentenceJoiner.Step {
        val j = SentenceJoiner()
        j.next(ellipsisA, 0, 1_000, joining)
        return j.next(bUpper, 2_000, 3_000, model, quiet)
    }

    @Test
    fun anEllipsisAtTheEndOfThePreviousPieceIsRemovedAndTheLogSaysSo() {
        val step = ellipsisStep(ellipsisModel())
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_ELLIPSIS, "s tebou spojit ve video hovoru.", bUpper), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: ellipsis removed, language model "))
        assertTrue("no text: $logs", logs.none { "tebou" in it || "nějak" in it })
        ellipsisStep(ellipsisModel(), quiet = true)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: ellipsis removed (quiet cut), language model "))
    }

    @Test
    fun anEllipsisStaysOrAnEllipsisBecomesACommaOrASentenceEndWhenTheModelClearlyPrefersIt() {
        assertEquals(SentenceJoin(true, PreviousEnd.KEEP, "s tebou spojit ve video hovoru.", bUpper), ellipsisStep(ellipsisModel(keep = -37.0)).join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: ellipsis kept, language model "))
        assertNull("a new sentence as written", ellipsisStep(ellipsisModel(split = -37.0)).join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: split, language model "))
        val comma: (List<String>) -> DoubleArray? = { texts -> DoubleArray(texts.size) { if ("nějak, s tebou" in texts[it]) -37.0 else if (" S tebou" in texts[it]) -42.0 else -40.0 } }
        assertEquals(SentenceJoin(true, PreviousEnd.ELLIPSIS_TO_COMMA, "s tebou spojit ve video hovoru.", bUpper), ellipsisStep(comma).join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: ellipsis to comma, language model "))
    }

    @Test
    fun aTrailingCommaOfThePreviousPieceGoesWhenTheModelClearlyPrefersTheTextWithoutIt() {
        val model: (List<String>) -> DoubleArray? = { texts ->
            DoubleArray(texts.size) {
                val t = texts[it]
                when {
                    "nějak s tebou" in t -> -40.0
                    "nějak S tebou" in t -> -45.0
                    "nějak, s tebou" in t -> -42.0
                    else -> -44.0
                }
            }
        }
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak,", 0, 1_000, joining)
        val step = j.next(bUpper, 2_000, 3_000, model)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_COMMA, "s tebou spojit ve video hovoru.", bUpper), step.join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: comma removed, language model "))
        // the comma is better: as before, the small letter after it
        val keeps: (List<String>) -> DoubleArray? = { texts -> DoubleArray(texts.size) { if ("nějak, s tebou" in texts[it]) -38.0 else if ("nějak, S tebou" in texts[it]) -45.0 else -42.0 } }
        val k = SentenceJoiner()
        k.next("Jasně, ale můžeš nějak,", 0, 1_000, joining)
        assertEquals(SentenceJoin(true, PreviousEnd.KEEP, "s tebou spojit ve video hovoru.", bUpper), k.next(bUpper, 2_000, 3_000, keeps).join)
    }

    /** After a short pause the ellipsis leans to being removed, after a long one (2.0 s or more between the speeches) the comparison is neutral. */
    @Test
    fun anEllipsisAfterALongPauseIsComparedWithoutAPriorToRemovingIt() {
        fun step(pauseMs: Long, model: (List<String>) -> DoubleArray?, quiet: Boolean = false): SentenceJoiner.Step {
            val j = SentenceJoiner()
            j.next(ellipsisA, 0, 1_000, joining)
            return j.next(bUpper, 1_000 + pauseMs, 2_000 + pauseMs, model, quiet)
        }
        val keepsByAHair = ellipsisModel(keep = -39.0)         // the ellipsis kept is likelier than the removal by 1.0
        assertEquals("1.9 s: the prior removes it", PreviousEnd.REMOVE_ELLIPSIS, step(1_900, keepsByAHair).join!!.previousEnd)
        assertEquals("2.0 s: neutral, the likelier reading wins", PreviousEnd.KEEP, step(2_000, keepsByAHair).join!!.previousEnd)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: ellipsis kept, language model "))
        assertEquals("a quiet cut is a pause under 2 s by definition", PreviousEnd.REMOVE_ELLIPSIS, step(2_500, keepsByAHair, quiet = true).join!!.previousEnd)
        // a new sentence that is only a little likelier wins after a long pause
        val splitByAHair2 = ellipsisModel(split = -39.0)
        assertNull("2.5 s: the sentence ends", step(2_500, splitByAHair2).join)
        assertTrue(lastBoundaryLog(), lastBoundaryLog().startsWith("sentence boundary: split, language model "))
        assertEquals(PreviousEnd.REMOVE_ELLIPSIS, step(1_000, splitByAHair2).join!!.previousEnd)
    }

    @Test
    fun aPieceAfterACutOnSpeechRemovesTheEllipsisOfThePreviousPiece() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak...", 0, 4_000, joining)
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_ELLIPSIS, "dnes přijdu.", "Dnes přijdu."), j.afterCut("Dnes přijdu.", 4_000, 8_000).join)
    }

    @Test
    fun aPieceAfterACutKeepsAQuestionMarkAndAnExclamationMarkAndTheLetter() {
        for (end in listOf("Co mám dělat?", "Stůj!")) {
            val j = SentenceJoiner()
            j.next(end, 0, 4_000, joining)
            assertNull(end, j.afterCut("Dnes přijdu.", 4_000, 8_000).join)
            assertEquals("nothing was applied, so nothing is logged: $logs", 0, logs.count { it.startsWith("sentence boundary: continued") })
        }
    }

    @Test
    fun aPieceAfterACutOnAnEndWithoutAPeriodOnlyLowersTheFirstLetter() {
        val j = SentenceJoiner()
        j.next("Jasně, ale můžeš nějak", 0, 4_000, joining)
        assertEquals(SentenceJoin(true, PreviousEnd.KEEP, "dnes přijdu", "Dnes přijdu"), j.afterCut("Dnes přijdu", 4_000, 8_000).join)
    }

    @Test
    fun aPieceAfterACutKeepsAnAcronymAndTheTextAfterItIsTheNextPrevious() {
        val j = SentenceJoiner()
        j.next("Jasně.", 0, 4_000, joining)
        assertEquals("ČR", j.afterCut("ČR to řeší.", 4_000, 8_000).join!!.continuingText.substringBefore(' '))
        assertEquals(PreviousEnd.REMOVE_PERIOD, j.afterCut("Teď přijde.", 8_000, 12_000).join!!.previousEnd)
    }

    @Test
    fun aPieceAfterACutContinuesNothingAfterAFailureOrLostSpeechOrWithoutAPrevious() {
        val first = SentenceJoiner()
        assertNull(first.afterCut("Dnes přijdu.", 4_000, 8_000).join)
        val failed = SentenceJoiner()
        failed.next("Jasně.", 0, 4_000, joining)
        failed.reset()
        assertNull(failed.afterCut("Dnes přijdu.", 4_000, 8_000).join)
        val lost = SentenceJoiner()
        lost.next("Jasně.", 0, 4_000, joining)
        lost.lost(5_000)
        assertNull(lost.afterCut("Dnes přijdu.", 6_000, 8_000).join)
    }
}
