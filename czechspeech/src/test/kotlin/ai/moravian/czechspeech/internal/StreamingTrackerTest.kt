package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * The rules of a streaming session without threads, native code or clocks: the tests play the capture thread (frames,
 * ends of pieces, offers to the final queue), the whisper thread (absorbing the passes) and the recognition loop by hand,
 * in virtual time (positions on the 32 ms frame grid, as in a session). A piece starts at 480 ms with its audio at
 * 160 ms unless said otherwise.
 */
class StreamingTrackerTest {
    private val events = mutableListOf<ListenEvent>()

    private fun tracker(
        previews: Boolean = true,
        guard: StreamingGuard? = null,
        language: LanguageSession? = null,
        emit: (ListenEvent) -> Unit = { events += it },
    ): StreamingTracker = if (guard == null) {
        StreamingTracker(StreamingConfig(), 384, previews, language = language, emit = emit)
    } else {
        StreamingTracker(StreamingConfig(), 384, previews, guard, language, emit)
    }

    private val threeLanguages = listOf("cs", "sk", "en")
    private fun languageSession(current: String = "cs") = LanguageSession(
        LanguageRule(threeLanguages, ai.moravian.czechspeech.CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f))),
    ).also { if (current != "cs") it.delivered(current, rescued = false) }

    /** Feeds frames, taking the passes the tracker requests as the whisper thread would. */
    private class Rig(val tracker: StreamingTracker) {
        var positionMs = 0L
        val snapshotsFrom = mutableListOf<Long>()
        private val token = AbortToken(FakeNative())

        fun frame(pieceMs: Long = 480, audioStartMs: Long = 160, quietFrames: Int = 0): StreamingTracker.Step {
            positionMs += 32
            val position = positionMs
            return tracker.onFrame(pieceMs, audioStartMs, position, quietFrames = quietFrames) { from ->
                snapshotsFrom += from
                FloatArray(((position - from) * 16).toInt()) { 0.3f }
            }
        }

        /** Frames until the tracker asks for something; it must be a pass, within [limitMs] of audio. */
        fun nextPass(pieceMs: Long = 480, audioStartMs: Long = 160, limitMs: Long = positionMs + 20_000): StreamingPass {
            while (positionMs < limitMs) {
                when (val step = frame(pieceMs, audioStartMs)) {
                    is StreamingTracker.Step.Pass -> {
                        assertSame(step.pass, tracker.requests.tryReceive().getOrNull())
                        return step.pass
                    }
                    StreamingTracker.Step.Cut, StreamingTracker.Step.CutQuiet -> fail("a cut at $positionMs ms, a pass was expected")
                    StreamingTracker.Step.None -> {}
                }
            }
            fail("no pass until $limitMs ms")
            error("unreachable")
        }

        /** Frames until [untilMs] of audio, none of which may ask for anything. */
        fun quiet(untilMs: Long, pieceMs: Long = 480, audioStartMs: Long = 160) {
            while (positionMs < untilMs) assertEquals("at $positionMs ms", StreamingTracker.Step.None, frame(pieceMs, audioStartMs))
        }

        /** The whisper thread's part of a pass: attaches it, absorbs the result and publishes it. */
        fun answer(pass: StreamingPass, text: String, recogniserMs: Long = 500, language: String = TEST_LANGUAGE, decisive: Boolean = true): Absorbed? {
            assertTrue(pass.attach(token))
            tracker.languageFor(pass)        // the engine asks before the recogniser runs
            val absorbed = tracker.absorb(pass, PassResult(text, recogniserMs, language, decisive))
            if (absorbed != null) tracker.publish(absorbed, null, 0)
            return absorbed
        }
    }

    private fun words(n: Int) = (1..n).joinToString(" ") { "w$it" }

    private fun utterance(
        startMs: Long = 480,
        endMs: Long = 2_900,
        audioStartMs: Long = 160,
        audioMs: Long = endMs + 600 - audioStartMs,
        quietCut: Boolean = false,
    ) = Utterance(FloatArray((audioMs * 16).toInt()) { 0.3f }, startMs, endMs, audioStartMs = audioStartMs, quietCut = quietCut)

    private fun transcript(text: String, startMs: Long, endMs: Long, processingMs: Long) =
        SpeechEvent.Transcript(Transcript(text, startMs, endMs, processingMs, TEST_LANGUAGE, null))

    /** Two passes, "dobrý den jak" and "dobrý den, jak se": the first word is committed after the second. */
    private fun twoPasses(rig: Rig): Pair<StreamingPass, StreamingPass> {
        val first = rig.nextPass()
        rig.answer(first, "dobrý den jak", 400)
        val second = rig.nextPass()
        rig.answer(second, "dobrý den, jak se", 900)
        return first to second
    }

    @Test
    fun theFirstPassComesOneSecondIntoTheSpeechAndTheNextAfterOneMoreSecond() {
        val rig = Rig(tracker())
        val first = rig.nextPass()
        assertEquals(1504, rig.positionMs)
        assertEquals(160, first.startMs)
        assertEquals(1504, first.endMs)
        assertEquals(480, first.pieceStartMs)
        assertEquals((1504 - 160) * 16, first.samples.size)
        rig.quiet(4_000) // none while it is in flight
        rig.answer(first, "a b c")
        val second = rig.nextPass()
        assertEquals(4_032, rig.positionMs) // frames went on to 4 000 meanwhile: due long ago, asked at once
        assertEquals(4_032, second.endMs)
    }

    @Test
    fun theNextPassComesAtTheFirstFrameOneSecondAfterTheEndOfThePreviousOne() {
        val rig = Rig(tracker())
        val first = rig.nextPass()
        rig.answer(first, "a b c")
        val second = rig.nextPass()
        assertEquals(2_528, second.endMs) // the first frame at or after 1 504 + 1 000
        assertEquals(listOf(160L, 160L), rig.snapshotsFrom)
    }

    @Test
    fun wordsTwoPassesAgreeOnGoOutAsATranscriptAndTheRestAsAPreview() {
        val rig = Rig(tracker())
        val first = rig.nextPass()
        rig.answer(first, "dobrý den jak", 400)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "dobrý den jak", 0)), events)
        events.clear()
        val second = rig.nextPass()
        rig.answer(second, "dobrý den, jak se", 900)
        assertEquals(
            listOf(transcript("dobrý", 480, 1504, 900), SpeechPreview(480, "den, jak se", 0)),
            events,
        )
    }

    @Test
    fun withoutPreviewsOnlyTranscriptsGoOut() {
        val rig = Rig(tracker(previews = false))
        twoPasses(rig)
        assertEquals(listOf<ListenEvent>(transcript("dobrý", 480, 1504, 900)), events)
    }

    @Test
    fun theEndOfTheSpeechAbortsTheRunningPassAndDropsItsResult() {
        val t = tracker()
        val rig = Rig(t)
        val pass = rig.nextPass()
        assertTrue(pass.attach(AbortToken(FakeNative())))
        t.pieceEnded()
        assertTrue(pass.isCancelled)
        assertNull(t.absorb(pass, PassResult("x y", 900)))
        assertEquals(emptyList<ListenEvent>(), events)
        assertEquals(1, t.stats().abortedPasses)
        assertTrue("its audio is zeroed", pass.samples.all { it == 0f })
    }

    @Test
    fun aPassThatFinishesAfterTheSpeechEndedIsDroppedAndTheFinalRecognisesItsWords() {
        val t = tracker()
        val rig = Rig(t)
        val first = rig.nextPass()
        rig.answer(first, "a b c")
        val second = rig.nextPass()
        // the recogniser had finished the second pass when the speech ended and the pass was cancelled
        t.pieceEnded()
        val u = utterance()
        t.finalComing(u)
        assertNull(t.absorb(second, PassResult("a b c d", 500)))
        val plan = t.finalPlan(u)
        assertFalse(plan.hadChunks)
        val chunk = t.absorbFinal(plan, "A b c d.", 900)
        assertEquals(Chunk("A b c d.", 480, 2_900, 900, true), chunk)
        assertEquals(0, t.stats().abortedPasses) // never attached: not counted as aborted
    }

    @Test
    fun noPassWhileAFinalIsPendingAndTheFinalCoversOnlyTheRest() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        rig.quiet(6_000, pieceMs = 4_000, audioStartMs = 3_800)
        val plan = t.finalPlan(u)
        assertEquals(0, plan.offset)
        assertTrue(plan.hadChunks)
        val chunk = t.absorbFinal(plan, "Dobrý den, jak se máte?", 1_200)
        assertEquals(Chunk("den, jak se máte?", 480, maxOf(u.endMs, 1504), 1_200, false), chunk)
        t.finalDone(u, true)
        val next = rig.nextPass(pieceMs = 4_000, audioStartMs = 3_800)
        assertEquals(3_800, next.startMs)
    }

    @Test
    fun afterACutTheFinalPlanStartsAtTheBufferStart() {
        val t = tracker()
        val rig = Rig(t)
        val ends = mutableListOf<Long>()
        for (k in 1..6) {
            val pass = rig.nextPass()
            ends += pass.endMs
            rig.answer(pass, words(2 * k + 1))
        }
        assertEquals(listOf(1504L, 2528L, 3552L, 4576L, 5600L, 6624L), ends)
        assertEquals(1, t.stats().cuts)
        val next = rig.nextPass()
        assertEquals(3_167, next.startMs) // the first window of [3 152, 3 652) ms, its centre 15 ms in
        val u = utterance(endMs = 6_900, audioMs = 7_000 - 160)
        t.pieceEnded()
        t.finalComing(u)
        val plan = t.finalPlan(u)
        assertEquals((3_167 - 160) * 16, plan.offset)
        assertEquals(u.samples.size - plan.offset, plan.samples().size)
    }

    @Test
    fun aPassAbsorbedAsItsPieceEndsSendsItsChunkAndTheFinalSkipsIt() {
        val t = tracker()
        val rig = Rig(t)
        val first = rig.nextPass()
        rig.answer(first, "dobrý den jak")
        events.clear()
        val second = rig.nextPass()
        assertTrue(second.attach(AbortToken(FakeNative())))
        val absorbed = t.absorb(second, PassResult("dobrý den jak se", 700))!!
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.publish(absorbed, null, 0)
        assertEquals(listOf<ListenEvent>(transcript("dobrý den jak", 480, 1504, 700)), events) // no preview after it
        val chunk = t.absorbFinal(t.finalPlan(u), "dobrý den jak se máte", 800)
        assertEquals(Chunk("se máte", 480, 2_900, 800, false), chunk)
    }

    @Test
    fun aChunkAbsorbedWhileItsPieceWasStillOpenButPublishedAfterTheFinalHasBegunStillGoesOut() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        val second = rig.nextPass()
        val absorbed = t.absorb(second, PassResult("a b c d", 500))!!
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        val plan = t.finalPlan(u)
        val chunk = t.absorbFinal(plan, "a b c d e", 600)
        t.publish(absorbed, null, 0)
        assertEquals("a b c", (events.last() as SpeechEvent.Transcript).transcript.text)
        assertEquals("d e", chunk!!.text)
        assertFalse(chunk.firstOfPiece)
    }

    @Test
    fun chunkEndsNeverDecreaseAndTheLastIsTheUtterancesEnd() {
        val t = tracker()
        val rig = Rig(t)
        for (text in listOf("a b c", "a b c d", "a b c d e f", "a b c d e f g")) rig.answer(rig.nextPass(), text)
        val u = utterance(endMs = 4_000)
        t.pieceEnded()
        t.finalComing(u)
        val last = t.absorbFinal(t.finalPlan(u), "a b c d e f g h", 500)!!
        val ends = events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.endMs } + last.endMs
        assertEquals(listOf(1_504L, 2_528L, 3_552L, 4_000L), ends)
        assertEquals(listOf(480L), events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.startMs }.distinct())
    }

    @Test
    fun theFinalsChunkEndsAtTheLastChunksEndWhenTheUtteranceEndsEarlier() {
        val t = tracker()
        val rig = Rig(t)
        for (text in listOf("a b c", "a b c d", "a b c d e f", "a b c d e f g")) rig.answer(rig.nextPass(), text)
        val u = utterance(endMs = 3_000)
        t.pieceEnded()
        t.finalComing(u)
        assertEquals(3_552, t.absorbFinal(t.finalPlan(u), "a b c d e f g h", 500)!!.endMs)
    }

    @Test
    fun aSlowDeviceGetsNoPassesAndALagCutAtEightSeconds() {
        val guard = StreamingGuard(15_000, 384).also { it.recorded(6_000, 15_000) }
        assertTrue(guard.passesPaused)
        val t = tracker(guard = guard)
        val rig = Rig(t)
        var cutAt = -1L
        while (rig.positionMs < 20_000 && cutAt < 0) {
            when (rig.frame()) {
                is StreamingTracker.Step.Pass -> fail("a pass on a slow device")
                StreamingTracker.Step.Cut, StreamingTracker.Step.CutQuiet -> cutAt = rig.positionMs
                StreamingTracker.Step.None -> {}
            }
        }
        assertEquals(8_160, cutAt) // 8 000 ms after the start of the buffer (160)
        rig.quiet(12_000) // once for the piece
        // the cut piece goes away, a new one starts at 12 000
        val secondStart = 12_000L
        var secondCut = -1L
        while (rig.positionMs < 30_000 && secondCut < 0) {
            if (rig.frame(secondStart, secondStart - 320) == StreamingTracker.Step.Cut) secondCut = rig.positionMs
        }
        assertEquals(secondStart - 320 + 8_000, secondCut)
        assertEquals(2, t.stats().forcedCuts)
    }

    @Test
    fun aBufferIsCutByTwelveSecondsAtTheLatestWhateverTheSpeed() {
        val t = tracker()
        val rig = Rig(t)
        var cutAt = -1L
        while (rig.positionMs < 30_000 && cutAt < 0) {
            when (rig.frame()) {
                is StreamingTracker.Step.Pass -> {} // never answered: the recogniser hangs
                StreamingTracker.Step.Cut, StreamingTracker.Step.CutQuiet -> cutAt = rig.positionMs
                StreamingTracker.Step.None -> {}
            }
        }
        assertTrue("cut at $cutAt", cutAt - 160 in 8_000..12_031)
    }

    @Test
    fun aCutThatComesDuringAPassCancelsItWhenThePieceEndsAndTheFinalCoversTheBuffer() {
        val guard = StreamingGuard(15_000, 384)
        val t = tracker(guard = guard)
        val rig = Rig(t)
        val pass = rig.nextPass()
        assertTrue(pass.attach(AbortToken(FakeNative())))
        // the recogniser is slow: the pass is still running when the buffer must be cut
        guard.recorded(6_000, 15_000)
        var cut = false
        while (!cut && rig.positionMs < 20_000) cut = rig.frame() == StreamingTracker.Step.Cut
        assertTrue(cut)
        assertFalse("the pass runs on until the piece ends", pass.isCancelled)
        t.pieceEnded()
        assertTrue(pass.isCancelled)
        assertNull(t.absorb(pass, PassResult("a b c", 500)))
        val u = utterance(endMs = 8_000, audioMs = 8_100)
        t.finalComing(u)
        assertEquals(0, t.finalPlan(u).offset)
    }

    @Test
    fun passesPauseOnASlowDeviceAndResumeWhenItIsFastAgain() {
        val t = tracker()
        val rig = Rig(t)
        val first = rig.nextPass() // 1 344 ms of audio, in the window of 384 frames (7 680 ms)
        rig.answer(first, "a b c", recogniserMs = 2_100)
        assertEquals(1, t.stats().pauses)
        rig.quiet(rig.positionMs + 2_400) // no pass any more (and not yet the cut at the target length of 4 s, at 4 128 ms)
        for (i in 1..3) {
            val u = utterance(startMs = 20_000L * i, endMs = 20_000L * i + 900, audioStartMs = 20_000L * i - 320, audioMs = 1_344)
            t.finalComing(u)
            t.absorbFinal(t.finalPlan(u), "x", 1_000)
            t.finalDone(u, true)
        }
        assertEquals(1, t.stats().pauses)
        val resumedAt = rig.positionMs
        val next = rig.nextPass(pieceMs = resumedAt, audioStartMs = resumedAt - 320, limitMs = resumedAt + 3_000)
        assertTrue(next.endMs > resumedAt)
    }

    @Test
    fun aPieceThatChangesWithoutAnUtteranceWithdrawsItsPreview() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        rig.frame(pieceMs = 9_000, audioStartMs = 8_700)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "")), events)
    }

    @Test
    fun aPieceThatEndedWithoutAnUtteranceWithdrawsItsPreviewAtTheNextFrame() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        t.pieceEnded()
        assertEquals(emptyList<ListenEvent>(), events)
        rig.frame(pieceMs = -1)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "")), events)
    }

    @Test
    fun stopAbortsTheRunningPassAndNothingIsAskedAfterIt() {
        val t = tracker()
        val rig = Rig(t)
        val pass = rig.nextPass()
        assertTrue(pass.attach(AbortToken(FakeNative())))
        t.stop()
        assertTrue(pass.isCancelled)
        // a lag cut is still asked for after stop() (a piece must not run on to the length cap), but no pass is
        val limit = rig.positionMs + 20_000
        while (rig.positionMs < limit) assertFalse("a pass after stop() at ${rig.positionMs} ms", rig.frame() is StreamingTracker.Step.Pass)
        assertNull(t.absorb(pass, PassResult("a b c", 500)))
        assertEquals(1, t.stats().abortedPasses)
    }

    @Test
    fun aFinalWithoutTextWithdrawsTheShownPreviewAndCountsTheLostWords() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        assertNull(t.absorbFinal(t.finalPlan(u), "", 800))
        t.finalDone(u, false)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, ""), SpeechEvent.UtteranceDone(480)), events)
        assertEquals(1, t.stats().emptyFinals)
        assertEquals(3, t.stats().lostPendingWords)
    }

    @Test
    fun aFinalThatIsNotComingWithdrawsAndForgetsThePiece() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.finalNotComing(u)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, ""), SpeechEvent.UtteranceDone(480)), events)
        val next = rig.nextPass(pieceMs = 5_000, audioStartMs = 4_700) // finals pending is back to 0
        assertEquals(4_700, next.startMs)
    }

    @Test
    fun passesNobodyTookAreZeroed() {
        val t = tracker()
        val rig = Rig(t)
        val pass = rig.nextPass() // taken by "nobody": put back
        assertTrue(t.requests.trySend(pass).isSuccess)
        assertTrue(pass.samples.any { it != 0f })
        t.discardRequests()
        assertTrue(pass.samples.all { it == 0f })
        assertTrue(t.requests.isClosedForSend)
    }

    @Test
    fun theAudioOfAnAbsorbedPassIsZeroedAfterTheQuietPointWasFoundOnIt() {
        val t = tracker()
        val rig = Rig(t)
        for (k in 1..5) rig.answer(rig.nextPass(), words(2 * k + 1))
        val sixth = rig.nextPass()
        assertTrue(sixth.samples.all { it == 0.3f })
        val absorbed = t.absorb(sixth, PassResult(words(13), 500))!!
        assertEquals(3_167, absorbed.outcome.cut!!.atMs) // found on the audio...
        assertTrue(sixth.samples.all { it == 0f }) // ...which is zeroed now
    }

    @Test
    fun aPassIsAbsorbedOnceAndOnlyOverTheCurrentBufferStart() {
        val t = tracker()
        val rig = Rig(t)
        for (k in 1..5) rig.answer(rig.nextPass(), words(2 * k + 1))
        val sixth = rig.nextPass()
        assertNotNull(t.absorb(sixth, PassResult(words(13), 500)))
        assertNull("the same pass twice", t.absorb(sixth, PassResult(words(13), 500)))
        // a pass over the buffer as it was before the cut
        val old = StreamingPass(480, 160, 7_000, FloatArray(16_000))
        assertNull(t.absorb(old, PassResult(words(15), 500)))
        assertTrue(old.samples.all { it == 0f })
        assertEquals(1, t.stats().stalePasses)
        // and the next pass over the new buffer still works
        val next = rig.nextPass()
        assertEquals(3_167, next.startMs)
        assertNotNull(rig.answer(next, (8..15).joinToString(" ") { "w$it" }))
    }

    @Test
    fun aPassOverAPieceThatEndedIsDroppedEvenWhenItWasNotCancelled() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        t.pieceEnded()
        val stray = StreamingPass(480, 160, 3_000, FloatArray(8_000) { 0.3f })
        assertNull(t.absorb(stray, PassResult("a b c d", 500)))
        assertEquals(1, t.stats().stalePasses)
        assertTrue(stray.samples.all { it == 0f })
    }

    @Test
    fun anEmptyPassCommitsNothingAndWithdrawsThePreview() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        val absorbed = rig.answer(rig.nextPass(), "")!!
        assertEquals(emptyList<String>(), absorbed.outcome.committed)
        assertNull(absorbed.chunk)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "")), events)
    }

    @Test
    fun aPassOfPunctuationOnlyIsEmptyBeforeTheCommitterSeesIt() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        events.clear()
        val absorbed = rig.answer(rig.nextPass(), " … ")!!
        assertEquals(emptyList<String>(), absorbed.outcome.pending)
        assertEquals(emptyList<String>(), absorbed.outcome.committed)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "")), events)
        // a final of punctuation only is empty too
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        assertNull(t.absorbFinal(t.finalPlan(u), ".", 500))
    }

    @Test
    fun endingTheSessionDuringAPassCancelsItAndSendsNothing() {
        val t = tracker()
        val rig = Rig(t)
        val pass = rig.nextPass()
        assertTrue(pass.attach(AbortToken(FakeNative())))
        t.end()
        assertTrue(pass.isCancelled)
        assertNull(t.absorb(pass, PassResult("a b c", 500)))
        assertEquals(emptyList<ListenEvent>(), events)
        assertEquals(StreamingTracker.Step.None, rig.frame())
        assertTrue(t.requests.isClosedForSend)
    }

    @Test
    fun anEventSinkThatThrowsDoesNotBringTheTrackerDown() {
        val t = tracker(emit = { throw IllegalStateException("closed") })
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        rig.answer(rig.nextPass(), "a b c d")
        assertEquals(3, t.stats().errors) // a preview, then a transcript and a preview: all lost, all counted
    }

    @Test
    fun aSnapshotThatFailsGivesNoPassAndNoException() {
        val t = tracker()
        var position = 0L
        var failed = 0
        while (position < 2_000) {
            position += 32
            val step = t.onFrame(480, 160, position) { failed++; throw IllegalStateException("no audio") }
            assertEquals(StreamingTracker.Step.None, step)
        }
        assertTrue(failed > 0)
        assertEquals(failed, t.stats().errors)
    }

    @Test
    fun theFinalIsAbsorbedOnce() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        val plan = t.finalPlan(u)
        assertNotNull(t.absorbFinal(plan, "dobrý den jak se máte", 500))
        assertNull(t.absorbFinal(plan, "dobrý den jak se máte", 500))
    }

    @Test
    fun aFinalOfAnUnknownPieceRecognisesTheWholeAudio() {
        val t = tracker()
        val u = utterance()
        t.finalComing(u)
        val plan = t.finalPlan(u)
        assertEquals(0, plan.offset)
        assertFalse(plan.hadChunks)
        assertEquals(Chunk("Dobrý den.", 480, 2_900, 700, true), t.absorbFinal(plan, "Dobrý den.", 700))
    }

    @Test
    fun theCostEstimateOfTheDefaultGuardUsesTheModelsWindowFloor() {
        // a final over 8 s of audio took 2 200 ms: the window is 8 640 ms at the floor 384 (a pass over 6 s is then
        // estimated at 1 956 ms, not slow) and 10 240 ms at the floor 512 (2 200 ms for 6 s: slow, passes pause)
        fun pausesAfterSlowFinal(floor: Int): Int {
            val t = StreamingTracker(StreamingConfig(), floor, true) { }
            val u = utterance(audioMs = 8_000)
            t.finalComing(u)
            t.absorbFinal(t.finalPlan(u), "x", 2_200)
            return t.stats().pauses
        }
        assertEquals(0, pausesAfterSlowFinal(384))
        assertEquals(1, pausesAfterSlowFinal(512))
    }

    @Test
    fun passTextWithoutLettersIsEmptyAndLoopsAreRemoved() {
        assertEquals("", PassText.clean(" … "))
        assertEquals("", PassText.clean("."))
        assertEquals("", PassText.clean(""))
        assertEquals("a b c", PassText.clean(" a b c "))
        val loop = "x " + "dobrý den jak se ".repeat(4)
        assertEquals("x dobrý den jak se", PassText.clean(loop).trim())
        assertEquals(PassText.clean(loop), PassText.clean(PassText.clean(loop)))
    }

    @Test
    fun statsCountPassesTranscriptsWordsCutsAndDelays() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.absorbFinal(t.finalPlan(u), "Dobrý den, jak se máte?", 1_200)
        val s = t.stats()
        assertEquals(2, s.passes)
        assertEquals(0, s.abortedPasses)
        assertEquals(400L + 900L, s.passRecogniserMs)
        assertEquals(2, s.transcripts) // the pass chunk and the final chunk
        assertEquals(1, s.wordsWhileSpeaking)
        assertEquals(4, s.wordsAfter)
        assertEquals(0, s.cuts)
        assertEquals(0, s.forcedCuts)
        assertEquals(0, s.pauses)
        assertEquals(2_528L - 160, s.maxBufferMs)
        assertEquals(listOf(2_528L - 1_504L, 0L), s.chunkDelaysMs) // the final chunk ends after the position: no delay
        assertEquals(0, s.finalSeams)
    }

    @Test
    fun aSeamDecidedByTheFinalIsCounted() {
        val t = tracker()
        val rig = Rig(t)
        for (k in 1..6) rig.answer(rig.nextPass(), words(2 * k + 1)) // the sixth cuts; no pass after it
        val u = utterance(endMs = 6_900, audioMs = 7_000 - 160)
        t.pieceEnded()
        t.finalComing(u)
        val chunk = t.absorbFinal(t.finalPlan(u), (8..15).joinToString(" ") { "w$it" }, 500)!!
        assertEquals("w12 w13 w14 w15", chunk.text)
        assertEquals(1, t.stats().finalSeams)
    }

    @Test
    fun eventsAreEmittedUnderTheLockSoTheyNeverInterleaveWithAnotherCall() {
        // emit runs inside publish: a thread that calls into the tracker from emit would deadlock only if the lock
        // were not reentrant; here emit reads the stats (reentrant) and the order of events stays intact
        lateinit var t: StreamingTracker
        val seen = mutableListOf<Int>()
        t = tracker(emit = { events += it; seen += t.stats().transcripts })
        val rig = Rig(t)
        twoPasses(rig)
        assertEquals(listOf<ListenEvent>(SpeechPreview(480, "dobrý den jak", 0), transcript("dobrý", 480, 1504, 900), SpeechPreview(480, "den, jak se", 0)), events)
        assertEquals(3, seen.size)
    }

    /**
     * The capture thread and the whisper thread really run side by side: an ideal recogniser (it hears every word
     * that lies wholly in its audio, one word per 400 ms) over 60 s of speech. Whatever the interleaving, nothing is
     * thrown and the chunks put together are the words, each once, in order. No sleeping: the threads meet at
     * semaphores, and the capture thread goes on with a few frames while a pass is recognised.
     */
    @Test
    fun theCaptureAndTheWhisperThreadsTogetherWriteEveryWordOnceInOrder() {
        for (seed in 1..6) {
            val emitted = java.util.Collections.synchronizedList(mutableListOf<ListenEvent>())
            val t = StreamingTracker(StreamingConfig(), 384, true) { emitted += it }
            val total = 150 // words of 400 ms each, from 480 ms
            fun hear(fromMs: Long, toMs: Long): String =
                (0 until total).filter { 480L + 400 * it >= fromMs && 480L + 400 * it + 400 <= toMs }
                    .joinToString(" ") { "v$it" }
            val failure = AtomicReference<Throwable?>(null)
            val handed = Semaphore(0)
            val answered = Semaphore(0)
            val whisper = thread(name = "whisper-$seed") {
                try {
                    while (true) {
                        if (!handed.tryAcquire(5, TimeUnit.SECONDS)) break
                        val pass = t.requests.tryReceive().getOrNull() ?: break
                        val absorbed = t.absorb(pass, PassResult(hear(pass.startMs, pass.endMs), 300))
                        if (absorbed != null) t.publish(absorbed, null, 0)
                        answered.release()
                    }
                } catch (e: Throwable) {
                    failure.set(e)
                    answered.release(1000)
                }
            }
            var position = 0L
            val rng = java.util.Random(seed.toLong())
            val speechEnd = 480L + 400L * total
            var waitingFor = 0
            var frames = 0
            while (position < speechEnd + 600 && failure.get() == null) {
                position += 32
                val pos = position
                val step = t.onFrame(480, 160, pos) { from -> FloatArray(((pos - from) * 16).toInt()) { 0.3f } }
                if (step is StreamingTracker.Step.Pass) {
                    handed.release()
                    waitingFor++
                    frames = rng.nextInt(40)
                } else if (waitingFor > 0 && --frames < 0) {
                    assertTrue(answered.tryAcquire(10, TimeUnit.SECONDS))
                    waitingFor--
                }
            }
            while (waitingFor > 0 && failure.get() == null) {
                assertTrue(answered.tryAcquire(10, TimeUnit.SECONDS))
                waitingFor--
            }
            val u = Utterance(FloatArray(((speechEnd - 160) * 16).toInt()) { 0.3f }, 480, speechEnd, audioStartMs = 160)
            t.pieceEnded()
            t.finalComing(u)
            val plan = t.finalPlan(u)
            val last = t.absorbFinal(plan, hear(plan.utterance.audioStartMs + plan.offset / 16, speechEnd), 500)
            t.finalDone(u, last != null)
            t.end()
            handed.release(10)
            whisper.join(10_000)
            failure.get()?.let { throw AssertionError("seed $seed: a thread failed", it) }
            val chunks = emitted.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript.text } + listOfNotNull(last?.text)
            assertEquals("seed $seed", (0 until total).joinToString(" ") { "v$it" }, chunks.joinToString(" "))
            assertTrue("seed $seed: passes ran", t.stats().passes > 5 && t.stats().cuts > 0)
        }
    }

    // ---- where a cut may fall, forced cuts that must not be lost, delays of the finals' chunks ----------------------

    private fun slowGuard() = StreamingGuard(15_000, 384).also { it.recorded(6_000, 15_000) }

    /** Frames until the tracker asks for a cut; the position of that frame. */
    private fun Rig.untilCut(pieceMs: Long = 480, audioStartMs: Long = 160, limitMs: Long = positionMs + 20_000): Long {
        while (positionMs < limitMs) if (frame(pieceMs, audioStartMs) == StreamingTracker.Step.Cut) return positionMs
        fail("no cut until $limitMs ms")
        error("unreachable")
    }

    @Test
    fun aForcedCutMayNotFallBeforeEightSecondsAfterTheStartOfTheBuffer() {
        val t = tracker(guard = slowGuard())
        val rig = Rig(t)
        assertEquals("anywhere before the piece is asked", 0, t.cutNotBeforeMs())
        rig.untilCut()
        assertEquals(160 + 8_000, t.cutNotBeforeMs())
    }

    /** Passes paused (a pass over 6 s takes 3 s), finals over 6 s 2.25 s (the lowest window): the target is 5.5 s (pieces of 4 to 5.5 s). */
    private fun chunkModeGuard(finalMs: Long = 2_250) = StreamingGuard(15_000, 384).also {
        it.recorded(6_000, 3_000)
        it.recordedFinal(6_000, finalMs)
    }

    @Test
    fun inChunkModeAPieceIsCutAsSoonAsItsTargetLengthIsReachedAndTheCutIsSoughtInTheLastSecondAndAHalfBeforeIt() {
        val t = tracker(guard = chunkModeGuard())
        val rig = Rig(t)
        val at = rig.untilCut()
        assertTrue("5.5 s of the buffer, which starts at 160 ms (frame grid of 32 ms): cut at $at", at - 160 in 5_500..5_531)
        assertEquals("the cut is sought from 4 s of the buffer: no piece is shorter", 160 + 4_000L, t.cutNotBeforeMs())
        assertTrue(t.cutNotBeforeMs() - 160 >= StreamingGuard.MIN_PIECE_MS)
    }

    // ---- a good cut point is used as soon as it exists: from the floor of the cut window on, on a quiet stretch ----

    /** Frames with no quiet stretch until [untilMs] (None each), then the first frame of the run of quiet frames. */
    private fun Rig.speakUntil(untilMs: Long) {
        while (positionMs < untilMs) assertEquals("at $positionMs ms", StreamingTracker.Step.None, frame())
    }

    @Test
    fun aQuietStretchFromTheFloorOfTheCutWindowOnCutsAtOnceInsteadOfWaitingForTheTarget() {
        val t = tracker(guard = chunkModeGuard())                  // target 5.5 s: the window opens at 4 s of the buffer (160 + 4 000)
        val rig = Rig(t)
        rig.speakUntil(160 + 4_000 - 32)
        // the hold: StreamingGuard.QUIET_HOLD_FRAMES consecutive quiet frames, not one frame that dipped inside a word
        for (n in 1 until StreamingGuard.QUIET_HOLD_FRAMES) assertEquals("quiet frame $n", StreamingTracker.Step.None, rig.frame(quietFrames = n))
        assertEquals(StreamingTracker.Step.CutQuiet, rig.frame(quietFrames = StreamingGuard.QUIET_HOLD_FRAMES))
        assertTrue("cut at ${rig.positionMs}, 1.5 s before the target", rig.positionMs - 160 < 5_500 - 1_000)
        assertEquals("never before 4 s of the buffer", 160 + 4_000L, t.cutNotBeforeMs())
    }

    @Test
    fun aQuietStretchBeforeTheFloorOfTheCutWindowCutsNothing() {
        val t = tracker(guard = chunkModeGuard())
        val rig = Rig(t)
        while (rig.positionMs < 160 + 3_900) assertEquals(StreamingTracker.Step.None, rig.frame(quietFrames = 9))
        assertEquals("a piece under 4 s is never cut", StreamingTracker.Step.None, rig.frame(quietFrames = 9))
    }

    @Test
    fun theWindowOfTheCutOpensWhereTheTargetsDoes() {
        // finals of 3.1 s: the target is 6.7 s, so the window opens at 5.2 s
        val t = tracker(guard = chunkModeGuard(finalMs = 3_100))
        val rig = Rig(t)
        while (rig.positionMs + 32 - 160 < 5_200) assertEquals(StreamingTracker.Step.None, rig.frame(quietFrames = 9))
        assertEquals("the first frame with 5.2 s of the buffer", StreamingTracker.Step.CutQuiet, rig.frame(quietFrames = 9))
        assertEquals(160 + 5_200L, t.cutNotBeforeMs())
    }

    @Test
    fun withoutAQuietStretchTheTargetCutComesAsBefore() {
        val t = tracker(guard = chunkModeGuard())
        val rig = Rig(t)
        val at = rig.untilCut()
        assertTrue("cut at $at", at - 160 in 5_500..5_531)
    }

    @Test
    fun aQuietStretchCutsNothingWhileTwoFinalsArePendingNorWhilePassesRun() {
        val two = tracker(guard = chunkModeGuard())
        two.finalComing(utterance(startMs = 100, endMs = 900, audioStartMs = 0))
        two.finalComing(utterance(startMs = 200, endMs = 950, audioStartMs = 0))
        val rig = Rig(two)
        while (rig.positionMs < 160 + 5_000) assertEquals(StreamingTracker.Step.None, rig.frame(quietFrames = 9))
        val running = Rig(tracker())                            // the default guard: passes run, no target
        while (running.positionMs < 160 + 6_000) {
            val step = running.frame(quietFrames = 9)
            if (step is StreamingTracker.Step.Pass) { running.tracker.requests.tryReceive(); continue }
            assertEquals(StreamingTracker.Step.None, step)
        }
    }

    @Test
    fun theTargetIsFoundAgainForEveryPieceAsThePhoneSlowsDown() {
        val g = chunkModeGuard()
        val t = tracker(guard = g)
        val rig = Rig(t)
        assertTrue(rig.untilCut() - 160 in 5_500..5_531)
        g.recordedFinal(6_000, 3_250)                   // the phone warmed up (the median of two is the larger)
        val target = g.targetPieceMs()!!
        assertEquals("a new, longer target", 7_000L, target)
        t.cutRefused()
        rig.quiet(rig.positionMs + 1_900)
        val second = rig.untilCut()
        // the retry after 2 s comes after the target of 7 s was reached already
        assertTrue("the next piece is longer than the 5.5 s of before: cut at $second ms", second - 160 in target..target + 2_100)
    }

    @Test
    fun aTargetCutWaitsWhileTwoFinalsArePendingAndTheLagRuleStillBacksItUp() {
        val one = tracker(guard = chunkModeGuard())
        one.finalComing(utterance(startMs = 100, endMs = 900, audioStartMs = 0))
        val atOne = Rig(one).untilCut()
        assertTrue("one final pending: the target cut at $atOne", atOne - 160 in 5_500..5_531)
        val two = tracker(guard = chunkModeGuard())
        two.finalComing(utterance(startMs = 100, endMs = 900, audioStartMs = 0))
        two.finalComing(utterance(startMs = 200, endMs = 950, audioStartMs = 0))
        val atTwo = Rig(two).untilCut()
        assertTrue("two pending: no target cut (5.5 s), the lag rule at $atTwo", atTwo - 160 in 10_000..12_000)
    }

    @Test
    fun withoutATargetTheLagLimitDecidesAsBefore() {
        val t = tracker(guard = chunkModeGuard(finalMs = 5_000))     // 65 % at any length: no target
        val rig = Rig(t)
        val at = rig.untilCut()
        assertTrue("cut at $at", at - 160 > 8_000)
        assertEquals(160 + 8_000, t.cutNotBeforeMs())
    }

    @Test
    fun passesThatRunKeepTheLagRuleAndNoTargetCut() {
        val t = tracker()                               // the default guard: passes run
        val rig = Rig(t)
        val at = rig.untilCut()
        assertTrue("cut at $at", at - 160 in 8_000..12_031)
    }

    // ---- a piece that starts where a cut ended continues speech: its first chunk says so ----

    /** Cuts the piece in progress (as the endpointer does at [cutMs]) and offers its utterance; the next piece starts there. */
    private fun cutAt(t: StreamingTracker, rig: Rig, cutMs: Long): Utterance {
        val u = utterance(startMs = 480, endMs = cutMs, audioStartMs = 160, audioMs = cutMs - 160)
        t.finalComing(u)
        return u
    }

    private fun firstChunkOf(t: StreamingTracker, u: Utterance): Chunk {
        val chunk = t.absorbFinal(t.finalPlan(u), "Slovo", 100)!!
        t.finalDone(u, true)
        return chunk
    }

    @Test
    fun thePieceAfterAForcedCutContinuesAndTheFirstPieceAndOneAfterSilenceDoNot() {
        val t = tracker(guard = chunkModeGuard())
        val rig = Rig(t)
        val at = rig.untilCut()
        val first = cutAt(t, rig, at - 1_000)
        assertFalse("the first piece of a speech", firstChunkOf(t, first).continuesCut)
        rig.frame(pieceMs = at - 1_000, audioStartMs = at - 1_000)      // the next piece starts where the cut was
        val second = utterance(startMs = at - 1_000, endMs = at + 3_000, audioStartMs = at - 1_000, audioMs = 4_000)
        t.finalComing(second)
        val chunk = firstChunkOf(t, second)
        assertTrue(chunk.toString(), chunk.firstOfPiece && chunk.continuesCut)
        // one that ends in silence (the endpointer's SpeechEnded comes first), and the one that starts after it
        rig.frame(pieceMs = at + 3_000, audioStartMs = at + 3_000)
        t.pieceEnded()
        val third = utterance(startMs = at + 3_000, endMs = at + 5_000, audioStartMs = at + 3_000, audioMs = 2_000)
        t.finalComing(third)
        assertTrue("it continues the one before it", firstChunkOf(t, third).continuesCut)
        rig.frame(pieceMs = at + 9_000, audioStartMs = at + 8_700)
        val fourth = utterance(startMs = at + 9_000, endMs = at + 11_000, audioStartMs = at + 8_700, audioMs = 2_300)
        t.finalComing(fourth)
        assertFalse("after silence: the language model decides", firstChunkOf(t, fourth).continuesCut)
    }

    @Test
    fun aPieceAfterACutOnAQuietFrameDoesNotContinueTheOneBeforeButOneAfterAFrameOfSpeechDoes() {
        for ((quiet, continues) in listOf(true to false, false to true)) {
            val t = tracker(guard = chunkModeGuard())
            val rig = Rig(t)
            val at = rig.untilCut()
            val first = utterance(startMs = 480, endMs = at - 1_000, audioStartMs = 160, audioMs = at - 1_160, quietCut = quiet)
            t.finalComing(first)
            firstChunkOf(t, first)
            rig.frame(pieceMs = at - 1_000, audioStartMs = at - 1_000)
            val second = utterance(startMs = at - 1_000, endMs = at + 2_000, audioStartMs = at - 1_000, audioMs = 3_000)
            t.finalComing(second)
            val chunk = firstChunkOf(t, second)
            assertEquals("quiet cut: $quiet", continues, chunk.continuesCut)
            assertEquals("the first chunk of the piece after a quiet cut says so (the language model decides with a prior)", quiet, chunk.quietCut)
        }
    }

    @Test
    fun aPieceThatStartsAtTheEndOfASilentPieceDoesNotContinueIt() {
        val t = tracker(guard = chunkModeGuard())
        val rig = Rig(t)
        rig.frame()
        t.pieceEnded()
        val u = utterance(startMs = 480, endMs = 2_000)
        t.finalComing(u)
        firstChunkOf(t, u)
        rig.frame(pieceMs = 2_000, audioStartMs = 1_700)               // by chance exactly where the one before ended
        val next = utterance(startMs = 2_000, endMs = 3_500, audioStartMs = 1_700, audioMs = 1_800)
        t.finalComing(next)
        assertFalse(firstChunkOf(t, next).continuesCut)
    }

    @Test
    fun theTrackerKnowsHowManyFinalsArePendingAndBooksARetryInTheGuard() {
        val g = chunkModeGuard()
        val t = tracker(guard = g)
        assertEquals(0, t.pendingFinals())
        val a = utterance(startMs = 100, endMs = 900, audioStartMs = 0)
        val b = utterance(startMs = 200, endMs = 950, audioStartMs = 0)
        t.finalComing(a)
        t.finalComing(b)
        assertEquals(2, t.pendingFinals())
        t.finalDone(a, true)
        assertEquals(1, t.pendingFinals())
        t.retryTime(700)
        assertEquals(1, g.retryCount)
        assertEquals(700L, g.retryRecogniserMs)
    }

    @Test
    fun theFinalPlanSaysWhetherWordsWerePendingWhenAPieceThatWroteWordsEnds() {
        val u = utterance(endMs = 2_900)
        // words committed and others still pending (the last words of the last pass): what remains may be speech
        val pending = tracker()
        twoPasses(Rig(pending))
        val withPending = pending.finalPlan(u)
        assertTrue(withPending.hadChunks)
        assertTrue(withPending.hadPending)
        // the last pass found nothing after the committed words: what remains is the tail of the piece (a breath)
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        rig.answer(rig.nextPass(), "", 300)
        val plan = t.finalPlan(utterance(endMs = 2_900))
        assertTrue(plan.hadChunks)
        assertFalse(plan.hadPending)
        // a piece that wrote nothing has no chunks and no pending words
        val fresh = tracker().finalPlan(utterance(startMs = 20_000, endMs = 21_500, audioStartMs = 19_700))
        assertFalse(fresh.hadChunks)
        assertFalse(fresh.hadPending)
    }

    @Test
    fun theStatisticsHoldTheMedianLengthOfThePieces() {
        val t = tracker()
        for ((i, ms) in listOf(2_000L, 6_000L, 3_000L).withIndex()) {
            val start = 480L + i * 10_000
            val u = utterance(startMs = start, endMs = start + ms, audioStartMs = start - 320, audioMs = ms)
            t.finalComing(u)
            t.absorbFinal(t.finalPlan(u), "a", 100)
        }
        assertEquals(3_000L, t.stats().pieceMedianMs)
    }

    @Test
    fun noCutMayFallBeforeTheEndOfTheAudioWhoseWordsWereCommitted() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)                                  // "dobrý" is committed, over the audio up to 1 504 ms
        assertEquals(1_504, t.cutNotBeforeMs())
    }

    @Test
    fun passesAbsorbedAfterTheCutWasDecidedCommitNothing() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val third = rig.nextPass()
        t.cutNotBeforeMs()                              // the endpointer is cutting the piece now
        assertNull(t.absorb(third, PassResult("dobrý den jak se máte", 500)))
        assertEquals(1, t.stats().stalePasses)
        assertEquals(1_504, t.cutNotBeforeMs())          // and what was committed stays as it was
    }

    @Test
    fun aRefusedForcedCutIsAskedAgainTwoSecondsLater() {
        val t = tracker(guard = slowGuard())
        val rig = Rig(t)
        val at = rig.untilCut()
        assertEquals(1, t.stats().forcedCuts)
        t.cutRefused()
        assertEquals(0, t.stats().forcedCuts)
        rig.quiet(at + 1_984)
        assertEquals(StreamingTracker.Step.Cut, rig.frame())
        assertEquals(1, t.stats().forcedCuts)
    }

    @Test
    fun aBrokenPieceIsStillCutAtTheLagLimit() {
        val t = tracker()
        val rig = Rig(t)
        rig.frame()
        t.finalPlan(Utterance(FloatArray(16_000), 480, 900, audioStartMs = 0))      // not this piece's audio: it is broken
        assertEquals(1, t.stats().errors)
        val at = rig.untilCut()
        assertTrue("cut at $at", at - 160 in 8_000..12_031)
    }

    @Test
    fun aStoppedSessionStillCutsAtTheLagLimit() {
        val t = tracker()
        val rig = Rig(t)
        rig.frame()
        t.stop()
        val at = rig.untilCut()
        assertTrue("cut at $at", at - 160 in 8_000..12_031)
    }

    @Test
    fun aForcedCutComesWhileAFinalIsWaiting() {
        val t = tracker(guard = slowGuard())
        val rig = Rig(t)
        t.finalComing(utterance(startMs = 100, endMs = 900, audioStartMs = 0))      // the piece before is waiting for its final
        assertEquals(8_160, rig.untilCut())
    }

    @Test
    fun theDelayOfTheChunkOfAFinalIsRecordedWhileTheSessionLasts() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)                                  // position 2 528 ms
        val u = utterance(endMs = 2_000)
        t.pieceEnded()
        t.finalComing(u)
        t.absorbFinal(t.finalPlan(u), "Dobrý den, jak se máte?", 1_200)
        assertEquals(listOf(1_024L, 528L), t.stats().chunkDelaysMs)

        val t2 = tracker()
        val rig2 = Rig(t2)
        twoPasses(rig2)
        t2.pieceEnded()
        t2.finalComing(u)
        t2.end()
        t2.absorbFinal(t2.finalPlan(u), "Dobrý den, jak se máte?", 1_200)
        assertEquals("after the end of the session the position is not the time", listOf(1_024L), t2.stats().chunkDelaysMs)
    }

    // ---- UtteranceDone: exactly one per piece, after its last Transcript -------------------------------------------

    private fun done() = events.filterIsInstance<SpeechEvent.UtteranceDone>()

    @Test
    fun theDoneEventComesAfterTheTranscriptOfTheFinalAndOnlyOnce() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        assertTrue(t.absorbFinal(t.finalPlan(u), "Dobrý den, jak se máte?", 1_200) != null)
        assertEquals("not before the final is done with", emptyList<SpeechEvent.UtteranceDone>(), done())
        t.finalDone(u, true)
        assertEquals(listOf(SpeechEvent.UtteranceDone(480)), done())
        assertTrue(events.last() is SpeechEvent.UtteranceDone)
    }

    @Test
    fun aFinalWithoutWordsStillEndsItsPiece() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.finalPlan(u)
        assertNull(t.absorbFinal(t.finalPlan(u), "", 800))
        t.finalDone(u, false)
        assertEquals(listOf(SpeechEvent.UtteranceDone(480)), done())
    }

    @Test
    fun aFinalThatIsNotComingEndsItsPieceToo() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.finalNotComing(u)
        assertEquals(listOf(SpeechEvent.UtteranceDone(480)), done())
    }

    @Test
    fun theDoneEventWaitsForAChunkAbsorbedButNotYetPublished() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        val second = rig.nextPass()
        assertTrue(second.attach(AbortToken(FakeNative())))
        val absorbed = t.absorb(second, PassResult("a b c d", 500))!!
        val u = utterance()
        t.pieceEnded()
        t.finalComing(u)
        t.finalNotComing(u)                       // the queue was full: the capture thread's decision, meanwhile
        assertEquals(emptyList<SpeechEvent.UtteranceDone>(), done())
        t.publish(absorbed, null, 0)
        assertEquals(listOf(SpeechEvent.UtteranceDone(480)), done())
        assertTrue(events.last() is SpeechEvent.UtteranceDone)
    }

    @Test
    fun aPieceDroppedWithoutAFinalIsEndedOnlyWhenItWroteWords() {
        val t = tracker()
        val rig = Rig(t)
        twoPasses(rig)                            // "dobrý" was written
        rig.frame(pieceMs = 9_000, audioStartMs = 8_700)   // the piece changed without an utterance for it
        assertEquals(listOf(SpeechEvent.UtteranceDone(480)), done())

        val t2 = tracker()
        val rig2 = Rig(t2)
        rig2.answer(rig2.nextPass(), "a b c")      // only a preview
        rig2.frame(pieceMs = 9_000, audioStartMs = 8_700)
        assertEquals("a piece that wrote nothing needs none", 1, done().size)
    }

    // ---- speed: the passes cancelled after they finished, the finals, the language model ---------------------------

    @Test
    fun theSpeedOfAPassCancelledAfterItFinishedIsStillMeasured() {
        val g = StreamingGuard(15_000, 384)
        val t = tracker(guard = g)
        val rig = Rig(t)
        val pass = rig.nextPass()
        assertTrue(pass.attach(AbortToken(FakeNative())))
        t.pieceEnded()                                   // cancels it, but the recogniser had finished (2.1 s)
        assertNull(t.absorb(pass, PassResult("a b", 2_100)))
        assertTrue("the slow pass counted", g.passesPaused)
    }

    @Test
    fun aFinalIsMeasuredAsAFinalAndTheLanguageModelIsRemembered() {
        val g = StreamingGuard(15_000, 384)
        val t = tracker(guard = g)
        val u = utterance(audioMs = 6_000)
        t.finalComing(u)
        t.absorbFinal(t.finalPlan(u), "x", 1_000)
        assertTrue("a final cost is known", g.finalCostPerWindowMs != null)
        t.languageModelTime(300)
        assertEquals(300L, g.languageModelMs)

        val g2 = StreamingGuard(15_000, 384)
        val t2 = tracker(guard = g2)
        val rig = Rig(t2)
        rig.answer(rig.nextPass(), "a b c")
        val second = rig.nextPass()
        assertTrue(second.attach(AbortToken(FakeNative())))
        t2.publish(t2.absorb(second, PassResult("a b c d", 500))!!, null, 250)
        assertEquals(250L, g2.languageModelMs)
    }

    // ---- why a pass was dropped (for its log line) ------------------------------------------------------------------

    @Test
    fun aDroppedPassSaysWhy() {
        val t = tracker()
        val rig = Rig(t)
        rig.answer(rig.nextPass(), "a b c")
        // its result came after the cancel
        val cancelled = rig.nextPass()
        assertTrue(cancelled.attach(AbortToken(FakeNative())))
        cancelled.cancel()
        assertNull(t.absorb(cancelled, PassResult("a b c d", 500)))
        assertEquals("cancelled after it finished", cancelled.dropReason)
        // its buffer start is not the committer's
        val old = StreamingPass(480, 100, 7_000, FloatArray(16_000))
        assertNull(t.absorb(old, PassResult("a b c d", 500)))
        assertEquals("its buffer was cut meanwhile", old.dropReason)
        // the endpointer cut the piece
        val third = rig.nextPass()
        t.cutNotBeforeMs()
        assertNull(t.absorb(third, PassResult("a b c d", 500)))
        assertEquals("its utterance was being cut", third.dropReason)
        // the piece ended
        t.pieceEnded()
        val stray = StreamingPass(480, 160, 3_000, FloatArray(8_000))
        assertNull(t.absorb(stray, PassResult("a b c d", 500)))
        assertEquals("its utterance has ended", stray.dropReason)
    }

    @Test
    fun aPassOfABrokenPieceOrAnEndedSessionSaysSo() {
        val t = tracker()
        val rig = Rig(t)
        rig.nextPass()
        t.finalPlan(Utterance(FloatArray(16_000), 480, 900, audioStartMs = 0))      // breaks the piece
        val stray = StreamingPass(480, 160, 3_000, FloatArray(8_000))
        assertNull(t.absorb(stray, PassResult("a b", 500)))
        assertEquals("its utterance fell back to the final recognition", stray.dropReason)

        val t2 = tracker()
        val rig2 = Rig(t2)
        val pass2 = rig2.nextPass()
        t2.end()
        assertNull(t2.absorb(pass2, PassResult("a b", 500)))
        assertTrue(pass2.dropReason == "cancelled after it finished" || pass2.dropReason == "the session has ended")
    }

    /**
     * The ends of pieces (with their finals), stop() and end() on the capture thread, run side by side with the
     * passes and finals on the whisper thread, in the order the engine hands them over (one queue). An ideal
     * recogniser over six pieces of 15 words; one seed in two also stops the session in the middle of a piece. Whatever
     * the interleaving, nothing is thrown, every final is done with, and the chunks together are the words, each once.
     */
    @Test
    fun piecesEndingStopAndEndWhileThePassesRunWriteEveryWordOnceInOrder() {
        for (seed in 1..8) {
            val sent = java.util.Collections.synchronizedList(mutableListOf<String>())
            val t = StreamingTracker(StreamingConfig(), 384, true) { if (it is SpeechEvent.Transcript) sent += it.transcript.text }
            val pieces = 6
            val perPiece = 15
            fun startOf(k: Int) = 480L + 10_000L * k
            fun hear(fromMs: Long, toMs: Long): String =
                (0 until pieces * perPiece).filter {
                    val s = startOf(it / perPiece) + 500L * (it % perPiece)
                    s >= fromMs - 50 && s + 400 <= toMs
                }.joinToString(" ") { "v$it" }
            val queue = java.util.concurrent.LinkedBlockingQueue<Any>()
            val failure = AtomicReference<Throwable?>(null)
            val finalsDone = java.util.concurrent.atomic.AtomicInteger()
            val whisper = thread(name = "whisper-$seed") {
                try {
                    while (true) {
                        when (val task = queue.poll(10, TimeUnit.SECONDS) ?: break) {
                            is StreamingPass -> {
                                val absorbed = t.absorb(task, PassResult(hear(task.startMs, task.endMs), 300))
                                if (absorbed != null) t.publish(absorbed, null, 0)
                            }
                            is Utterance -> {
                                val plan = t.finalPlan(task)
                                val chunk = t.absorbFinal(plan, hear(task.audioStartMs + plan.offset / 16, task.audioStartMs + task.samples.size / 16), 400)
                                if (chunk != null) sent += chunk.text
                                t.finalDone(task, chunk != null)
                                finalsDone.incrementAndGet()
                            }
                            else -> break
                        }
                    }
                } catch (e: Throwable) {
                    failure.set(e)
                }
            }
            val rng = java.util.Random(seed.toLong())
            val stopIn = if (seed % 2 == 0) rng.nextInt(200) else -1       // frames into the last piece
            var position = 0L
            for (k in 0 until pieces) {
                val from = startOf(k)
                val speechEnd = from + 7_500
                position = maxOf(position, from - 32)
                var frames = 0
                while (position < speechEnd + 200 && failure.get() == null) {
                    position += 32
                    val pos = position
                    val step = t.onFrame(from, from - 320, pos) { f -> FloatArray(((pos - f) * 16).toInt()) { 0.3f } }
                    if (step is StreamingTracker.Step.Pass) {
                        t.requests.tryReceive()
                        queue.add(step.pass)
                    }
                    if (k == pieces - 1 && stopIn >= 0 && frames++ == stopIn) t.stop()
                    if (rng.nextInt(4) == 0) Thread.yield()
                }
                t.pieceEnded()
                val u = Utterance(FloatArray(((speechEnd + 600 - (from - 320)) * 16).toInt()) { 0.3f }, from, speechEnd, audioStartMs = from - 320)
                t.finalComing(u)
                queue.add(u)
                t.onFrame(-1, 0, position + 32) { FloatArray(0) }
            }
            t.end()
            queue.add(Any())
            whisper.join(20_000)
            failure.get()?.let { throw AssertionError("seed $seed: a thread failed", it) }
            assertEquals("seed $seed: every final was done with", pieces, finalsDone.get())
            val all = (0 until pieces * perPiece).joinToString(" ") { "v$it" }
            assertEquals("seed $seed", all, synchronized(sent) { sent.joinToString(" ") })
        }
    }

    // ---- the language of a piece ------------------------------------------------------------------------------------

    @Test
    fun wordsCommitOnlyBetweenTwoPassesInTheSameLanguage() {
        val rig = Rig(tracker(language = languageSession()))
        val first = rig.nextPass()
        rig.answer(first, "a b c", language = "cs")
        val second = rig.nextPass()
        val held = rig.answer(second, "a b c", language = "en")
        assertTrue("cs then en: nothing is committed", held!!.outcome.committed.isEmpty())
        assertEquals(listOf("a", "b", "c"), held.outcome.pending)
        val third = rig.nextPass()
        assertEquals("en then en: the words are agreed on", listOf("a", "b", "c"), rig.answer(third, "a b c", language = "en")!!.outcome.committed)
        assertEquals("the first two passes were held", 2, rig.tracker.stats().passesHeldByLanguage)
    }

    @Test
    fun theSamePassesInOneLanguageCommitAtTheSecond() {
        val rig = Rig(tracker(language = languageSession()))
        rig.answer(rig.nextPass(), "a b c", language = "cs")
        assertEquals(listOf("a", "b", "c"), rig.answer(rig.nextPass(), "a b c", language = "cs")!!.outcome.committed)
    }

    @Test
    fun thePieceLanguageChangesAfterTwoAgreeingPasses() {
        // en, cs, en: one deviating pass in between, the piece stays in cs
        val deviating = Rig(tracker(language = languageSession()))
        deviating.answer(deviating.nextPass(), "a b c", language = "en")
        deviating.answer(deviating.nextPass(), "a b c", language = "cs")
        deviating.answer(deviating.nextPass(), "a b c", language = "en")
        assertEquals("cs", deviating.tracker.languageFor(deviating.nextPass()))
        assertEquals(0, deviating.tracker.stats().languageSwitches)

        // en, en: the piece changes to en
        val agreeing = Rig(tracker(language = languageSession()))
        agreeing.answer(agreeing.nextPass(), "a b c", language = "en")
        agreeing.answer(agreeing.nextPass(), "a b c", language = "en")
        assertEquals("en", agreeing.tracker.languageFor(agreeing.nextPass()))
        assertEquals(1, agreeing.tracker.stats().languageSwitches)
    }

    @Test
    fun theFirstPassOfAPieceTakesTheLanguageOfTheSession() {
        val session = languageSession("en")
        val rig = Rig(tracker(language = session))
        val first = rig.nextPass()
        assertEquals("en", rig.tracker.languageFor(first))
        // the language of the piece is kept when the session changes meanwhile
        session.delivered("cs", rescued = false)
        assertEquals("en", rig.tracker.languageFor(first))
    }

    @Test
    fun nothingCommitsBeforeTheFirstDecisivePass() {
        val rig = Rig(tracker(language = languageSession()))
        rig.answer(rig.nextPass(), "a b c", language = "cs", decisive = false)
        assertTrue(rig.answer(rig.nextPass(), "a b c", language = "cs", decisive = false)!!.outcome.committed.isEmpty())
        // the first decisive pass, in the same language as the one before: the words are agreed on
        assertEquals(listOf("a", "b", "c"), rig.answer(rig.nextPass(), "a b c", language = "cs", decisive = true)!!.outcome.committed)
        // later passes that are not decisive commit all the same: the piece has settled
        rig.answer(rig.nextPass(), "a b c d e", language = "cs", decisive = false)
        assertEquals(listOf("d", "e"), rig.answer(rig.nextPass(), "a b c d e f", language = "cs", decisive = false)!!.outcome.committed)
    }

    @Test
    fun withoutALanguageSessionTheTrackerIsUnchanged() {
        val rig = Rig(tracker())
        rig.answer(rig.nextPass(), "a b c", language = "cs", decisive = false)
        assertEquals("a pass in another language and not decisive commits all the same", listOf("a", "b", "c"), rig.answer(rig.nextPass(), "a b c", language = "en", decisive = false)!!.outcome.committed)
        assertEquals(0, rig.tracker.stats().languageSwitches)
        assertEquals(0, rig.tracker.stats().passesHeldByLanguage)
    }

    @Test
    fun theFinalOfAPieceThatCommittedWordsIsForcedInThePieceLanguage() {
        val rig = Rig(tracker(language = languageSession()))
        rig.answer(rig.nextPass(), "a b c", language = "en")
        rig.answer(rig.nextPass(), "a b c d", language = "en")       // the piece is in en now and committed a b c
        val u = utterance()
        rig.tracker.pieceEnded()
        rig.tracker.finalComing(u)
        val plan = rig.tracker.finalPlan(u)
        assertEquals("en", plan.language)
        assertEquals("en", plan.current)
        assertEquals("en", rig.tracker.lastChunkLanguage(u.startMs))
    }

    @Test
    fun theFinalOfAPieceWithoutCommittedWordsIsNotForced() {
        val rig = Rig(tracker(language = languageSession()))
        rig.answer(rig.nextPass(), "a b c", language = "en")
        val u = utterance()
        rig.tracker.pieceEnded()
        rig.tracker.finalComing(u)
        val plan = rig.tracker.finalPlan(u)
        assertNull(plan.language)
        assertEquals("the language of the piece the thresholds are made from", "cs", plan.current)
        assertNull(rig.tracker.lastChunkLanguage(u.startMs))
    }
}
