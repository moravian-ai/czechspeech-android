package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.EarlyStart
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What streaming transcription needs of the endpointer: where a piece's audio starts, a snapshot from a position and a
 * cut on request. Each frame's samples are filled with its index, so the audio shows which frames it holds.
 */
class EndpointerStreamingTest {
    private class Run(config: CzechSpeechConfig) {
        val events = mutableListOf<EndpointerEvent>()
        val endpointer = Endpointer(config) { events += it }
        private var index = 0
        private val frame = FloatArray(FRAME_SAMPLES)

        fun feed(prob: Float, frames: Int) {
            repeat(frames) {
                frame.fill(index.toFloat())
                endpointer.accept(frame, prob)
                index++
            }
        }

        val utterances get() = events.filterIsInstance<EndpointerEvent.UtteranceReady>().map { it.utterance }
        val handedOut get() = events.filterIsInstance<EndpointerEvent.UtteranceEarly>().map { it.utterance }
    }

    /** The audio of [u] starts at the session ms it says: its first sample is the index of that frame. */
    private fun assertAudioStartIsTrue(u: Utterance) =
        assertEquals(u.audioStartMs / FRAME_MS, u.samples[0].toLong())

    @Test fun theAudioOfAPieceStartsAtItsPreRoll() {          // 20 silent frames, then speech; pre-roll 300 ms = 10 frames
        val r = Run(CzechSpeechConfig()); r.feed(0f, 20); r.feed(0.9f, 20)
        assertEquals(320, r.endpointer.audioStartMs); assertEquals(640, r.endpointer.pieceMs)
    }

    @Test fun aSnapshotFromAPositionCopiesTheAudioFromThere() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 20); r.feed(0.9f, 20)
        assertEquals(20f, r.endpointer.snapshotFrom(640)[0])
        assertEquals(30 * FRAME_SAMPLES - 5_280, r.endpointer.snapshotFrom(650).size)
        assertArrayEquals(r.endpointer.snapshot(), r.endpointer.snapshotFrom(0), 0f)
        assertEquals(0, r.endpointer.snapshotFrom(r.endpointer.positionMs).size)
    }

    @Test fun cutNowEndsThePieceAtTheLowestProbabilityOfTheLastTwoSeconds() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.2f, 1); r.feed(0.9f, 20)
        assertTrue(r.endpointer.cutNow())
        val u = r.utterances.single()
        assertEquals((10 + 200 + 1) * 32L, u.endMs)             // after the frame with 0.2
        assertEquals(u.endMs, r.endpointer.pieceMs); assertEquals(u.endMs, r.endpointer.audioStartMs)
        assertEquals(0L, u.audioStartMs)                          // pre-roll from frame 0
    }

    @Test fun anUtteranceSaysHowManyMillisecondsOfItsFramesWereSpeech() {
        // 40 frames of speech (1 280 ms), 10 of a gap above negThreshold (0.2) and 10 of speech, then silence ends it
        val r = Run(CzechSpeechConfig()); r.feed(0f, 5); r.feed(0.9f, 40); r.feed(0.2f, 10); r.feed(0.9f, 10); r.feed(0f, 40)
        assertEquals(50 * FRAME_MS, r.utterances.single().speechMs)
        // a cut counts the speech frames of the piece it ends
        val c = Run(CzechSpeechConfig()); c.feed(0f, 10); c.feed(0.9f, 200); c.feed(0.2f, 1); c.feed(0.9f, 20)
        assertTrue(c.endpointer.cutNow())
        assertEquals(200 * FRAME_MS, c.utterances.single().speechMs)
    }

    @Test fun quietFramesCountsTheFramesBelowNegThresholdAtTheEndOfTheSpeechInProgress() {
        val r = Run(CzechSpeechConfig())
        r.feed(0.02f, 5); assertEquals("no speech in progress", 0, r.endpointer.quietFrames)
        r.feed(0.9f, 20); assertEquals(0, r.endpointer.quietFrames)
        r.feed(0.02f, 3); assertEquals(3, r.endpointer.quietFrames)
        r.feed(0.2f, 1); assertEquals("0.2 is above negThreshold: a gap between words", 0, r.endpointer.quietFrames)
        r.feed(0.02f, 1); r.feed(0.9f, 1); assertEquals(0, r.endpointer.quietFrames)
        r.feed(0.02f, 18); assertEquals("just under minSilenceMs: still in the speech", 18, r.endpointer.quietFrames)
        r.feed(0.02f, 1); assertEquals("the utterance ended", 0, r.endpointer.quietFrames)
    }

    // ---- the quiet-point cut falls inside the quiet stretch that triggered it ----

    /** 10 silent frames, 200 frames of speech, a dip of 2 frames as low as the silence (inside the last 1.5 s window), 30 frames of speech, a quiet stretch of 8 frames. */
    private fun withADipBeforeTheLastWords(): Run {
        val r = Run(CzechSpeechConfig())
        r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.02f, 2); r.feed(0.9f, 30); r.feed(0.02f, StreamingGuard.QUIET_HOLD_FRAMES)
        return r
    }

    @Test fun aQuietPointCutFallsInsideTheQuietStretchNotAtAnEarlierDipOfTheWindow() {
        // the quietest frame of the window (earliest on a tie) is the dip: the 30 frames of the last words are left behind
        val old = withADipBeforeTheLastWords()
        assertTrue(old.endpointer.cutNow())
        assertEquals(10 + 200 + 1, old.utterances.single().samples.size / FRAME_SAMPLES)
        assertEquals(200 * FRAME_MS, old.utterances.single().speechMs)
        // the cut in the quiet stretch keeps all speech and QUIET_CUT_TAIL_MS of the stretch
        val r = withADipBeforeTheLastWords()
        assertTrue(r.endpointer.cutAtQuietStretch())
        val piece = r.utterances.single()
        assertEquals(10 + 200 + 2 + 30 + QUIET_CUT_TAIL_FRAMES, piece.samples.size / FRAME_SAMPLES)
        assertEquals((200 + 30) * FRAME_MS, piece.speechMs)
        assertTrue("a cut on a quiet frame", piece.quietCut)
        // the rest of the stretch is the start of the next piece, which holds only what comes after it
        r.feed(0.9f, 20); r.feed(0.02f, 60)
        assertEquals(2, r.utterances.size)
        assertEquals(20 * FRAME_MS, r.utterances[1].speechMs)
        assertEquals(piece.endMs, r.utterances[1].audioStartMs)
    }

    @Test fun aQuietPointCutIsNeverBeforeTheEarliestCutAndNeverOutsideTheStretch() {
        val r = withADipBeforeTheLastWords()
        val stretchStart = 10 + 200 + 2 + 30
        r.endpointer.cutNotBefore = { (stretchStart + 6) * FRAME_MS }       // later than the tail would reach
        assertTrue(r.endpointer.cutAtQuietStretch())
        assertEquals(stretchStart + 6, r.utterances.single().samples.size / FRAME_SAMPLES)
        // a floor beyond the end of the buffer cuts at the end of the stretch, not after it
        val late = withADipBeforeTheLastWords()
        late.endpointer.cutNotBefore = { 100_000 }
        assertTrue(late.endpointer.cutAtQuietStretch())
        assertEquals(stretchStart + StreamingGuard.QUIET_HOLD_FRAMES, late.utterances.single().samples.size / FRAME_SAMPLES)
    }

    @Test fun aQuietPointCutNeedsAQuietStretchAndEnoughAudio() {
        val noStretch = Run(CzechSpeechConfig()); noStretch.feed(0f, 10); noStretch.feed(0.9f, 200)
        assertFalse(noStretch.endpointer.cutAtQuietStretch())
        val little = Run(CzechSpeechConfig()); little.feed(0f, 10); little.feed(0.9f, 20); little.feed(0.02f, 8)
        assertFalse(little.endpointer.cutAtQuietStretch())
        assertTrue(little.utterances.isEmpty())
    }

    @Test fun theQuietHoldIsTheStopTailAndTheTailKeptIsBetween96And160Ms() {
        assertEquals(STOP_TAIL_SILENCE_MS / FRAME_MS.toInt(), StreamingGuard.QUIET_HOLD_FRAMES)
        assertEquals(8, StreamingGuard.QUIET_HOLD_FRAMES)
        assertTrue(QUIET_CUT_TAIL_MS in 96..160)
        assertTrue("the tail is kept inside the stretch the hold guarantees", QUIET_CUT_TAIL_FRAMES < StreamingGuard.QUIET_HOLD_FRAMES)
        assertEquals(2_000L, SentenceBoundary.ELLIPSIS_PAUSE_LIMIT_MS)
    }

    /** 40 frames of speech, a pause of 28 frames (896 ms), 40 frames of speech, silence. */
    private fun withAPauseOf900Ms(config: CzechSpeechConfig): Run {
        val r = Run(config); r.feed(0f, 10); r.feed(0.9f, 40); r.feed(0.02f, 28); r.feed(0.9f, 40); r.feed(0.02f, 80)
        r.endpointer.finish()
        return r
    }

    @Test fun aPauseOf900MsEndsTheUtteranceAt600MsAndNotAt1200Ms() {
        assertEquals(600, CzechSpeechConfig().minSilenceMs)
        assertEquals("two utterances at the default of 600 ms", 2, withAPauseOf900Ms(CzechSpeechConfig()).utterances.size)
        assertEquals("one at 1200 ms: the pause is inside the speech", 1, withAPauseOf900Ms(CzechSpeechConfig(minSilenceMs = 1_200)).utterances.size)
        // the one utterance holds both stretches and ends with the longer silence
        val one = withAPauseOf900Ms(CzechSpeechConfig(minSilenceMs = 1_200)).utterances.single()
        assertTrue(one.samples.size / FRAME_SAMPLES >= 10 + 40 + 28 + 40)
    }

    @Test fun theRangeOfMinSilenceMsAndEarlyStartWithALongPause() {
        for (ms in listOf(400, 600, 2_000)) assertEquals(ms, CzechSpeechConfig(minSilenceMs = ms).minSilenceMs)
        for (ms in listOf(31, 10_001)) {
            try { CzechSpeechConfig(minSilenceMs = ms); org.junit.Assert.fail("$ms accepted") } catch (e: IllegalArgumentException) { /* expected */ }
        }
        // early start keeps its quarter of a second of pause: with 1200 ms it offers the utterance in the pause of 900 ms,
        // retracts it when the speech goes on, and offers the last one again (used)
        val r = withAPauseOf900Ms(CzechSpeechConfig(minSilenceMs = 1_200, earlyStart = EarlyStart()))
        assertEquals(2, r.handedOut.size)
        assertEquals(1, r.events.count { it is EndpointerEvent.EarlyRetracted })
        assertEquals(1, r.utterances.size)
    }

    @Test fun aCutOnAQuietFrameSaysSoAndOneOnAFrameOfSpeechDoesNot() {
        // a frame under negThreshold (0.10): a pause between sentences shorter than minSilenceMs
        val quiet = Run(CzechSpeechConfig()); quiet.feed(0f, 10); quiet.feed(0.9f, 200); quiet.feed(0.02f, 1); quiet.feed(0.9f, 20)
        assertTrue(quiet.endpointer.cutNow()); assertTrue(quiet.utterances.single().quietCut)
        // 0.2 is above negThreshold: the lowest frame of the window is still speech (a gap between words)
        val speech = Run(CzechSpeechConfig()); speech.feed(0f, 10); speech.feed(0.9f, 200); speech.feed(0.2f, 1); speech.feed(0.9f, 20)
        assertTrue(speech.endpointer.cutNow()); assertFalse(speech.utterances.single().quietCut)
        // an utterance that ends by silence is not a cut
        val ended = Run(CzechSpeechConfig()); ended.feed(0.9f, 40); ended.feed(0f, 40)
        assertFalse(ended.utterances.single().quietCut)
    }

    @Test fun cutNowDoesNothingOutsideSpeechOrWithTooLittleAudio() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); assertFalse(r.endpointer.cutNow())
        r.feed(0.9f, 40); assertFalse(r.endpointer.cutNow()); assertTrue(r.utterances.isEmpty())
    }

    @Test fun cutNowNeedsMoreThanTheSearchWindowButNotOneFrameMore() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 53)   // 63 frames buffered = the 2 s window
        assertFalse(r.endpointer.cutNow())
        r.feed(0.9f, 1)
        assertTrue(r.endpointer.cutNow())
    }

    @Test fun afterACutNowTheRestIsTheNextPieceAndSnapshotsFollowIt() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.2f, 1); r.feed(0.9f, 20)
        assertTrue(r.endpointer.cutNow())
        val cutMs = r.endpointer.audioStartMs
        assertEquals(211f, r.endpointer.snapshotFrom(0)[0])       // frame 211 is the first one kept
        assertEquals(20, r.endpointer.snapshot().size / FRAME_SAMPLES)
        assertEquals(20 * FRAME_SAMPLES, r.endpointer.snapshotFrom(cutMs).size)
        assertFalse("a second cut needs more audio again", r.endpointer.cutNow())
    }

    @Test fun everyEmittedUtteranceCarriesTheStartOfItsAudio() {
        // ended by silence
        val a = Run(CzechSpeechConfig()); a.feed(0f, 20); a.feed(0.9f, 20); a.feed(0.02f, 25)
        val ended = a.utterances.single()
        assertEquals(320L, ended.audioStartMs); assertAudioStartIsTrue(ended)

        // cut at the length cap: the first piece and the rest that follows
        val c = Run(CzechSpeechConfig()); c.feed(0f, 10); c.feed(0.9f, 900); c.feed(0.02f, 25)
        assertEquals(2, c.utterances.size)
        c.utterances.forEach { assertAudioStartIsTrue(it) }
        assertEquals(0L, c.utterances[0].audioStartMs)
        assertEquals(c.utterances[0].endMs, c.utterances[1].audioStartMs)

        // handed out early, and then confirmed as that very utterance
        val e = Run(CzechSpeechConfig(earlyStart = EarlyStart())); e.feed(0f, 20); e.feed(0.9f, 30); e.feed(0.02f, 8)
        val early = e.handedOut.single()
        assertEquals(320L, early.audioStartMs); assertAudioStartIsTrue(early)
        e.feed(0.02f, 12)
        assertTrue(e.utterances.single() === early)

        // cut on request
        val n = Run(CzechSpeechConfig()); n.feed(0f, 10); n.feed(0.9f, 200); n.feed(0.2f, 1); n.feed(0.9f, 20)
        n.endpointer.cutNow()
        assertAudioStartIsTrue(n.utterances.single())
    }

    // The earliest point a cut may fall on, asked of the streaming coordinator inside every cut.

    @Test fun aCutFallsNoEarlierThanTheFloorOfItsCoordinator() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.2f, 1); r.feed(0.9f, 20)
        r.endpointer.cutNotBefore = { 6_800 }                      // the quietest frame (after it: 6 752 ms) lies before it
        assertTrue(r.endpointer.cutNow())
        assertEquals(213 * FRAME_MS, r.utterances.single().endMs)   // the first frame whose end is at or after 6 800 ms
    }

    @Test fun aFloorBeyondTheBufferCutsAtItsEnd() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.2f, 1); r.feed(0.9f, 20)
        r.endpointer.cutNotBefore = { 1_000_000 }
        assertTrue(r.endpointer.cutNow())
        assertEquals(r.endpointer.positionMs, r.utterances.single().endMs)
        r.feed(0.9f, 3)                                             // the piece goes on from there
        assertEquals(r.endpointer.positionMs - 3 * FRAME_MS, r.endpointer.pieceMs)
        assertEquals(3 * FRAME_SAMPLES, r.endpointer.snapshot().size)
    }

    @Test fun aFloorBeforeTheWindowChangesNothing() {
        val r = Run(CzechSpeechConfig()); r.feed(0f, 10); r.feed(0.9f, 200); r.feed(0.2f, 1); r.feed(0.9f, 20)
        r.endpointer.cutNotBefore = { 100 }
        assertTrue(r.endpointer.cutNow())
        assertEquals((10 + 200 + 1) * 32L, r.utterances.single().endMs)
    }

    @Test fun theLengthCapAlsoKeepsToTheFloor() {
        // the buffer holds 875 frames: the cap cuts at the 876th; the dip at frame 830 lies in its last 2 s
        val withFloor = Run(CzechSpeechConfig())
        withFloor.endpointer.cutNotBefore = { 26_900 }
        withFloor.feed(0f, 10); withFloor.feed(0.9f, 820); withFloor.feed(0.2f, 1); withFloor.feed(0.9f, 45)
        assertEquals(841 * FRAME_MS, withFloor.utterances.single().endMs)

        val without = Run(CzechSpeechConfig())
        without.feed(0f, 10); without.feed(0.9f, 820); without.feed(0.2f, 1); without.feed(0.9f, 45)
        assertEquals(831 * FRAME_MS, without.utterances.single().endMs)
    }
}
