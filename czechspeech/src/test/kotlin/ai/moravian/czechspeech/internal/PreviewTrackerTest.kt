package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SpeechPreview
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule

/**
 * The rules of previews without threads or native code: the tests play the capture thread (frames, the ends of
 * pieces, offers to the final queue), the preview runner ([PreviewTracker.finished]) and the recognition loop
 * ([PreviewTracker.finalDone]) by hand. Positions are on the 32 ms frame grid, as in a session.
 */
class PreviewTrackerTest {
    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val emitted = mutableListOf<SpeechPreview>()

    private fun tracker(config: PreviewConfig = PreviewConfig()) = PreviewTracker(config) { emitted += it }

    /** Feeds frames to a tracker and takes the requests it makes, as the capture thread and the runner would. */
    private class Driver(val tracker: PreviewTracker) {
        var positionMs = 0L
        val requests = mutableListOf<PreviewRun>()
        val snapshots = mutableListOf<FloatArray>()

        /**
         * Frames of the piece that started at [pieceMs] (-1: none in progress) until [untilMs] of audio; [speaking]:
         * frames of speech, not of a pause; [capMs]: where the length cap cuts the piece at the latest.
         */
        fun frames(
            pieceMs: Long,
            untilMs: Long,
            speaking: Boolean = true,
            capMs: Long = Long.MAX_VALUE,
            onRequest: (PreviewRun) -> Unit = {},
        ) {
            while (positionMs < untilMs) {
                positionMs += FRAME_MS
                val position = positionMs
                val requested = tracker.onFrame(pieceMs, position, capMs, speaking) {
                    FloatArray(((position - pieceMs) * SAMPLE_RATE / 1000).toInt()) { 0.5f }.also { snapshots += it }
                }
                val run = tracker.requests.tryReceive().getOrNull()
                assertSame("onFrame() returns the run it requested", run, requested)
                if (run != null) {
                    requests += run
                    onRequest(run)
                }
            }
        }
    }

    private fun utterance(startMs: Long, endMs: Long) = Utterance(FloatArray(0), startMs, endMs)

    @Test
    fun theFirstPreviewAfterFiveSecondsOfSpeechThenAtMostEveryFiveSecondsOfAudio() {
        val t = tracker()
        val d = Driver(t)
        // A piece from 0 to 28 s (the length cap); each preview finishes at once.
        d.frames(pieceMs = 0, untilMs = 28_000) { t.finished(it, "text") }

        assertEquals(listOf(5_024L, 10_048L, 15_072L, 20_096L, 25_120L), d.requests.map { it.endMs })
        assertTrue(d.requests.all { it.startMs == 0L })
        assertEquals("a run holds the audio of the piece so far", d.snapshots, d.requests.map { it.samples })
        // the same text again: all of it is stable from the second preview on
        assertEquals(listOf(SpeechPreview(0, "text")) + List(4) { SpeechPreview(0, "text", 4) }, emitted)
        // The work of previews for a 28 s piece: 5 runs over 75.4 s of audio, 2.7 times the piece itself.
        assertEquals(75_360L, d.requests.sumOf { it.samples.size.toLong() } * 1000 / SAMPLE_RATE)
    }

    @Test
    fun whileARecognitionStartedInAPauseWaitsOrRunsNoTimedPreviewStartsAndTheyCountFromThatPause() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 4_000)
        d.frames(pieceMs = 0, untilMs = 4_256, speaking = false)
        t.finalStartingEarly()
        t.earlyStarted(utterance(0, 4_000))
        d.frames(pieceMs = 0, untilMs = 8_000)                      // the speech goes on; the recognition still runs
        assertTrue("none at 5 s: the recogniser is busy with the pause's", d.requests.isEmpty())
        assertTrue(t.fromPause(0, "from the pause"))
        t.earlyEnded()
        d.frames(pieceMs = 0, untilMs = 8_992)
        assertTrue("nor before 5 s after the pause began", d.requests.isEmpty())
        d.frames(pieceMs = 0, untilMs = 9_024)
        assertEquals(listOf(9_024L), d.requests.map { it.endMs })
        assertEquals(listOf(SpeechPreview(0, "from the pause")), emitted)
    }

    @Test
    fun withoutFromPausesTimedPreviewsKeepTheirTimes() {
        val t = tracker(PreviewConfig(fromPauses = false))
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 4_000)
        t.finalStartingEarly()
        t.earlyStarted(utterance(0, 4_000))
        t.earlyEnded()                                              // dropped when the speech went on
        d.frames(pieceMs = 0, untilMs = 5_024)
        assertEquals(listOf(5_024L), d.requests.map { it.endMs })
    }

    @Test
    fun aPreviewFromAPauseIsShownOnlyWhileItsPieceIsInProgressAndIsWithdrawnLikeAnyOther() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 2_000)
        assertFalse("another piece's", t.fromPause(960, "text"))
        assertFalse("nothing to show", t.fromPause(0, ""))
        assertTrue(t.fromPause(0, "text"))
        t.pieceEnded()
        assertFalse("after the end of its speech", t.fromPause(0, "late"))
        // the utterance gives no transcript: the preview from the pause is withdrawn
        val u = utterance(0, 2_000)
        t.finalComing(u)
        t.finalDone(u, transcribed = false)
        assertEquals(listOf(SpeechPreview(0, "text"), SpeechPreview(0, "")), emitted)

        val stopped = tracker()
        Driver(stopped).frames(pieceMs = 0, untilMs = 2_000)
        stopped.stop()
        assertFalse("after stop()", stopped.fromPause(0, "text"))
    }

    @Test
    fun aTimedPreviewRunningWhenAPauseIsHandedOutEarlyIsCancelled() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        val run = d.requests.single()
        t.finalStartingEarly()
        assertTrue(run.isCancelled)
        t.finished(run, "too late")
        assertTrue(emitted.isEmpty())
    }

    @Test
    fun theStartThatTwoPreviewsInARowAgreeOnIsStableWhetherTimedOrFromAPause() {
        val t = tracker(PreviewConfig(firstAfterMs = 2_000, intervalMs = 2_000))
        val d = Driver(t)
        val texts = ArrayDeque(listOf("Pošlu to zákaz", "Pošlu to zákazníkovi zítra"))
        d.frames(pieceMs = 0, untilMs = 5_000) { t.finished(it, texts.removeFirst()) }
        assertTrue(t.fromPause(0, "Pošlu to zákazníkovi zítra ráno."))
        assertEquals(listOf(0, 8, 26), emitted.map { it.stableLength })
        assertEquals(listOf("", "Pošlu to", "Pošlu to zákazníkovi zítra"), emitted.map { it.stableText })
        assertEquals(listOf("Pošlu to zákaz", " zákazníkovi zítra", " ráno."), emitted.map { it.unstableText })
        assertTrue("text is the whole preview, as before", emitted.all { it.text == it.stableText + it.unstableText })

        // a preview that fails or has no words leaves the last text to compare with
        d.frames(pieceMs = 0, untilMs = 7_000) { t.finished(it, "") }
        d.frames(pieceMs = 0, untilMs = 9_000) { t.finished(it, "Pošlu to zákazníkovi hned") }
        assertEquals("Pošlu to zákazníkovi", emitted.last().stableText)

        // the stable part can shrink: it is a preview
        d.frames(pieceMs = 0, untilMs = 11_000) { t.finished(it, "Posílám") }
        assertEquals(0, emitted.last().stableLength)
    }

    @Test
    fun theFirstPreviewOfTheNextUtteranceHasNoStablePartAndAWithdrawnOneNeither() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 11_000) { t.finished(it, "dobrý den") }
        assertEquals(listOf(0, 9), emitted.map { it.stableLength })
        t.pieceEnded()
        val u = utterance(0, 11_000)
        t.finalComing(u)
        t.finalDone(u, transcribed = false)
        assertEquals(SpeechPreview(0, ""), emitted.last())
        d.frames(pieceMs = 12_000, untilMs = 18_000) { t.finished(it, "dobrý den") }
        assertEquals(SpeechPreview(12_000, "dobrý den", 0), emitted.last())
    }

    @Test
    fun theStableLengthIsWithinTheText() {
        assertEquals("abc", SpeechPreview(0, "abc def", 3).stableText)
        assertEquals(" def", SpeechPreview(0, "abc def", 3).unstableText)
        assertEquals("", SpeechPreview(0, "abc").stableText)
        failure<IllegalArgumentException> { SpeechPreview(0, "abc", 4) }
        failure<IllegalArgumentException> { SpeechPreview(0, "abc", -1) }
    }

    @Test
    fun continuousPreviewsStartAsSoonAsThePreviousOneHasEndedAndASecondOfAudioHasCome() {
        val config = PreviewConfig.continuous()
        assertEquals(PreviewConfig(firstAfterMs = 2_000, intervalMs = 1_000), config)

        // a recogniser faster than a second: a preview for every second of audio
        val fast = tracker(config)
        val d = Driver(fast)
        d.frames(pieceMs = 0, untilMs = 6_200) { fast.finished(it, "text") }
        assertEquals(listOf(2_016L, 3_040L, 4_064L, 5_088L, 6_112L), d.requests.map { it.endMs })

        // a phone that takes 2.5 s for a pass: the next pass starts with the frame after the previous one ended
        val phone = tracker(config)
        val d2 = Driver(phone)
        var running: PreviewRun? = null
        var endsAtMs = 0L
        while (d2.positionMs < 12_000) {
            running?.let {
                if (d2.positionMs >= endsAtMs) {
                    phone.finished(it, "text", 2_500)
                    running = null
                }
            }
            d2.frames(pieceMs = 0, untilMs = d2.positionMs + FRAME_MS) {
                running = it
                endsAtMs = it.endMs + 2_500
            }
        }
        assertEquals(listOf(2_016L, 4_576L, 7_136L, 9_696L), d2.requests.map { it.endMs })
    }

    @Test
    fun theTimesComeFromTheConfig() {
        val t = tracker(PreviewConfig(firstAfterMs = 2_000, intervalMs = 3_000))
        val d = Driver(t)
        d.frames(pieceMs = 1_000, untilMs = 12_000) { t.finished(it, "text") }
        assertEquals(listOf(3_008L, 6_016L, 9_024L), d.requests.map { it.endMs })
    }

    @Test
    fun aPreviewThatWouldStillRunAtTheLengthCapIsNotStarted() {
        // A phone that takes 0.35 ms per ms of encoder window (10.24 s at least), as estimated from the last preview:
        // previews at 5, 10, 15 and 20 s of a piece cut at 28 s. The one due at 25 s would end at about 34 s, after
        // the cut, which brings the final text and would abort it.
        fun cost(perMs: Double) = { run: PreviewRun -> (perMs * maxOf(run.endMs - run.startMs, 10_240)).toLong() }
        val phone = cost(0.35)
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 28_000, capMs = 28_000) { t.finished(it, "text", phone(it)) }
        assertEquals(listOf(5_024L, 10_048L, 15_072L, 20_096L), d.requests.map { it.endMs })

        // A faster one gets the fifth; so does the first preview of a session, with nothing to estimate from.
        val fast = cost(0.05)
        val t2 = tracker()
        val d2 = Driver(t2)
        d2.frames(pieceMs = 0, untilMs = 28_000, capMs = 28_000) { t2.finished(it, "text", fast(it)) }
        assertEquals(5, d2.requests.size)
        val t3 = tracker()
        val d3 = Driver(t3)
        d3.frames(pieceMs = 0, untilMs = 5_024, capMs = 5_100)
        assertEquals(1, d3.requests.size)
        // A run that was cancelled or failed teaches nothing: 0 keeps the estimate.
        t.finished(PreviewRun(0, 25_120, FloatArray(0)), "", 0)
        d.frames(pieceMs = 0, untilMs = 28_000 + FRAME_MS, capMs = 28_000)
        assertEquals(4, d.requests.size)
    }

    @Test
    fun aPreviewNeverStartsInAPauseWhichMayBeTheEndOfTheUtterance() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 4_992)
        d.frames(pieceMs = 0, untilMs = 5_568, speaking = false)    // 576 ms of pause when the preview is due
        assertTrue(d.requests.isEmpty())
        d.frames(pieceMs = 0, untilMs = 5_600)                       // the speech goes on
        assertEquals(listOf(5_600L), d.requests.map { it.endMs })
    }

    @Test
    fun onePreviewAtATimeAndAPreviewDueMeanwhileStartsAtTheNextFrame() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 20_000)            // the first preview does not finish meanwhile
        assertEquals(1, d.requests.size)
        t.finished(d.requests[0], "a")
        d.frames(pieceMs = 0, untilMs = 20_032)
        assertEquals(listOf(5_024L, 20_032L), d.requests.map { it.endMs })
    }

    @Test
    fun noPreviewWhileAFinalRecognitionWaitsOrRuns() {
        val t = tracker()
        val d = Driver(t)
        // An utterance from 1 to 3 s goes to the final recognition, which runs until 9 s.
        d.frames(pieceMs = 992, untilMs = 3_584)
        t.pieceEnded()
        val earlier = utterance(992, 3_000)
        t.finalComing(earlier)
        d.frames(pieceMs = -1, untilMs = 3_808)
        d.frames(pieceMs = 3_808, untilMs = 8_992)         // the next utterance: 5.2 s of speech at 9 s
        assertTrue("finals first", d.requests.isEmpty())
        t.finalDone(earlier, transcribed = true)
        d.frames(pieceMs = 3_808, untilMs = 9_024)
        assertEquals(listOf(9_024L), d.requests.map { it.endMs })

        // An offer that fails (a full or closed queue) leaves no final recognition behind.
        val t2 = tracker()
        val d2 = Driver(t2)
        val skipped = utterance(0, 2_000)
        t2.finalComing(skipped)
        t2.finalNotComing(skipped)
        d2.frames(pieceMs = 2_400, untilMs = 7_424)
        assertEquals(listOf(7_424L), d2.requests.map { it.endMs })
    }

    @Test
    fun aResultReachesTheAppOnlyWhileItsUtteranceIsInProgress() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        t.finished(d.requests.single(), "první")
        assertEquals(listOf(SpeechPreview(0, "první")), emitted)

        d.frames(pieceMs = 0, untilMs = 10_048)
        val second = d.requests[1]
        t.pieceEnded()                                       // SpeechEnded comes before the result
        assertTrue("the run of an utterance that has ended is cancelled", second.isCancelled)
        t.finished(second, "druhý")
        assertEquals("a result after the end is dropped", 1, emitted.size)

        // Ended at the length cap: handed to the final recognition, the speech goes on as a new piece.
        val t2 = tracker()
        val d2 = Driver(t2)
        d2.frames(pieceMs = 0, untilMs = 5_024)
        val run = d2.requests.single()
        t2.finalComing(utterance(0, 5_024))
        assertTrue(run.isCancelled)
        d2.frames(pieceMs = 5_024, untilMs = 5_056)
        t2.finished(run, "pozdě")
        assertTrue("a result after the cut is dropped", emitted.size == 1)
    }

    @Test
    fun anEmptyResultIsNotEmittedAndLeavesNothingToWithdraw() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        t.finished(d.requests.single(), "")
        val u = utterance(0, 5_100)
        t.pieceEnded()
        t.finalComing(u)
        t.finalDone(u, transcribed = false)
        assertTrue(emitted.isEmpty())
    }

    @Test
    fun theEndOfAnUtteranceAbortsItsRunningPreviewAndARunNotStartedNeverStarts() {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        val run = d.requests.single()
        val token = AbortToken(native)
        assertTrue(run.attach(token))
        assertTrue(token.begin(handle))                      // the recogniser is running
        t.pieceEnded()
        assertEquals("whisper.cpp is told to stop", listOf(token.generation), native.aborts)
        token.end()

        // A run cancelled before the whisper thread got to it does not start.
        val t2 = tracker()
        val d2 = Driver(t2)
        d2.frames(pieceMs = 0, untilMs = 5_024)
        val late = d2.requests.single()
        t2.pieceEnded()
        assertFalse(late.attach(AbortToken(native)))
    }

    @Test
    fun aPreviewIsWithdrawnWhenItsUtteranceGivesNoTranscript() {
        fun previewed(t: PreviewTracker, d: Driver) {
            d.frames(pieceMs = 0, untilMs = 5_024)
            t.finished(d.requests.single(), "náhled")
        }
        val withdrawn = SpeechPreview(0, "")

        // The transcript replaces it: nothing to withdraw.
        tracker().let { t ->
            previewed(t, Driver(t))
            val u = utterance(0, 5_100)
            t.pieceEnded()
            t.finalComing(u)
            t.finalDone(u, transcribed = true)
            t.end()
            assertEquals(listOf(SpeechPreview(0, "náhled")), emitted)
        }
        // No words, a failed recognition: withdrawn after the final recognition.
        emitted.clear()
        tracker().let { t ->
            val d = Driver(t)
            previewed(t, d)
            val u = utterance(0, 5_100)
            t.pieceEnded()
            t.finalComing(u)
            d.frames(pieceMs = -1, untilMs = 6_000)
            assertEquals("not before the final recognition has ended", 1, emitted.size)
            t.finalDone(u, transcribed = false)
            assertEquals(listOf(SpeechPreview(0, "náhled"), withdrawn), emitted)
        }
        // Skipped (a full queue): withdrawn at once.
        emitted.clear()
        tracker().let { t ->
            previewed(t, Driver(t))
            val u = utterance(0, 5_100)
            t.pieceEnded()
            t.finalComing(u)
            t.finalNotComing(u)
            assertEquals(listOf(SpeechPreview(0, "náhled"), withdrawn), emitted)
        }
        // Dropped by the endpointer after SpeechEnded (too little speech): withdrawn at the next frame.
        emitted.clear()
        tracker().let { t ->
            val d = Driver(t)
            previewed(t, d)
            t.pieceEnded()
            d.frames(pieceMs = -1, untilMs = 5_056)
            assertEquals(listOf(SpeechPreview(0, "náhled"), withdrawn), emitted)
        }
        // Dropped at the length cap: the piece in progress changes with no utterance for it.
        emitted.clear()
        tracker().let { t ->
            val d = Driver(t)
            previewed(t, d)
            d.frames(pieceMs = 5_024, untilMs = 5_056)
            assertEquals(listOf(SpeechPreview(0, "náhled"), withdrawn), emitted)
        }
        // Dropped when capture ends.
        emitted.clear()
        tracker().let { t ->
            previewed(t, Driver(t))
            t.pieceEnded()
            t.end()
            assertEquals(listOf(SpeechPreview(0, "náhled"), withdrawn), emitted)
        }
        // Never shown, never withdrawn.
        emitted.clear()
        tracker().let { t ->
            val d = Driver(t)
            d.frames(pieceMs = 0, untilMs = 4_000)
            t.pieceEnded()
            d.frames(pieceMs = -1, untilMs = 4_100)
            val u = utterance(4_500, 9_600)
            d.frames(pieceMs = 4_500, untilMs = 9_600)       // its preview is requested and never finishes
            t.pieceEnded()
            t.finalComing(u)
            t.finalDone(u, transcribed = false)
            t.end()
            assertTrue(emitted.isEmpty())
        }
    }

    @Test
    fun aPieceCutAtTheLengthCapAndTheNextOneHaveTheirOwnPreviews() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 26_016) { t.finished(it, "a") }
        val first = utterance(0, 26_016)
        t.finalComing(first)                                 // the cut: the first piece goes to the final recognition
        d.frames(pieceMs = 26_016, untilMs = 32_000) { t.finished(it, "b") }
        assertEquals("finals first", 5, d.requests.size)
        t.finalDone(first, transcribed = true)
        d.frames(pieceMs = 26_016, untilMs = 32_032) { t.finished(it, "b") }
        assertEquals(
            listOf(SpeechPreview(0, "a")) + List(4) { SpeechPreview(0, "a", 1) } + SpeechPreview(26_016, "b"),
            emitted,
        )
    }

    @Test
    fun afterStopNoPreviewStartsAndTheRunningOneIsCancelledAtOnce() {
        // stop() ends the utterance in progress within 1 s; a preview started then would only delay its final text.
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        val run = d.requests.single()
        t.stop()
        assertTrue("cancelled at once, before its utterance ends", run.isCancelled)
        t.finished(run, "pozdě")
        d.frames(pieceMs = 0, untilMs = 12_000)
        assertEquals("none after stop()", 1, d.requests.size)
        assertTrue(emitted.isEmpty())

        // A preview shown before stop() is still replaced or withdrawn as usual.
        val t2 = tracker()
        val d2 = Driver(t2)
        d2.frames(pieceMs = 0, untilMs = 5_024)
        t2.finished(d2.requests.single(), "náhled")
        t2.stop()
        t2.pieceEnded()
        d2.frames(pieceMs = -1, untilMs = 5_056)
        assertEquals(listOf(SpeechPreview(0, "náhled"), SpeechPreview(0, "")), emitted)
    }

    @Test
    fun endCancelsTheRunAndClosesTheRequests() {
        val t = tracker()
        val d = Driver(t)
        d.frames(pieceMs = 0, untilMs = 5_024)
        val run = d.requests.single()
        t.end()
        assertTrue(run.isCancelled)
        assertTrue("closed", t.requests.tryReceive().isClosed)
        assertNull(t.onFrame(0, 20_000, Long.MAX_VALUE, true) { error("no snapshot after the end") })
        t.finished(run, "pozdě")
        assertTrue(emitted.isEmpty())
        t.end()                                              // again: nothing
    }

    @Test
    fun aRequestNobodyTookIsZeroedWhenTheSessionEnds() {
        // The runner never took it (its coroutine had not started when the session was cancelled).
        val t = tracker()
        var samples = FloatArray(0)
        var position = 0L
        while (position < 5_024) {
            position += FRAME_MS
            t.onFrame(0, position, Long.MAX_VALUE, true) { FloatArray(512) { 0.5f }.also { samples = it } }
        }
        assertTrue(samples.isNotEmpty() && samples.all { it == 0.5f })
        t.end()
        t.discardRequests()
        assertTrue("zeroed", samples.all { it == 0f })
        assertTrue("and gone", t.requests.tryReceive().isClosed)
        t.discardRequests()                                  // again: nothing
    }

    @Test
    fun aRequestThatCannotBeDeliveredIsZeroedAndPreviewsStop() {
        val t = tracker()
        val d = Driver(t)
        t.requests.cancel()                                  // the runner is gone
        d.frames(pieceMs = 0, untilMs = 12_000)
        assertEquals("one snapshot, then no more", 1, d.snapshots.size)
        assertTrue("its audio is zeroed", d.snapshots.single().all { it == 0f })
        assertTrue(d.requests.isEmpty())
    }

    @Test
    fun previewConfigDefaultsAndLimits() {
        assertEquals(PreviewConfig(firstAfterMs = 5_000, intervalMs = 5_000), PreviewConfig())
        PreviewConfig(firstAfterMs = 1_000, intervalMs = 29_000)
        PreviewConfig(firstAfterMs = 29_000, intervalMs = 1_000)
        for (bad in listOf(999, 29_001, 0, -1)) {
            val first = failure<IllegalArgumentException> { PreviewConfig(firstAfterMs = bad) }
            assertTrue(first.message, first.message!!.contains("firstAfterMs"))
            val interval = failure<IllegalArgumentException> { PreviewConfig(intervalMs = bad) }
            assertTrue(interval.message, interval.message!!.contains("intervalMs"))
        }
    }

    @Test
    fun aPreviewDoesNotPrintItsText() {
        assertEquals(
            "SpeechPreview(startMs=480, text=<9 chars>, stableLength=5)",
            SpeechPreview(480, "dobrý den", 5).toString(),
        )
    }
}
