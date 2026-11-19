package ai.moravian.czechspeech.internal

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout

/** The decision about an utterance handed out early, and the times in its log lines. Times are given, in ns. */
class EarlyRunTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(30)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private fun ms(value: Long) = value * 1_000_000

    private fun token(native: FakeNative = FakeNative()) = AbortToken(native)

    @Test
    fun confirmedWhileTheRecognitionRunsSavesTheTimeItRanBeforeTheConfirmation() = runBlocking {
        val run = EarlyRun()
        assertTrue(run.begin(token(), ms(1_000)))
        run.confirm(ms(1_350))
        assertTrue(run.isConfirmed)
        assertTrue(run.hadHeadStart)
        assertEquals("still running", 350, run.savedMs(ms(2_000)))
        run.end(ms(3_200))
        assertEquals(350, run.savedMs(ms(9_999)))
        assertEquals(EarlyRun.Outcome.USED, withTimeout(1_000) { run.awaitOutcome() })
    }

    @Test
    fun aRecognitionThatEndedBeforeTheConfirmationSavesAllOfItsTime() {
        val run = EarlyRun()
        run.begin(token(), ms(1_000))
        run.end(ms(1_200))
        run.confirm(ms(1_350))
        assertEquals(200, run.savedMs(ms(5_000)))
    }

    @Test
    fun aRecognitionThatStartedAfterTheConfirmationSavesNothing() {
        // the recogniser was busy with the utterance before: as without early start
        val run = EarlyRun()
        run.confirm(ms(1_350))
        assertFalse("not begun yet", run.hadHeadStart)
        assertTrue("confirmed, so it runs", run.begin(token(), ms(2_000)))
        assertEquals("too late to save anything", ", early start too late", run.note(ms(2_500)))
        run.end(ms(4_000))
        assertEquals(0, run.savedMs(ms(5_000)))
        assertFalse(run.hadHeadStart)
    }

    @Test
    fun retractedBeforeItBeganItNeverBeginsAndTheCallerZeroesTheAudio() = runBlocking {
        val run = EarlyRun()
        assertTrue("nobody took the audio", run.retract(ms(1_100)))
        assertTrue(run.isDead)
        assertFalse(run.begin(token(), ms(1_200)))
        assertFalse(run.hasBegun)
        assertEquals(EarlyRun.Outcome.DROPPED, withTimeout(1_000) { run.awaitOutcome() })
        assertEquals(0, run.recogniserMs(ms(2_000)))
        assertEquals(0, run.savedMs(ms(2_000)))
    }

    @Test
    fun retractedWhileItRunsAbortsTheRecognitionAndLeavesTheAudioToTheWhisperThread() {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        val token = token(native)
        val run = EarlyRun()
        run.begin(token, ms(1_000))
        assertTrue(token.begin(handle))
        assertFalse("the recogniser still reads the audio", run.retract(ms(1_300)))
        assertEquals(listOf(token.generation), native.aborts.toList())
        assertTrue(token.isCancelled)
        run.end(ms(1_320))
        assertEquals("the recogniser stopped 20 ms after the abort", 20, run.abortMs(ms(9_000)))
        assertEquals("recogniser time spent for nothing", 320, run.recogniserMs(ms(9_000)))
        assertEquals(0, run.savedMs(ms(9_000)))
    }

    @Test
    fun retractedAfterTheRecognitionEndedAbortsNothing() {
        val native = FakeNative()
        val token = token(native)
        val run = EarlyRun()
        run.begin(token, ms(1_000))
        run.end(ms(1_100))
        assertFalse(run.retract(ms(1_300)))
        assertTrue("no abort: the token is no longer the run's", native.aborts.isEmpty())
        assertFalse(token.isCancelled)
    }

    @Test
    fun anAbortThatFindsTheNativeCallOverOrThatStoppedNothingIsNotCounted() {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        // the native call has returned; the recognition is in its last steps (the language model, the filters)
        val late = EarlyRun()
        val lateToken = token(native)
        late.begin(lateToken, ms(1_000))
        assertTrue(lateToken.begin(handle))
        lateToken.end()
        late.retract(ms(1_300))
        assertTrue("nothing native to abort", native.aborts.isEmpty())
        late.end(ms(1_400))
        assertEquals(-1, late.abortMs(ms(2_000)))
        assertEquals(", early start discarded", late.note(ms(2_000)))

        // the abort reached whisper.cpp, which returned its text all the same
        val finished = EarlyRun()
        val token = token(native)
        finished.begin(token, ms(1_000))
        assertTrue(token.begin(handle))
        finished.retract(ms(1_300))
        assertEquals(1, native.aborts.size)
        assertEquals(10, finished.abortMs(ms(1_310)))
        finished.abortStoppedNothing()
        finished.end(ms(1_400))
        assertEquals(-1, finished.abortMs(ms(2_000)))
    }

    @Test
    fun cancellingATokenTellsWhetherANativeCallWasRunning() {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        val idle = token(native)
        assertFalse(idle.cancel())
        val running = token(native)
        assertTrue(running.begin(handle))
        assertTrue(running.cancel())
        assertEquals(listOf(running.generation), native.aborts.toList())
    }

    @Test
    fun theFirstDecisionStands() {
        val run = EarlyRun()
        run.confirm(ms(1))
        assertFalse(run.retract(ms(2)))
        assertTrue(run.isConfirmed)
        val other = EarlyRun()
        other.retract(ms(1))
        other.confirm(ms(2))
        assertTrue(other.isDead)
        assertFalse(other.isConfirmed)
    }

    @Test
    fun aDemotedRunGoesOnForAPreviewUntilItIsStopped() = runBlocking {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        val token = token(native)
        val run = EarlyRun()
        run.begin(token, ms(1_000))
        assertTrue(token.begin(handle))
        run.demote(ms(1_300))
        assertEquals(EarlyRun.Outcome.PREVIEW, withTimeout(1_000) { run.awaitOutcome() })
        assertTrue("nothing is aborted", native.aborts.isEmpty())
        assertFalse(run.isDead)
        assertFalse(run.isOver)
        assertEquals(-1, run.abortMs(ms(1_400)))
        assertEquals(", early start for a preview", run.note(ms(1_400)))

        assertFalse("the recogniser still reads the audio", run.stop(ms(2_000)))
        assertEquals(listOf(token.generation), native.aborts.toList())
        assertTrue(run.wasStopped)
        assertEquals("still stopping", 40, run.abortMs(ms(2_040)))
        assertEquals(", early start for a preview, aborted, stopped 45 ms after the abort", run.note(ms(2_045)))
        run.end(ms(2_060))
        assertEquals("what the abort cost the final text behind it", 60, run.abortMs(ms(9_000)))
        assertTrue(run.isOver)
    }

    @Test
    fun aDemotedRunThatEndedKeepsItsResultAndAStopChangesNothing() {
        val native = FakeNative()
        val run = EarlyRun()
        run.begin(token(native), ms(1_000))
        run.demote(ms(1_300))
        run.end(ms(2_500))
        assertFalse(run.stop(ms(3_000)))
        assertFalse(run.wasStopped)
        assertTrue(native.aborts.isEmpty())
        assertEquals(-1, run.abortMs(ms(4_000)))
    }

    @Test
    fun aDemotedRunStoppedBeforeItBeganNeverBegins() {
        val run = EarlyRun()
        run.demote(ms(1_300))
        assertTrue("nobody took the audio: the caller zeroes it", run.stop(ms(1_500)))
        assertTrue(run.isDead)
        assertTrue(run.isOver)
        assertFalse(run.begin(token(), ms(1_600)))
        assertFalse("stopping twice", run.stop(ms(1_700)))
    }

    @Test
    fun aRetractedRunCannotBeDemotedNorAConfirmedOneStopped() {
        val retracted = EarlyRun()
        retracted.retract(ms(1))
        retracted.demote(ms(2))
        assertTrue(retracted.isDead)
        val confirmed = EarlyRun()
        confirmed.begin(token(), ms(1))
        confirmed.confirm(ms(2))
        assertFalse(confirmed.stop(ms(3)))
        assertFalse(confirmed.isDead)
    }

    @Test
    fun theLogNoteSaysWhatIsKnownWhenItIsWrittenAndRemembersWhetherTheDecisionWasInIt() {
        val pending = EarlyRun()
        pending.begin(token(), ms(1_000))
        assertEquals(", early start not decided yet", pending.note(ms(1_100)))
        assertFalse(pending.wasNoted)

        val used = EarlyRun()
        used.begin(token(), ms(1_000))
        used.confirm(ms(1_352))
        assertEquals(", early start used, 352 ms saved", used.note(ms(3_000)))
        assertTrue(used.wasNoted)

        val native = FakeNative()
        val running = token(native)
        val dropped = EarlyRun()
        dropped.begin(running, ms(1_000))
        assertTrue(running.begin(native.whisperInit("model")))
        dropped.retract(ms(1_100))
        assertEquals(", early start discarded, stopped 100 ms after the abort", dropped.note(ms(1_200)))
        assertTrue(dropped.wasNoted)
    }
}
