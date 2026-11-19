package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.EarlyStart
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The endpointer with [CzechSpeechConfig.earlyStart]: the utterance is handed out once the pause that may end it has
 * lasted 256 ms (8 frames), and then either confirmed as it is or retracted. Each frame's samples are its index + 1,
 * so an utterance shows which frames it holds, and zeros are the silence that had not been recorded yet.
 */
class EndpointerEarlyStartTest {
    private val early = CzechSpeechConfig(earlyStart = EarlyStart())
    private val frameMs = FRAME_MS.toInt()

    private class Run(config: CzechSpeechConfig) {
        val events = mutableListOf<EndpointerEvent>()
        val endpointer = Endpointer(config) { events += it }
        private var index = 0
        private val frame = FloatArray(FRAME_SAMPLES)

        fun feed(prob: Float, frames: Int) {
            repeat(frames) {
                frame.fill((index + 1).toFloat())
                endpointer.accept(frame, prob)
                index++
            }
        }

        val ready get() = events.filterIsInstance<EndpointerEvent.UtteranceReady>().map { it.utterance }
        val handedOut get() = events.filterIsInstance<EndpointerEvent.UtteranceEarly>().map { it.utterance }
        val retracted get() = events.filterIsInstance<EndpointerEvent.EarlyRetracted>().map { it.utterance }

        /** The frame index of each frame of [samples], or -1 for a frame of zeros. */
        fun frames(samples: FloatArray): List<Int> =
            (0 until samples.size / FRAME_SAMPLES).map { samples[it * FRAME_SAMPLES].toInt() - 1 }

        fun kinds(): List<String> = events.map { it::class.java.simpleName }
    }

    @Test
    fun offByDefaultAndThenNoUtteranceIsHandedOutEarly() {
        assertNull(CzechSpeechConfig().earlyStart)
        val run = Run(CzechSpeechConfig())
        run.feed(0.02f, 20)
        run.feed(0.9f, 30)
        run.feed(0.02f, 19)
        assertEquals(listOf("SpeechStarted", "SpeechEnded", "UtteranceReady"), run.kinds())
        assertNull(run.ready.single().early)
        assertEquals("the microphone's own audio to the end", (10..68).toList(), run.frames(run.ready.single().samples))
    }

    @Test
    fun theDefaultIs256MsOfSilence() {
        assertEquals(256, EarlyStart().silenceMs)
    }

    @Test
    fun theUtteranceIsHandedOutAfterTheEarlySilenceWithZerosForTheRestOfThePause() {
        val run = Run(early)
        run.feed(0.02f, 20)             // frames 0..19
        run.feed(0.9f, 30)              // speech, frames 20..49
        run.feed(0.02f, 7)              // 224 ms of silence: not yet
        assertTrue(run.handedOut.isEmpty())
        run.feed(0.02f, 1)              // frame 57: 256 ms
        val utterance = run.handedOut.single()
        assertNotNull(utterance.early)
        assertEquals(20L * frameMs, utterance.startMs)
        assertEquals(50L * frameMs, utterance.endMs)
        // pre-roll from frame 10, speech, the 8 frames of the pause so far, zeros for the 11 still to come
        assertEquals((10..57).toList() + List(11) { -1 }, run.frames(utterance.samples))
        assertTrue("nothing else yet", run.ready.isEmpty() && run.retracted.isEmpty())
    }

    @Test
    fun whenThePauseBecomesTheEndTheSameUtteranceIsTheFinalOne() {
        val run = Run(early)
        run.feed(0.02f, 20)
        run.feed(0.9f, 30)
        run.feed(0.02f, 8)
        val handedOut = run.handedOut.single()
        val before = handedOut.samples.copyOf()
        run.feed(0.02f, 10)             // 18 frames of silence: still open
        assertTrue(run.ready.isEmpty())
        run.feed(0.02f, 1)              // 19 frames: the end
        assertSame("the very utterance that was handed out", handedOut, run.ready.single())
        assertTrue("and untouched", before.contentEquals(handedOut.samples))
        assertEquals(listOf("SpeechStarted", "UtteranceEarly", "SpeechEnded", "UtteranceReady"), run.kinds())
        assertTrue(run.retracted.isEmpty())
    }

    @Test
    fun theFinalUtteranceIsAsLongAsWithoutEarlyStartWithTheSameAudioUpToTheEarlyPoint() {
        val with = Run(early)
        val without = Run(CzechSpeechConfig())
        for (run in listOf(with, without)) {
            run.feed(0.02f, 20)
            run.feed(0.9f, 30)
            run.feed(0.02f, 19)
        }
        val a = with.ready.single()
        val b = without.ready.single()
        assertEquals(b.samples.size, a.samples.size)
        assertEquals(b.startMs, a.startMs)
        assertEquals(b.endMs, a.endMs)
        val audible = 48 * FRAME_SAMPLES        // frames 10..57
        assertTrue(a.samples.copyOf(audible).contentEquals(b.samples.copyOf(audible)))
        assertTrue("zeros for the rest", a.samples.drop(audible).all { it == 0f })
        assertTrue("where the microphone's audio would be", b.samples.drop(audible).none { it == 0f })
    }

    @Test
    fun whenTheSpeechGoesOnTheUtteranceIsRetractedAndTheRealOneHoldsEverything() {
        val run = Run(early)
        run.feed(0.02f, 20)
        run.feed(0.9f, 30)              // frames 20..49
        run.feed(0.02f, 12)             // a pause of 384 ms, frames 50..61
        val first = run.handedOut.single()
        run.feed(0.9f, 1)               // frame 62: the speech goes on
        assertSame(first, run.retracted.single())
        assertTrue(run.ready.isEmpty())
        run.feed(0.9f, 9)               // frames 63..71
        run.feed(0.02f, 8)              // the next pause is handed out again, with all the audio
        val second = run.handedOut[1]
        assertNotSame(first, second)
        assertEquals((10..79).toList() + List(11) { -1 }, run.frames(second.samples))
        assertEquals(72L * frameMs, second.endMs)
        run.feed(0.02f, 11)
        assertSame(second, run.ready.single())
        assertEquals(
            listOf("SpeechStarted", "UtteranceEarly", "EarlyRetracted", "UtteranceEarly", "SpeechEnded", "UtteranceReady"),
            run.kinds(),
        )
    }

    @Test
    fun aFrameBetweenTheThresholdsCountsAsSpeechAndRetracts() {
        val run = Run(early)
        run.feed(0.9f, 30)
        run.feed(0.02f, 8)
        run.feed(0.15f, 1)              // at or above negThreshold
        assertEquals(1, run.retracted.size)
    }

    @Test
    fun aRunTooShortToBeAnUtteranceIsNeverHandedOut() {
        val run = Run(early)
        run.feed(0.9f, 5)               // 160 ms: under minSpeechMs
        run.feed(0.02f, 19)
        assertTrue(run.handedOut.isEmpty())
        assertTrue(run.ready.isEmpty())
    }

    @Test
    fun atTheEndOfTheAudioTheUtteranceHandedOutIsTheFinalOne() {
        // stop() or the end of the stream in the pause: the utterance ends as if silence followed, zeros included
        val run = Run(early)
        run.feed(0.9f, 30)
        run.feed(0.02f, 10)
        val handedOut = run.handedOut.single()
        run.endpointer.finish()
        assertSame(handedOut, run.ready.single())
        assertEquals((30 + 19) * FRAME_SAMPLES, handedOut.samples.size)
    }

    @Test
    fun atTheEndOfTheAudioBeforeTheEarlyPointTheUtteranceIsAsWithoutEarlyStart() {
        val run = Run(early)
        run.feed(0.9f, 30)
        run.feed(0.02f, 5)
        run.endpointer.finish()
        assertTrue(run.handedOut.isEmpty())
        assertEquals((0..34).toList(), run.frames(run.ready.single().samples))
        assertNull(run.ready.single().early)
    }

    @Test
    fun noUtteranceIsHandedOutWhenTheLengthCapWouldCutItFirst() {
        // the cap is 875 frames: a pause that starts at frame 860 cannot reach 19 frames within it
        val run = Run(early)
        run.feed(0.9f, 860)
        run.feed(0.02f, 8)
        assertTrue(run.handedOut.isEmpty())
        run.feed(0.02f, 11)
        assertTrue(run.retracted.isEmpty())
        assertTrue("the cap cut it, as without early start", run.ready.isNotEmpty())
        assertTrue(run.ready.all { it.early == null })
    }

    @Test
    fun anUtteranceThatJustFitsUnderTheCapIsHandedOut() {
        val run = Run(early)
        run.feed(0.9f, 856)             // with 19 frames of silence: 875, the cap exactly
        run.feed(0.02f, 8)
        val handedOut = run.handedOut.single()
        assertEquals(875 * FRAME_SAMPLES, handedOut.samples.size)
        run.feed(0.02f, 11)
        assertSame(handedOut, run.ready.single())
    }

    @Test
    fun clearForgetsTheUtteranceHandedOutWhichItsReceiverOwns() {
        val run = Run(early)
        run.feed(0.9f, 30)
        run.feed(0.02f, 8)
        val handedOut = run.handedOut.single()
        run.endpointer.clear()
        run.endpointer.finish()
        assertTrue(run.ready.isEmpty())
        assertTrue(run.retracted.isEmpty())
        assertTrue("not zeroed by the endpointer: the receiver does that", handedOut.samples.any { it != 0f })
    }

    @Test
    fun theTrailingSilenceStillBecomesThePreRollOfAQuickNextUtterance() {
        val run = Run(early)
        run.feed(0.9f, 30)
        run.feed(0.02f, 19)             // frames 30..48
        run.feed(0.9f, 10)              // frames 49..58
        run.feed(0.02f, 19)
        assertEquals("the microphone's silence, not zeros", (39..48).toList(), run.frames(run.ready[1].samples).take(10))
    }

    @Test
    fun aSilenceThatRoundsUpToTheEndOfTheUtteranceIsRefused() {
        // 580 ms is 19 frames, as is minSilenceMs 600: nothing could ever start early, so the configuration says so
        val e = assertThrows(IllegalArgumentException::class.java) {
            CzechSpeechConfig(earlyStart = EarlyStart(silenceMs = 580))
        }
        assertTrue(e.message, e.message!!.contains("32 ms frames"))
        CzechSpeechConfig(earlyStart = EarlyStart(silenceMs = 576))     // 18 frames: one frame early
    }

    @Test
    fun theSilenceIsConfigurableAndCheckedAgainstTheEndOfTheUtterance() {
        val run = Run(CzechSpeechConfig(earlyStart = EarlyStart(silenceMs = 400)))      // 13 frames
        run.feed(0.9f, 30)
        run.feed(0.02f, 12)
        assertTrue(run.handedOut.isEmpty())
        run.feed(0.02f, 1)
        assertEquals(1, run.handedOut.size)

        assertThrows(IllegalArgumentException::class.java) { EarlyStart(silenceMs = 64) }
        assertThrows(IllegalArgumentException::class.java) { CzechSpeechConfig(earlyStart = EarlyStart(silenceMs = 600)) }
        assertThrows(IllegalArgumentException::class.java) {
            CzechSpeechConfig(minSilenceMs = 200, earlyStart = EarlyStart())
        }
        CzechSpeechConfig(minSilenceMs = 300, earlyStart = EarlyStart())
    }
}
