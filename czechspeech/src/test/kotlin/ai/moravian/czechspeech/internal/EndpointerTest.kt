package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Endpointer with synthetic probability sequences. Each frame's samples are filled with its frame index, so the
 * audio of an utterance shows exactly which frames it contains.
 */
class EndpointerTest {
    private val config = CzechSpeechConfig()
    private val frameMs = FRAME_MS.toInt()

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

        fun feedMs(prob: Float, ms: Int) = feed(prob, ms / FRAME_MS.toInt())

        val utterances get() = events.filterIsInstance<EndpointerEvent.UtteranceReady>().map { it.utterance }
        val started get() = events.filterIsInstance<EndpointerEvent.SpeechStarted>()
        val ended get() = events.filterIsInstance<EndpointerEvent.SpeechEnded>()

        /** Frame indices contained in an utterance, from the sample values. */
        fun frames(u: Utterance): List<Int> = (0 until u.samples.size / FRAME_SAMPLES).map { u.samples[it * FRAME_SAMPLES].toInt() }
    }

    @Test
    fun startAndEnd() {
        val run = Run(config)
        run.feed(0.02f, 50)             // 1.6 s silence, frames 0..49
        run.feed(0.9f, 40)              // speech, frames 50..89
        run.feed(0.02f, 30)             // silence

        assertEquals(listOf(EndpointerEvent.SpeechStarted(50L * frameMs)), run.started)
        assertEquals(1, run.ended.size)
        assertEquals(90L * frameMs, run.ended[0].atMs)
        assertTrue(run.ended[0].confirmed)

        val u = run.utterances.single()
        assertEquals(50L * frameMs, u.startMs)
        assertEquals(90L * frameMs, u.endMs)
        // pre-roll (10 frames = 320 ms >= 300 ms) + 40 speech + 19 silence frames (608 ms >= 600 ms)
        assertEquals((40..108).toList(), run.frames(u))
        // order: started, ended, utterance
        assertTrue(run.events[0] is EndpointerEvent.SpeechStarted)
        assertTrue(run.events[1] is EndpointerEvent.SpeechEnded)
        assertTrue(run.events[2] is EndpointerEvent.UtteranceReady)
    }

    @Test
    fun speechContinuesWhileAboveNegThreshold() {
        val run = Run(config)
        run.feed(0.9f, 20)
        run.feed(0.15f, 40)             // between negThreshold (0.10) and threshold (0.25): still speech
        run.feed(0.9f, 20)
        run.feed(0.02f, 25)
        assertEquals(1, run.utterances.size)
        assertEquals(80L * frameMs, run.utterances[0].endMs)
    }

    @Test
    fun shortSilenceDoesNotEndTheUtterance() {
        val run = Run(config)
        run.feed(0.9f, 20)
        run.feed(0.02f, 18)             // 576 ms < 600 ms
        run.feed(0.9f, 20)
        run.feed(0.02f, 19)
        assertEquals(1, run.utterances.size)
        assertEquals(1, run.ended.size)
    }

    @Test
    fun shortBlipsAreDiscarded() {
        val run = Run(config)
        run.feed(0.02f, 20)
        run.feed(0.95f, 3)              // 96 ms click
        run.feed(0.02f, 30)
        run.feed(0.6f, 7)               // 224 ms < 250 ms of speech frames
        run.feed(0.02f, 30)
        run.feed(0.2f, 100)             // never reaches threshold
        run.feed(0.02f, 30)
        assertTrue(run.started.isEmpty())
        assertTrue(run.utterances.isEmpty())
        assertTrue(run.ended.all { !it.confirmed })
        assertEquals(2, run.ended.size) // the two runs above threshold end (VAD reset), unreported
    }

    @Test
    fun eightSpeechFramesAreEnough() {
        val run = Run(config)
        run.feed(0.6f, 8)               // 256 ms >= 250 ms
        run.feed(0.02f, 20)
        assertEquals(1, run.utterances.size)
    }

    @Test
    fun preRollCoversTheAudioBeforeTheTrigger() {
        val run = Run(config.copy(preRollMs = 300))
        run.feed(0.02f, 30)
        run.feed(0.9f, 20)              // trigger at frame 30
        run.feed(0.02f, 19)
        val u = run.utterances.single()
        val frames = run.frames(u)
        assertEquals(20, frames.first())  // 10 frames of pre-roll: 20..29
        assertEquals(30L * frameMs, u.startMs)
    }

    @Test
    fun preRollIsShorterAtTheStartOfTheStream() {
        val run = Run(config)
        run.feed(0.02f, 3)
        run.feed(0.9f, 20)
        run.feed(0.02f, 19)
        assertEquals(0, run.frames(run.utterances.single()).first())
    }

    @Test
    fun noPreRoll() {
        val run = Run(config.copy(preRollMs = 0))
        run.feed(0.02f, 30)
        run.feed(0.9f, 20)
        run.feed(0.02f, 19)
        assertEquals(30, run.frames(run.utterances.single()).first())
    }

    @Test
    fun twoUtterancesSeparatedByTwoSeconds() {
        val run = Run(config)
        run.feedMs(0.02f, 1_000)        // frames 0..30
        run.feedMs(0.9f, 2_000)         // 31..92
        run.feedMs(0.02f, 2_000)        // 93..154
        run.feedMs(0.9f, 2_000)         // 155..216
        run.feedMs(0.02f, 1_000)

        assertEquals(2, run.started.size)
        assertEquals(2, run.ended.size)
        val (a, b) = run.utterances
        assertEquals(31L * frameMs, a.startMs)
        assertEquals(93L * frameMs, a.endMs)
        assertEquals(155L * frameMs, b.startMs)
        assertEquals(217L * frameMs, b.endMs)
        assertEquals(145, run.frames(b).first())     // pre-roll from the silence in between
        // events alternate: started, ended, utterance, started, ended, utterance
        assertEquals(
            listOf("S", "E", "U", "S", "E", "U"),
            run.events.map {
                when (it) {
                    is EndpointerEvent.SpeechStarted -> "S"
                    is EndpointerEvent.SpeechEnded -> "E"
                    is EndpointerEvent.UtteranceReady -> "U"
                }
            },
        )
    }

    @Test
    fun longSpeechIsCutAtTheQuietestFrameBeforeTheCap() {
        val run = Run(config)
        val maxFrames = config.maxUtteranceMs / frameMs      // 875 frames = 28.0 s
        run.feed(0.9f, 843)
        run.feed(0.3f, 1)                                     // frame 843: the quietest within the last 2 s
        run.feed(0.9f, 1250 - 844)                            // speech until 40 s
        run.feed(0.02f, 19)

        assertEquals(1, run.started.size)
        assertEquals(1, run.ended.size)
        val (first, second) = run.utterances
        assertEquals(844 * FRAME_SAMPLES, first.samples.size)
        assertEquals(0L, first.startMs)
        assertEquals(844L * frameMs, first.endMs)
        assertEquals((0..843).toList(), run.frames(first))
        assertEquals(844L * frameMs, second.startMs)
        assertEquals(1250L * frameMs, second.endMs)
        assertEquals(844, run.frames(second).first())
        assertTrue(run.utterances.all { it.samples.size <= maxFrames * FRAME_SAMPLES })
    }

    @Test
    fun capWithoutAQuietFrameCutsAtTheStartOfTheSearchWindow() {
        val run = Run(config)
        run.feed(0.9f, 2000)            // 64 s of uniform speech
        run.feed(0.02f, 19)
        val utterances = run.utterances
        assertTrue(utterances.size >= 3)
        assertTrue(utterances.all { it.samples.size <= 875 * FRAME_SAMPLES })
        // pieces are contiguous and cover the whole run
        for (i in 1 until utterances.size) assertEquals(utterances[i - 1].endMs, utterances[i].startMs)
        assertEquals(2000L * frameMs, utterances.last().endMs)
        assertEquals(1, run.started.size)
    }

    @Test
    fun finishClosesAnOpenUtterance() {
        val run = Run(config)
        run.feed(0.02f, 10)
        run.feed(0.9f, 30)
        run.endpointer.finish()
        assertEquals(1, run.utterances.size)
        assertEquals(40L * frameMs, run.utterances[0].endMs)
        assertEquals(1, run.ended.size)
    }

    @Test
    fun clearDropsEverything() {
        val run = Run(config)
        run.feed(0.9f, 30)
        run.endpointer.clear()
        run.endpointer.finish()
        assertTrue(run.utterances.isEmpty())
    }

    @Test
    fun emittedUtterancesAreCopies() {
        val run = Run(config)
        run.feed(0.9f, 20)
        run.feed(0.02f, 19)
        val u = run.utterances.single()
        run.feed(0.9f, 20)              // reuse of the internal buffer must not touch the emitted copy
        run.feed(0.02f, 19)
        assertEquals(0, run.frames(u).first())
    }
}
