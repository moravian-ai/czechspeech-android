package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.StreamingConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The seam of a cut inside speech in pass mode (a fast phone: the passes run, so words are sent while the speech goes on): the
 * words the passes committed are among the words the piece delivered, and a continuation with no chunk of its own at its final
 * still gets the context. Played by hand, with the passes answered as the whisper thread would.
 */
class StreamingTrackerPassSeamTest {
    private val events = mutableListOf<ListenEvent>()
    private val released = mutableListOf<FloatArray>()
    private val token = AbortToken(FakeNative())
    private var positionMs = 0L

    private fun tracker() = StreamingTracker(StreamingConfig(), 384, false) { events += it }.also { t ->
        t.onContextReleased = { released += it }
    }

    private fun frame(t: StreamingTracker, pieceMs: Long, audioStartMs: Long): StreamingTracker.Step {
        positionMs += 32
        val position = positionMs
        return t.onFrame(pieceMs, audioStartMs, position) { from -> FloatArray(((position - from) * 16).toInt()) { 0.3f } }
    }

    /** Frames until the tracker asks for a pass; the whisper thread takes it and answers with [text]. */
    private fun answeredPass(t: StreamingTracker, pieceMs: Long, audioStartMs: Long, text: String) {
        val limit = positionMs + 5_000
        while (positionMs < limit) {
            val step = frame(t, pieceMs, audioStartMs)
            if (step is StreamingTracker.Step.Pass) {
                val pass = t.requests.tryReceive().getOrNull()
                if (pass == null) {
                    fail("no pass in the queue")
                    return
                }
                assertTrue(pass.attach(token))
                t.languageFor(pass)
                val absorbed = t.absorb(pass, PassResult(text, 500))
                if (absorbed != null) t.publish(absorbed, null, 0)
                return
            }
        }
        fail("no pass until $limit ms")
    }

    private fun words(n: Int) = (1..n).joinToString(" ") { "w$it" }

    @Test
    fun theWordsThePassesCommittedAreAmongTheDeliveredWordsOfTheSeamAndAContinuationWithoutChunksGetsTheContext() {
        val t = tracker()
        answeredPass(t, 480, 160, words(4))
        answeredPass(t, 480, 160, words(6))
        assertTrue("the passes committed words while the speech went on", t.stats().wordsWhileSpeaking > 0)
        // the piece is cut inside speech: its final writes the rest
        val first = Utterance(FloatArray(((3_000 - 160) * 16)) { 0.3f }, 480, 3_000, audioStartMs = 160)
        t.finalComing(first)
        val plan1 = t.finalPlan(first)
        assertTrue(plan1.hadChunks)
        t.absorbFinal(plan1, words(8), 0)
        t.releasePlanContext(plan1)
        t.finalDone(first, true)
        // the continuation has no chunk when its final comes
        positionMs = 3_000
        frame(t, 3_000, 3_000)
        val second = Utterance(FloatArray(((4_000 - 3_000) * 16)) { 0.3f }, 3_000, 4_000, audioStartMs = 3_000)
        t.finalComing(second)
        val plan2 = t.finalPlan(second)
        assertNotNull("the context of the piece before", plan2.context)
        assertEquals("the words of the passes and of the final, in order, once each", (1..8).map { "w$it" }, plan2.seam!!.delivered)
        assertEquals(false, plan2.hadChunks)
        t.releasePlanContext(plan2)
    }
}
