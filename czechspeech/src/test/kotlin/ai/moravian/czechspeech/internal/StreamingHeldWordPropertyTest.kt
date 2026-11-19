package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

/**
 * The exactly-once property of the held word ([CutSeam.CONTEXT_AND_HELD_WORD]), over random sequences of events played by
 * hand on the tracker, with the capture thread's and the whisper thread's steps interleaved at random: pieces cut inside
 * speech, cut on a quiet frame, ended by silence or dropped; finals that anchor (the held word replaced), that fall back
 * after a flush, that come out empty or fail; the end of capture or a teardown at any point. Every word has its own token, so
 * the concatenation of the delivered texts shows what was written how many times.
 *
 * Asserted for each sequence that ran to its end: every token of a text a final produced is delivered exactly once and in
 * order (a held word either replaced by its re-decoded copy in the next text or flushed, never both, never neither);
 * every piece that was offered to the final queue has exactly one UtteranceDone, after all its texts (the held word's too);
 * and no audio the seam kept survives: every array is zeroed and none is retained. For a sequence cut off by a teardown:
 * no token twice, no event after the teardown (also when a whisper task absorbs its text after it), and the same for the audio.
 */
class StreamingHeldWordPropertyTest {
    private class Piece(val index: Int, val kind: Kind, val startMs: Long, val endMs: Long) {
        enum class Kind { CUT, QUIET, SILENCE, DROPPED }

        lateinit var utterance: Utterance
        var plan: FinalPlan? = null
        var planned = false
        var absorbed = false
        var done = false
        var offered = false
        var started = false
        var ended = false
        var outcome = Outcome.PLAIN
        var transcribed = false
        var ownWords = 0

        override fun toString() = "piece $index $kind $startMs..$endMs outcome=$outcome words=$ownWords"
    }

    private enum class Outcome { PLAIN, REPLACE, CASCADE, FALLBACK, EMPTY, FAIL }

    private class Case(val seed: Long) {
        val events = mutableListOf<ListenEvent>()
        val released = mutableListOf<FloatArray>()
        val planContexts = mutableListOf<FloatArray>()
        var tornDown = false
        var eventsAfterTeardown = 0
        val rnd = Random(seed)
        val tracker: StreamingTracker
        var nextToken = 1
        val producedTokens = mutableListOf<Int>()      // tokens of the texts handed to absorbFinal, in order
        var trace = StringBuilder()

        init {
            val guard = StreamingGuard(15_000, 384)
            guard.recorded(6_000, 3_000)                 // passes paused: chunk mode, where a word is held
            tracker = StreamingTracker(StreamingConfig(cutSeam = CutSeam.CONTEXT_AND_HELD_WORD), 384, false, guard, null) {
                if (tornDown) eventsAfterTeardown++
                events += it
            }
            tracker.onContextReleased = { released += it }
        }

        fun token(): String = "t${nextToken++}"
    }

    private fun sample(startMs: Long, endMs: Long) = FloatArray(((endMs - startMs) * 16).toInt()) { 0.001f + (it % 991) / 1000f }

    private fun run(seed: Long, cancelAt: Int?): StreamingStats {
        val c = Case(seed)
        val t = c.tracker
        val rnd = c.rnd
        // the pieces of the session
        val n = 1 + rnd.nextInt(6)
        val pieces = ArrayList<Piece>()
        var at = 1_000L
        for (i in 0 until n) {
            val kind = when (rnd.nextInt(10)) {
                in 0..5 -> Piece.Kind.CUT
                6 -> Piece.Kind.QUIET
                7 -> Piece.Kind.SILENCE
                8 -> Piece.Kind.DROPPED
                else -> Piece.Kind.CUT
            }
            val length = if (kind == Piece.Kind.DROPPED) 300L else 3_000L + 1_000L * rnd.nextInt(3)
            val p = Piece(i, kind, at, at + length)
            if (kind != Piece.Kind.DROPPED) p.utterance = Utterance(sample(at, at + length), at, at + length, audioStartMs = at, quietCut = kind == Piece.Kind.QUIET)
            pieces += p
            // a piece after a cut starts where the cut was; after anything else, after a gap
            at = if (kind == Piece.Kind.CUT || kind == Piece.Kind.QUIET) at + length else at + length + 1_000
        }
        for (p in pieces) {
            p.outcome = Outcome.values()[rnd.nextInt(Outcome.values().size)]
            p.ownWords = if (rnd.nextInt(8) == 0) 0 else 1 + rnd.nextInt(7)
        }
        val finals = pieces.filter { it.kind != Piece.Kind.DROPPED }
        var capturePosition = 0          // capture steps done: 3 per piece (start, end/offer) and the end of capture
        val stopAfter = if (rnd.nextInt(4) == 0) rnd.nextInt(n * 2 + 1) else -1
        var stopped = false
        var captureEnded = false
        var captureStep = 0
        var finalIndex = 0               // the next final to process on the whisper thread
        var step = 0
        // capture steps: for piece i, 2i = its frame (start), 2i + 1 = its end and offer; then the end of capture
        val captureSteps = pieces.size * 2 + 1

        // The position of the session clock: it advances with the pieces, and now and then jumps far ahead (a slow phone), so that the
        // lag rule of the continuation's final plan flushes the held word (never goes back).
        var lastPosition = 0L
        fun frameOf(p: Piece) {
            val jump = if (rnd.nextInt(5) == 0) 13_000L else 0L
            lastPosition = maxOf(lastPosition, p.startMs + 32 + jump)
            t.onFrame(p.startMs, p.startMs, lastPosition) { FloatArray(0) }
            p.started = true
        }

        fun endOf(p: Piece) {
            when (p.kind) {
                Piece.Kind.DROPPED -> {}                // no utterance: the next piece's frame replaces it
                Piece.Kind.SILENCE -> {
                    t.pieceEnded()
                    t.finalComing(p.utterance)
                }
                else -> t.finalComing(p.utterance)
            }
            p.ended = true
            p.offered = p.kind != Piece.Kind.DROPPED
        }

        fun canCapture(): Boolean = captureStep < captureSteps

        fun doCapture() {
            if (captureStep == captureSteps - 1) {
                t.end()
                captureEnded = true
            } else {
                val p = pieces[captureStep / 2]
                if (captureStep % 2 == 0) frameOf(p) else endOf(p)
            }
            captureStep++
        }

        // whisper steps for the final under way: 0 = plan, 1 = absorb, 2 = done
        var whisperStep = 0
        fun currentFinal(): Piece? = finals.getOrNull(finalIndex)
        fun canWhisper(): Boolean {
            val p = currentFinal() ?: return false
            return p.offered
        }

        fun ownText(p: Piece, withHeld: String?): String {
            val words = ArrayList<String>()
            withHeld?.let { words += it }
            repeat(p.ownWords) {
                val tok = c.token()
                words += tok
            }
            c.producedTokens += words.map { it.drop(1).toInt() }
            return words.joinToString(" ")
        }

        fun doWhisper() {
            val p = currentFinal()!!
            when (whisperStep) {
                0 -> {
                    p.plan = t.finalPlan(p.utterance)
                    p.plan!!.context?.let { c.planContexts += it }
                    p.planned = true
                    whisperStep = 1
                }
                1 -> {
                    val plan = p.plan!!
                    val held = plan.seam?.heldWord
                    var text = ""
                    var replaced = false
                    var runSamples = 0
                    when (p.outcome) {
                        Outcome.REPLACE -> if (held != null) {
                            text = ownText(p, held)
                            replaced = true
                        } else {
                            text = ownText(p, null)
                        }
                        Outcome.CASCADE, Outcome.FALLBACK -> {
                            if (held != null) t.flushHeld("no anchor")
                            text = ownText(p, null)
                        }
                        Outcome.EMPTY -> {
                            if (held != null) t.flushHeld("empty")
                        }
                        Outcome.FAIL -> {}
                        Outcome.PLAIN -> text = ownText(p, null)
                    }
                    if (p.outcome != Outcome.FAIL) {
                        // the engine zeroes the context right after the run that used it
                        t.releasePlanContext(plan)
                        val chunk = t.absorbFinal(plan, text, 0, TEST_LANGUAGE, runSamples, replaced)
                        // the engine sends the chunk as a Transcript, right after absorbing it
                        if (chunk != null) {
                            if (c.tornDown) c.eventsAfterTeardown++
                            c.events += SpeechEvent.Transcript(Transcript(chunk.text, chunk.startMs, chunk.endMs, 0, chunk.language, null))
                            p.transcribed = true
                        }
                    }
                    p.absorbed = true
                    whisperStep = 2
                }
                else -> {
                    val plan = p.plan!!
                    t.releasePlanContext(plan)
                    t.finalDone(p.utterance, p.transcribed)
                    p.done = true
                    whisperStep = 0
                    finalIndex++
                }
            }
        }

        var steps = 0
        val limit = cancelAt ?: Int.MAX_VALUE
        while (steps < limit) {
            val cap = canCapture()
            val whi = canWhisper()
            if (!cap && !whi) break
            // a whisper step that needs the capture side first (plan(i) needs finalComing(i), which canWhisper checks)
            val takeCapture = cap && (!whi || rnd.nextBoolean())
            if (stopAfter >= 0 && !stopped && captureStep >= stopAfter) {
                t.stop()
                stopped = true
            }
            if (takeCapture) doCapture() else doWhisper()
            steps++
        }
        val teardownCut = cancelAt != null && (canCapture() || canWhisper())
        if (!teardownCut) {
            // the end of the session: the queue is drained, then teardown
            t.finalsOver()
        } else {
            c.tornDown = true
        }
        // what the engine does for a plan it was working on when the session was cancelled: releases its context
        for (p in pieces) p.plan?.let { t.releasePlanContext(it) }
        t.discardRequests()
        // A whisper task that was past its cancellation check goes on to absorb its text after the teardown: nothing is kept
        // from it (the copy of the audio would never be zeroed) and nothing is sent.
        if (teardownCut) {
            for (p in pieces) {
                val plan = p.plan
                if (plan != null && !p.absorbed) t.absorbFinal(plan, ownText(p, null), 0, TEST_LANGUAGE, 0, false)
            }
        }
        val where = "seed $seed cancelAt $cancelAt: ${pieces.joinToString("; ")}"

        val delivered = c.events.filterIsInstance<SpeechEvent.Transcript>()
            .flatMap { e -> e.transcript.text.split(' ').filter { it.isNotEmpty() }.map { it.drop(1).toInt() } }
        val doneOf = c.events.withIndex().filter { it.value is SpeechEvent.UtteranceDone }
        if (!teardownCut) {
            assertEquals("$where: every token once, in order: $delivered", c.producedTokens.distinct().sorted(), delivered)
            for (p in finals) {
                val dones = doneOf.filter { (it.value as SpeechEvent.UtteranceDone).startMs == p.startMs }
                assertEquals("$where: exactly one UtteranceDone for piece ${p.index}", 1, dones.size)
                val lastText = c.events.withIndex().lastOrNull { (it.value as? SpeechEvent.Transcript)?.transcript?.startMs == p.startMs }
                if (lastText != null) assertTrue("$where: the UtteranceDone of piece ${p.index} comes after its texts", dones[0].index > lastText.index)
            }
        } else {
            assertEquals("$where: no token twice: $delivered", delivered.distinct(), delivered)
            val dones = doneOf.map { (it.value as SpeechEvent.UtteranceDone).startMs }
            assertEquals("$where: no UtteranceDone twice", dones.distinct(), dones)
        }
        assertTrue("$where: every array the seam kept was zeroed", c.released.all { a -> a.all { it == 0f } })
        assertTrue("$where: every context of a plan was zeroed", c.planContexts.all { a -> a.all { it == 0f } })
        assertEquals("$where: no audio retained", 0, t.retainedContextArrays())
        assertTrue("$where: no held word left behind after the end", teardownCut || !t.heldWordPending())
        if (teardownCut) assertEquals("$where: no event after the teardown", 0, c.eventsAfterTeardown)
        return t.stats()
    }

    @Test
    fun everyHeldWordIsDeliveredExactlyOnceWhateverTheOrderOfEventsAndEveryBufferIsZeroedAtTheEnd() {
        var replaced = 0
        var kept = 0
        var lagFlushes = 0
        for (seed in 1L..600L) {
            val stats = run(seed, cancelAt = null)
            replaced += stats.heldReplaced
            kept += stats.heldKept
            lagFlushes += stats.heldKeptBy["lag"] ?: 0
        }
        // not vacuous: with holding switched off nothing is ever held, and every seed would pass
        assertTrue("words were held and replaced: $replaced", replaced > 0)
        assertTrue("words were held and sent as they were: $kept", kept > 0)
        assertTrue("the lag rule flushed held words: $lagFlushes", lagFlushes > 0)
    }

    @Test
    fun aTeardownAtAnyPointDeliversNothingTwiceAndLeavesNoAudio() {
        for (seed in 1L..400L) run(seed, cancelAt = Random(seed).nextInt(30))
    }
}
