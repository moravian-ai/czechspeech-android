package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.EarlyStart
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * [PreviewConfig.fromPauses] with [CzechSpeechConfig.earlyStart]: a recognition started in a pause that the speech then
 * ran over is not aborted; its text comes as a preview, and it gives way only when its utterance ends first.
 *
 * The audio is that of [EarlyStartSessionTest]: one utterance with a pause of 0.4 s inside, speech in frames 15..46
 * and 59..90; the pause is handed out early with frame 54 and run over with frame 59, the pause after it is handed out
 * with frame 98 and ends the utterance with frame 109. onFrame(n) runs when the frames up to n-2 have been processed.
 */
class PausePreviewSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val on = CzechSpeechConfig(earlyStart = EarlyStart())
    private val startMs = 480L
    private val endMs = 91 * FRAME_MS

    private fun quiet(seconds: Double, seed: Long) = Signals.whiteNoise(seconds, amplitude = 0.01f, seed = seed)

    private fun withPause() =
        Signals.concat(quiet(0.5, 1), Signals.tone(1.0), quiet(0.4, 2), Signals.tone(1.0), quiet(1.5, 3))

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val logs: MutableList<String> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }

    private fun seen(predicate: (ListenEvent) -> Boolean): Boolean = synchronized(events) { events.any(predicate) }

    private fun waitOnCapture(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (!condition()) {
            if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for $what")
            Thread.sleep(1)
        }
    }

    private fun listen(engine: SpeechEngine, previews: PreviewConfig = PreviewConfig()) {
        logs.clear()
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            runBlocking { withContext(Dispatchers.Default) { engine.listenWithPreviews(previews).collect { events += it } } }
        } finally {
            CsLog.sink = sink
        }
    }

    private class Awaited : SpeechEngine.Hooks {
        val awaited = AtomicInteger()
        override fun earlyStartAwaited() {
            awaited.incrementAndGet()
        }
    }

    private fun assertAllZero(arrays: List<FloatArray>) {
        assertTrue("whisper was called", arrays.isNotEmpty())
        for (a in arrays) assertTrue("audio buffer zeroed after use", a.all { it == 0f })
    }

    @Test
    fun onByDefault() {
        assertTrue(PreviewConfig().fromPauses)
    }

    @Test
    fun aRecognitionThatEndedBeforeTheSpeechWentOnComesAsAPreview() {
        val native = FakeNative().apply { texts += listOf("dobrý", "dobrý den") }
        val hooks = Awaited()
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition's end") { hooks.awaited.get() == 1 }
            // the speech went on with frame 59: the preview comes before the utterance ends
            if (n == 64) waitOnCapture("the preview") { seen { it is SpeechPreview } }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        listen(engine)

        val transcript = transcripts.single()
        assertEquals(
            listOf(
                SpeechEvent.SpeechStarted(startMs),
                SpeechPreview(startMs, "dobrý"),
                SpeechEvent.SpeechEnded(endMs),
                SpeechEvent.Transcript(transcript),
            ),
            events.toList(),
        )
        assertEquals("dobrý den", transcript.text)
        assertEquals("the recognition of the pause, then the utterance's", 2, native.transcribeCalls.get())
        assertEquals("both by the call of the final text", 2, native.greedyCalls.get())
        assertEquals(0, native.previewCalls.get())
        assertTrue("nothing was aborted", native.aborts.isEmpty())
        assertEquals(1, engine.earlyStartsShown)
        assertEquals(0, engine.earlyStartsDiscarded)
        assertEquals(0, engine.earlyStartsAborted)
        assertEquals(1, engine.earlyStartsUsed + engine.earlyStartsTooLate)
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertAllZero(native.passedToWhisper)
        assertTrue(logs.joinToString("\n"), logs.any { it == "utterance: early start shown as a preview" })
        assertTrue(
            logs.joinToString("\n"),
            logs.any {
                Regex("""^early start: \d used, \d+ ms saved, \d too late; 1 shown as a preview; 0 discarded, 0 aborted, 0 ms of recogniser time$""")
                    .matches(it)
            },
        )
        assertTrue("never the text", logs.none { "dobrý" in it })
    }

    @Test
    fun aRecognitionStillRunningWhenTheSpeechGoesOnIsNotAbortedAndComesAsAPreviewWhenItEnds() {
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            texts += listOf("dobrý", "dobrý den")
            onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) { "never released" } }
        }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 64) {          // the speech went on with frame 59
                check(native.aborts.isEmpty()) { "aborted when the speech went on" }
                release.countDown()
                waitOnCapture("the preview") { seen { it is SpeechPreview } }
            }
        })
        val engine = fakeEngine(native, config = on) { source }
        listen(engine)
        assertEquals(listOf(SpeechPreview(startMs, "dobrý")), events.filterIsInstance<SpeechPreview>())
        assertEquals("dobrý den", transcripts.single().text)
        assertTrue(native.aborts.isEmpty())
        assertEquals(1, engine.earlyStartsShown)
        assertTrue(logs.any { it.startsWith("utterance: ") && it.endsWith(", transcript kept, early start for a preview") })
    }

    @Test
    fun theNextPauseAbortsItAndStartsItsOwnRecognitionAtOnceWhichSavesTimeWhenTheUtteranceEndsThere() {
        // The recognition of the first pause runs until it is aborted. Final text first: the second pause may be the
        // end, so its recognition must not wait behind a preview.
        val ended = CountDownLatch(1)
        val native = FakeNative().apply {
            abortableCalls = setOf(0)
            onTranscribe = { if (it == 1) check(ended.await(10, TimeUnit.SECONDS)) { "the utterance never ended" } }
        }
        val source = ArrayAudioSource(withPause(), sleepPerFrameMs = 5, onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 97) check(native.aborts.isEmpty()) { "aborted before the next pause" }
            if (n == 101) {         // the second pause was handed out with frame 98; the utterance ends with frame 109
                waitOnCapture("the second pause's recognition") { native.transcribeCalls.get() == 2 }
                check(!seen { it is SpeechEvent.SpeechEnded }) { "the utterance has ended already" }
            }
            if (n == 111) ended.countDown()
        })
        val engine = fakeEngine(native, config = on) { source }
        listen(engine)
        assertTrue("no preview", events.none { it is SpeechPreview })
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(2, native.transcribeCalls.get())
        assertEquals("aborted once, by the second pause", 1, native.aborts.toSet().size)
        assertEquals(1, engine.earlyStartsAborted)
        assertEquals(0, engine.earlyStartsShown)
        assertEquals(0, engine.earlyStartsDiscarded)
        assertEquals("the second pause's recognition ran ahead of the end", 1, engine.earlyStartsUsed)
        assertEquals(0, engine.earlyStartsTooLate)
        assertTrue("saved ${engine.earlyStartSavedMs} ms", engine.earlyStartSavedMs in 20..5_000)
        assertEquals(1, engine.aborts)
        assertTrue("stopped ${engine.maxAbortMs} ms after the abort", engine.maxAbortMs in 0..5_000)
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertAllZero(native.passedToWhisper)
        assertFalse(engine.isModelLoaded)
        val cancelled = Regex(
            """^utterance: \d+ ms of audio, recogniser cancelled after \d+ ms, .*, early start for a preview, aborted, """ +
                """stopped \d+ ms after the abort$""",
        )
        assertTrue(logs.joinToString("\n"), logs.any { cancelled.matches(it) })
        assertTrue(logs.any { Regex("""^aborts: 1, the slowest stopped the recogniser after \d+ ms$""").matches(it) })
        assertTrue(logs.any { Regex("""^early start: 1 used, \d+ ms saved, 0 too late; 0 shown as a preview; 0 discarded, 1 aborted, .*""").matches(it) })
    }

    @Test
    fun theAbortsOfASessionAreCountedForThatSessionOnly() {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val first = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
        })
        val sources = ArrayDeque(listOf(first, ArrayAudioSource(withPause())))
        val engine = fakeEngine(native, config = on) { sources.removeFirst() }
        listen(engine)
        assertTrue(logs.any { it.startsWith("aborts: 1, ") })
        events.clear()
        listen(engine)
        assertEquals("dobrý den", transcripts.single().text)
        assertTrue("no abort in the second session, so no line", logs.none { it.startsWith("aborts: ") })
        assertEquals("the engine's count stays", 1, engine.aborts)
    }

    @Test
    fun theStablePartSpansAPreviewFromAPauseAndATimedOne() {
        // 4 s of speech (to frame 140), a pause of 0.4 s handed out with frame 148 and run over with frame 153, 3 s of
        // speech. The pause's recognition gives the first preview; the timed one, 2 s after the pause began, the
        // second, and what the two start with is stable.
        val native = FakeNative().apply { texts += listOf("dobrý", "dobrý den", "dobrý den, jak se máte") }
        val audio = Signals.concat(quiet(0.5, 1), Signals.tone(4.0), quiet(0.4, 2), Signals.tone(3.0), quiet(1.5, 3))
        val awaited = AtomicInteger()
        val hooks = object : SpeechEngine.Hooks {
            override fun earlyStartAwaited() {
                awaited.incrementAndGet()
            }

            override fun previewRequested(samples: FloatArray) {
                waitOnCapture("the timed preview") { synchronized(events) { events.count { it is SpeechPreview } } == 2 }
            }
        }
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 151) waitOnCapture("the early recognition's end") { awaited.get() == 1 }
            if (n == 157) waitOnCapture("the preview from the pause") { seen { it is SpeechPreview } }
        })
        val engine = fakeEngine(native, hooks, config = on) { source }
        listen(engine, PreviewConfig(intervalMs = 2_000))
        assertEquals(
            listOf(SpeechPreview(startMs, "dobrý"), SpeechPreview(startMs, "dobrý den", stableLength = 5)),
            events.filterIsInstance<SpeechPreview>(),
        )
        assertEquals("dobrý den, jak se máte", transcripts.single().text)
        assertEquals("the pause's by the final call, the timed one by the preview call", 1, native.previewCalls.get())
        assertEquals(1, engine.earlyStartsShown)
    }

    @Test
    fun continuousPreviewsInASessionComeForEverySecondOfSpeechWhenTheRecogniserKeepsUp() {
        // 6 s of speech: previews after 2 s, then each a second of audio after the one before, which has ended by then.
        val native = FakeNative()
        val audio = Signals.concat(quiet(0.5, 1), Signals.tone(6.0), quiet(1.5, 2))
        val requested = AtomicInteger()
        val hooks = object : SpeechEngine.Hooks {
            override fun previewRequested(samples: FloatArray) {
                val n = requested.incrementAndGet()
                waitOnCapture("preview $n") { synchronized(events) { events.count { it is SpeechPreview } } >= n }
            }
        }
        val engine = fakeEngine(native, hooks, config = on) { ArrayAudioSource(audio) }
        listen(engine, PreviewConfig.continuous())
        val previews = events.filterIsInstance<SpeechPreview>()
        assertEquals("at 2, 3, 4 and 5 s of the utterance", 4, previews.size)
        assertEquals("the same text again is stable to its end", listOf(0, 9, 9, 9), previews.map { it.stableLength })
        assertEquals(4, native.previewCalls.get())
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals("the final text once, started in the closing pause", 1, native.greedyCalls.get())
        assertEquals(1, native.maxConcurrentRecognitions.get())
        assertAllZero(native.passedToWhisper)
    }

    @Test
    fun aPreviewFromAPauseThatFinishesAfterItsUtteranceEndedIsNotShown() {
        // The recognition is held until the utterance has ended; the abort came, but whisper.cpp returned text.
        val release = CountDownLatch(1)
        val native = FakeNative().apply {
            texts += listOf("pozdě", "dobrý den")
            onTranscribe = { if (it == 0) check(release.await(10, TimeUnit.SECONDS)) { "never released" } }
        }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 112) release.countDown()           // the utterance ended with frame 109
        })
        val engine = fakeEngine(native, config = on) { source }
        listen(engine)
        assertTrue(events.none { it is SpeechPreview })
        assertEquals(3, events.size)
        assertEquals("dobrý den", transcripts.single().text)
        assertTrue(events.none { it.toString().contains("pozdě") })
        assertTrue(logs.none { "pozdě" in it })
        assertEquals(0, engine.earlyStartsShown)
        assertEquals("the abort stopped nothing: whisper.cpp ran to its end", 0, engine.aborts)
        assertTrue(logs.none { it.startsWith("aborts: ") })
    }

    @Test
    fun withoutFromPausesItIsAbortedAsSoonAsTheSpeechGoesOn() {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        val source = ArrayAudioSource(withPause(), onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 64) waitOnCapture("the abort") { native.aborts.isNotEmpty() }
        })
        val engine = fakeEngine(native, config = on) { source }
        listen(engine, PreviewConfig(fromPauses = false))
        assertTrue(events.none { it is SpeechPreview })
        assertEquals("dobrý den", transcripts.single().text)
        assertEquals(1, engine.earlyStartsDiscarded)
        assertEquals(0, engine.earlyStartsAborted)
    }

    @Test
    fun stopAbortsARecognitionKeptForAPreview() {
        val native = FakeNative().apply { abortableCalls = setOf(0) }
        lateinit var engine: SpeechEngine
        // after the pause the speech never ends by itself
        val audio = Signals.concat(quiet(0.5, 1), Signals.tone(1.0), quiet(0.4, 2))
        val source = ArrayAudioSource(audio, tail = { Signals.toneAt(it) }, onFrame = { n ->
            if (n == 57) waitOnCapture("the early recognition") { native.transcribeCalls.get() == 1 }
            if (n == 70) {
                check(native.aborts.isEmpty()) { "aborted before stop()" }
                engine.stop()
            }
            if (n == 73) waitOnCapture("the abort at stop()") { native.aborts.isNotEmpty() }
        })
        engine = fakeEngine(native, config = on) { source }
        listen(engine)
        assertEquals("dobrý den", transcripts.single().text)
        assertTrue(events.none { it is SpeechPreview })
        assertEquals(1, engine.earlyStartsAborted)
        assertAllZero(native.passedToWhisper)
    }
}
