package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.EarlyStart
import ai.moravian.czechspeech.ListenEvent
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.Transcript
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
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
 * Streaming transcription in a [SpeechEngine] session, on the fake native layer. "Speech" is [Signals.clock] from 500 ms,
 * whose words are `slovo0`, `slovo1`, ... each spoken over [500 + 500k, 900 + 500k) ms; the fake recogniser returns the
 * words that lie in the audio it is given ([FakeNative.clockWords]), as the simulation's fake recogniser does. Where a
 * test needs every pass to reach the app before capture goes on, the capture thread waits for it at each request
 * ([Paced]), so nothing depends on timing.
 */
class StreamingSessionTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val streaming = CzechSpeechConfig(streaming = StreamingConfig())

    private fun speech(seconds: Double) = Signals.concat(Signals.zeros(0.5), Signals.clock(500, seconds), Signals.zeros(1.5))

    /** Every word `slovo<id>` of the blocks `(fromMs, seconds)` of clock speech that lies in `[startMs, endMs)` of the audio. */
    private fun clockWords(vararg blocks: Pair<Long, Double>): (Long, Long) -> String = { startMs, endMs ->
        blocks.flatMap { (from, seconds) -> (0 until (seconds * 2).toInt()).map { from + 500L * it } }
            .filter { it + 400 <= endMs && it >= startMs - 50 }
            .joinToString(" ") { "slovo${(it - 500) / 500}" }
    }

    private fun allWords(seconds: Double) = (0 until (seconds * 2).toInt()).map { "slovo$it" }

    private val events: MutableList<ListenEvent> = Collections.synchronizedList(mutableListOf())
    private val transcripts: List<Transcript>
        get() = synchronized(events) { events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript } }
    private val previews: List<SpeechPreview> get() = synchronized(events) { events.filterIsInstance<SpeechPreview>() }
    private val words: List<String> get() = transcripts.flatMap { it.text.split(' ') }.filter { it.isNotEmpty() }
    private val dones: List<SpeechEvent.UtteranceDone> get() = synchronized(events) { events.filterIsInstance<SpeechEvent.UtteranceDone>() }

    /** Blocks the capture thread until [condition] holds; fails the session after 10 s. */
    private fun waitOnCapture(what: String, condition: () -> Boolean) {
        val deadline = System.nanoTime() + 10_000_000_000L
        while (!condition()) {
            if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for $what")
            Thread.sleep(1)
        }
    }

    /**
     * Records the audio of every pass requested; [paced]: capture goes on only once each pass has been absorbed and
     * published (or dropped), as the engine reports it.
     */
    private inner class Paced(private val paced: Boolean = true) : SpeechEngine.Hooks {
        val requested = AtomicInteger()
        val finished = AtomicInteger()
        val recognised = AtomicInteger()
        val boundaries = AtomicInteger()
        val earlyOffers = AtomicInteger()
        val seen: MutableList<FloatArray> = Collections.synchronizedList(mutableListOf())

        override fun streamingPassRequested(samples: FloatArray) {
            seen += samples
            val n = requested.incrementAndGet()
            if (paced) waitOnCapture("pass $n") { finished.get() >= n }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }

        override fun utteranceRecognised() {
            recognised.incrementAndGet()
        }

        override fun sentenceBoundaryComing() {
            boundaries.incrementAndGet()
        }

        override fun earlyStartOffered(queued: Boolean) {
            earlyOffers.incrementAndGet()
        }
    }

    private suspend fun collect(flow: Flow<ListenEvent>) = withContext(Dispatchers.Default) { flow.collect { events += it } }

    private fun withLog(block: () -> Unit): List<String> {
        val lines = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        try {
            block()
        } finally {
            CsLog.sink = sink
        }
        return lines.toList()
    }

    /**
     * [got] holds the spoken words in order, at most [atMostLost] of them lost and at most [atMostRepeated] repeated
     * (none by default: no cut of a piece, the length cap or the lag cut, may write a word twice).
     */
    private fun assertInOrder(expected: List<String>, got: List<String>, atMostLost: Int, atMostRepeated: Int = 0) {
        val index = got.map { expected.indexOf(it) }
        assertTrue("every word is one of the spoken ones: $got", index.all { it >= 0 })
        val distinct = got.distinct()
        assertTrue("in order: $got", distinct.map { expected.indexOf(it) }.zipWithNext().all { (a, b) -> a < b })
        assertTrue("at most $atMostRepeated repeated: ${got.groupBy { it }.filter { it.value.size > 1 }.keys}", got.size - distinct.size <= atMostRepeated)
        assertTrue("at most $atMostLost lost, got ${expected - got.toSet()}", expected.size - distinct.size <= atMostLost)
    }

    @Test
    fun withoutStreamingListenRunsNoPass() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 12.0) }
        val hooks = Paced()
        val engine = fakeEngine(native, hooks) { ArrayAudioSource(speech(12.0)) }
        collect(engine.listen())
        assertEquals(0, native.previewCalls.get())
        assertEquals(0, hooks.requested.get())
        assertEquals("one recognition of the utterance", 1, native.transcribeCalls.get())
        assertEquals(allWords(12.0), words)
        assertEquals(1, transcripts.size)
        assertTrue("no UtteranceDone without streaming", events.none { it is SpeechEvent.UtteranceDone })
    }

    @Test
    fun aLongSpeechIsWrittenWhileSpokenWithoutLosingOrRepeatingAWord() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 20.0) }
        val hooks = Paced()
        val engine = fakeEngine(native, hooks, config = streaming) { ArrayAudioSource(speech(20.0)) }
        collect(engine.listen())
        val ended = events.filterIsInstance<SpeechEvent.SpeechEnded>().single()
        val started = events.filterIsInstance<SpeechEvent.SpeechStarted>().single()
        // what was spoken, once and in order (unique words: no seam can repeat or lose one)
        assertEquals(allWords(20.0), words)
        val beforeEnd = events.indexOf(ended)
        assertTrue("written while spoken: $transcripts", events.subList(0, beforeEnd).count { it is SpeechEvent.Transcript } >= 2)
        assertTrue(transcripts.all { it.startMs == started.atMs })
        assertEquals("endMs never decreases", transcripts.map { it.endMs }.sorted(), transcripts.map { it.endMs })
        assertEquals(ended.atMs, transcripts.last().endMs)
        assertTrue("the buffer stays short: ${native.previewSampleCounts}", native.previewSampleCounts.max() < 9 * 16_000)
        assertEquals("every pass was taken by the runner", hooks.requested.get(), hooks.finished.get())
        assertEquals(hooks.requested.get(), native.previewCalls.get())
        assertTrue("the audio of every pass is zeroed", hooks.seen.all { s -> s.all { it == 0f } })
        assertTrue("everything the recogniser got is zeroed, the rests for the finals too", native.passedToWhisper.all { s -> s.all { it == 0f } })
    }

    @Test
    fun previewsShowOnlyTheWordsAfterTheLastTranscript() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 20.0) }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(20.0)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        val started = events.filterIsInstance<SpeechEvent.SpeechStarted>().single()
        assertTrue("previews came", previews.isNotEmpty())
        var sent = emptySet<String>()
        var previewsAfterTheEnd = 0
        var ended = false
        for (event in events.toList()) {
            when (event) {
                is SpeechEvent.Transcript -> sent = sent + event.transcript.text.split(' ')
                is SpeechEvent.SpeechEnded -> ended = true
                is SpeechPreview -> {
                    if (ended) previewsAfterTheEnd++
                    assertEquals(0, event.stableLength)
                    assertEquals(started.atMs, event.startMs)
                    val shown = event.text.split(' ').filter { it.isNotEmpty() }
                    assertTrue("$shown overlaps $sent", shown.none { it in sent })
                }
                else -> Unit
            }
        }
        assertEquals("no preview after the SpeechEnded", 0, previewsAfterTheEnd)
        assertEquals(allWords(20.0), words)
        val last = events.indexOfLast { it is SpeechEvent.Transcript }
        assertTrue("nothing is left showing after the last Transcript", events.drop(last + 1).none { it is SpeechPreview && it.text.isNotEmpty() })
    }

    @Test
    fun theLengthCapEndsAPieceAndTheNextStartsAfresh() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 40.0) }
        val hooks = Paced()
        // The cap (28 s) is long past at frame 900: capture waits there for the final of the first piece, so that the
        // next piece gets its passes as it would with a microphone (no pass is asked for while a final waits).
        val source = ArrayAudioSource(speech(40.0), onFrame = { n ->
            if (n == 900) waitOnCapture("the first piece recognised") { hooks.recognised.get() >= 1 }
        })
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        collect(engine.listen())
        assertEquals("two utterances", 2, transcripts.map { it.startMs }.distinct().size)
        assertInOrder(allWords(40.0), words, atMostLost = 1)   // the fake VAD has no gaps: a cut may split a word (StreamingVirtualTimeTest has gaps)
        assertTrue("the second piece was written while spoken", transcripts.count { it.startMs != transcripts.first().startMs } >= 2)
    }

    @Test
    fun withoutAgreementTheBufferIsCutAtTwelveSeconds() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 20.0)
            previewOverride = { i, _ -> "x$i" }
        }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(20.0)) }
        collect(engine.listen())
        assertEquals("two utterances: $transcripts", 2, transcripts.map { it.startMs }.distinct().size)
        val first = transcripts.first()
        assertTrue("the first piece is at most 12 s of audio: $first", first.endMs - first.startMs <= 12_000 + 32)
        assertInOrder(allWords(20.0), words, atMostLost = 1)   // the fake VAD has no gaps: the cut may split a word
    }

    @Test
    fun stopDuringAPassWritesThePendingWordsInTheLastTranscript() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 20.0)
            abortableCalls = setOf(0)
        }
        lateinit var engine: SpeechEngine
        // Capture waits, with the first pass running, and stops the session there.
        val source = ArrayAudioSource(speech(20.0), onFrame = { n ->
            if (n == 70) {
                check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no pass" }
                engine.stop()
            }
        })
        engine = fakeEngine(native, Paced(paced = false), config = streaming) { source }
        collect(engine.listen())
        val ended = events.filterIsInstance<SpeechEvent.SpeechEnded>().single()
        val spoken = allWords(20.0).filterIndexed { k, _ -> 500 + 500 * k + 400 <= ended.atMs }
        assertTrue("some words: $spoken", spoken.size >= 4)
        assertEquals(spoken, words)
        assertEquals(ended.atMs, transcripts.last().endMs)
        assertTrue("the running pass was aborted", engine.abortedPreviews >= 1)
        assertEquals(1, native.previewCalls.get())
    }

    @Test
    fun cancellingDuringAPassSendsNothingMoreAndZeroesItsAudio() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 20.0)
            abortableCalls = setOf(0)
        }
        val hooks = Paced(paced = false)
        val closing = CountDownLatch(1)
        // Capture waits, with the pass running, until the collection has been cancelled.
        val source = ArrayAudioSource(speech(20.0), onFrame = { n ->
            if (n == 70) {
                check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no pass" }
                check(closing.await(10, TimeUnit.SECONDS)) { "not cancelled" }
            }
        })
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        val collected = async(Dispatchers.Default) { runCatching { engine.listen().collect { events += it } } }
        assertTrue("the pass is running", native.transcribeStarted.await(10, TimeUnit.SECONDS))
        awaitCondition("the pass was requested") { hooks.requested.get() >= 1 }
        engine.close()
        closing.countDown()
        val outcome = collected.awaitWithin(2_000, "the collection ended")
        assertTrue("$outcome", outcome.exceptionOrNull() is CancellationException)
        engine.awaitClosed()
        val atTheEnd = events.size
        Thread.sleep(50)
        assertEquals("no event after the cancellation", atTheEnd, events.size)
        assertTrue("a pass was seen", hooks.seen.isNotEmpty())
        assertTrue("every pass's audio is zeroed", hooks.seen.all { s -> s.all { it == 0f } })
        assertEquals(0, native.whisperFrees.get() - native.whisperInits.get())
    }

    @Test
    fun aPassOfPunctuationOnlyCommitsNothing() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 12.0)
            previewOverride = { i, w -> if (i == 2) "…" else w }
        }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(12.0)) }
        collect(engine.listen())
        assertTrue("no Transcript has it: ${transcripts.map { it.text }}", transcripts.none { it.text.contains("…") })
        assertEquals(allWords(12.0), words)
    }

    @Test
    fun sentenceJoiningDecidesOnlyTheFirstTranscriptOfAnUtterance() {
        // two utterances 1 s apart, the first long (12 s from 0.5 s, the second from 13.5 s)
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 12.0), Signals.zeros(1.0), Signals.clock(13_500, 3.0), Signals.zeros(1.5),
        )
        val native = FakeNative().apply { clockWords = clockWords(500L to 12.0, 13_500L to 3.0) }
        val hooks = Paced()
        // The language model is loaded after the first utterance; the second one starts only once it is there.
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 430) waitOnCapture("the language model") { native.lmInits.get() >= 1 }
        })
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm), streaming = StreamingConfig())
        val engine = fakeEngine(native, hooks, config = config) { source }
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        val starts = transcripts.map { it.startMs }.distinct()
        assertEquals(2, starts.size)
        for (start in starts) {
            val ofUtterance = transcripts.filter { it.startMs == start }
            assertTrue("every later Transcript of an utterance has no decision", ofUtterance.drop(1).all { it.sentenceJoin == null })
        }
        assertTrue("the first utterance came in several", transcripts.count { it.startMs == starts[0] } >= 2)
        assertEquals(1, lines.count { it.startsWith("sentence boundary:") })
        assertEquals("one decision per utterance, the language model only for the second", 2, hooks.boundaries.get())
        assertEquals(allWords(12.0) + (26 until 32).map { "slovo$it" }, words)
    }


    /**
     * Chunk mode (passes paused) on a phone whose finals take 2.25 s: pieces of 4 s are cut inside the speech; the model
     * writes each as a sentence (a capital and a period), as Whisper does. The pieces after a cut continue the one before
     * with no language model; the speech after the pause is decided by it as before.
     */
    @Test
    fun piecesAfterAForcedCutContinueThePreviousOneAndTheSpeechAfterAPauseIsDecidedByTheLanguageModel() {
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 13.0), Signals.zeros(1.2), Signals.clock(14_700, 3.0), Signals.zeros(1.5),
        )
        val words = clockWords(500L to 13.0, 14_700L to 3.0)
        val native = FakeNative().apply {
            clockWords = { from, to -> words(from, to).let { if (it.isEmpty()) it else it.replaceFirstChar { c -> c.uppercase() } + "." } }
            // a model that likes the small letter: fewer tokens, and with equal tokens the larger ids (the lower-case letters)
            lmScoreOf = { ids -> -2.0 * ids.size + 0.05 * ids.sum() }
        }
        val hooks = Paced()
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 430) waitOnCapture("the language model") { native.lmInits.get() >= 1 }
        })
        // pinned to NONE: the fake models a Whisper that capitalises every piece start, which is what the joiner decides about after a cut
        // without the context; with the context the text after the anchor is mid-sentence (CutSeamSessionTest covers the seam)
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm), streaming = StreamingConfig(cutSeam = ai.moravian.czechspeech.CutSeam.NONE))
        val engine = fakeEngine(native, hooks, config = config) { source }
        engine.streamingGuard!!.recorded(6_000, 3_000)                  // passes are paused: chunk mode
        engine.streamingGuard!!.recordedFinal(6_000, 2_250)             // pieces of 4 s
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        val all = transcripts
        assertTrue("forced cuts: ${all.size} transcripts", lines.count { it.startsWith("streaming: forced cut at ") } >= 2)
        val firstSpeech = all.filter { it.startMs < 14_000 }
        assertNull("the speech starts", firstSpeech.first().sentenceJoin)
        for (t in firstSpeech.drop(1)) {
            val join = t.sentenceJoin!!
            assertEquals(t.text, true, join.continuesPrevious)
            assertEquals(PreviousEnd.REMOVE_PERIOD, join.previousEnd)
            // the model prefers the small letter by more than 1.0, so the first word is lowered
            assertTrue(join.continuingText, join.continuingText[0].isLowerCase() && t.text[0].isUpperCase() && join.continuingText.drop(1) == t.text.drop(1))
        }
        assertEquals("one decision of the language model: the speech after the pause", 1, lines.count { it.startsWith("sentence boundary: ") && !it.contains("continued") })
        assertEquals(firstSpeech.size - 1, lines.count { it.startsWith("sentence boundary: continued (cut)") })
        assertTrue("after a cut the language model is asked about the capital only", lines.none { it.contains("(cut)") && !it.contains("language model") })
        assertEquals(
            "the boundary of the language model: the first piece, the pause, and the capital of each piece after a cut",
            2 + (firstSpeech.size - 1), hooks.boundaries.get(),
        )
    }

    /**
     * A cut on a quiet frame (a pause between sentences shorter than minSilenceMs) is not forced to continue: the language
     * model decides the boundary as after a pause. The speech has a pause of 0.3 s where the first piece is cut.
     */
    @Test
    fun aPieceCutOnAQuietFrameIsDecidedByTheLanguageModelNotForcedToContinue() {
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 4.0), Signals.zeros(0.5), Signals.clock(5_000, 4.0), Signals.zeros(1.5),
        )
        val words = clockWords(500L to 4.0, 5_000L to 4.0)
        val native = FakeNative().apply {
            clockWords = { from, to -> words(from, to).let { if (it.isEmpty()) it else it.replaceFirstChar { c -> c.uppercase() } + "." } }
        }
        val hooks = Paced()
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 300) waitOnCapture("the language model") { native.lmInits.get() >= 1 }
        })
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm), streaming = StreamingConfig())
        val engine = fakeEngine(native, hooks, config = config) { source }
        engine.streamingGuard!!.recorded(6_000, 3_000)
        engine.streamingGuard!!.recordedFinal(6_000, 2_250)
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        // the first cut is on the pause of 0.5 s (under minSilenceMs) once 256 ms of it have passed after the floor of the window (quiet point, before the target);
        // the second is where the speech ends: the first quiet stretch after the floor, 0.5 s before the endpointer ends the utterance
        val cuts = lines.filter { it.startsWith("streaming: forced cut at ") }
        assertEquals(lines.toString(), 2, cuts.size)
        assertTrue(cuts.toString(), cuts.all { it.contains("reason: target") && it.contains("quiet point") })
        assertEquals("the cut was on the pause: nothing is forced to continue", 0, lines.count { it.startsWith("sentence boundary: continued") })
        assertEquals("the language model decided the boundary after the cut", 1, lines.count { it.startsWith("sentence boundary: ") })
        assertTrue("the line says it was a quiet cut: $lines", lines.single { it.startsWith("sentence boundary: ") }.contains(" (quiet cut), language model "))
        assertEquals("the boundary of the first piece and of the one after the cut", 2, hooks.boundaries.get())
    }

    @Test
    fun earlyStartIsIgnoredWhileStreaming() {
        // a pause of 1 s in the middle of speech: early start would offer an utterance there
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 3.0), Signals.zeros(1.0), Signals.clock(4_500, 3.0), Signals.zeros(1.5),
        )
        val withEarlyStart = CzechSpeechConfig(earlyStart = EarlyStart())
        val control = Paced()
        runBlocking {
            collect(fakeEngine(FakeNative(), control, config = withEarlyStart) { ArrayAudioSource(audio) }.listen())
        }
        assertTrue("without streaming the pause is offered early", control.earlyOffers.get() >= 1)

        events.clear()
        val hooks = Paced()
        val config = CzechSpeechConfig(earlyStart = EarlyStart(), streaming = StreamingConfig())
        val native = FakeNative().apply { clockWords = clockWords(500L to 3.0, 4_500L to 3.0) }
        val lines = withLog { runBlocking { collect(fakeEngine(native, hooks, config = config) { ArrayAudioSource(audio) }.listen()) } }
        assertEquals(0, hooks.earlyOffers.get())
        assertEquals(1, lines.count { it == "listen: early start is off while streaming" })
        assertFalse("no early start line", lines.any { it.startsWith("early start:") })
        assertTrue(transcripts.isNotEmpty())
    }

    @Test
    fun aFailingPassCostsOnlyThatPass() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 12.0)
            failNext = 1                    // the first recognition, a pass, fails
        }
        val hooks = Paced()
        val engine = fakeEngine(native, hooks, config = streaming) { ArrayAudioSource(speech(12.0)) }
        collect(engine.listen())
        assertEquals(allWords(12.0), words)
        assertTrue("passes went on after the failed one", hooks.requested.get() >= 3)
        assertEquals(hooks.requested.get(), hooks.finished.get())
        assertTrue(events.none { it is SpeechEvent.Error })
    }

    @Test
    fun aFailingFinalDoesNotStopThePassesOfTheNextUtterance() = runBlocking {
        // 0.9 s of speech, then 3 s more from 2.5 s; the first final recognition (the first greedy one) fails
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 0.9), Signals.zeros(1.1), Signals.clock(2_500, 3.0), Signals.zeros(1.5),
        )
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 0.9, 2_500L to 3.0)
            onTranscribe = { if (greedyCalls.get() == 1) failNext = 1 }
        }
        val hooks = Paced()
        val engine = fakeEngine(native, hooks, config = streaming) { ArrayAudioSource(audio) }
        collect(engine.listen())
        assertEquals("the failure is reported", 1, events.count { it is SpeechEvent.Error })
        assertEquals("the loop went on", 2, hooks.recognised.get())
        assertTrue("the second utterance got its passes", hooks.requested.get() >= 1)
        assertEquals((4 until 10).map { "slovo$it" }, words)
        val errorAt = events.indexOfFirst { it is SpeechEvent.Error }
        assertEquals("each utterance ends once, the failed one too", events.filterIsInstance<SpeechEvent.SpeechStarted>().map { it.atMs }, dones.map { it.startMs })
        assertTrue("after its error", events.indexOf(dones[0]) > errorAt)
    }

    @Test
    fun everyUtteranceEndsWithOneDoneEventAfterItsLastTranscriptWhateverEndedIt() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 40.0) }
        val hooks = Paced()
        val source = ArrayAudioSource(speech(40.0), onFrame = { n ->
            if (n == 900) waitOnCapture("the first piece recognised") { hooks.recognised.get() >= 1 }
        })
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        collect(engine.listen())
        val started = events.filterIsInstance<SpeechEvent.SpeechStarted>().single()
        val starts = transcripts.map { it.startMs }.distinct()
        assertEquals("two utterances: the length cap cut the speech", 2, starts.size)
        assertEquals("exactly one Done for each, in order, with the startMs of its Transcripts", starts, dones.map { it.startMs })
        for (done in dones) {
            val lastTranscript = events.indexOfLast { it is SpeechEvent.Transcript && it.transcript.startMs == done.startMs }
            assertTrue("after the last Transcript of ${done.startMs}", events.indexOf(done) > lastTranscript)
        }
        assertEquals("the first piece starts where its speech started", started.atMs, starts[0])
        assertEquals("a piece after a cut starts where the one before ended", transcripts.last { it.startMs == starts[0] }.endMs, starts[1])
    }

    @Test
    fun anUtteranceWhoseFinalFindsNoWordsStillEnds() = runBlocking {
        val native = FakeNative().apply { clockWords = { _, _ -> "" } }
        val audio = Signals.concat(Signals.zeros(0.5), Signals.clock(500, 0.9), Signals.zeros(1.5))
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(audio) }
        collect(engine.listen())
        assertEquals(emptyList<Transcript>(), transcripts)
        assertEquals(listOf(SpeechEvent.UtteranceDone(events.filterIsInstance<SpeechEvent.SpeechStarted>().single().atMs)), dones)
        assertTrue(events.indexOf(dones.single()) > events.indexOfFirst { it is SpeechEvent.SpeechEnded })
    }

    @Test
    fun theMeasuredSpeedOutlivesTheSessionInOneEngine() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 6.0); recognitionMs = 20 }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(6.0)) }
        val guard = engine.streamingGuard!!
        assertEquals(StreamingGuard.DEFAULT_COST, guard.costPerWindowMs, 0.0)
        collect(engine.listen())
        val measured = guard.costPerWindowMs
        assertTrue("the first session measured the speed: $measured", measured < StreamingGuard.DEFAULT_COST)
        events.clear()
        collect(engine.listen())
        assertTrue("the second session worked with the same estimate", engine.streamingGuard === guard)
        assertEquals(allWords(6.0), words)
    }

    // ---- a pass that is running when its piece ends, whatever ends it; finals that fail or find nothing -------------

    /**
     * Like [Paced], but the first pass requested once [armed] is set is the one capture runs ahead of: the fake
     * recogniser holds it (until its abort) and capture goes on, so the piece ends while that pass is running.
     */
    private inner class Blocking(private val native: FakeNative) : SpeechEngine.Hooks {
        @Volatile var armed = false
        val requested = AtomicInteger()
        val finished = AtomicInteger()
        val blocked = AtomicInteger()
        val recognised = AtomicInteger()

        override fun streamingPassRequested(samples: FloatArray) {
            val n = requested.incrementAndGet()
            if (armed && blocked.get() == 0) {
                armed = false
                blocked.incrementAndGet()
                val index = native.transcribeCalls.get()          // every earlier pass has finished: this is its index
                native.abortableCalls = setOf(index)
                waitOnCapture("the blocked pass started") { native.transcribeCalls.get() > index }
            } else {
                waitOnCapture("pass $n") { finished.get() >= n }
            }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }

        override fun utteranceRecognised() {
            recognised.incrementAndGet()
        }
    }

    @Test
    fun aPassRunningWhenTheSpeechEndsIsAbortedAndTheFinalWritesItsWordsOnce() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 3.0) }
        val hooks = Blocking(native)
        val source = ArrayAudioSource(speech(3.0), onFrame = { n -> if (n == 55) hooks.armed = true })    // 1.76 s
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        collect(engine.listen())
        assertEquals("the first pass finished, the second was running", 2, hooks.requested.get())
        assertEquals(1, hooks.blocked.get())
        assertEquals("one pass interrupted", 1, engine.abortedPreviews)
        assertEquals(allWords(3.0), words)
        assertEquals(1, dones.size)
        assertTrue(events.none { it is SpeechEvent.Error })
    }

    @Test
    fun aPassRunningAtTheLengthCapIsAbortedAndNoWordIsWrittenTwice() = runBlocking {
        val native = FakeNative().apply { clockWords = clockWords(500L to 40.0) }
        val hooks = Blocking(native)
        val source = ArrayAudioSource(speech(40.0), onFrame = { n ->
            if (n == 800) hooks.armed = true
            if (n == 900) waitOnCapture("the first piece recognised") { hooks.recognised.get() >= 1 }
        })
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        collect(engine.listen())
        assertEquals(1, hooks.blocked.get())
        assertEquals("one pass interrupted", 1, engine.abortedPreviews)
        assertEquals("two utterances", 2, dones.size)
        assertInOrder(allWords(40.0), words, atMostLost = 1)
    }

    @Test
    fun aPassRunningAtAForcedCutIsAbortedAndNoWordIsWrittenTwice() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 14.0)
            previewOverride = { i, _ -> "x$i" }            // the passes never agree: the buffer grows
        }
        val hooks = Blocking(native)
        val source = ArrayAudioSource(speech(14.0), onFrame = { n ->
            if (n == 200) hooks.armed = true
            if (n == 330) waitOnCapture("the first piece recognised") { hooks.recognised.get() >= 1 }
        })
        val engine = fakeEngine(native, hooks, config = streaming) { source }
        // a phone whose finals are dear: a buffer of 8 s is cut at once, and its passes are cheap enough to run
        engine.streamingGuard!!.recorded(6_000, 100)
        engine.streamingGuard!!.recordedFinal(12_000, 12_000)
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        assertEquals(lines.toString(), 1, lines.count { it.startsWith("streaming: forced cut at ") })
        assertEquals(1, hooks.blocked.get())
        assertEquals("one pass interrupted", 1, engine.abortedPreviews)
        assertEquals("two utterances", 2, dones.size)
        assertInOrder(allWords(14.0), words, atMostLost = 1)
    }

    @Test
    fun aFinalThatFailsAfterChunksWereSentLeavesTheWordsSentAndEndsTheUtterance() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 8.0)
            onTranscribe = { if (greedyCalls.get() == 1) failNext = 1 }     // the first final recognition fails
        }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(8.0)) }
        collect(engine.listen())
        assertTrue("chunks went out while speaking", transcripts.isNotEmpty())
        assertEquals("the failure is reported", 1, events.count { it is SpeechEvent.Error })
        assertEquals("what was sent is the start of the speech, once", allWords(8.0).take(words.size), words)
        assertTrue("the rest was lost with the failed final", words.size < allWords(8.0).size)
        assertEquals(1, dones.size)
        assertTrue("after the error", events.indexOf(dones[0]) > events.indexOfFirst { it is SpeechEvent.Error })
    }

    @Test
    fun aFinalWithoutWordsWithdrawsThePreviewAndEndsTheUtterance() = runBlocking {
        val inRange = clockWords(500L to 2.5)
        val native = FakeNative().apply {
            // the passes hear the words; the final, over the audio to the end of the speech and its pause, nothing
            clockWords = { startMs, endMs -> if (endMs > 3_300) "" else inRange(startMs, endMs) }
        }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(2.5)) }
        collect(engine.listenWithPreviews(PreviewConfig()))
        assertTrue("a preview was shown", previews.any { it.text.isNotEmpty() })
        assertEquals("withdrawn at the end", "", previews.last().text)
        assertEquals(1, dones.size)
        assertTrue(events.indexOf(dones[0]) > events.indexOf(previews.last()))
    }

    @Test
    fun anUtteranceSkippedBecauseRecognitionFellBehindStillEnds() = runBlocking {
        // four utterances of 0.3 s of speech (too short for a pass), 3 s apart; the first recognition is held while the others end, and the
        // queue holds 2 s of audio: the second utterance (1.2 s) waits, the third and fourth are skipped
        val hold = CountDownLatch(1)
        val native = FakeNative().apply {
            clockWords = { _, _ -> "w" }
            holdRecognition = hold
        }
        val audio = Signals.concat(*Array(4) { k -> Signals.concat(Signals.zeros(0.5), Signals.clock(500L + 3_000L * k, 0.3), Signals.zeros(1.6)) })
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 94) check(native.transcribeStarted.await(10, TimeUnit.SECONDS)) { "no recognition" }
        })
        val engine = fakeEngine(native, Paced(), maxQueuedAudioMs = 2_000, config = streaming) { source }
        val collected = async(Dispatchers.Default) { engine.listen().collect { events += it } }
        try {
            awaitCondition("two utterances skipped") { engine.skippedUtterances == 2 }
        } finally {
            hold.countDown()
        }
        collected.awaitWithin(10_000, "the session")
        assertEquals(2, events.count { it is SpeechEvent.Error })
        val started = events.filterIsInstance<SpeechEvent.SpeechStarted>().map { it.atMs }
        assertEquals("each of the four utterances ends, the skipped ones too (at once, before the recognised ones)", started, dones.map { it.startMs }.sorted())
        assertEquals("the recognised ones", 2, transcripts.size)
    }

    @Test
    fun anErrorInAPassStillSettlesItAndZeroesItsAudio() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 12.0)
            onTranscribe = { if (it == 0) throw OutOfMemoryError("test") }       // the first recognition, a pass
        }
        val hooks = Paced(paced = false)
        val engine = fakeEngine(native, hooks, config = streaming) { ArrayAudioSource(speech(12.0)) }
        val outcome = runCatching { collect(engine.listen()) }
        assertTrue("the Error is not swallowed: $outcome", outcome.isFailure)
        assertTrue("a pass was requested", hooks.seen.isNotEmpty())
        assertTrue("the audio of every pass is zeroed", hooks.seen.all { s -> s.all { it == 0f } })
        assertTrue("the pass that failed was settled", hooks.finished.get() >= 1)
        // Unpaced, the capture thread can request one more pass between the Error and the end of the session: the tracker
        // keeps one pass in flight, so at most one, and it is dropped with the queue (its audio is zeroed, asserted above)
        // without a finish report. The earlier equality was a race of the test.
        assertTrue("requested ${hooks.requested.get()}, finished ${hooks.finished.get()}", hooks.requested.get() - hooks.finished.get() in 0..1)
    }
    /**
     * The same session with the default [ai.moravian.czechspeech.CutSeam.CONTEXT] (the keyboard's real configuration): the text after a
     * cut comes from the run with the context, so it starts mid-sentence in the case Whisper gave it; the language model is not asked
     * about a capital that is not there, and the join still continues the previous piece and takes its period off.
     */
    @Test
    fun withTheContextOnlyPiecesAfterACutStillContinueThePreviousOneAndTheModelIsNotAskedAboutALowerCaseRest() {
        val audio = Signals.concat(
            Signals.zeros(0.5), Signals.clock(500, 13.0), Signals.zeros(1.2), Signals.clock(14_700, 3.0), Signals.zeros(1.5),
        )
        val words = clockWords(500L to 13.0, 14_700L to 3.0)
        val native = FakeNative().apply {
            clockWords = { from, to -> words(from, to).let { if (it.isEmpty()) it else it.replaceFirstChar { c -> c.uppercase() } + "." } }
            lmScoreOf = { ids -> -2.0 * ids.size + 0.05 * ids.sum() }
        }
        val hooks = Paced()
        val source = ArrayAudioSource(audio, onFrame = { n ->
            if (n == 430) waitOnCapture("the language model") { native.lmInits.get() >= 1 }
        })
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm), streaming = StreamingConfig(cutSeam = ai.moravian.czechspeech.CutSeam.CONTEXT))
        val engine = fakeEngine(native, hooks, config = config) { source }
        engine.streamingGuard!!.recorded(6_000, 3_000)
        engine.streamingGuard!!.recordedFinal(6_000, 2_250)
        val lines = withLog { runBlocking { collect(engine.listen()) } }
        val firstSpeech = transcripts.filter { it.startMs < 14_000 }
        assertTrue("forced cuts: ${firstSpeech.size} transcripts", firstSpeech.size >= 3)
        assertTrue("the pieces after a cut ran with the context: $lines", lines.any { it.startsWith("utterance: seam context ") })
        for (t in firstSpeech.drop(1)) {
            val join = t.sentenceJoin!!
            assertEquals(t.text, true, join.continuesPrevious)
            assertEquals(PreviousEnd.REMOVE_PERIOD, join.previousEnd)
            assertTrue("the rest starts as Whisper wrote it, in the middle of the sentence: ${t.text}", t.text[0].isLowerCase())
            assertEquals(t.text, join.continuingText)
        }
        assertEquals(firstSpeech.size - 1, lines.count { it.startsWith("sentence boundary: continued (cut)") })
        assertTrue("the language model is not asked about a lower-case rest", lines.none { it.startsWith("sentence boundary: continued (cut)") && it.contains("language model") })
        assertEquals("the boundary of the language model: the first piece and the pause", 2, hooks.boundaries.get())
    }

    /**
     * Pass mode (a fast phone), no two passes agree, so the buffer is cut at its 12 s: the piece after the cut has under a second of speech,
     * no pass of its own, and its final runs with the context of the piece before. No word is lost or written twice and no preview outlives the last Transcript.
     */
    @Test
    fun aContinuationWithoutPassesOfItsOwnInPassModeRunsWithTheContextAndLeavesNoPreview() = runBlocking {
        val native = FakeNative().apply {
            clockWords = clockWords(500L to 12.9)
            previewOverride = { i, _ -> "x$i" }
        }
        val engine = fakeEngine(native, Paced(), config = streaming) { ArrayAudioSource(speech(12.9)) }
        assertTrue("passes run", !engine.streamingGuard!!.passesPaused)
        val lines = withLog { runBlocking { collect(engine.listenWithPreviews(PreviewConfig())) } }
        assertEquals("two utterances: $transcripts", 2, transcripts.map { it.startMs }.distinct().size)
        assertTrue("the second final ran with the context: $lines", lines.any { it.startsWith("utterance: seam context ") })
        assertInOrder(allWords(12.9), words, atMostLost = 1)
        val last = events.indexOfLast { it is SpeechEvent.Transcript }
        assertTrue("nothing is left showing after the last Transcript", events.drop(last + 1).none { it is SpeechPreview && it.text.isNotEmpty() })
        assertTrue("everything the recogniser got is zeroed", native.passedToWhisper.all { s -> s.all { it == 0f } })
    }
}
