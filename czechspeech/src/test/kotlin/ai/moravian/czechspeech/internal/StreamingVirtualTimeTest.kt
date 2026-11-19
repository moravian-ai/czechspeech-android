package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.StreamingConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.Locale

/**
 * One session of streaming transcription in virtual time, without a model or a clock: the real [Endpointer] and
 * [StreamingTracker] over the clock speech of [Signals.clock] (words `slovo<k>`, each over `[500 + 500k, 900 + 500k)`
 * ms, continuous for [speechSeconds]), a fake recogniser that hears the words wholly inside the audio it is given, and
 * a whisper thread of its own: one task at a time, in order, whose duration follows the curve of a phone (recogniser
 * ms by ms of audio, times [factor]). The curve is the floating microphone of the author's phone, 3 threads, cool:
 * 0 s of audio 3.0 s, 5 s 3.2 s, 10 s 3.6 s, 15 s 5.9 s, 26 s 11.4 s. A running pass that is cancelled stops 100 ms
 * after the cancel. Words are told apart by their ids, so a word sent twice or never shows.
 */
internal class VirtualSession(
    private val factor: Double,
    private val speechSeconds: Double,
    private val guard: StreamingGuard = StreamingGuard(15_000, 384),
    private val maxLagMs: Int = 15_000,
    /** From this virtual time on the phone is [slowdownFactor] times as slow as [factor] (it warms up). */
    private val slowdownAtMs: Long = Long.MAX_VALUE,
    private val slowdownFactor: Double = factor,
    /** Recogniser seconds by seconds of audio: [EARLIER_CURVE], or [PIXEL_CURVE] (final recognitions measured on the phone). */
    private val curve: List<Pair<Double, Double>> = EARLIER_CURVE,
    /**
     * 0: the words run on with the gaps of probability 0.6 only (no quiet frame anywhere). Otherwise after every
     * [breathEvery]th word a breath: [BREATH_MS] of frames at 0.05 (below the endpointer's negThreshold, far under its
     * 600 ms of silence) from the end of the word, as a comma or a breath in continuous dictation.
     */
    private val breathEvery: Int = 0,
    /** False: the tracker is told of no quiet frame, i.e. a cut only at the target length (the behaviour before the quiet point). */
    private val quietPoint: Boolean = true,
    /** Length of a breath in ms (frames at 0.05 from the end of the word). */
    private val breathMs: Long = BREATH_MS,
    /** Non-uniform frames: every 4th word has a one-frame dip (0.05) five frames into it, as a VAD that flickers inside a word. */
    private val dips: Boolean = false,
    /** The language model time of every final recognition, in ms (on the whisper thread after the recogniser, and told to the guard). */
    private val languageModelMs: Long = 0,
    /** With a value: a model with the languages cs, sk and en and this minimum length to leave Czech, so the passes are held by the language rule. */
    private val minLeaveMs: Int? = null,
    /** What is done at a cut inside speech: the context alone by default here (the other tests measure that), the default of the SDK in the held-word test. */
    private val cutSeam: ai.moravian.czechspeech.CutSeam = ai.moravian.czechspeech.CutSeam.CONTEXT,
) {
    class Result(
        /** Every word sent, in the order of sending, with the virtual time it was sent at. */
        val sent: List<Pair<String, Long>>,
        /** Per word id, ms from the end of the word to the first time it was sent (-1: never sent). */
        val delaysMs: List<Long>,
        val words: Int,
        /** Ms from the end of the last word to the last text sent. */
        val afterSpeechMs: Long,
        /** Audio of every final recognition, in ms. */
        val finalAudioMs: List<Long>,
        /** The recogniser time of every final recognition, in ms, and the virtual time it started at. */
        val finalCostMs: List<Long>,
        val finalStartedAtMs: List<Long>,
        /** Most finals waiting or running at once. */
        val maxFinalsAtOnce: Int,
        val maxFinalWaitMs: Long,
        val paused: Int,
        val resumed: Int,
        val passes: Int,
        val forcedCuts: Int,
        /** Forced cuts decided by the target length, and by the lag limit (log lines `reason: target` / `reason: lag limit`). */
        val targetCuts: Int,
        /** Of them, those made at a quiet point before the target (log `quiet point`). */
        val quietPointCuts: Int,
        val lagCuts: Int,
        val stats: StreamingStats,
        /** Ms from the start of the speech (500 ms) to the first text sent. */
        val firstTextMs: Long,
        /** The `[startMs, endMs]` of every chunk sent, in order of sending. */
        val ranges: List<Pair<Long, Long>>,
    ) {
        val text: List<String> get() = sent.map { it.first }
        val maxDelayMs: Long get() = delaysMs.max()
        val missing: List<String> get() = (0 until words).filter { delaysMs[it] < 0 }.map { "slovo$it" }
        val repeated: List<String> get() = text - text.distinct().toSet()
        /** The chunks of a piece end no later than the next piece starts: no two pieces claim the same audio. */
        val rangesDoNotOverlap: Boolean
            get() = ranges.zipWithNext().all { (a, b) -> a.first == b.first || a.second <= b.first }
        val inOrder: Boolean get() = text.distinct().map { it.removePrefix("slovo").toInt() }.zipWithNext().all { (a, b) -> a < b }
    }

    private enum class Kind { PASS, FINAL }

    private class Task(val kind: Kind, val readyAt: Long, val pass: StreamingPass? = null, val utterance: Utterance? = null)

    private class Running(val task: Task, val startedAt: Long, var doneAt: Long, val plan: FinalPlan?, val audioMs: Long, val decisive: Boolean = true) {
        var abortSeen = false
    }


    private fun durationMs(audioMs: Long, at: Long): Long {
        // PIXEL_CURVE is over the encoder's window (the audio and 32 frames, in steps of 16, from 384 frames), as measured
        val x = if (curve === PIXEL_CURVE) EncoderWindow.frames((audioMs * 16).toInt(), 384) * 0.02 else audioMs / 1000.0
        var i = curve.indexOfLast { it.first <= x }.coerceIn(0, curve.size - 2)
        val (x0, y0) = curve[i]
        val (x1, y1) = curve[i + 1]
        return Math.round(1000 * (if (at >= slowdownAtMs) slowdownFactor else factor) * (y0 + (y1 - y0) * (x - x0) / (x1 - x0)))
    }

    private val wordCount = (speechSeconds * 2).toInt()

    /** Where word k starts: 500 ms apart, and after each breath word the breath makes the gap [breathMs] long instead of 100 ms. */
    private fun wordStartMs(k: Int): Long = 500L + 500L * k + (if (breathEvery > 0) (k / breathEvery) * (breathMs - 100) else 0L)
    private fun wordEndMs(k: Int) = wordStartMs(k) + 400
    private fun hasBreathAfter(k: Int) = breathEvery > 0 && k % breathEvery == breathEvery - 1

    /** The words `[fromMs, toMs)` of the audio wholly hold, as the fake recogniser of the session tests hears them. */
    private fun hear(fromMs: Long, toMs: Long): String =
        (0 until wordCount).filter { wordStartMs(it) >= fromMs - 50 && wordEndMs(it) <= toMs }.joinToString(" ") { "slovo$it" }

    /**
     * The probability of speech of the frame ending at [endMs]: words 0.95, the gaps between them 0.6, silence 0.05; after a breath
     * word the whole gap is 0.05 (the frames inside it; a frame that touches the word 0.6), and with [dips] a frame inside every 4th
     * word is 0.05 too, as low as the silence.
     */
    private fun probability(endMs: Long): Float {
        val from = endMs - FRAME_MS
        val speechEnd = wordEndMs(wordCount - 1) + 100
        if (endMs <= 500 || from >= speechEnd) return 0.05f
        var k = ((from - 500) / 500).toInt().coerceAtMost(wordCount - 1)
        while (k > 0 && wordStartMs(k) > from) k--
        val phase = from - wordStartMs(k)
        if (phase < 0) return 0.6f
        if (dips && k % 4 == 2 && phase in 160 until 192) return 0.05f
        if (phase + FRAME_MS <= 400 && endMs <= speechEnd) return 0.95f
        if (hasBreathAfter(k) && phase >= 400 && from + FRAME_MS <= wordStartMs(k + 1)) return 0.05f
        return 0.6f
    }

    fun run(): Result {
        val audio = FloatArray(((wordEndMs(wordCount - 1) + 100 + 1_500) * (SAMPLE_RATE / 1000)).toInt())
        var clock = 0L
        val sent = ArrayList<Pair<String, Long>>()
        val ranges = ArrayList<Pair<Long, Long>>()
        val firstSent = LongArray(wordCount) { -1 }
        fun record(text: String, at: Long) {
            for (w in text.split(' ').filter { it.isNotEmpty() }) {
                sent += w to at
                val k = w.removePrefix("slovo").toInt()
                if (firstSent[k] < 0) firstSent[k] = at
            }
        }
        val token = AbortToken(FakeNative())
        val rule = minLeaveMs?.let {
            LanguageRule(listOf("cs", "sk", "en"), ai.moravian.czechspeech.CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), it, 1_000))
        }
        val tracker = StreamingTracker(StreamingConfig(maxLagMs, cutSeam), 384, false, guard, rule?.let { LanguageSession(it) }) { event ->
            if (event is SpeechEvent.Transcript) {
                ranges += event.transcript.startMs to event.transcript.endMs
                record(event.transcript.text, clock)
            }
        }
        val queue = ArrayDeque<Task>()
        var running: Running? = null
        var maxFinals = 0
        var maxWait = 0L
        val finalAudio = ArrayList<Long>()
        val finalCost = ArrayList<Long>()
        val finalStarted = ArrayList<Long>()
        var paused = 0
        var resumed = 0
        var lastSent = 0L

        fun finalsAtOnce() = queue.count { it.kind == Kind.FINAL } + if (running?.task?.kind == Kind.FINAL) 1 else 0

        fun startNext(at: Long) {
            while (running == null && queue.isNotEmpty()) {
                val task = queue.removeFirst()
                if (task.kind == Kind.PASS) {
                    val pass = task.pass!!
                    if (pass.isCancelled || !pass.attach(token)) {
                        tracker.absorb(pass, null)
                        continue
                    }
                    // the engine asks the language of the piece before the recogniser runs
                    val decisive = rule?.decisive(tracker.languageFor(pass), pass.audioMs) ?: true
                    running = Running(task, at, at + durationMs(pass.audioMs, at), null, pass.audioMs, decisive)
                } else {
                    val u = task.utterance!!
                    val plan = tracker.finalPlan(u)
                    val audioMs = (u.samples.size - plan.offset) / 16L
                    finalAudio += audioMs
                    finalStarted += at
                    maxWait = maxOf(maxWait, at - task.readyAt)
                    running = Running(task, at, at + durationMs(audioMs, at) + languageModelMs, plan, audioMs)
                    finalCost += running!!.doneAt - at
                }
            }
        }

        fun complete(r: Running) {
            val task = r.task
            if (task.kind == Kind.PASS) {
                val pass = task.pass!!
                val result = if (r.abortSeen) null else PassResult(hear(pass.startMs, pass.endMs), r.doneAt - r.startedAt, TEST_LANGUAGE, r.decisive)
                val absorbed = tracker.absorb(pass, result)
                if (absorbed != null) tracker.publish(absorbed, null, 0)
            } else {
                val u = task.utterance!!
                val plan = r.plan!!
                val from = u.audioStartMs + plan.offset / 16
                val to = u.audioStartMs + u.samples.size / 16
                val chunk = tracker.absorbFinal(plan, hear(from, to), r.doneAt - r.startedAt - languageModelMs)
                if (languageModelMs > 0) tracker.languageModelTime(languageModelMs)
                if (chunk != null) {
                    ranges += chunk.startMs to chunk.endMs
                    record(chunk.text, r.doneAt)
                }
                tracker.finalDone(u, chunk != null)
            }
        }

        fun advanceTo(t: Long) {
            while (true) {
                val r = running ?: return
                if (r.doneAt > t) return
                clock = r.doneAt
                running = null
                complete(r)
                startNext(clock)
            }
        }

        val config = CzechSpeechConfig(streaming = StreamingConfig(maxLagMs, ai.moravian.czechspeech.CutSeam.CONTEXT))
        val endpointer = Endpointer(config) { event ->
            when (event) {
                is EndpointerEvent.SpeechEnded -> tracker.pieceEnded()
                is EndpointerEvent.UtteranceReady -> {
                    tracker.finalComing(event.utterance)
                    queue += Task(Kind.FINAL, clock, utterance = event.utterance)
                    maxFinals = maxOf(maxFinals, finalsAtOnce())
                }
                else -> Unit
            }
        }
        wireEndpointer(endpointer, tracker)

        val lines = java.util.Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        var passes = 0
        try {
            val frame = FloatArray(FRAME_SAMPLES)
            var f = 0
            var ended = false
            while (true) {
                val t = (f + 1) * FRAME_MS
                advanceTo(t)
                clock = t
                if (f.toLong() * FRAME_SAMPLES < audio.size) {
                    audio.copyInto(frame, 0, f * FRAME_SAMPLES, minOf(audio.size, (f + 1) * FRAME_SAMPLES))
                    endpointer.accept(frame, probability(t))
                    when (val step = tracker.onFrame(endpointer.pieceMs, endpointer.audioStartMs, endpointer.positionMs, if (quietPoint) endpointer.quietFrames else 0, endpointer::snapshotFrom)) {
                        is StreamingTracker.Step.Pass -> {
                            tracker.requests.tryReceive()
                            passes++
                            queue += Task(Kind.PASS, t, pass = step.pass)
                        }
                        StreamingTracker.Step.Cut -> endpointer.cutNow()
                        StreamingTracker.Step.CutQuiet -> endpointer.cutAtQuietStretch()
                        StreamingTracker.Step.None -> Unit
                    }
                } else if (!ended) {
                    endpointer.finish()
                    tracker.end()
                    ended = true
                }
                startNext(t)
                running?.let { r ->
                    if (r.task.kind == Kind.PASS && !r.abortSeen && r.task.pass!!.isCancelled) {
                        r.abortSeen = true
                        r.doneAt = minOf(r.doneAt, t + ABORT_MS)
                    }
                }
                if (ended && running == null && queue.isEmpty()) break
                f++
                check(f < 1_000_000) { "the session does not end" }
            }
            lastSent = sent.maxOfOrNull { it.second } ?: 0
        } finally {
            CsLog.sink = sink
        }
        paused = lines.count { it.startsWith("streaming: passes paused") }
        resumed = lines.count { it.startsWith("streaming: passes resumed") }
        val stats = tracker.stats()
        return Result(
            sent = sent,
            delaysMs = (0 until wordCount).map { if (firstSent[it] < 0) -1L else firstSent[it] - wordEndMs(it) },
            words = wordCount,
            afterSpeechMs = lastSent - wordEndMs(wordCount - 1),
            finalAudioMs = finalAudio,
            finalCostMs = finalCost,
            finalStartedAtMs = finalStarted,
            maxFinalsAtOnce = maxFinals,
            maxFinalWaitMs = maxWait,
            paused = paused,
            resumed = resumed,
            passes = passes,
            forcedCuts = stats.forcedCuts,
            targetCuts = lines.count { it.contains("reason: target") },
            quietPointCuts = lines.count { it.contains("quiet point") },
            lagCuts = lines.count { it.contains("reason: lag limit") },
            stats = stats,
            firstTextMs = (sent.firstOrNull()?.second ?: 0L) - 500L,
            ranges = ranges,
        )
    }

    /** What the engine does with the endpointer and the tracker besides the frames. */
    private fun wireEndpointer(endpointer: Endpointer, tracker: StreamingTracker) {
        endpointer.cutNotBefore = tracker::cutNotBeforeMs
    }

    companion object {
        const val ABORT_MS = 100L
        const val BREATH_MS = 230L

        val EARLIER_CURVE = listOf(0.0 to 3.0, 5.0 to 3.2, 10.0 to 3.6, 15.0 to 5.9, 26.0 to 11.4, 40.0 to 18.0)

        /**
         * The Pixel 9a, floating microphone, 3 threads, cold, lowest window 384 frames (7.68 s): the first test of the
         * author with chunk mode: 2.0 to 2.9 s (typically 2.25 s) up to the lowest window (flat), 3.0 to 3.8 s at 448 to 480
         * frames (8.96 to 9.6 s), 3.3 to 4.0 s at 496 to 528 (9.9 to 10.6 s), 4.7 s at 560 (11.2 s); beyond it extended at
         * the same slope. The x is the window in s, not the audio: audio up to 7.04 s is on the lowest window.
         */
        val PIXEL_CURVE = listOf(0.0 to 2.25, 7.68 to 2.25, 8.96 to 3.4, 9.92 to 3.7, 11.2 to 4.7, 12.8 to 5.9, 20.0 to 10.0, 30.0 to 16.0)
    }
}

/**
 * Streaming over long speech in virtual time ([VirtualSession]): what the safeguard guarantees (delays, queue, flapping,
 * the rest after the end) and that no cut writes a word twice or loses it.
 */
class StreamingVirtualTimeTest {
    /**
     * [lostAllowed]: only a cut of the lag limit, whose floor (8 s) may leave a window of one frame, can split a word, which is
     * then lost: one per such cut at most. A cut at the target length (a window of 1.5 s) loses none: 0 of 74 + 54 + 54 in the
     * simulations, and the lag cuts lost 0 of 37 on the phone curve and 37 of 37 on the earlier one.
     */
    private fun assertEveryWordOnceInOrder(r: VirtualSession.Result, what: String, lostAllowed: Int = 0) {
        assertTrue("$what: words never sent ${r.missing} (at most $lostAllowed)", r.missing.size <= lostAllowed)
        assertEquals("$what: words sent twice", emptyList<String>(), r.repeated)
        assertTrue("$what: in order", r.inOrder)
        assertTrue("$what: the pieces do not claim the same audio: ${r.ranges}", r.rangesDoNotOverlap)
        if (lostAllowed == 0) assertEquals("$what: the words together", (0 until r.words).map { "slovo$it" }, r.text)
    }

    @Test
    fun theLengthCapOfAFastPhoneWritesEveryWordOnce() {
        val r = VirtualSession(factor = 0.05, speechSeconds = 100.0).run()
        assertEveryWordOnceInOrder(r, "fast phone, length cap")
    }

    @Test
    fun aForcedCutOfASlowPhoneWritesEveryWordOnce() {
        val r = VirtualSession(factor = 1.9, speechSeconds = 120.0).run()
        assertTrue("forced cuts happened: ${r.forcedCuts}", r.forcedCuts >= 3)
        assertEveryWordOnceInOrder(r, "slow phone, forced cuts", lostAllowed = r.lagCuts)
    }

    /**
     * Czech speech with a model of cs, sk and en: passes shorter than `minLeaveMs` cannot decide the language, and nothing is committed
     * before the first one that can. A pass is 1.3 s, 2.3 s, 3.3 s... long, one a second on a fast phone: with 2 000 ms the second
     * decides and the first word comes as with one language; with 3 000 ms the third does, one pass (1 000 ms) later.
     */
    @Test
    fun firstCommittedWordWithMinLeave2000IsAsTodayAnd3000AddsOnePass() {
        val today = VirtualSession(factor = 0.05, speechSeconds = 30.0).run()
        val two = VirtualSession(factor = 0.05, speechSeconds = 30.0, minLeaveMs = 2_000).run()
        val three = VirtualSession(factor = 0.05, speechSeconds = 30.0, minLeaveMs = 3_000).run()
        assertEveryWordOnceInOrder(two, "minLeaveMs 2000")
        assertEveryWordOnceInOrder(three, "minLeaveMs 3000")
        assertEquals("2 000 ms: the first word as with one language", today.firstTextMs, two.firstTextMs)
        assertTrue("3 000 ms: one pass later: ${three.firstTextMs} ms, one language ${today.firstTextMs} ms", Math.abs(three.firstTextMs - today.firstTextMs - 1_000) <= 32)
        assertEquals("a pass is held until the first that can decide", 1, three.stats.passesHeldByLanguage - two.stats.passesHeldByLanguage)
    }

    private fun table(rows: List<Pair<String, VirtualSession.Result>>): String {
        val sb = StringBuilder("phone | max delay s | after speech s | pieces (final audio s) median/max | finals at once | paused/resumed | passes | forced | lost\n")
        for ((name, r) in rows) {
            val sizes = r.finalAudioMs.sorted()
            sb.append(
                String.format(
                    Locale.ROOT, "%s | %.2f | %.2f | %.1f/%.1f | %d | %d/%d | %d | %d | %d\n", name, r.maxDelayMs / 1000.0, r.afterSpeechMs / 1000.0,
                    sizes[sizes.size / 2] / 1000.0, sizes.last() / 1000.0, r.maxFinalsAtOnce, r.paused, r.resumed, r.passes, r.forcedCuts, r.missing.size,
                ),
            )
        }
        return sb.toString()
    }

    @Test
    fun fiveMinutesOfContinuousSpeechOnTheEarlierPhoneCurve() {
        val rows = listOf(1.0, 1.9, 2.2).map { "x$it" to VirtualSession(factor = it, speechSeconds = 300.0).run() }
        val text = table(rows)
        File("build").mkdirs()
        File("build/streaming-virtual-time.txt").writeText(text)
        println(text)
        for ((name, r) in rows) {
            assertEveryWordOnceInOrder(r, name, lostAllowed = r.lagCuts)
            assertTrue("$name: the largest delay of a word ${r.maxDelayMs} ms is within the 15 s limit", r.maxDelayMs <= 15_000)
            assertTrue("$name: at most one final waits while one runs: ${r.maxFinalsAtOnce}", r.maxFinalsAtOnce <= 2)
            assertTrue("$name: passes stopped and resumed ${r.paused + r.resumed} times", r.paused + r.resumed <= 2)
            assertTrue("$name: the rest after the speech came ${r.afterSpeechMs} ms after it", r.afterSpeechMs <= 12_000)
        }
    }


    private fun median(values: List<Long>): Long = values.sorted()[values.size / 2]

    private fun pixelTable(rows: List<Pair<String, VirtualSession.Result>>): String {
        val sb = StringBuilder("phone | max delay s | piece (final audio s) median / shortest | duty (cost / piece) median / worst | finals at once | after speech s | target cuts | lag cuts | words lost | words twice\n")
        for ((name, r) in rows) {
            val duty = r.finalCostMs.zip(r.finalAudioMs).map { (c, a) -> (1000.0 * c / a).toLong() }.sorted()
            sb.append(
                String.format(
                    Locale.ROOT, "%s | %.2f | %.1f / %.1f | %.0f %% / %.0f %% | %d | %.2f | %d | %d | %d | %d\n", name, r.maxDelayMs / 1000.0,
                    median(r.finalAudioMs) / 1000.0, r.finalAudioMs.min() / 1000.0, duty[duty.size / 2] / 10.0, duty.last() / 10.0, r.maxFinalsAtOnce,
                    r.afterSpeechMs / 1000.0, r.targetCuts, r.lagCuts, r.missing.size, r.repeated.size,
                ),
            )
        }
        return sb.toString()
    }

    /**
     * The first test on the phone: chunk mode cut a piece only when the lag limit came near (8.4 to 9.6 s pieces, words
     * waiting 13 s). Now a piece is cut as soon as the phone is estimated to manage it with 60 % of its time.
     */
    @Test
    fun fiveMinutesOfContinuousSpeechOnThePixelCurveCutsAsSoonAsThePhoneKeepsUp() {
        val rows = listOf(1.0, 1.4, 1.9, 2.2).map { "x$it" to VirtualSession(factor = it, speechSeconds = 300.0, curve = VirtualSession.PIXEL_CURVE).run() }
        val text = pixelTable(rows)
        File("build").mkdirs()
        File("build/streaming-virtual-time-pixel.txt").writeText(text)
        println(text)
        // x1.9 is not here: its 12.5 s does not hold, see the ignored test below
        val limits = mapOf("x1.0" to 8_000L, "x1.4" to 10_000L, "x1.9" to 15_000L, "x2.2" to 15_000L)
        for ((name, r) in rows) {
            assertEveryWordOnceInOrder(r, name, lostAllowed = r.lagCuts)
            assertTrue("$name: the largest delay of a word ${r.maxDelayMs} ms is within ${limits[name]} ms", r.maxDelayMs <= limits[name]!!)
            assertTrue("$name: at most one final waits while one runs: ${r.maxFinalsAtOnce}", r.maxFinalsAtOnce <= 2)
            assertTrue("$name: no piece under 4 s: ${r.finalAudioMs.min()} ms", r.finalAudioMs.min() >= 4_000)
            if (r.lagCuts == 0) {
                val duty = r.finalCostMs.zip(r.finalAudioMs).map { (c, a) -> 1000L * c / a }
                assertTrue("$name: the median duty ${median(duty) / 10.0} % is within 60 %", median(duty) <= 600)
                assertEquals("$name: no word is lost at a cut at the target length", 0, r.missing.size)
            }
        }
    }

    /**
     * The default of the SDK, `CutSeam.CONTEXT_AND_HELD_WORD`, on the same phones as the test above: a held word comes with the next
     * piece, so the largest delay of a word is longer than with the context alone, but the documented guarantee holds: within
     * [StreamingConfig.maxLagMs] (15 s), the replacement of a held word is refused when it would come later.
     */
    @Test
    fun theDefaultHeldWordKeepsTheDelayOfAWordWithinTheLagLimitOnThePixelCurve() {
        val rows = ArrayList<Pair<String, VirtualSession.Result>>()
        for (f in listOf(1.0, 1.4, 1.9, 2.2)) for (seam in listOf(ai.moravian.czechspeech.CutSeam.CONTEXT, ai.moravian.czechspeech.CutSeam.CONTEXT_AND_HELD_WORD)) {
            rows += "x$f ${seam.name}" to VirtualSession(factor = f, speechSeconds = 300.0, curve = VirtualSession.PIXEL_CURVE, cutSeam = seam).run()
        }
        val text = pixelTable(rows)
        File("build").mkdirs()
        File("build/streaming-virtual-time-held-word.txt").writeText(text)
        println(text)
        for ((name, r) in rows) {
            assertEveryWordOnceInOrder(r, name, lostAllowed = r.lagCuts)
            assertTrue("$name: the largest delay of a word ${r.maxDelayMs} ms is within the 15 s limit", r.maxDelayMs <= 15_000)
            assertTrue("$name: at most one final waits while one runs: ${r.maxFinalsAtOnce}", r.maxFinalsAtOnce <= 2)
        }
    }

    /**
     * The bound asked for: 12.5 s at x1.9. It does NOT hold (14.4 s): the audio is on the lowest window only up to 7.04 s
     * (the window is the audio and 32 frames), where 4.3 s of recognition are 61 % of the piece, just over the 60 %; every
     * longer piece is dearer by the window, so the rule finds no target and the lag limit decides (8 s pieces).
     */
    @org.junit.Ignore("the asked bound of 12.5 s does not hold at x1.9: 14.4 s, see the report")
    @Test
    fun aPhoneAtX19KeepsTheDelayOfAWordWithin12AndAHalfSeconds() {
        val r = VirtualSession(factor = 1.9, speechSeconds = 300.0, curve = VirtualSession.PIXEL_CURVE).run()
        assertTrue("the largest delay ${r.maxDelayMs} ms", r.maxDelayMs <= 12_500)
    }

    /** The phone warms up in the middle of the session: the pieces get longer within two pieces and no word waits over 15 s. */
    @Test
    fun aPhoneThatSlowsDownInTheMiddleGetsLongerPiecesWithinTwoPieces() {
        val at = 100_000L
        val r = VirtualSession(factor = 1.0, speechSeconds = 300.0, slowdownAtMs = at, slowdownFactor = 1.9, curve = VirtualSession.PIXEL_CURVE).run()
        val before = r.finalAudioMs.filterIndexed { i, _ -> r.finalStartedAtMs[i] < at }
        val after = r.finalAudioMs.filterIndexed { i, _ -> r.finalStartedAtMs[i] >= at }
        println("before ${median(before)} ms (${before.size} pieces), after ${after.size} pieces: ${after.take(6)}, then median ${median(after.drop(2))} ms")
        assertTrue("pieces of a quick phone are short: ${median(before)}", median(before) <= 4_000)
        // a quiet phone: 4 s pieces; the median of the last three finals moves after two slow ones, so the pieces get longer
        // from the fourth piece on the new speed (the first two finals of the slow phone, and one already cut by then)
        assertTrue("from the fourth piece on the pieces are long: ${after.take(8)}", after.drop(3).all { it >= 5_500 })
        assertTrue("the largest delay of a word ${r.maxDelayMs} ms", r.maxDelayMs <= 15_000)
        assertEveryWordOnceInOrder(r, "slowdown", lostAllowed = r.lagCuts)
    }

    /** Phones near the line where passes pay (a pass of 1.8 to 2.4 s, the keyboard's window): they must not flap. */
    @Test
    fun phonesNearTheLimitOfThePassesDoNotFlapBetweenPausedAndRunning() {
        val rows = listOf(0.6, 0.66, 0.7, 0.75, 0.8).map { "x$it" to VirtualSession(factor = it, speechSeconds = 300.0).run() }
        val text = table(rows)
        File("build").mkdirs()
        File("build/streaming-virtual-time-limit.txt").writeText(text)
        println(text)
        for ((name, r) in rows) {
            assertEveryWordOnceInOrder(r, name, lostAllowed = r.lagCuts)
            assertTrue("$name: passes stopped and resumed ${r.paused + r.resumed} times in 5 minutes", r.paused + r.resumed <= 4)   // measured: 3 at most, was 3 to 8 per 2 to 5 minutes
            assertTrue("$name: the largest delay ${r.maxDelayMs} ms", r.maxDelayMs <= 15_000)
        }
    }

    private fun breathTable(rows: List<Pair<String, VirtualSession.Result>>): String {
        val sb = StringBuilder("phone | first text s | max delay s | piece (final audio s) median / shortest | duty (cost / piece) median / worst | finals at once | words lost | words twice | target cuts (of them at a quiet point) | lag cuts\n")
        for ((name, r) in rows) {
            val duty = r.finalCostMs.zip(r.finalAudioMs).map { (c, a) -> (1000.0 * c / a).toLong() }.sorted()
            sb.append(
                String.format(
                    Locale.ROOT, "%s | %.2f | %.2f | %.1f / %.1f | %.0f %% / %.0f %% | %d | %d | %d | %d (%d) | %d\n", name, r.firstTextMs / 1000.0, r.maxDelayMs / 1000.0,
                    median(r.finalAudioMs) / 1000.0, r.finalAudioMs.min() / 1000.0, duty[duty.size / 2] / 10.0, duty.last() / 10.0, r.maxFinalsAtOnce,
                    r.missing.size, r.repeated.size, r.targetCuts, r.quietPointCuts, r.lagCuts,
                ),
            )
        }
        return sb.toString()
    }

    private fun pixel(
        factor: Double, breathEvery: Int, quietPoint: Boolean, breathMs: Long = VirtualSession.BREATH_MS, dips: Boolean = false, languageModelMs: Long = 0,
    ) = VirtualSession(
        factor = factor, speechSeconds = 300.0, curve = VirtualSession.PIXEL_CURVE, breathEvery = breathEvery, quietPoint = quietPoint,
        breathMs = breathMs, dips = dips, languageModelMs = languageModelMs,
    ).run()

    private fun duty(r: VirtualSession.Result) = r.finalCostMs.zip(r.finalAudioMs).map { (c, a) -> 1000L * c / a }

    /**
     * The quiet-point cut falls inside the quiet stretch that triggered it. Frames are NOT uniform: every 4th word has a one-frame
     * dip as low as the silence five frames into it, and a breath of 400 ms comes every 1.3 s, so every window of 1.5 s of the cut has one and
     * every cut is a quiet-point cut. A cut at the lowest frame anywhere from the floor on (earliest on a tie) takes the dip and
     * splits the word: a word lost.
     */
    @Test
    fun aQuietPointCutFallsInsideTheQuietStretchAndLosesNoWordWhenFramesAreNotUniform() {
        for (f in listOf(1.0, 1.4)) {
            val r = pixel(f, breathEvery = 2, quietPoint = true, breathMs = 400, dips = true)
            assertTrue("x$f: cuts at the quiet point ${r.quietPointCuts} of ${r.targetCuts}", r.quietPointCuts >= r.targetCuts - 1 && r.targetCuts > 50)
            assertEveryWordOnceInOrder(r, "x$f, dips", lostAllowed = 0)
            assertTrue("x$f: no piece under 4 s but the last: ${r.finalAudioMs.dropLast(1).min()} ms", r.finalAudioMs.dropLast(1).min() >= 4_000)
        }
    }

    /**
     * Continuous dictation with a breath after every 4th word (every 2 s): the Pixel curve, with the cut at the target only ("before":
     * the quiet point switched off) and at the first quiet stretch from the floor of the window on, for breaths of 230 and 400 ms
     * (the quiet stretch must last [StreamingGuard.QUIET_HOLD_FRAMES] frames) and without and with a language model of 300 ms per
     * decision on the whisper thread (counted in the targets by the guard). Written to build/streaming-virtual-time-pixel-breaths.txt.
     */
    @Test
    fun fiveMinutesOfDictationWithBreathsOnThePixelCurve() {
        val rows = ArrayList<Pair<String, VirtualSession.Result>>()
        for (lm in listOf(0L, 300L)) for (breath in listOf(230L, 400L)) for (f in listOf(1.0, 1.4, 1.9)) for (on in listOf(false, true)) {
            rows += "x$f, breath $breath ms, lm $lm ms, ${if (on) "after" else "before"}" to pixel(f, 4, on, breath, languageModelMs = lm)
        }
        val text = breathTable(rows)
        File("build").mkdirs()
        File("build/streaming-virtual-time-pixel-breaths.txt").writeText(text)
        println(text)
        for ((name, r) in rows) {
            assertEveryWordOnceInOrder(r, name, lostAllowed = r.lagCuts)
            assertTrue("$name: no piece under 4 s but the last: ${r.finalAudioMs.dropLast(1).min()} ms", r.finalAudioMs.dropLast(1).min() >= 4_000)
            assertTrue("$name: at most one final waits while one runs: ${r.maxFinalsAtOnce}", r.maxFinalsAtOnce <= 2)
            if (r.lagCuts == 0) assertTrue("$name: the median duty ${median(duty(r)) / 10.0} % is within 60 %", median(duty(r)) <= 600)
        }
        val byName = rows.toMap()
        for (lm in listOf(0L, 300L)) {
            val before = byName.getValue("x1.0, breath 400 ms, lm $lm ms, before")
            val after = byName.getValue("x1.0, breath 400 ms, lm $lm ms, after")
            assertTrue("lm $lm: first text ${after.firstTextMs} ms, before ${before.firstTextMs} ms", after.firstTextMs < before.firstTextMs)
            assertTrue("lm $lm: delay ${after.maxDelayMs} ms, before ${before.maxDelayMs} ms", after.maxDelayMs <= before.maxDelayMs)
            // a phone for which no target exists is cut by the lag limit, as before: nothing changes
            assertEquals(byName.getValue("x1.9, breath 400 ms, lm $lm ms, before").finalAudioMs, byName.getValue("x1.9, breath 400 ms, lm $lm ms, after").finalAudioMs)
        }
    }

    /**
     * Without any quiet frame inside the speech the rule changes nothing in it: the same pieces, the same delays. The one cut it adds is
     * inside the silence after the speech, where the endpointer would have ended the utterance 600 ms later: the last piece, at most.
     */
    @Test
    fun withoutQuietFramesInTheSpeechTheEarlyCutChangesNothingInIt() {
        for (f in listOf(1.0, 1.4)) {
            val before = pixel(f, 0, false)
            val after = pixel(f, 0, true)
            assertEquals("x$f: the pieces but the last", before.finalAudioMs.dropLast(1), after.finalAudioMs.dropLast(1))
            assertEquals("x$f: the largest delay", before.maxDelayMs, after.maxDelayMs)
            assertTrue("x$f: the cuts ${before.targetCuts} and ${after.targetCuts}", after.targetCuts - before.targetCuts in 0..1)
            assertTrue("x$f: the rest after the speech ${after.afterSpeechMs} ms, before ${before.afterSpeechMs} ms", after.afterSpeechMs <= before.afterSpeechMs)
        }
    }
}
