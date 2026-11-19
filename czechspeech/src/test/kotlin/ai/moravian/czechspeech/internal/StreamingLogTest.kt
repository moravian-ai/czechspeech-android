package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.StreamingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger

/**
 * The log lines and the statistics of a streaming session: counts and times, never text. Speech is the clock speech of
 * [StreamingSessionTest] (words `slovo<k>`), so a line that quoted a word would show.
 */
class StreamingLogTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val streaming = CzechSpeechConfig(streaming = StreamingConfig())

    private fun speech(seconds: Double) = Signals.concat(Signals.zeros(0.5), Signals.clock(500, seconds), Signals.zeros(1.5))

    private fun clockWords(seconds: Double): (Long, Long) -> String = { startMs, endMs ->
        (0 until (seconds * 2).toInt()).map { 500L + 500L * it }
            .filter { it + 400 <= endMs && it >= startMs - 50 }
            .joinToString(" ") { "slovo${(it - 500) / 500}" }
    }

    /** Capture waits for every pass to be done; keeps what [streamingEnded] was told. */
    private class Paced : SpeechEngine.Hooks {
        val requested = AtomicInteger()
        val finished = AtomicInteger()
        val ended: MutableList<StreamingStats> = Collections.synchronizedList(mutableListOf())

        override fun streamingPassRequested(samples: FloatArray) {
            val n = requested.incrementAndGet()
            val deadline = System.nanoTime() + 10_000_000_000L
            while (finished.get() < n) {
                if (System.nanoTime() > deadline) throw AssertionError("timed out waiting for pass $n")
                Thread.sleep(1)
            }
        }

        override fun streamingPassFinished() {
            finished.incrementAndGet()
        }

        override fun streamingEnded(stats: StreamingStats) {
            ended += stats
        }
    }

    private fun session(native: FakeNative, hooks: Paced, seconds: Double, config: CzechSpeechConfig = streaming): List<String> {
        val lines = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> lines += message }
        try {
            val engine = fakeEngine(native, hooks, config = config) { ArrayAudioSource(speech(seconds)) }
            runBlocking { withContext(Dispatchers.Default) { engine.listen().collect { } } }
        } finally {
            CsLog.sink = sink
        }
        return lines.toList()
    }

    @Test
    fun everyPassHasOneLineAndTheSessionASummaryWithoutText() {
        val native = FakeNative().apply { clockWords = clockWords(14.0) }
        val lines = session(native, Paced(), 14.0)
        val passes = lines.filter { it.startsWith("pass: ") }
        assertEquals(native.previewCalls.get(), passes.size)
        assertTrue(passes.size >= 5)
        val shape = Regex("""^pass: \d+ ms of audio from \d+ ms, (recogniser \d+ ms|recogniser cancelled after \d+ ms), window \d+ frames, .*""")
        assertTrue(passes.toString(), passes.all { shape.matches(it) })
        assertTrue("words committed are counted: ${passes.last()}", passes.any { Regex("""; \d+ word\(s\) committed, \d+ pending""").containsMatchIn(it) })
        assertEquals(1, lines.count { it.startsWith("streaming: utterance of ") })
        assertEquals(1, lines.count { Regex("""^streaming: \d+ passes""").containsMatchIn(it) })
        assertTrue("no text in the log", lines.none { "slovo" in it })
    }

    @Test
    fun theSessionLineAndTheStatsAgree() {
        val native = FakeNative().apply { clockWords = clockWords(14.0) }
        val hooks = Paced()
        val lines = session(native, hooks, 14.0)
        val stats = hooks.ended.single()
        val line = lines.single { Regex("""^streaming: \d+ passes""").containsMatchIn(it) }
        val m = Regex(
            """^streaming: (\d+) passes \((\d+) aborted, (\d+) ms of recogniser time\), (\d+) transcripts, (\d+) \+ (\d+) words, """ +
                """(\d+) cuts, (\d+) forced cuts, passes paused (\d+) times, buffer up to (\d+) ms, """ +
                """median piece (\d+) ms, text after its audio median (\d+) ms, max (\d+) ms.*""",
        ).matchEntire(line)
        assertNotNull(line, m)
        val g = m!!.groupValues.drop(1).map { it.toLong() }
        assertEquals(
            listOf(stats.passes, stats.abortedPasses, stats.transcripts, stats.wordsWhileSpeaking, stats.wordsAfter, stats.cuts, stats.forcedCuts, stats.pauses)
                .map { it.toLong() },
            listOf(g[0], g[1], g[3], g[4], g[5], g[6], g[7], g[8]),
        )
        assertEquals(stats.passRecogniserMs, g[2])
        assertEquals(stats.maxBufferMs, g[9])
        assertTrue("passes ran: $stats", stats.passes >= 5 && stats.transcripts >= 2)
        // the situations the algorithm does not guard are in the line, as counts
        assertTrue(line, Regex(""", \d+ seam\(s\) decided by a final, \d+ empty final\(s\) \(\d+ pending word\(s\) lost\), \d+ stale pass\(es\), \d+ error\(s\)""").containsMatchIn(line))
    }

    @Test
    fun aForcedCutAndAPauseAreLogged() {
        // no two passes agree: the buffer grows to 12 s and is cut there
        val noAgreement = FakeNative().apply {
            clockWords = clockWords(20.0)
            previewOverride = { i, _ -> "x$i" }
        }
        val forced = session(noAgreement, Paced(), 20.0)
        assertEquals(forced.toString(), 1, forced.count { it.startsWith("streaming: forced cut at ") })
        assertTrue(forced.any { Regex("""^streaming: forced cut at \d+ ms, \d+ ms not committed, closing estimated at \d+ ms \(lag limit 15000 ms\), reason: lag limit 15000 ms""").matches(it) })

        // a slow recogniser: the first pass is measured at about 2.2 s for the 7 680 ms window, passes pause
        val slow = FakeNative().apply {
            clockWords = clockWords(18.0)
            recognitionMs = 2_200
        }
        // pinned to NONE: this is about when chunk mode cuts (the target length, the lag limit); with the context the lag rule counts 2.5 s more
        // audio in the closing recognition and cuts a piece the pending finals kept waiting sooner (StreamingGuardContextTest and StreamingTrackerSeamTest cover that)
        val lines = session(slow, Paced(), 18.0, CzechSpeechConfig(streaming = StreamingConfig(cutSeam = ai.moravian.czechspeech.CutSeam.NONE)))
        assertEquals(lines.toString(), 1, lines.count { Regex("""^streaming: passes paused, a pass over 6 s would take 2\d{3} ms""").matches(it) })
        assertEquals("no pass after the pause", 1, lines.count { it.startsWith("pass: ") })
        // chunk mode cuts at the target length (5.5 s: a final takes 2.2 s, pieces of 4 to 5.5 s), not only at the lag limit
        val cuts = lines.filter { it.startsWith("streaming: forced cut at ") }
        assertTrue(cuts.toString(), cuts.size in 2..4)           // 18 s of speech in pieces of 4 to 5.5 s
        assertTrue(cuts.toString(), cuts.all { Regex(""", reason: target \d+ ms, duty \d+ %$""").containsMatchIn(it) })
    }
    @Test
    fun theSlowPhoneCutsAtTheTargetOrTheLagLimitAlsoWithTheDefaultContext() {
        // the twin of the part of aForcedCutAndAPauseAreLogged that is pinned to NONE: the same slow recogniser, the default cutSeam
        val slow = FakeNative().apply {
            clockWords = clockWords(18.0)
            recognitionMs = 2_200
        }
        val lines = session(slow, Paced(), 18.0)
        val cuts = lines.filter { it.startsWith("streaming: forced cut at ") }
        assertTrue(cuts.toString(), cuts.size in 2..4)
        assertTrue(cuts.toString(), cuts.all { Regex(""", reason: (target \d+ ms, duty \d+ %|lag limit \d+ ms)""").containsMatchIn(it) })
        assertTrue("no utterance was skipped: $lines", lines.none { it.startsWith("utterance skipped") })
    }
}
