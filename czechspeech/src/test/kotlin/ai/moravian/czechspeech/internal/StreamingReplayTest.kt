package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

/**
 * Replays every pass of the measurement's simulation (`scripts/streaming/export_replay.py passes`) through
 * [StreamCommitter] and demands the simulation's decisions: how many words of each pass were the committed ones still
 * in the buffer, how many were newly committed, whether the buffer was cut (and that the cut is one the simulation
 * made, at a point inside the range asked for), which word each seam resolved to, and in the end the same committed
 * text.
 * Where the recording is at hand the quiet point is computed by [QuietPoint] too and compared with the recorded cut.
 * Skipped (naming the variable) without `CZECHSPEECH_STREAMING_WORK` and the exported file. A synthetic sample in the
 * same format (`src/test/resources/streaming/replay-synthetic.tsv`, made by `make_synthetic_replay.py` from the
 * simulation's own code over invented words, a fake recogniser and no recordings) is replayed always.
 */
class StreamingReplayTest {
    private class Record(val kind: String, val bufStart: Long, val end: Long, val m: Int, val n: Int, val text: String)

    private class Utterance(val sid: String, val cond: String, val index: Int, val a0: Long, val words: String) {
        val records = ArrayList<Record>()

        /** pass index (among the records) -> the seam word the following record resolved, from the simulation's `seams`. */
        val seams = HashMap<Int, Int>()
    }

    private fun load(file: File): List<Utterance> {
        val result = ArrayList<Utterance>()
        file.forEachLine(Charsets.UTF_8) { line ->
            val f = line.split('\t', limit = 7)
            when (f[0]) {
                "U" -> result.add(Utterance(f[1], f[2], f[3].toInt(), f[4].toLong(), f.getOrElse(5) { "" }))
                "S" -> result.last().seams[f[2].toInt()] = f[1].toInt()
                "P" -> result.last().records.add(
                    Record(f[1], f[2].toLong(), f[3].toLong(), f[4].toInt(), f[5].toInt(), f.getOrElse(6) { "" }),
                )
            }
        }
        return result
    }

    private fun ms(samples: Long, what: String): Long {
        assertEquals("$what ($samples samples) is a whole number of ms", 0L, samples % 16)
        return samples / 16
    }

    private fun replayExported(cfg: String) {
        val file = File(File(TestEnv.streamingWork, "sdk"), "replay-$cfg.tsv")
        TestEnv.assumeFile(file, "replay data $cfg (CZECHSPEECH_STREAMING_WORK, scripts/streaming/export_replay.py)")
        replay(cfg, load(file), withRecordings = true)
    }

    private fun replay(cfg: String, utterances: List<Utterance>, withRecordings: Boolean) {
        var passes = 0
        var finals = 0
        var cuts = 0
        var quietChecked = 0
        var quietExact = 0
        var quietFar = 0
        val audio = HashMap<String, FloatArray?>()
        for (u in utterances) {
            val where = "$cfg ${u.sid} ${u.cond} utterance ${u.index}"
            val wav = if (withRecordings) File(TestEnv.streamingWork, "wav/${u.cond}/${u.sid}.wav") else null
            val samples = if (wav != null && wav.exists()) audio.getOrPut(wav.path) { Wav.read(wav) } else null
            val committer = StreamCommitter(ms(u.a0, "$where a0"))
            val committedWords = ArrayList<String>()
            for ((i, r) in u.records.withIndex()) {
                val at = "$where record $i (${r.kind}, buffer ${r.bufStart}..${r.end})"
                val start = ms(r.bufStart, at)
                val nextStart = u.records.getOrNull(i + 1)?.bufStart
                if (r.kind == "final") {
                    val outcome = committer.absorbFinal(r.text)
                    assertEquals("$at: prefix", r.m, outcome.prefix)
                    assertEquals("$at: words", r.n, outcome.words.size)
                    assertEquals("$at: seam word", u.seams[i - 1], outcome.seamWord)
                    committedWords += outcome.words
                    finals++
                    continue
                }
                val recordedCut = nextStart != null && nextStart != r.bufStart
                val outcome = committer.absorbPass(start, ms(r.end, at), r.text) { lo, hi ->
                    // the recorded cut decides; ours is only compared
                    if (!recordedCut) fail("$at: a cut the simulation did not make was asked for")
                    val recorded = ms(nextStart!!, at)
                    // the simulation's quiet point is the centre of a 30 ms window inside the range
                    assertTrue("$at: the recorded cut $recorded lies in [${lo + 15}, ${hi - 15}]", recorded in lo + 15..hi - 15)
                    if (samples != null) {
                        quietChecked++
                        val ours = QuietPoint.find(samples, (lo * 16).toInt(), (hi * 16).toInt()) / 16L
                        if (ours == recorded) quietExact++
                        if (Math.abs(ours - recorded) > 10) quietFar++
                    }
                    recorded
                }
                assertEquals("$at: prefix", r.m, outcome.prefix)
                assertEquals("$at: committed", r.n, outcome.committed.size)
                assertEquals("$at: seam word", u.seams[i - 1], outcome.seamWord)
                assertEquals("$at: cut decided", recordedCut, outcome.cut != null)
                if (recordedCut) {
                    assertEquals("$at: cut at", ms(nextStart!!, at), outcome.cut!!.atMs)
                    cuts++
                }
                committedWords += outcome.committed
                passes++
            }
            assertEquals("$where: committed text", u.words, committedWords.joinToString(" "))
        }
        println(
            "replay $cfg: utterances=${utterances.size} passes=$passes finals=$finals cuts=$cuts " +
                "quiet points checked=$quietChecked exact=$quietExact farther than 10 ms=$quietFar",
        )
        assertTrue("passes were replayed", passes > 0 && cuts > 0)
        assertTrue("quiet points within 10 ms in >= 99.5 % of cuts", quietFar <= quietChecked / 200)
    }

    @Test
    fun theCommitterMakesTheSimulationsDecisionsOnEveryPassOfBPauza() = replayExported("b-pauza")

    @Test
    fun theCommitterMakesTheSimulationsDecisionsAtThe2500msPassToo() = replayExported("b-pauza+t2.5")

    @Test
    fun theCommitterMakesTheSimulationsDecisionsOnTheSyntheticSample() {
        val utterances = load(File("src/test/resources/streaming/replay-synthetic.tsv"))
        assertTrue("seams were recorded", utterances.sumOf { it.seams.size } > 20)
        assertTrue("an empty pass is in the sample", utterances.any { u -> u.records.any { it.kind == "pass" && it.text.isEmpty() } })
        replay("synthetic", utterances, withRecordings = false)
    }
}
