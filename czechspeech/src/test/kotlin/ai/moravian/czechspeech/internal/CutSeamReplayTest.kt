package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Replays the anchor decisions the measurement stored (`zbytek_po_strihu.py export-anchor`: one row per tail and arm,
 * L2.5 and H1) through [SeamAnchor] and demands the same decision: how many delivered words the context repeats, how many
 * leading words of its text they take, whether the anchor is accepted, and the text after it. Hypotheses stored, no decoding.
 * The real data is skipped (naming the variable) without `CZECHSPEECH_STREAMING_WORK` and `sdk/replay-anchor.tsv` in it;
 * a synthetic sample in the same format (`src/test/resources/streaming/replay-anchor-synthetic.tsv`, made by
 * `make_synthetic_anchor_replay.py` from the measurement's own functions over invented words) is replayed always.
 */
class CutSeamReplayTest {
    private class Row(
        val id: String, val arm: String, val pieceWords: List<String>, val pieceMs: Long, val held: Int,
        val hyp: List<String>, val inBuffer: Int, val prefix: Int, val ok: Boolean, val rest: List<String>,
    )

    private fun words(s: String) = s.split(' ').filter { it.isNotEmpty() }

    private fun load(file: File): List<Row> = file.readLines(Charsets.UTF_8).filter { it.isNotBlank() }.map { line ->
        val f = line.split('\t')
        Row(f[0], f[1], words(f[5]), f[3].toLong(), f[4].toInt(), words(f[6]), f[7].toInt(), f[8].toInt(), f[9] == "1", words(f.getOrElse(10) { "" }))
    }

    /** The row's number (from 1), its id and arm for every difference. */
    private fun replay(rows: List<Row>): List<String> {
        val mismatches = ArrayList<String>()
        for ((i, r) in rows.withIndex()) {
            val delivered = r.pieceWords.dropLast(r.held)
            val expected = SeamAnchor.expected(r.pieceWords.size, r.pieceMs, r.held)
            val d = SeamAnchor.decide(r.hyp, delivered, expected)
            val same = d.inBuffer == r.inBuffer && d.prefix == r.prefix && d.ok == r.ok &&
                (!d.ok || r.hyp.drop(d.prefix) == r.rest)
            if (!same) {
                mismatches += "row ${i + 1} ${r.id} ${r.arm}: in_buf ${d.inBuffer}/${r.inBuffer}, prefix ${d.prefix}/${r.prefix}, " +
                    "ok ${d.ok}/${r.ok} (Kotlin/stored)"
            }
        }
        println("CutSeamReplayTest: ${rows.size} decisions, ${mismatches.size} mismatches")
        mismatches.take(10).forEach(::println)
        return mismatches
    }

    @Test
    fun theSyntheticSampleIsReplayedExactly() {
        val file = File("src/test/resources/streaming/replay-anchor-synthetic.tsv")
        assertTrue("$file exists", file.isFile)
        val rows = load(file)
        assertTrue("both arms, accepted and refused: ${rows.size} rows", rows.size >= 40 && rows.any { it.ok } && rows.any { !it.ok } && rows.any { it.held == 1 })
        assertEquals(emptyList<String>(), replay(rows))
    }

    @Test
    fun theStoredDecisionsOfTheMeasurementAreReplayedExactly() {
        val file = File(File(TestEnv.streamingWork, "sdk"), "replay-anchor.tsv")
        TestEnv.assumeFile(file, "anchor decisions (CZECHSPEECH_STREAMING_WORK, zbytek_po_strihu.py export-anchor)")
        val rows = load(file)
        assertEquals("392 decisions in the measurement", 392, rows.size)
        assertEquals(emptyList<String>(), replay(rows))
    }
}
