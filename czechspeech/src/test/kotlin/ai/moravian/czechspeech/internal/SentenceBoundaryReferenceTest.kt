package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.internal.SentenceBoundary.Choice
import ai.moravian.czechspeech.internal.SentenceBoundary.Form
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The port against the original: src/test/resources/sentences/reference-boundaries.tsv holds the decisions of the
 * measurement's research/dekodovani/hranice.py (scripts/sentences/make_reference.py) on invented boundaries with fixed
 * candidate scores, one row per boundary and score set. Each must come out the same here: the form of the boundary,
 * the language model's texts, the margin, the decision and the texts a client ends up with.
 */
class SentenceBoundaryReferenceTest {
    private class Row(
        val a: String,
        val b: String,
        val set: String,
        val form: Form,
        val scores: Map<Choice, Double>,
        val choice: Choice,
        val margin: Double?,
        val newA: String,
        val newB: String,
        val texts: Map<Choice, String>,
    ) {
        override fun toString() = "set $set: $a | $b"
    }

    private val forms = mapOf(
        "PU" to Form.PERIOD_UPPER, "PL" to Form.PERIOD_LOWER, "NU" to Form.UPPER, "CU" to Form.COMMA_UPPER,
        "other" to Form.OTHER,
    )
    private val choices = mapOf(
        "split" to Choice.SPLIT, "join" to Choice.JOIN, "join_keep" to Choice.JOIN_KEEP, "comma" to Choice.COMMA,
        "raw" to Choice.RAW,
    )
    private val textColumns = listOf(Choice.SPLIT, Choice.JOIN_KEEP, Choice.JOIN, Choice.COMMA)

    private val allRows: List<Row> = File(TestEnv.lmResources.parentFile, "sentences/reference-boundaries.tsv")
        .readLines(Charsets.UTF_8)
        .filter { it.isNotEmpty() }
        .map { line ->
            val c = line.split('\t')
            check(c.size == 13) { "13 columns: $line" }
            Row(
                a = c[0], b = c[1], set = c[2], form = forms.getValue(c[3]),
                scores = c[4].split(',').filter { it.isNotEmpty() }.associate { kv ->
                    val (k, v) = kv.split('=')
                    choices.getValue(k) to v.toDouble()
                },
                choice = choices.getValue(c[5]),
                margin = c[6].takeIf { it != "-" }?.toDouble(),
                newA = c[7], newB = c[8],
                texts = textColumns.zip(c.subList(9, 13)).filter { it.second != "-" }.toMap(),
            )
        }

    /**
     * The measured decisions are for the boundaries the measurement knew: a previous text ending in an ellipsis was "anything
     * else, left as it is" there and is decided by the language model now (ellipsis rows: see [SentenceBoundaryTest]).
     */
    private val rows: List<Row> = allRows.filter { SentenceBoundary.words(it.a).lastOrNull()?.trimEnd { c -> c in "\"'“”„»«)]" }?.let { w -> w.endsWith("…") || w.endsWith("..") } != true }

    @Test
    fun theReferenceCoversEveryFormAndDecision() {
        assertEquals(176, allRows.size)
        assertEquals("16 rows end in an ellipsis", 160, rows.size)
        assertEquals(Form.entries.toSet() - setOf(Form.ELLIPSIS, Form.COMMA_LOWER), rows.map { it.form }.toSet())
        assertEquals(Choice.entries.toSet() - setOf(Choice.ELLIPSIS, Choice.UNCOMMA, Choice.UNCOMMA_KEEP), rows.map { it.choice }.toSet())
    }

    @Test
    fun formsAndLanguageModelTextsAreTheMeasurementsOnes() {
        for (r in rows) {
            assertEquals("$r", r.form, SentenceBoundary.form(r.a, r.b))
            val texts = SentenceBoundary.languageModelTexts(r.a, r.b)
            when (r.form) {
                Form.OTHER -> assertTrue("$r", texts.isEmpty())
                Form.UPPER -> assertEquals("$r", r.texts.filterKeys { it == Choice.JOIN || it == Choice.JOIN_KEEP }, texts)
                // "A, B": the measurement's two texts, exactly, and the two without the comma that the SDK added: the same texts
                // with the comma at the end of A's window taken out
                Form.COMMA_UPPER -> {
                    val measured = r.texts.filterKeys { it == Choice.JOIN || it == Choice.JOIN_KEEP }
                    val comma = SentenceBoundary.words(r.a).takeLast(SentenceBoundary.WORDS_BEFORE).joinToString(" ").length - 1
                    val expected = measured + mapOf(
                        Choice.UNCOMMA to measured.getValue(Choice.JOIN).removeRange(comma, comma + 1),
                        Choice.UNCOMMA_KEEP to measured.getValue(Choice.JOIN_KEEP).removeRange(comma, comma + 1),
                    )
                    assertEquals("$r", expected, texts)
                }
                else -> assertEquals("$r", r.texts, texts)
            }
        }
    }

    @Test
    fun marginsAndDecisionsAreTheMeasurementsOnes() {
        for (r in rows) {
            val margin = SentenceBoundary.margin(r.scores)
            if (r.margin == null) assertEquals("$r", null, margin) else assertEquals("$r", r.margin, margin!!, 0.0)
            assertEquals("$r", r.choice, SentenceBoundary.decide(r.form, r.scores))
            // the whole way: the texts the SDK scores, the scores the measurement gave those texts, the decision
            // the texts without the comma that the SDK adds for "A, B" are scored 10 below their text with it: the path is exercised and
            // the decision is the measured one (a comma goes only by a margin)
            val scoreOf = r.texts.entries.associate { (choice, text) -> text to r.scores.getValue(choice) }
            val ask = SentenceBoundary.languageModelTexts(r.a, r.b)
            val extra = ask.filterKeys { it == Choice.UNCOMMA || it == Choice.UNCOMMA_KEEP }.entries.associate { (choice, text) ->
                text to r.scores.getValue(if (choice == Choice.UNCOMMA) Choice.JOIN else Choice.JOIN_KEEP) - 10.0
            }
            val decided = SentenceBoundary.decide(r.a, r.b) { asked -> DoubleArray(asked.size) { (scoreOf + extra).getValue(asked[it]) } }
            assertEquals("$r", r.choice, decided)
        }
    }

    @Test
    fun theClientEndsUpWithTheMeasurementsTexts() {
        for (r in rows) {
            val join = SentenceBoundary.sentenceJoin(r.a, r.b, r.choice)
            val (newA, newB) = when {
                join == null -> r.a to r.b
                join.continuesPrevious -> join.previousEnd.applyTo(r.a) to join.continuingText
                else -> r.a to join.standaloneText
            }
            assertEquals("$r", r.newA to r.newB, newA to newB)
        }
    }

    /**
     * The only evidence of the repository for the quiet-cut prior: the 160 boundaries of the reference (invented texts, fixed
     * candidate scores, 8 score sets per text: from "the model clearly prefers the split" to "clearly prefers a join"), decided as
     * after a pause and as after a quiet cut. There is no reference punctuation for them (no "true" comma or period), so no count
     * of commas added or removed against a reference can be given; the table only shows what the prior changes. Written to
     * build/sentence-boundary-quiet.txt.
     */
    @Test
    fun theQuietCutPriorMovesSplitsToGoingOnAndTheCommaNeedsAMargin() {
        val boundaries = rows.filter { it.form == Form.PERIOD_UPPER || it.form == Form.PERIOD_LOWER }
        fun count(quiet: Boolean, commaMargin: Boolean): Map<String, Int> {
            val out = linkedMapOf("split" to 0, "comma" to 0, "plain" to 0)
            for (r in boundaries) {
                val choice = SentenceBoundary.decide(r.form, r.scores, quiet, if (commaMargin) SentenceBoundary.COMMA_OVER_PLAIN_MARGIN else 0.0)
                out[when (choice) { Choice.SPLIT -> "split"; Choice.COMMA -> "comma"; else -> "plain" }] = out.getValue(when (choice) { Choice.SPLIT -> "split"; Choice.COMMA -> "comma"; else -> "plain" }) + 1
            }
            return out
        }
        val pause = count(quiet = false, commaMargin = false)
        val quietNoMargin = count(quiet = true, commaMargin = false)
        val quiet = count(quiet = true, commaMargin = true)
        val text = "after a pause (measured): $pause\nquiet cut, comma without a margin: $quietNoMargin\nquiet cut, comma needs ${SentenceBoundary.COMMA_OVER_PLAIN_MARGIN} over plain: $quiet\n"
        File("build").mkdirs()
        File("build/sentence-boundary-quiet.txt").writeText(text)
        println(text)
        assertEquals(boundaries.size, pause.values.sum())
        assertTrue(text, quiet.getValue("split") < pause.getValue("split"))
        assertTrue(text, quiet.getValue("comma") <= quietNoMargin.getValue("comma"))
        assertEquals("the same boundaries stay joined or split but for the comma", quietNoMargin.getValue("split"), quiet.getValue("split"))
    }

    /**
     * The 16 rows of the reference that end in an ellipsis ("To bylo všechno... | Další bod.", "Bylo to tak… | Že jsme odešli.", 8 score
     * sets each) were "left as it is" by the measurement and are decided by the language model now. Each row's NEW intended decision, by
     * score set (the sets have no score for the ellipsis kept, which the measurement did not know): the best removal wins unless the new
     * sentence beats it by more than 2.0 (set 4), a comma needs 1.5 over the plain continuation (set 3: 2.0), a capital stays when the small
     * letter does not win by more than 1.0 (sets 1 and 5). A later change of a margin shows here, row by row.
     */
    @Test
    fun theEllipsisRowsOfTheReferenceHaveTheirNewIntendedDecisions() {
        val ellipsisRows = allRows.filter { it !in rows }
        assertEquals(16, ellipsisRows.size)
        val expected = mapOf(
            "0" to Choice.JOIN, "1" to Choice.JOIN_KEEP, "2" to Choice.JOIN, "3" to Choice.COMMA,
            "4" to Choice.SPLIT, "5" to Choice.JOIN_KEEP, "6" to Choice.JOIN, "7" to Choice.JOIN,
        )
        for (r in ellipsisRows) {
            assertEquals("$r", Form.ELLIPSIS, SentenceBoundary.form(r.a, r.b))
            val choice = SentenceBoundary.decide(Form.ELLIPSIS, r.scores)
            assertEquals("$r", expected.getValue(r.set), choice)
            val join = SentenceBoundary.sentenceJoin(r.a, r.b, choice)
            when (choice) {
                Choice.SPLIT -> assertEquals("$r: the capital is already there", null, join)
                Choice.COMMA -> assertEquals("$r", PreviousEnd.ELLIPSIS_TO_COMMA, join!!.previousEnd)
                else -> assertEquals("$r", PreviousEnd.REMOVE_ELLIPSIS, join!!.previousEnd)
            }
        }
    }
}
