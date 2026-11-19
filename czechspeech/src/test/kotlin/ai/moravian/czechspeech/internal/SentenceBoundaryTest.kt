package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import ai.moravian.czechspeech.internal.SentenceBoundary.Choice
import ai.moravian.czechspeech.internal.SentenceBoundary.Form
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The rules of sentence joining without the engine. The first tests are those of the measurement
 * (of the measurement's own code), so the port decides as the measured code did.
 */
class SentenceBoundaryTest {
    private val b = SentenceBoundary

    // --- the measurement's tests, ported ---------------------------------------------------------------------

    @Test
    fun lowerFirstKeepsAcronymsAndLowersPlainWords() {
        assertEquals("spojit", b.lowerFirst("Spojit"))
        assertEquals("„ano", b.lowerFirst("„Ano"))
        assertEquals("ODS", b.lowerFirst("ODS"))
        assertEquals("ČR", b.lowerFirst("ČR"))
        assertEquals("2013", b.lowerFirst("2013"))
        assertEquals("Že", b.upperFirst("že"))
    }

    @Test
    fun aJoinableEndNeedsOnePeriodThatIsNoOrdinalNorAbbreviation() {
        fun joinable(a: String, bb: String) = b.form(a, bb) == Form.PERIOD_UPPER
        assertTrue(joinable("ale můžeš nějak.", "S tebou spojit"))
        assertTrue(joinable("v roce.", "2013 jsme"))
        assertFalse(joinable("ale můžeš nějak", "S tebou"))
        assertFalse(joinable("ale můžeš nějak...", "S tebou"))
        assertFalse(joinable("opravdu?", "Ano"))
        assertFalse(joinable("dne 17.", "Listopadu"))
        assertFalse(joinable("nějak.", "s tebou"))
        assertFalse(joinable("např.", "Praha"))
    }

    @Test
    fun theFormOfWhispersBoundary() {
        assertEquals(Form.PERIOD_UPPER, b.form("ale můžeš nějak.", "S tebou"))
        assertEquals(Form.PERIOD_LOWER, b.form("ale můžeš nějak.", "s tebou"))     // the author's report
        assertEquals(Form.PERIOD_UPPER, b.form("v roce.", "2013 jsme"))
        assertEquals(Form.UPPER, b.form("tomto hlasování, protože", "To, co"))
        assertEquals(Form.OTHER, b.form("tomto hlasování, protože", "ODS"))
        assertEquals(Form.OTHER, b.form("letošního roku", "a přijal"))
        assertEquals(Form.COMMA_UPPER, b.form("uvedl,", "Že"))
        assertEquals(Form.COMMA_LOWER, b.form("uvedl,", "že"))      // only the comma may go
        assertEquals(
            mapOf(
                Choice.JOIN_KEEP to ("uvedl," to "Že to"), Choice.JOIN to ("uvedl," to "že to"),
                Choice.UNCOMMA_KEEP to ("uvedl" to "Že to"), Choice.UNCOMMA to ("uvedl" to "že to"),
            ),
            b.candidates("uvedl,", "Že to"),
        )
        assertEquals(Choice.JOIN, decide(Form.COMMA_UPPER, Choice.JOIN to -1.0, Choice.JOIN_KEEP to -5.0))
        assertEquals(Form.OTHER, b.form("dne 17.", "listopadu"))
        assertEquals(Form.OTHER, b.form("", "Ano"))
        assertEquals(Form.OTHER, b.form("Ano.", "–"))
    }

    @Test
    fun candidatesDropThePeriodAndLowerTheFirstLetter() {
        val c = b.candidates("ale můžeš nějak.", "S tebou spojit")
        assertEquals("ale můžeš nějak." to "S tebou spojit", c[Choice.SPLIT])
        assertEquals("ale můžeš nějak" to "s tebou spojit", c[Choice.JOIN])
        assertEquals("ale můžeš nějak" to "S tebou spojit", c[Choice.JOIN_KEEP])
        assertEquals("ale můžeš nějak," to "s tebou spojit", c[Choice.COMMA])
        assertEquals("řekl „ano“" to "pak", b.candidates("řekl „ano.“", "Pak")[Choice.JOIN])
        assertFalse(Choice.JOIN in b.candidates("zdraví.", "ODS"))
        assertEquals("zdraví," to "ODS", b.candidates("zdraví.", "ODS")[Choice.COMMA])
        // "A. b" and "A B" have the candidates of "A. B"
        assertEquals(b.candidates("ale můžeš nějak.", "S tebou spojit"), b.candidates("ale můžeš nějak.", "s tebou spojit"))
        assertEquals(b.candidates("ale můžeš nějak.", "S tebou spojit"), b.candidates("ale můžeš nějak", "S tebou spojit"))
    }

    @Test
    fun theLanguageModelSeesAWindowOfTwentyAndTwelveWords() {
        val a = (1..25).joinToString(" ") { "slovo$it" } + "."
        val bb = "Další " + (1..15).joinToString(" ") { "věta$it" }
        val texts = b.languageModelTexts(a, bb)
        assertEquals(listOf(Choice.SPLIT, Choice.JOIN_KEEP, Choice.JOIN, Choice.COMMA), texts.keys.toList())
        val split = texts.getValue(Choice.SPLIT)
        assertEquals((6..25).joinToString(" ") { "slovo$it" } + ". Další " + (1..11).joinToString(" ") { "věta$it" }, split)
        assertEquals(32, b.words(split).size)
        assertTrue(texts.getValue(Choice.JOIN).contains("slovo25 další věta1"))
        assertTrue(texts.getValue(Choice.COMMA).contains("slovo25, další"))
        // "A B" and "A, B" only decide the case: two texts
        assertEquals(setOf(Choice.JOIN, Choice.JOIN_KEEP), b.languageModelTexts("a já", "Petr přijde").keys)
        assertEquals("and the comma that may go: the same two plus the two without it", setOf(Choice.JOIN, Choice.JOIN_KEEP, Choice.UNCOMMA, Choice.UNCOMMA_KEEP), b.languageModelTexts("uvedl,", "Že přijde").keys)
        assertTrue(b.languageModelTexts("opravdu?", "Ano").isEmpty())
    }

    @Test
    fun decideAndDecideByForm() {
        // decide(): join when the best joined candidate beats the end of the sentence by more than θ
        val s = scores(Choice.SPLIT to -10.0, Choice.JOIN to -8.0, Choice.JOIN_KEEP to -9.0)
        assertEquals(Choice.JOIN, decide(Form.PERIOD_UPPER, Choice.SPLIT to -10.0, Choice.JOIN to -10.0 + 5.6, Choice.JOIN_KEEP to -30.0))
        assertEquals(Choice.SPLIT, decide(Form.PERIOD_UPPER, *s))
        // "A. b" without a join gets a capital ("A. B")
        assertEquals(Choice.SPLIT, decide(Form.PERIOD_LOWER, *s))
        // "A B" only ever gets a small letter: no period whatever the language model says
        assertEquals(Choice.RAW, decide(Form.UPPER, Choice.SPLIT to 0.0, Choice.JOIN to -9.0, Choice.JOIN_KEEP to -8.0))
        assertEquals(Choice.JOIN, decide(Form.UPPER, Choice.SPLIT to 99.0, Choice.JOIN to -5.0, Choice.JOIN_KEEP to -8.0))
        // a small letter only with the case margin: otherwise joined with the capital (a name, an institution)
        assertEquals(Choice.JOIN_KEEP, decide(Form.PERIOD_UPPER, Choice.SPLIT to -20.0, Choice.JOIN to -8.0, Choice.JOIN_KEEP to -9.0))
        assertEquals(Choice.RAW, decide(Form.UPPER, Choice.JOIN to -8.0, Choice.JOIN_KEEP to -9.0))
        // "A. b": Whisper's small letter stays when the small letter wins
        assertEquals(Choice.JOIN, decide(Form.PERIOD_LOWER, Choice.SPLIT to -20.0, Choice.JOIN to -8.0, Choice.JOIN_KEEP to -9.0))
        assertEquals(Choice.JOIN_KEEP, decide(Form.PERIOD_LOWER, Choice.SPLIT to -20.0, Choice.JOIN to -9.0, Choice.JOIN_KEEP to -5.0))
        // ... and, as in the measured code (decide_class), the capital also wins by less than the case margin when it
        // is the best candidate; the report's wording ("a capital only by more than 2.5") is stricter
        assertEquals(Choice.JOIN_KEEP, decide(Form.PERIOD_LOWER, Choice.SPLIT to -20.0, Choice.JOIN to -9.0, Choice.JOIN_KEEP to -8.0))
        assertEquals(Choice.RAW, decide(Form.OTHER, *s))
        assertEquals(Choice.RAW, b.decide(Form.PERIOD_UPPER, null))
    }

    // --- the thresholds of the measurement -------------------------------------------------------------------

    @Test
    fun theThresholdsAreTheMeasuredOnes() {
        assertEquals("the tuned margin of the measured code", 5.5557098388671875, SentenceBoundary.JOIN_MARGIN, 0.0)
        assertEquals(3.6588, SentenceBoundary.JOIN_MARGIN_AFTER_LOWER, 5e-5)
        assertEquals(2.5, SentenceBoundary.CASE_MARGIN, 0.0)
        assertEquals(20, SentenceBoundary.WORDS_BEFORE)
        assertEquals(12, SentenceBoundary.WORDS_AFTER)
        assertEquals(3_000L, SentenceBoundary.MAX_PAUSE_MS)
    }

    @Test
    fun aJoinNeedsMoreThanTheMarginOverTheEndOfTheSentence() {
        fun periodUpper(margin: Double) =
            decide(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -50.0 + margin, Choice.JOIN_KEEP to -60.0)
        assertEquals(Choice.SPLIT, periodUpper(5.55))
        assertEquals(Choice.JOIN, periodUpper(5.56))
        fun periodLower(margin: Double) =
            decide(Form.PERIOD_LOWER, Choice.SPLIT to -50.0, Choice.JOIN to -50.0 + margin, Choice.JOIN_KEEP to -60.0)
        assertEquals(Choice.SPLIT, periodLower(3.65))
        assertEquals(Choice.JOIN, periodLower(3.66))
        // the comma candidate counts as a join
        assertEquals(
            Choice.COMMA,
            decide(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -60.0, Choice.JOIN_KEEP to -70.0, Choice.COMMA to -40.0),
        )
    }

    // --- names -----------------------------------------------------------------------------------------------

    @Test
    fun namesKeepTheirCapitalUnlessTheLanguageModelPrefersTheSmallLetterByMoreThanTheMargin() {
        for ((a, bb) in listOf("Pozdravuj." to "Honzu a Petru.", "Dneska jsem byl v." to "Praze na úřadě.")) {
            // the join wins clearly; the small letter by 2.5 only, which is not more than the margin
            val keep = b.decide(a, bb, lm(a, bb, Choice.JOIN to -40.0, Choice.JOIN_KEEP to -42.5, Choice.SPLIT to -60.0))
            assertEquals(Choice.JOIN_KEEP, keep)
            val join = b.sentenceJoin(a, bb, keep)!!
            assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, bb, bb), join)
            assertEquals(Choice.JOIN, b.decide(a, bb, lm(a, bb, Choice.JOIN to -40.0, Choice.JOIN_KEEP to -42.6, Choice.SPLIT to -60.0)))
        }
        // "A B": Praha stays Praha unless the small letter wins by more than the margin
        assertEquals(Choice.RAW, b.decide("Jedu do", "Prahy zítra.", lm("Jedu do", "Prahy zítra.", Choice.JOIN to -30.0, Choice.JOIN_KEEP to -32.4)))
        assertEquals(Choice.JOIN, b.decide("Jedu do", "Prahy zítra.", lm("Jedu do", "Prahy zítra.", Choice.JOIN to -30.0, Choice.JOIN_KEEP to -32.6)))
        assertNull(b.sentenceJoin("Jedu do", "Prahy zítra.", Choice.RAW))
        // an acronym or a number is never lowered: nothing to decide
        assertEquals(Form.OTHER, b.form("Hlasovala", "ODS."))
        assertEquals("ODS přijde", b.sentenceJoin("Zdraví.", "ODS přijde", Choice.JOIN_KEEP)!!.continuingText)
    }

    // --- what a client gets ----------------------------------------------------------------------------------

    @Test
    fun theReportedCaseIsJoined() {
        // "…ale můžem se nějak s tebou spojit…" came out as "…můžeš nějak. s tebou spojit…"; the language model
        // prefers the join by 12.6 in the measurement
        val a = "Jasně, ale můžeš nějak."
        val bb = "s tebou spojit ve video hovoru."
        assertEquals(Form.PERIOD_LOWER, b.form(a, bb))
        val choice = b.decide(a, bb, lm(a, bb, Choice.JOIN to -40.0, Choice.JOIN_KEEP to -50.0, Choice.SPLIT to -52.6, Choice.COMMA to -47.0))
        assertEquals(Choice.JOIN, choice)
        val join = b.sentenceJoin(a, bb, choice)!!
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, bb, "S tebou spojit ve video hovoru."), join)
        assertEquals("Jasně, ale můžeš nějak s tebou spojit ve video hovoru.", join.previousEnd.applyTo(a) + " " + join.continuingText)
    }

    @Test
    fun eachChoiceTellsTheClientWhatToDo() {
        val a = "Myslím, že."
        val bb = "To nebude problém."
        assertNull(b.sentenceJoin(a, bb, Choice.RAW))
        assertNull("the end of a sentence already written so", b.sentenceJoin(a, bb, Choice.SPLIT))
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, "to nebude problém.", bb), b.sentenceJoin(a, bb, Choice.JOIN))
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_PERIOD, bb, bb), b.sentenceJoin(a, bb, Choice.JOIN_KEEP))
        assertEquals(SentenceJoin(true, PreviousEnd.PERIOD_TO_COMMA, "to nebude problém.", bb), b.sentenceJoin(a, bb, Choice.COMMA))
        // "A. b" left as the end of a sentence: the capital Whisper left out, both texts the same
        assertEquals(
            SentenceJoin(false, PreviousEnd.KEEP, "To nebude problém.", "To nebude problém."),
            b.sentenceJoin(a, "to nebude problém.", Choice.SPLIT),
        )
        // "A B" and "A, B": nothing to change at the end of A
        assertEquals(
            SentenceJoin(true, PreviousEnd.KEEP, "to nebude problém.", bb),
            b.sentenceJoin("Myslím, že", bb, Choice.JOIN),
        )
        assertEquals(SentenceJoin(true, PreviousEnd.KEEP, "že to", "Že to"), b.sentenceJoin("uvedl,", "Že to", Choice.JOIN))
    }

    @Test
    fun theEndOfThePreviousTextIsEditedBeforeClosersAndWhitespace() {
        assertEquals("nějak", PreviousEnd.REMOVE_PERIOD.applyTo("nějak."))
        assertEquals("nějak,", PreviousEnd.PERIOD_TO_COMMA.applyTo("nějak."))
        assertEquals("nějak. ", PreviousEnd.KEEP.applyTo("nějak. "))
        assertEquals("nějak ", PreviousEnd.REMOVE_PERIOD.applyTo("nějak. "))
        assertEquals("řekl „ano“", PreviousEnd.REMOVE_PERIOD.applyTo("řekl „ano.“"))
        assertEquals("řekl „ano“, ", PreviousEnd.PERIOD_TO_COMMA.applyTo("řekl „ano.“ "))
        assertEquals("no period", "nějak", PreviousEnd.REMOVE_PERIOD.applyTo("nějak"))
        assertEquals("an ellipsis stays", "nějak...", PreviousEnd.PERIOD_TO_COMMA.applyTo("nějak..."))
        // the candidate texts the language model scored
        assertEquals(b.candidates("řekl „ano.“", "Pak")[Choice.COMMA]!!.first, PreviousEnd.PERIOD_TO_COMMA.applyTo("řekl „ano.“"))
    }

    @Test
    fun onlyAPeriodThatEndsASentenceIsEdited() {
        // what the SDK never calls the end of a sentence comes back as it is: nothing, an ellipsis, an abbreviation,
        // another sentence end, an ordinal
        for (end in listOf(PreviousEnd.REMOVE_PERIOD, PreviousEnd.PERIOD_TO_COMMA)) {
            for (text in listOf("", "   ", "…", "Bylo to tak…", "a tak dále...", "atd.", "Koupil jablka, hrušky atd.",
                "č.", "Bydlím v č.", "?", "Opravdu?", "!", "17.", "Sešli jsme se 17.", "v roce 2019.", "Ing.", "(12.5.",
                "nějak", "nějak,", "atd.  ", "17. \n")) {
                assertEquals("$end of \"$text\"", text, end.applyTo(text))
            }
        }
        // whitespace after a sentence-ending period stays where it was
        assertEquals("nějak  ", PreviousEnd.REMOVE_PERIOD.applyTo("nějak.  "))
        assertEquals("nějak,\n", PreviousEnd.PERIOD_TO_COMMA.applyTo("nějak.\n"))
        assertEquals("To je vše“ ", PreviousEnd.REMOVE_PERIOD.applyTo("To je vše.“ "))
    }

    @Test
    fun noDecisionWithoutTheLanguageModel() {
        assertEquals(Choice.RAW, b.decide("Myslím, že.", "To nebude problém.") { null })
        assertEquals(Choice.RAW, b.decide("Myslím, že.", "To nebude problém.") { DoubleArray(1) })     // wrong size
        assertEquals(Choice.RAW, b.decide("Myslím, že.", "To nebude problém.") { DoubleArray(it.size) { Double.NaN } })
        var called = false
        assertEquals(Choice.RAW, b.decide("Opravdu?", "Ano.") { called = true; DoubleArray(it.size) })
        assertFalse("nothing to decide, no language model", called)
    }

    @Test
    fun textFollowsPythonsStrMethods() {
        assertEquals(listOf("a", "b", "c"), b.words("  a\tb  c \n"))
        assertTrue(b.words(" \n ").isEmpty())
        assertEquals("", b.endPunct("slovo"))
        assertEquals(".", b.endPunct("slovo.“"))
        assertEquals("...", b.endPunct("slovo..."))
        assertEquals("?", b.endPunct("(proč?)"))
        assertTrue(b.periodEnd("To je vše.“"))
        assertFalse(b.periodEnd("dne 17."))
        assertFalse(b.periodEnd("(12.5."))
        assertFalse(b.periodEnd("Ing."))
        for (legal in listOf("podle čl.", "dle zák.", "firma Alfa s.r.o.", "Beta a.s.", "ve sp.")) assertFalse(legal, b.periodEnd(legal))
        assertTrue("a word that only looks like one: \"zákon.\"", b.periodEnd("Platí zákon."))
        assertTrue(b.periodEnd("Ingrid."))
        assertEquals("nějak“", b.stripPeriod("nějak.“  "))
        assertEquals("Ša", b.upperFirstWord("ša"))
        assertEquals("„že to", b.lowerFirstWord("„Že to"))
        assertEquals("ODS to", b.upperFirstWord("ODS to"))
        assertTrue(b.startsUpper("„Ano"))
        assertFalse(b.startsUpper("123"))
    }

    // --- a piece after a quiet cut inside continuous speech: the decision leans to going on ----------------------

    private fun quiet(form: Form, vararg s: Pair<Choice, Double>) = b.decide(form, s.toMap(), quiet = true)

    @Test
    fun theQuietCutConstantsAreNamed() {
        assertEquals(2.0, SentenceBoundary.QUIET_CUT_SPLIT_MARGIN, 0.0)
        assertEquals(1.0, SentenceBoundary.CASE_MARGIN_CONTINUED, 0.0)
    }

    @Test
    fun aQuietCutIsJoinedUnlessTheEndOfTheSentenceBeatsTheJoinByAClearMargin() {
        fun periodUpper(joinOverSplit: Double) =
            quiet(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -50.0 + joinOverSplit, Choice.JOIN_KEEP to -60.0)
        // the split wins only when it is better by more than the margin (2.0): the join may even be the worse of the two
        assertEquals(Choice.JOIN, periodUpper(5.6))
        assertEquals(Choice.JOIN, periodUpper(0.0))
        assertEquals(Choice.JOIN, periodUpper(-1.9))
        assertEquals(Choice.SPLIT, periodUpper(-2.1))
        // after a real pause the same scores split (unchanged)
        assertEquals(Choice.SPLIT, decide(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -60.0))
        // "A. b": the small letter of Whisper after the period is a sign of a false boundary, the margin grows by its shift
        val shift = SentenceBoundary.JOIN_MARGIN - SentenceBoundary.JOIN_MARGIN_AFTER_LOWER
        fun periodLower(joinOverSplit: Double) =
            quiet(Form.PERIOD_LOWER, Choice.SPLIT to -50.0, Choice.JOIN to -50.0 + joinOverSplit, Choice.JOIN_KEEP to -60.0)
        assertEquals(Choice.JOIN, periodLower(-2.0 - shift + 0.1))
        assertEquals(Choice.SPLIT, periodLower(-2.0 - shift - 0.1))
    }

    @Test
    fun aQuietCutCanChooseTheCommaAndTheBestJoinWins() {
        assertEquals(
            Choice.COMMA,
            quiet(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -52.0, Choice.JOIN_KEEP to -60.0, Choice.COMMA to -49.0),
        )
        // the comma is not the best: the plain continuation
        assertEquals(
            Choice.JOIN,
            quiet(Form.PERIOD_UPPER, Choice.SPLIT to -45.0, Choice.JOIN to -46.0, Choice.JOIN_KEEP to -60.0, Choice.COMMA to -47.0),
        )
        // the new sentence wins by a clear margin even against the comma
        assertEquals(
            Choice.SPLIT,
            quiet(Form.PERIOD_UPPER, Choice.SPLIT to -45.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -60.0, Choice.COMMA to -47.5),
        )
    }

    @Test
    fun aQuietCutLowersTheFirstWordWhenTheLanguageModelPrefersTheSmallLetterByMoreThanTheCutMargin() {
        // join (small letter) beats join_keep (capital) by 1.5: lowered after a quiet cut, kept after a real pause (2.5)
        val scores = arrayOf(Choice.SPLIT to -60.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -51.5)
        assertEquals(Choice.JOIN, quiet(Form.PERIOD_UPPER, *scores))
        assertEquals(Choice.JOIN_KEEP, decide(Form.PERIOD_UPPER, Choice.SPLIT to -50.0, Choice.JOIN to -40.0, Choice.JOIN_KEEP to -41.5))
        // a name: the capital stays when the model prefers it (or is nearly indifferent)
        assertEquals(Choice.JOIN_KEEP, quiet(Form.PERIOD_UPPER, Choice.SPLIT to -60.0, Choice.JOIN to -50.5, Choice.JOIN_KEEP to -50.0))
        assertEquals(Choice.JOIN_KEEP, quiet(Form.PERIOD_UPPER, Choice.SPLIT to -60.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -51.0))
        // without a period ("A B", "A, B") a quiet cut still never adds one and only lowers
        assertEquals(Choice.JOIN, quiet(Form.UPPER, Choice.SPLIT to 99.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -51.5))
        assertEquals(Choice.RAW, quiet(Form.UPPER, Choice.JOIN to -50.5, Choice.JOIN_KEEP to -50.0))
        assertEquals(Choice.RAW, b.decide(Form.OTHER, mapOf(Choice.SPLIT to 0.0), quiet = true))
        assertEquals(Choice.RAW, b.decide(Form.PERIOD_UPPER, null, quiet = true))
    }

    @Test
    fun withoutTheQuietFlagEveryDecisionIsTheMeasuredOne() {
        // the default is a real pause: the flag is the only thing that changes a decision
        val s = mapOf(Choice.SPLIT to -50.0, Choice.JOIN to -47.0, Choice.JOIN_KEEP to -60.0, Choice.COMMA to -49.0)
        for (form in Form.entries) assertEquals("$form", b.decide(form, s, quiet = false), b.decide(form, s))
        assertEquals(Choice.SPLIT, b.decide(Form.PERIOD_UPPER, s))
        assertEquals(Choice.JOIN, b.decide(Form.PERIOD_UPPER, s, quiet = true))
    }

    @Test
    fun theQuietFlagReachesTheDecisionByText() {
        val a = "Jasně, ale můžeš nějak."
        val bb = "S tebou spojit ve video hovoru."
        // split better than the join by 1 only: after a pause split, after a quiet cut joined
        val lm = lm(a, bb, Choice.SPLIT to -50.0, Choice.JOIN to -51.0, Choice.JOIN_KEEP to -60.0, Choice.COMMA to -60.0)
        assertEquals(Choice.SPLIT, b.decide(a, bb, lm))
        assertEquals(Choice.JOIN, b.decide(a, bb, lm, quiet = true))
    }

    // --- a piece after a cut on speech: plain continuation, the first word lowered by the language model -------

    @Test
    fun theCandidatesOfAContinuationAreTheSmallLetterAndTheCapital() {
        assertEquals(
            mapOf(
                Choice.JOIN to "Jasně, ale můžeš nějak dnes přijdu k tobě.",
                Choice.JOIN_KEEP to "Jasně, ale můžeš nějak Dnes přijdu k tobě.",
            ),
            b.continuationTexts("Jasně, ale můžeš nějak.", "Dnes přijdu k tobě."),
        )
        assertEquals("a previous text without a period", setOf(Choice.JOIN, Choice.JOIN_KEEP), b.continuationTexts("Jasně, ale", "Dnes přijdu.").keys)
        // nothing to decide: a small letter already, an acronym, a number
        assertEquals(emptyMap<Choice, String>(), b.continuationTexts("Jasně, ale můžeš nějak.", "dnes přijdu k tobě."))
        assertEquals(emptyMap<Choice, String>(), b.continuationTexts("Jasně, ale můžeš nějak.", "ČR to řeší."))
        assertEquals(emptyMap<Choice, String>(), b.continuationTexts("Jasně, ale můžeš nějak.", "17 lidí přišlo."))
    }

    // --- an ellipsis at the end of the previous piece: an artefact of the pause, the prior leans to removing it ----------

    @Test
    fun theFormOfAnEllipsisAndOfATrailingCommaFollowedByAnyLetter() {
        for (a in listOf("Bylo to tak…", "Bylo to tak...", "Bylo to tak.....", "(Bylo to tak...)", "„Bylo to tak…“", "Bylo to tak.. ")) {
            assertEquals(a, Form.ELLIPSIS, b.form(a, "Že jsme odešli."))
            assertEquals(a, Form.ELLIPSIS, b.form(a, "že jsme odešli."))
        }
        assertEquals("no letter before the dots", Form.OTHER, b.form("...", "Že"))
        assertEquals("no word after", Form.OTHER, b.form("Bylo to tak…", "…"))
        assertEquals(Form.COMMA_UPPER, b.form("Jasně,", "Ale tak"))
        assertEquals(Form.COMMA_LOWER, b.form("Jasně,", "ale tak"))
        assertEquals("a period is still a period", Form.PERIOD_UPPER, b.form("Bylo to tak.", "Že jsme odešli."))
    }

    @Test
    fun theEndOfThePreviousTextIsEditedForAnEllipsisAndACommaBeforeClosersAndWhitespace() {
        val removeEllipsis = PreviousEnd.REMOVE_ELLIPSIS
        assertEquals("nějak", removeEllipsis.applyTo("nějak..."))
        assertEquals("nějak", removeEllipsis.applyTo("nějak…"))
        assertEquals("nějak", removeEllipsis.applyTo("nějak.."))
        assertEquals("(nějak)", removeEllipsis.applyTo("(nějak...)"))
        assertEquals("„nějak“ ", removeEllipsis.applyTo("„nějak…“ "))
        assertEquals("nějak.", removeEllipsis.applyTo("nějak."))
        assertEquals("nějak,", removeEllipsis.applyTo("nějak,"))
        assertEquals("nějak", removeEllipsis.applyTo("nějak"))
        assertEquals("nějak,", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("nějak..."))
        assertEquals("nějak,", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("nějak…"))
        assertEquals("„nějak,“", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("„nějak…“"))
        assertEquals("nějak. ", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("nějak. "))
        assertEquals("nějak", PreviousEnd.REMOVE_COMMA.applyTo("nějak,"))
        assertEquals("(nějak) ", PreviousEnd.REMOVE_COMMA.applyTo("(nějak,) "))
        assertEquals("nějak...", PreviousEnd.REMOVE_COMMA.applyTo("nějak..."))
        assertEquals("nějak.", PreviousEnd.REMOVE_COMMA.applyTo("nějak."))
        // the old values do not touch an ellipsis or a comma
        assertEquals("nějak...", PreviousEnd.REMOVE_PERIOD.applyTo("nějak..."))
        assertEquals("nějak…", PreviousEnd.PERIOD_TO_COMMA.applyTo("nějak…"))
        assertEquals("nějak,", PreviousEnd.REMOVE_PERIOD.applyTo("nějak,"))
    }

    @Test
    fun theCandidatesOfAnEllipsisAndOfATrailingComma() {
        assertEquals(
            mapOf(
                Choice.SPLIT to "Bylo to tak… Že jsme odešli.",
                Choice.ELLIPSIS to "Bylo to tak… že jsme odešli.",
                Choice.JOIN_KEEP to "Bylo to tak Že jsme odešli.",
                Choice.JOIN to "Bylo to tak že jsme odešli.",
                Choice.COMMA to "Bylo to tak, že jsme odešli.",
            ),
            b.languageModelTexts("Bylo to tak…", "Že jsme odešli."),
        )
        assertEquals(
            mapOf(
                Choice.JOIN_KEEP to "Jasně, Ale tak",
                Choice.JOIN to "Jasně, ale tak",
                Choice.UNCOMMA_KEEP to "Jasně Ale tak",
                Choice.UNCOMMA to "Jasně ale tak",
            ),
            b.languageModelTexts("Jasně,", "Ale tak"),
        )
        assertEquals(setOf(Choice.JOIN, Choice.UNCOMMA), b.languageModelTexts("Jasně,", "ale tak").keys)
    }

    @Test
    fun theMarginsOfTheEllipsisAndTheCommaAreNamed() {
        assertEquals(2.0, SentenceBoundary.ELLIPSIS_KEEP_MARGIN, 0.0)
        assertEquals(2.0, SentenceBoundary.ELLIPSIS_SPLIT_MARGIN, 0.0)
        assertEquals(1.5, SentenceBoundary.COMMA_OVER_PLAIN_MARGIN, 0.0)
        assertEquals(1.5, SentenceBoundary.COMMA_REMOVE_MARGIN, 0.0)
    }

    private fun ellipsis(
        split: Double = -60.0, keep: Double = -60.0, join: Double = -50.0, joinKeep: Double = -60.0, comma: Double = -60.0,
    ) = decide(
        Form.ELLIPSIS, Choice.SPLIT to split, Choice.ELLIPSIS to keep, Choice.JOIN to join, Choice.JOIN_KEEP to joinKeep, Choice.COMMA to comma,
    )

    @Test
    fun anEllipsisIsRemovedUnlessAnotherReadingIsClearlyLikelier() {
        assertEquals("plain continuation", Choice.JOIN, ellipsis())
        assertEquals(Choice.JOIN, ellipsis(keep = -48.1))
        assertEquals("the ellipsis stays only when the model prefers it by more than 2", Choice.ELLIPSIS, ellipsis(keep = -47.9))
        assertEquals(Choice.JOIN, ellipsis(split = -48.1))
        assertEquals("a new sentence only by more than 2", Choice.SPLIT, ellipsis(split = -47.9))
        assertEquals(Choice.JOIN, ellipsis(comma = -48.6))
        assertEquals("a comma only by more than 1.5 over the plain continuation", Choice.COMMA, ellipsis(comma = -48.4))
    }

    @Test
    fun anEllipsisBeforeANameKeepsTheCapitalAndLowersTheOthers() {
        assertEquals(Choice.JOIN, ellipsis(join = -50.0, joinKeep = -52.0))
        assertEquals("a name", Choice.JOIN_KEEP, ellipsis(join = -50.5, joinKeep = -50.0))
        assertEquals("1.0 is not more than the margin", Choice.JOIN_KEEP, ellipsis(join = -50.0, joinKeep = -51.0))
        assertEquals(Choice.RAW, b.decide(Form.ELLIPSIS, null))
    }

    @Test
    fun theClientIsToldWhatToDoWithAnEllipsis() {
        val a = "Bylo to tak…"
        val bb = "Že jsme odešli."
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_ELLIPSIS, "že jsme odešli.", bb), b.sentenceJoin(a, bb, Choice.JOIN))
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_ELLIPSIS, bb, bb), b.sentenceJoin(a, bb, Choice.JOIN_KEEP))
        assertEquals(SentenceJoin(true, PreviousEnd.ELLIPSIS_TO_COMMA, "že jsme odešli.", bb), b.sentenceJoin(a, bb, Choice.COMMA))
        assertEquals("the ellipsis stays, the next word is small", SentenceJoin(true, PreviousEnd.KEEP, "že jsme odešli.", bb), b.sentenceJoin(a, bb, Choice.ELLIPSIS))
        assertNull("a new sentence as written", b.sentenceJoin(a, bb, Choice.SPLIT))
        assertEquals(SentenceJoin(false, PreviousEnd.KEEP, bb, bb), b.sentenceJoin(a, "že jsme odešli.", Choice.SPLIT))
        // an ellipsis before a word that cannot be lowered, all the same: the text goes on
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_ELLIPSIS, "17 lidí", "17 lidí"), b.sentenceJoin("Bylo nás...", "17 lidí", Choice.JOIN_KEEP))
        // the old period-dealing is untouched
        assertEquals(PreviousEnd.PERIOD_TO_COMMA, b.sentenceJoin("Bylo to tak.", bb, Choice.COMMA)!!.previousEnd)
    }

    // --- a comma at the end of the previous piece may go: a pause in the wrong place must not force it ----------

    @Test
    fun aTrailingCommaIsRemovedOnlyWhenTheLanguageModelPrefersItByMoreThanTheMargin() {
        fun commaUpper(uncomma: Double, uncommaKeep: Double = -60.0, join: Double = -50.0, joinKeep: Double = -60.0) = decide(
            Form.COMMA_UPPER, Choice.JOIN to join, Choice.JOIN_KEEP to joinKeep, Choice.UNCOMMA to uncomma, Choice.UNCOMMA_KEEP to uncommaKeep,
        )
        assertEquals("as before: the comma stays, the letter is lowered", Choice.JOIN, commaUpper(uncomma = -48.6))
        assertEquals("the comma goes", Choice.UNCOMMA, commaUpper(uncomma = -48.4))
        assertEquals("a name after the comma that goes", Choice.UNCOMMA_KEEP, commaUpper(uncomma = -52.0, uncommaKeep = -48.0))
        assertEquals("as before: nothing without a margin", Choice.RAW, commaUpper(uncomma = -60.0, join = -50.0, joinKeep = -49.0))
        fun commaLower(uncomma: Double) = decide(Form.COMMA_LOWER, Choice.JOIN to -50.0, Choice.UNCOMMA to uncomma)
        assertEquals(Choice.RAW, commaLower(-48.6))
        assertEquals(Choice.UNCOMMA, commaLower(-48.4))
        // the old "A, B" decision without the new candidates
        assertEquals(Choice.JOIN, decide(Form.COMMA_UPPER, Choice.JOIN to -5.0, Choice.JOIN_KEEP to -8.0))
        val a = "Jasně,"
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_COMMA, "ale tak", "Ale tak"), b.sentenceJoin(a, "Ale tak", Choice.UNCOMMA))
        assertEquals(SentenceJoin(true, PreviousEnd.REMOVE_COMMA, "Ale tak", "Ale tak"), b.sentenceJoin(a, "Ale tak", Choice.UNCOMMA_KEEP))
    }

    @Test
    fun aQuietCutChoosesTheCommaOnlyWhenTheModelClearlyPrefersItOverThePlainContinuation() {
        fun q(comma: Double) = quiet(Form.PERIOD_UPPER, Choice.SPLIT to -60.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -70.0, Choice.COMMA to comma)
        assertEquals(Choice.JOIN, q(-48.6))
        assertEquals(Choice.JOIN, q(-50.0))
        assertEquals(Choice.COMMA, q(-48.4))
        // after a real pause the measured decision: the best joined candidate, the comma without a margin
        assertEquals(Choice.COMMA, decide(Form.PERIOD_UPPER, Choice.SPLIT to -60.0, Choice.JOIN to -50.0, Choice.JOIN_KEEP to -70.0, Choice.COMMA to -49.9))
    }

    @Test
    fun aContinuationLowersTheFirstWordUnlessTheLanguageModelPrefersTheCapitalOrIsNearlyIndifferent() {
        assertTrue(b.lowerAfterCut(mapOf(Choice.JOIN to -40.0, Choice.JOIN_KEEP to -41.1)))
        assertFalse("1.0 is not more than the margin", b.lowerAfterCut(mapOf(Choice.JOIN to -40.0, Choice.JOIN_KEEP to -41.0)))
        assertFalse("a name", b.lowerAfterCut(mapOf(Choice.JOIN to -45.0, Choice.JOIN_KEEP to -40.0)))
        assertTrue("the language model failed: lowered as before", b.lowerAfterCut(null))
        assertTrue("no candidates: lowered as before", b.lowerAfterCut(emptyMap()))
    }

    // --- review of 672325d: the margins of a quiet cut are against the BEST continuing candidate ---------------

    @Test
    fun aQuietCutSplitsOnlyWhenTheNewSentenceBeatsTheBestContinuingCandidateByTheMarginWhateverTheCommaDid() {
        fun q(split: Double, plain: Double, comma: Double) =
            quiet(Form.PERIOD_UPPER, Choice.SPLIT to split, Choice.JOIN to plain, Choice.JOIN_KEEP to -80.0, Choice.COMMA to comma)
        // the comma is rejected (1.3 over the plain continuation, margin 1.5): the new sentence is then compared with the plain one
        assertEquals("2.5 over the plain continuation, 1.2 over the rejected comma", Choice.SPLIT, q(split = -48.0, plain = -50.5, comma = -49.2))
        assertEquals("1.9 over the plain continuation", Choice.JOIN, q(split = -48.6, plain = -50.5, comma = -49.2))
        assertEquals("2.1 over the plain continuation", Choice.SPLIT, q(split = -48.4, plain = -50.5, comma = -49.2))
        // the comma is accepted (1.6 over plain): the new sentence is compared with the comma, the best continuing candidate
        assertEquals("1.9 over the comma", Choice.COMMA, q(split = -46.5, plain = -50.0, comma = -48.4))
        assertEquals("2.1 over the comma", Choice.SPLIT, q(split = -46.3, plain = -50.0, comma = -48.4))
        // no comma at all
        assertEquals(Choice.JOIN, q(split = -48.1, plain = -50.0, comma = -90.0))
        assertEquals(Choice.SPLIT, q(split = -47.9, plain = -50.0, comma = -90.0))
    }

    // --- an abbreviation keeps its own period when an ellipsis goes ---------------------------------------------

    @Test
    fun anEllipsisAfterAnAbbreviationKeepsTheAbbreviationsPeriod() {
        assertEquals("atd.", PreviousEnd.REMOVE_ELLIPSIS.applyTo("atd..."))
        assertEquals("např.", PreviousEnd.REMOVE_ELLIPSIS.applyTo("např.."))
        assertEquals("atd.", PreviousEnd.REMOVE_ELLIPSIS.applyTo("atd…"))
        assertEquals("(atd.) ", PreviousEnd.REMOVE_ELLIPSIS.applyTo("(atd...) "))
        assertEquals("atd.,", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("atd..."))
        assertEquals("nějak,", PreviousEnd.ELLIPSIS_TO_COMMA.applyTo("nějak..."))
        assertEquals("Koupil jablka atd. a hrušky", "Koupil jablka atd.".let { PreviousEnd.REMOVE_ELLIPSIS.applyTo("$it..") } + " a hrušky")
        // the candidates the model scores say the same
        val texts = b.languageModelTexts("Koupil jablka atd...", "Hrušky taky.")
        assertEquals("Koupil jablka atd. hrušky taky.", texts.getValue(Choice.JOIN))
        assertEquals("Koupil jablka atd., hrušky taky.", texts.getValue(Choice.COMMA))
        assertEquals("Koupil jablka atd... Hrušky taky.", texts.getValue(Choice.SPLIT))
    }

    // --- helpers ---------------------------------------------------------------------------------------------

    private fun scores(vararg s: Pair<Choice, Double>) = s

    private fun decide(form: Form, vararg s: Pair<Choice, Double>) = b.decide(form, s.toMap())

    /** The language model for the boundary of [a] and [bb]: [scores] by candidate, -99 for the others. */
    private fun lm(a: String, bb: String, vararg scores: Pair<Choice, Double>): (List<String>) -> DoubleArray {
        val choiceOf = b.languageModelTexts(a, bb).entries.associate { (choice, text) -> text to choice }
        val byChoice = scores.toMap()
        return { texts -> DoubleArray(texts.size) { byChoice[choiceOf.getValue(texts[it])] ?: -99.0 } }
    }
}
