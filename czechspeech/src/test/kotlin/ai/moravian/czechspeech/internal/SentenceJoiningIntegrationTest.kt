package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.internal.SentenceBoundary.Choice
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * Sentence joining with the real Czech GPT-2 (q8_0, CZECHSPEECH_LM) on the host build: the invented pairs of the
 * measurement's report decide as reported there (PyTorch float32), and the time of one boundary on this machine.
 */
class SentenceJoiningIntegrationTest {
    private lateinit var lm: LanguageModel

    @Before
    fun load() {
        TestEnv.assumeNativeLibrary()
        TestEnv.assumeFile(TestEnv.languageModel, "Czech GPT-2 language model (CZECHSPEECH_LM)")
        lm = LanguageModel.load(JniNativeApi, TestEnv.languageModel)
    }

    @After
    fun free() {
        if (::lm.isInitialized) lm.free()
    }

    private fun joinedText(a: String, b: String, choice: Choice): String {
        val join = SentenceBoundary.sentenceJoin(a, b, choice) ?: return "$a $b"
        return if (join.continuesPrevious) "${join.previousEnd.applyTo(a)} ${join.continuingText}" else "$a ${join.standaloneText}"
    }

    /** log P of the best joined candidate less that of the end of the sentence, as in the report's table. */
    private fun margin(a: String, b: String): Double {
        val texts = SentenceBoundary.languageModelTexts(a, b)
        val scores = texts.keys.zip(lm.score(texts.values.toList(), TestEnv.threads)!!.toList()).toMap()
        return listOf(Choice.JOIN, Choice.JOIN_KEEP, Choice.COMMA).mapNotNull { scores[it] }.max() - scores.getValue(Choice.SPLIT)
    }

    @Test
    fun theReportsPairsDecideAsMeasured() {
        // (A, B, the result in the report, its margin there)
        val reported = listOf(
            Triple("Jasně, ale můžeš nějak.", "s tebou spojit ve video hovoru.", "Jasně, ale můžeš nějak s tebou spojit ve video hovoru.") to 12.6,
            Triple("Pozdravuj.", "Honzu a Petru.", "Pozdravuj Honzu a Petru.") to 8.4,
            Triple("Potkal jsem se s.", "Honzou v kavárně.", "Potkal jsem se s Honzou v kavárně.") to 12.0,
            Triple("Myslím, že.", "To nebude problém.", "Myslím, že to nebude problém.") to 15.7,
            Triple("Tomáši.", "Zdraví Honza.", "Tomáši. Zdraví Honza.") to -0.3,
            Triple("Ahoj, zítra nemůžu.", "Honza má narozeniny.", "Ahoj, zítra nemůžu. Honza má narozeniny.") to -0.4,
            Triple("Přijdu v pondělí.", "Do té doby to nestihnu.", "Přijdu v pondělí. Do té doby to nestihnu.") to 1.4,
            Triple("Díky za zprávu.", "Ozvu se večer.", "Díky za zprávu. Ozvu se večer.") to -0.9,
        )
        for ((pair, reportedMargin) in reported) {
            val (a, b, expected) = pair
            val choice = SentenceBoundary.decide(a, b) { lm.score(it, TestEnv.threads) }
            println("  margin %5.1f (report %5.1f), %-9s %s | %s".format(margin(a, b), reportedMargin, choice, a, b))
            assertEquals(expected, joinedText(a, b, choice))
        }
        // the report's other invented pairs, for the record
        for ((a, b) in listOf(
            "Dneska jsem byl v." to "Praze na úřadě.",
            "Přijdu až v pondělí, protože." to "Do té doby to nestihnu.",
            "Zavolej mi." to "Potřebuju s tebou mluvit.",
            "To je všechno." to "Měj se.",
        )) {
            val choice = SentenceBoundary.decide(a, b) { lm.score(it, TestEnv.threads) }
            println("  margin %5.1f, %-9s %s".format(margin(a, b), choice, joinedText(a, b, choice)))
        }
    }

    @Test
    fun theTimeOfOneBoundary() {
        // A boundary inside a long dictated sentence: the full window of 20 + 12 words, four candidates.
        val a = "Dobrý den, posílám vám podklady k zítřejší schůzce a prosím, abyste se na ně podívali ještě dnes, protože."
        val b = "Bychom je měli probrat hned na začátku, než přijdou ostatní kolegové z oddělení."
        for (threads in listOf(1, TestEnv.threads)) {
            repeat(2) { SentenceBoundary.decide(a, b) { lm.score(it, threads) } }     // warm up
            val ms = (1..9).map {
                val t0 = System.nanoTime()
                SentenceBoundary.decide(a, b) { lm.score(it, threads) }
                (System.nanoTime() - t0) / 1e6
            }.sorted()
            val candidates = SentenceBoundary.languageModelTexts(a, b).size
            println("  one boundary ($candidates candidates), $threads thread(s): median %.0f ms, min %.0f, max %.0f".format(ms[4], ms[0], ms[8]))
        }
    }
}
