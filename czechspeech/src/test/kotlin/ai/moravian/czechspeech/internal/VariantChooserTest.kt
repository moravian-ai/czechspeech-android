package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Choosing among Whisper's hypotheses as the measurement did:
 * score = log P_Whisper + λ · log P_LM + β · Whisper tokens, the hypotheses deduplicated by text and ordered with
 * Whisper's choice first, the first highest score winning.
 */
class VariantChooserTest {
    private fun window(vararg hyps: Hyp) = NBest.parse(NBestBytes.encode(listOf(hyps.toList())))[0]

    @Test
    fun parsesWindowsAndHypotheses() {
        val windows = NBest.parse(
            NBestBytes.encode(
                listOf(
                    listOf(Hyp(" Dobrý den.", -1.5, 6, best = true, score = -0.25), Hyp(" Dobrý den", -2.0, 5)),
                    listOf(),
                    listOf(Hyp(" Ahoj.", Double.NEGATIVE_INFINITY, 3, best = true)),
                ),
            ),
        )
        assertEquals(3, windows.size)
        assertEquals(2, windows[0].size)
        val h = windows[0][0]
        assertEquals(" Dobrý den.", h.text)
        assertEquals(-1.5, h.sumLogprob, 0.0)
        assertEquals(-0.25, h.score, 0.0)
        assertEquals(6, h.length)
        assertTrue(h.best)
        assertFalse(windows[0][1].best)
        assertEquals(0, windows[1].size)
        assertEquals(Double.NEGATIVE_INFINITY, windows[2][0].sumLogprob, 0.0)
    }

    @Test
    fun malformedBytesAreRejected() {
        val good = NBestBytes.encode(listOf(listOf(Hyp(" text", -1.0, 3, best = true))))
        failure<CzechSpeechException> { NBest.parse(good.copyOf(good.size - 1)) }
        failure<CzechSpeechException> { NBest.parse(good + 0.toByte()) }
        failure<CzechSpeechException> { NBest.parse(ByteArray(0)) }
    }

    @Test
    fun countsOutOfRangeAreRejected() {
        fun ints(vararg v: Int) = ByteBuffer.allocate(4 * v.size).order(ByteOrder.LITTLE_ENDIAN).apply { v.forEach { putInt(it) } }.array()
        for (bad in listOf(ints(-1), ints(1001), ints(1, -1), ints(1, 65))) {
            failure<CzechSpeechException> { NBest.parse(bad) }
        }
        val negativeText = ByteBuffer.allocate(4 + 4 + 8 + 8 + 4 + 4 + 4 + 4).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(1).putInt(1).putDouble(-1.0).putDouble(-0.5).putFloat(0f).putInt(2).putInt(1).putInt(-1).array()
        failure<CzechSpeechException> { NBest.parse(negativeText) }
    }

    @Test
    fun duplicateTextsMergeKeepingTheBetterProbabilityAndWhispersChoice() {
        // rescore.parse_nbest: same stripped text -> one candidate, the better log-probability, the best flag kept
        val c = VariantChooser.candidates(
            window(
                Hyp(" Ahoj.", -3.0, 4, score = -0.5),
                Hyp(" Ahoj. ", -2.0, 5, best = true, score = -0.4),
                Hyp(" Čau.", -2.5, 4, score = -0.6),
                Hyp("Ahoj.", -4.0, 4, score = -0.9),
            ),
        )
        assertEquals(listOf("Ahoj.", "Čau."), c.map { it.text })
        assertTrue(c[0].best)
        assertEquals(-2.0, c[0].sumLogprob, 0.0)
        assertEquals(5, c[0].length)
    }

    @Test
    fun whispersChoiceComesFirstThenByWhispersScore() {
        val c = VariantChooser.candidates(
            window(
                Hyp(" a", -1.0, 2, score = -0.9),
                Hyp(" b", -1.0, 2, score = -0.1),
                Hyp(" c", -1.0, 2, best = true, score = -0.5),
                Hyp(" d", -1.0, 2, score = -0.1),
            ),
        )
        assertEquals(listOf("c", "b", "d", "a"), c.map { it.text })
    }

    @Test
    fun theFormulaDecides() {
        val c = VariantChooser.candidates(
            window(
                Hyp(" Viděl jsem velký psy.", -2.0, 5, best = true, score = -0.4),
                Hyp(" Viděl jsem velké psy.", -2.3, 5, score = -0.46),
            ),
        )
        // whisper alone prefers the first; the language model the second
        assertEquals(1, VariantChooser.choose(c, doubleArrayOf(-50.0, -40.0), lambda = 0.3, beta = 0.3))
        assertEquals(0, VariantChooser.choose(c, doubleArrayOf(-50.0, -49.5), lambda = 0.3, beta = 0.3))
        assertEquals(0, VariantChooser.choose(c, doubleArrayOf(-50.0, -40.0), lambda = 0.0, beta = 0.3))
    }

    @Test
    fun theLengthBonusFavoursLongerHypotheses() {
        val c = VariantChooser.candidates(
            window(Hyp(" krátká", -1.0, 3, best = true), Hyp(" krátká věta", -1.5, 5)),
        )
        assertEquals(0, VariantChooser.choose(c, doubleArrayOf(0.0, 0.0), lambda = 0.3, beta = 0.0))
        assertEquals(1, VariantChooser.choose(c, doubleArrayOf(0.0, 0.0), lambda = 0.3, beta = 0.3))
    }

    @Test
    fun aTieGoesToTheEarlierCandidate() {
        val c = VariantChooser.candidates(window(Hyp(" a", -1.0, 2, best = true), Hyp(" b", -1.0, 2)))
        assertEquals(0, VariantChooser.choose(c, doubleArrayOf(-5.0, -5.0), lambda = 0.3, beta = 0.3))
        // impossible hypotheses never win, and when all are, Whisper's choice stays
        val inf = VariantChooser.candidates(
            window(Hyp(" a", Double.NEGATIVE_INFINITY, 2, best = true), Hyp(" b", Double.NEGATIVE_INFINITY, 2)),
        )
        assertEquals(0, VariantChooser.choose(inf, doubleArrayOf(-1.0, -1.0), lambda = 0.3, beta = 0.3))
    }

    @Test
    fun theLanguageModelIsNeededOnlyWhenTheWordsDiffer() {
        assertFalse(VariantChooser.needsLanguageModel(VariantChooser.candidates(window(Hyp(" Ahoj.", -1.0, 3, true)))))
        assertFalse(
            VariantChooser.needsLanguageModel(
                VariantChooser.candidates(
                    window(Hyp(" Ahoj, Petře.", -1.0, 5, true), Hyp(" ahoj Petře!", -1.2, 5), Hyp(" Ahoj Petře", -1.3, 4)),
                ),
            ),
        )
        assertTrue(
            VariantChooser.needsLanguageModel(
                VariantChooser.candidates(window(Hyp(" Ahoj, Petře.", -1.0, 5, true), Hyp(" Ahoj, Petr.", -1.2, 5))),
            ),
        )
    }

    @Test
    fun anEmptyWindowHasNoCandidates() {
        assertTrue(VariantChooser.candidates(emptyList()).isEmpty())
    }

    @Test
    fun aVariantWithoutLettersOrDigitsIsNeverACandidateUnlessWhisperChoseIt() {
        // With suppress_blank off, a beam can end the text at once; the language model gives the empty text 0, its
        // maximum, so it would win against any sentence and the utterance would be dropped.
        val sentence = Hyp(" Dobrý den, posílám vám zprávu z vlaku.", -5.0, 12, best = true, score = -0.4)
        val c = VariantChooser.candidates(window(sentence, Hyp("", -4.0, 1, score = -4.0), Hyp(" ...", -4.5, 2)))
        assertEquals(listOf("Dobrý den, posílám vám zprávu z vlaku."), c.map { it.text })
        // Whisper's own choice stays a candidate even when it is empty
        val chosenEmpty = VariantChooser.candidates(window(Hyp("", -0.1, 1, best = true), Hyp(" Ahoj.", -3.0, 3)))
        assertEquals(listOf("", "Ahoj."), chosenEmpty.map { it.text })
    }

    @Test
    fun theTemperatureOfAFallbackIsRead() {
        val w = window(Hyp(" a", -9.0, 2, best = true, temperature = 0.4f), Hyp(" b", -9.5, 2, temperature = 0.4f))
        assertEquals(0.4f, w[0].temperature, 0f)
        assertTrue(VariantChooser.atFallbackTemperature(w))
        assertFalse(VariantChooser.atFallbackTemperature(window(Hyp(" a", -1.0, 2, best = true))))
    }

    @Test
    fun windowsAreCounted() {
        assertFalse(NBest.hasWindows(NBestBytes.encode(emptyList())))
        assertFalse(NBest.hasWindows(ByteArray(0)))
        assertTrue(NBest.hasWindows(NBestBytes.encode(listOf(emptyList()))))
    }

    @Test
    fun whisperChoosingNoTextIsRecognised() {
        val noText = VariantChooser.candidates(window(Hyp(" ...", -0.2, 2, best = true), Hyp(" Ahoj.", -3.0, 3)))
        assertTrue(VariantChooser.whisperChoseNoText(noText))
        val text = VariantChooser.candidates(window(Hyp(" Ahoj.", -0.2, 3, best = true), Hyp("", -3.0, 1)))
        assertFalse(VariantChooser.whisperChoseNoText(text))
        val noBest = VariantChooser.candidates(window(Hyp(" Ahoj.", -0.2, 3), Hyp(" Čau.", -3.0, 3)))
        assertFalse(VariantChooser.whisperChoseNoText(noBest))
        assertFalse(VariantChooser.whisperChoseNoText(emptyList()))
    }

    @Test
    fun aWindowWithoutWhispersChoiceIsOrderedByScore() {
        val c = VariantChooser.candidates(window(Hyp(" a", -1.0, 2, score = -0.7), Hyp(" b", -1.0, 2, score = -0.2)))
        assertEquals(listOf("b", "a"), c.map { it.text })
        assertFalse(c.any { it.best })
        assertEquals(1, VariantChooser.choose(c, doubleArrayOf(-9.0, -1.0), lambda = 0.3, beta = 0.3))
    }
}
