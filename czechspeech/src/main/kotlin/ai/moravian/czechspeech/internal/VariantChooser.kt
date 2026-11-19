package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import java.nio.BufferOverflowException
import java.nio.BufferUnderflowException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One hypothesis of Whisper's beam search: its text as Whisper wrote it, the sum of its token log-probabilities
 * ([sumLogprob], end token included), whisper.cpp's ranking [score], its [length] in Whisper tokens, whether
 * whisper.cpp chose it ([best]), and the [temperature] of the decoding attempt (above 0 in a fallback, whose
 * log-probabilities are of the logits divided by it).
 */
internal class Hypothesis(
    val text: String,
    val sumLogprob: Double,
    val score: Double,
    val length: Int,
    val best: Boolean,
    val temperature: Float = 0f,
)

/** Reads the hypotheses of the native n-best call (layout in czechspeech_jni.c, cs_put_hypotheses), per window. */
internal object NBest {
    private const val MAX_WINDOWS = 1_000
    private const val MAX_HYPOTHESES = 64

    /** @throws CzechSpeechException when [bytes] are not in that layout (a bug, never expected). */
    fun parse(bytes: ByteArray): List<List<Hypothesis>> {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        try {
            val windows = buf.int
            if (windows !in 0..MAX_WINDOWS) malformed()
            val result = List(windows) {
                val n = buf.int
                if (n !in 0..MAX_HYPOTHESES) malformed()
                List(n) {
                    val sumLogprob = buf.double
                    val score = buf.double
                    val temperature = buf.float
                    val length = buf.int
                    val best = buf.int != 0
                    val textBytes = buf.int
                    if (textBytes < 0 || textBytes > buf.remaining()) malformed()
                    val text = String(bytes, buf.position(), textBytes, Charsets.UTF_8)
                    buf.position(buf.position() + textBytes)
                    Hypothesis(text, sumLogprob, score, length, best, temperature)
                }
            }
            if (buf.hasRemaining()) malformed()
            return result
        } catch (e: BufferUnderflowException) {
            malformed()
        } catch (e: BufferOverflowException) {
            malformed()
        }
    }

    /** Whether [bytes] hold at least one window: none when whisper_full stopped before the encoder. */
    fun hasWindows(bytes: ByteArray): Boolean =
        bytes.size >= 4 && ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt(0) > 0

    private fun malformed(): Nothing = throw CzechSpeechException("Malformed hypotheses from the recogniser")
}

/**
 * Variant selection, as measured: Whisper's hypotheses of a
 * window are deduplicated by their trimmed text and ordered with Whisper's choice first, then by whisper.cpp's score;
 * each gets score = log P_Whisper + λ · log P_LM + β · its length in Whisper tokens, and the first highest one wins.
 *
 * Unlike the measurement, a hypothesis without a letter or digit is no candidate unless Whisper chose it: with
 * suppress_blank off a beam can end the text at once, and the language model gives the empty text 0, its highest
 * score, so it would beat any sentence and the utterance would be dropped. The other way round, when Whisper's own
 * choice has no letter or digit, it stays ([whisperChoseNoText]).
 */
internal object VariantChooser {
    class Candidate(
        /** Trimmed: what the language model scores and what deduplicates. */
        val text: String,
        /** As Whisper wrote it (a leading space), joined to the other windows' choices. */
        val raw: String,
        val sumLogprob: Double,
        val score: Double,
        val length: Int,
        val best: Boolean,
    )

    fun candidates(window: List<Hypothesis>): List<Candidate> {
        val seen = LinkedHashMap<String, Candidate>()
        for (h in window) {
            val text = h.text.trim()
            val old = seen[text]
            if (old == null || h.sumLogprob > old.sumLogprob || h.best) {
                seen[text] = Candidate(
                    text = text,
                    raw = h.text,
                    sumLogprob = if (old == null) h.sumLogprob else maxOf(h.sumLogprob, old.sumLogprob),
                    score = h.score,
                    length = h.length,
                    best = h.best || (old?.best ?: false),
                )
            }
        }
        return seen.values
            .filter { it.best || it.text.any { c -> c.isLetterOrDigit() } }
            .sortedWith(compareBy<Candidate> { !it.best }.thenByDescending { it.score })
    }

    /**
     * True when Whisper decoded the window at a fallback temperature: its log-probabilities are then of the logits
     * divided by the temperature, another scale than λ and β were tuned for, so Whisper's choice stays.
     */
    fun atFallbackTemperature(window: List<Hypothesis>): Boolean = window.any { it.temperature > 0f }

    /**
     * True when Whisper's own choice (the first candidate) has no letter or digit: Whisper says there is no speech, and
     * the text of a beam it turned down must not override that, neither through the language model nor the formula.
     */
    fun whisperChoseNoText(candidates: List<Candidate>): Boolean =
        candidates.firstOrNull()?.let { it.best && it.text.none { c -> c.isLetterOrDigit() } } ?: false

    /**
     * False when the candidates have the same words (differing at most in case and punctuation): the language model
     * then cannot change the words, and Whisper's choice stays. The measurement found 16 % of utterances like that.
     */
    fun needsLanguageModel(candidates: List<Candidate>): Boolean =
        candidates.size > 1 && candidates.map { words(it.text) }.distinct().size > 1

    /** Index of the winner: the first candidate with the highest combined score. */
    fun choose(candidates: List<Candidate>, lm: DoubleArray, lambda: Double, beta: Double): Int {
        require(lm.size == candidates.size) { "one language model score per candidate" }
        var best = 0
        var bestScore = Double.NEGATIVE_INFINITY
        for ((i, c) in candidates.withIndex()) {
            val s = c.sumLogprob + lambda * lm[i] + beta * c.length
            if (s > bestScore + 1e-12) {
                best = i
                bestScore = s
            }
        }
        return best
    }

    private val NON_WORD = Regex(SdkPatterns.NON_WORD)

    private fun words(text: String): String =
        text.lowercase().split(NON_WORD).filter { it.isNotEmpty() }.joinToString(" ")
}
