package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.PreviousEnd
import ai.moravian.czechspeech.SentenceJoin
import kotlin.math.ln

/**
 * Sentence joining, the rules without the engine: does transcript B continue the sentence of the transcript before it,
 * A, although a pause split them? Whisper then often ends A with a period and starts B with a capital ("…můžeš nějak. |
 * S tebou spojit…"), or writes a capital without a period ("A B", "A, B").
 *
 * A port of the decision measured on 12 Nov 2026: the candidate texts of the boundary are scored
 * by the Czech GPT-2 of variant selection, on a window of the last [WORDS_BEFORE] words of A and the first
 * [WORDS_AFTER] of B, and A and B are joined when the best joined candidate beats the end of the sentence by more than
 * [JOIN_MARGIN] (natural log P), [JOIN_MARGIN_AFTER_LOWER] when Whisper wrote "A. b". A period is never added, and a
 * capital is lowered only when the language model prefers the small letter by more than [CASE_MARGIN]: in the
 * measurement no name of a person lost its capital, multi-word names and institutions did in 2 to 6 % of the cases
 * ("Ministerstva zemědělství"). On the measurement's test sets this fixed 42.5 % of the false periods and took the
 * period of 0.9 % of the true sentence ends (0.4 % of 770 in the simulation).
 *
 * Whitespace, letters, digits and case follow Python's `str` methods, as in the measurement.
 */
internal object SentenceBoundary {
    /** How Whisper wrote the boundary between A and B (`raw_class` of the measurement). */
    enum class Form {
        /** "A. B": A ends with a period that may end a sentence, B starts with a capital or a digit (PU). */
        PERIOD_UPPER,

        /** "A. b": the same period, B starts with a small letter (PL; the author's "…nějak. s tebou…"). */
        PERIOD_LOWER,

        /** "A B": no punctuation at the end of A, B starts with a capital (NU). */
        UPPER,

        /** "A, B": a comma at the end of A, B starts with a capital (CU). */
        COMMA_UPPER,

        /** "A, b": a comma at the end of A, B starts with a small letter: only the comma may go. */
        COMMA_LOWER,

        /**
         * "A… B" or "A... b": A ends in an ellipsis (two or more dots, or "…", closers after it allowed) that follows a
         * letter or digit. Usually an artefact of the pause, so it is decided by the language model with a prior for removing it.
         */
        ELLIPSIS,

        /** Anything else (a question mark, "A b", an abbreviation or ordinal before B …): left as it is. */
        OTHER,
    }

    /** The candidates of the measurement, and [RAW]: Whisper's text unchanged. */
    enum class Choice {
        /** The end of a sentence: "A. B" (`split`). */
        SPLIT,

        /** Joined with a small letter: "A b" (`join`). */
        JOIN,

        /** Joined with B's capital kept, as for a name: "A B" (`join_keep`). */
        JOIN_KEEP,

        /** Joined by a comma: "A, b" (`comma`). */
        COMMA,

        /** After an ellipsis: the ellipsis stays and the next word is small, "A… b". */
        ELLIPSIS,

        /** After a trailing comma: the comma goes and the next word is small, "A, b" → "A b". */
        UNCOMMA,

        /** After a trailing comma: the comma goes and the capital stays (a name), "A, B" → "A B". */
        UNCOMMA_KEEP,

        /** No decision: Whisper's text as it is (`raw`). */
        RAW,
    }

    /**
     * θ, natural log P: the lowest margin at which at most 1 % of the true sentence ends of the measurement's tuning set
     * lost their period (`tune_theta`, window of 20 + 12 words, with the comma candidate), exactly as the measured code
     * used it (the margin of one tuning boundary, a float32). The report rounds it to 5.56 and 5.6.
     */
    const val JOIN_MARGIN = 5.5557098388671875

    /**
     * θ for "A. b": [JOIN_MARGIN] less the log of how much likelier Whisper's small letter after a period makes a false
     * boundary (`prior_shift` on the tuning set: 16 false to 32 true boundaries written "A. b", 15 to 203 written "A. B",
     * each + 0.5). 3.6588; the report rounds it to 3.7.
     */
    val JOIN_MARGIN_AFTER_LOWER: Double = JOIN_MARGIN - ln((16.5 / 32.5) / (15.5 / 203.5))

    /**
     * δ: a capital is lowered only when "A b" beats "A B" by more than this (`tune_case`: at most 2 % of the capitals
     * inside sentences of the tuning set lowered).
     */
    const val CASE_MARGIN = 2.5

    /**
     * A piece that follows a cut on a quiet frame of continuous speech (not real silence): the quiet frame is a pause
     * shorter than the endpointer's minimum silence, typically a comma or a breath, so the decision leans to going on.
     * The joined candidates win unless the end of the sentence beats the best of them by MORE than this (natural log P,
     * a factor of 7.4): the threshold θ is then −[QUIET_CUT_SPLIT_MARGIN] (for "A. b" less by the same shift as there),
     * not +[JOIN_MARGIN]. NOT measured: the measurement covers pauses of 600 ms and more, where θ = 0 (no prior at all)
     * would take the period from 17 % of the true sentence ends; there is no labelled data of quiet cuts in the
     * repository. 2.0 is the conservative end: "a clear margin", a sentence end whose two readings the model finds
     * about equally good is not kept.
     */
    const val QUIET_CUT_SPLIT_MARGIN = 2.0

    /**
     * δ for the first word of a piece that continues the sentence (after a quiet cut that was decided to go on, and
     * after a cut on speech): its capital is the start of the piece's own doing, so it is lowered when the language model
     * prefers the small letter by more than this, much less than [CASE_MARGIN], which guards the capitals Whisper wrote
     * inside a sentence of its own (names). A name keeps its capital because the model prefers it. NOT measured either;
     * the measurement says that no name of a person lost its capital at 2.5, and the first word of a piece is a common
     * word in the vast majority of cases.
     */
    const val CASE_MARGIN_CONTINUED = 1.0

    /**
     * After an ellipsis (an artefact of the pause more often than a trailing off, and a real trailing off at the end of
     * the dictation has no next piece): the ellipsis stays ("A… b") only when the model prefers that over the best removal
     * by more than this, and a new sentence ("A… B") only when it beats the best removal by more than
     * [ELLIPSIS_SPLIT_MARGIN]. Not measured (no labelled data); the conservative end of "clearly".
     */
    const val ELLIPSIS_KEEP_MARGIN = 2.0
    const val ELLIPSIS_SPLIT_MARGIN = 2.0

    /**
     * The prior of the ellipsis margins ([ELLIPSIS_KEEP_MARGIN], [ELLIPSIS_SPLIT_MARGIN]) applies when the two speeches are less than
     * this far apart (the pause the endpointer knows: the start of the next piece less the end of the previous one) and always after
     * a quiet cut. After a longer real pause a trailing off is likelier real, so the readings are compared without it.
     */
    const val ELLIPSIS_PAUSE_LIMIT_MS = 2_000L

    /**
     * A comma ("A, b") is chosen over the plain continuation ("A b") only when the model prefers it by more than this: in
     * doubt, no comma. Used where the decision leans to going on (after a quiet cut, after an ellipsis). The measured
     * decision after a real pause (best joined candidate) is unchanged. Not measured.
     */
    const val COMMA_OVER_PLAIN_MARGIN = 1.5

    /**
     * A comma at the end of the previous piece goes ("A, b" → "A b") only when the model prefers the text without it by more
     * than this: Whisper writes it at a pause that may be in the wrong place. Not measured.
     */
    const val COMMA_REMOVE_MARGIN = 1.5

    /** Words of A and of B in the language model's window (`w20`). */
    const val WORDS_BEFORE = 20
    const val WORDS_AFTER = 12

    /** No join across a longer pause (B's start minus A's end, speech to speech): a safeguard of the report. */
    const val MAX_PAUSE_MS = 3_000L

    /** The joined candidates the decision compares with [Choice.SPLIT], in the measurement's order (`USE`). */
    private val JOINS = listOf(Choice.JOIN, Choice.JOIN_KEEP, Choice.COMMA)

    private const val CLOSERS = "\"'“”„»«)]"
    private const val SENTENCE_END = ".?!…"
    private val ORDINAL = Regex(SdkPatterns.ORDINAL)
    private val ABBREVIATIONS = setOf(
        "např", "tzv", "tj", "tzn", "atd", "apod", "resp", "cca", "č", "odst", "písm", "mj", "aj", "popř", "př",
        "str", "sb", "mld", "mil", "tis", "kč", "hod", "min", "ing", "mgr", "dr", "doc", "prof", "judr", "mudr",
        "phdr", "rndr", "st", "sv", "ul", "nám", "tř", "pí", "odd", "zn", "vč", "ev", "příp",
        // legal text: čl., zák., s.r.o. and a.s. (their points are not letters: "sro", "as"), sp. zn. is "sp"
        "čl", "zák", "sro", "as", "sp",
    )

    // --- text, as Python's str ----------------------------------------------------------------------------------

    /** `str.isalnum` of one character: a letter or any number (decimal, letter-like, other). */
    private fun isAlnum(c: Char): Boolean = c.isLetter() || when (Character.getType(c).toByte()) {
        Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
        else -> false
    }

    /** `str.split()`: the words between runs of whitespace. */
    fun words(text: String): List<String> {
        val out = ArrayList<String>()
        var i = 0
        while (i < text.length) {
            while (i < text.length && text[i].isWhitespace()) i++
            val start = i
            while (i < text.length && !text[i].isWhitespace()) i++
            if (i > start) out += text.substring(start, i)
        }
        return out
    }

    private fun firstAlnum(word: String): Int = word.indexOfFirst(::isAlnum)

    fun startsUpper(word: String): Boolean = firstAlnum(word).let { it >= 0 && word[it].isUpperCase() }

    private fun startsDigit(word: String): Boolean = firstAlnum(word).let { it >= 0 && word[it].isDigit() }

    /** More than one capital (ODS, ČR, McDonald): the first letter never changes. */
    private fun isAcronym(word: String): Boolean = word.count { it.isUpperCase() } > 1

    fun lowerFirst(word: String): String {
        val i = firstAlnum(word)
        if (i < 0 || !word[i].isUpperCase() || isAcronym(word)) return word
        return word.substring(0, i) + word[i].lowercase() + word.substring(i + 1)
    }

    fun upperFirst(word: String): String {
        val i = firstAlnum(word)
        if (i < 0 || !word[i].isLetter()) return word
        return word.substring(0, i) + word[i].uppercase() + word.substring(i + 1)
    }

    /** The punctuation at the end of a word, closing quotes and brackets left out ("", ".", "?", "..." …). */
    fun endPunct(word: String): String {
        val w = word.trimEnd { it in CLOSERS }
        var j = w.length
        while (j > 0 && !isAlnum(w[j - 1])) j--
        return w.substring(j)
    }

    private fun normWord(word: String): String = word.lowercase().filter { isAlnum(it) || it == '_' }

    /** A number or an acronym, or no letter at all: its first letter has no case to decide. */
    private fun caseless(word: String): Boolean = startsDigit(word) || isAcronym(word) || firstAlnum(word) < 0

    /** A ends with one period that may end a sentence: not "...", not an ordinal ("17.") nor an abbreviation ("např."). */
    fun periodEnd(a: String): Boolean {
        val last = words(a).lastOrNull()?.trimEnd { it in CLOSERS } ?: return false
        if (!last.endsWith(".") || last.endsWith("..")) return false
        return !ORDINAL.containsMatchIn(last) && normWord(last) !in ABBREVIATIONS
    }

    /** A without the period at its end (before closing quotes or brackets, which stay), trailing whitespace dropped. */
    fun stripPeriod(a: String): String {
        val t = a.trimEnd()
        val core = t.trimEnd { it in CLOSERS }
        val closers = t.substring(core.length)
        return (if (core.endsWith(".") && !core.endsWith("..")) core.dropLast(1) else core) + closers
    }

    /**
     * [PreviousEnd.applyTo]: the period at the end of [previous] deleted or made a comma (before closers and whitespace,
     * which stay); [previous] as it is when it does not end with a period that may end a sentence ([periodEnd]: not
     * an ellipsis, an abbreviation, an ordinal, nor any other punctuation).
     */
    fun editPreviousEnd(previous: String, end: PreviousEnd): String {
        when (end) {
            PreviousEnd.REMOVE_ELLIPSIS -> return replaceTail(previous, ellipsisLength(previous), ellipsisReplacement(previous, ""))
            PreviousEnd.ELLIPSIS_TO_COMMA -> return replaceTail(previous, ellipsisLength(previous), ellipsisReplacement(previous, ","))
            PreviousEnd.REMOVE_COMMA -> return replaceTail(previous, if (commaEnd(previous)) 1 else 0, "")
            else -> Unit
        }
        if (end == PreviousEnd.KEEP || !periodEnd(previous)) return previous
        val body = previous.trimEnd()
        val core = body.trimEnd { it in CLOSERS }
        if (!core.endsWith(".") || core.endsWith("..")) return previous
        val comma = if (end == PreviousEnd.PERIOD_TO_COMMA) "," else ""
        return core.dropLast(1) + body.substring(core.length) + comma + previous.substring(body.length)
    }

    /** [text] without trailing whitespace and closers: where its words end. */
    private fun core(text: String): String = text.trimEnd().trimEnd { it in CLOSERS }

    /**
     * How many characters of the ellipsis at the end of [text] (before closers and whitespace): a run of two or more dots, or
     * "…" (alone or after dots), that follows a letter or digit; 0 when there is none ("…" or "..." alone, one period).
     */
    private fun ellipsisLength(text: String): Int {
        val c = core(text)
        var j = c.length
        while (j > 0 && (c[j - 1] == '.' || c[j - 1] == '…')) j--
        val run = c.length - j
        if (run == 0 || j == 0 || !isAlnum(c[j - 1])) return 0
        val dots = c.substring(j)
        return if (dots.contains('…') || run >= 2) run else 0
    }

    /**
     * What replaces the ellipsis at the end of [text]: [with], after the period of an abbreviation that the dots belong to ("atd..."
     * → "atd." and "atd.,"; "např.." → "např."), which stays.
     */
    private fun ellipsisReplacement(text: String, with: String): String {
        val before = core(text).dropLast(ellipsisLength(text))
        val word = before.substring(before.lastIndexOfAny(charArrayOf(' ', '\t', '\n', ' ')) + 1)
        return (if (normWord(word) in ABBREVIATIONS) "." else "") + with
    }

    /** A ends in an ellipsis ([ellipsisLength]). */
    fun ellipsisEnd(a: String): Boolean = ellipsisLength(a) > 0

    /** A ends in a comma (before closers and whitespace). */
    fun commaEnd(a: String): Boolean = core(a).endsWith(",")

    /** [text] with [length] characters taken from the end of its core (before closers and whitespace) and [with] put there. */
    private fun replaceTail(text: String, length: Int, with: String): String {
        if (length == 0) return text
        val body = text.trimEnd()
        val c = core(text)
        return c.dropLast(length) + with + body.substring(c.length) + text.substring(body.length)
    }

    private fun mapFirstWord(b: String, f: (String) -> String): String {
        val space = b.indexOf(' ')
        return if (space < 0) f(b) else f(b.substring(0, space)) + b.substring(space)
    }

    fun lowerFirstWord(b: String): String = mapFirstWord(b, ::lowerFirst)

    fun upperFirstWord(b: String): String = mapFirstWord(b) { if (isAcronym(it)) it else upperFirst(it) }

    // --- the boundary --------------------------------------------------------------------------------------------

    fun form(a: String, b: String): Form {
        val aw = words(a)
        val bw = words(b)
        if (aw.isEmpty() || bw.isEmpty() || firstAlnum(bw[0]) < 0) return Form.OTHER
        if (periodEnd(a)) {
            return if (startsUpper(bw[0]) || caseless(bw[0])) Form.PERIOD_UPPER else Form.PERIOD_LOWER
        }
        if (ellipsisEnd(a)) return Form.ELLIPSIS
        val end = endPunct(aw.last())
        if ((end == "" || end == ",") && startsUpper(bw[0]) && !caseless(bw[0])) {
            return if (end == "") Form.UPPER else Form.COMMA_UPPER
        }
        if (end == "," && !caseless(bw[0])) return Form.COMMA_LOWER
        return Form.OTHER
    }

    /**
     * The candidates (new A, new B) of the boundary, by choice (`candidates` with the comma candidate): the same for
     * "A. B", "A. b" and "A B"; after a comma only [Choice.JOIN] ("A, b") and [Choice.JOIN_KEEP] ("A, B").
     * [Choice.JOIN] only when B's first letter can be lowered.
     */
    fun candidates(a: String, b: String): Map<Choice, Pair<String, String>> {
        val period = periodEnd(a)
        val a0 = if (period) stripPeriod(a) else a
        val upper = upperFirstWord(b)
        val lower = lowerFirstWord(b)
        val aw = words(a)
        val out = LinkedHashMap<Choice, Pair<String, String>>()
        if (ellipsisEnd(a)) {
            // an ellipsis: the new sentence as written, the ellipsis kept with a small letter, and the three ways of removing it
            val e0 = replaceTail(a, ellipsisLength(a), ellipsisReplacement(a, ""))
            out[Choice.SPLIT] = a to upper
            if (lower != upper) out[Choice.ELLIPSIS] = a to lower
            out[Choice.JOIN_KEEP] = e0 to upper
            if (lower != upper) out[Choice.JOIN] = e0 to lower
            out[Choice.COMMA] = replaceTail(a, ellipsisLength(a), ellipsisReplacement(a, ",")) to (if (lower != upper) lower else upper)
            return out
        }
        if (!period && aw.isNotEmpty() && endPunct(aw.last()) == ",") {
            // a comma: kept ("A, B", "A, b") or removed ("A B", "A b")
            val c0 = replaceTail(a, 1, "")
            out[Choice.JOIN_KEEP] = a to upper
            if (lower != upper) out[Choice.JOIN] = a to lower
            out[Choice.UNCOMMA_KEEP] = c0 to upper
            if (lower != upper) out[Choice.UNCOMMA] = c0 to lower
            return out
        }
        out[Choice.SPLIT] = (if (period) a else "$a0.") to upper
        out[Choice.JOIN_KEEP] = a0 to upper
        if (lower != upper) {
            out[Choice.JOIN] = a0 to lower
            out[Choice.COMMA] = "$a0," to lower
        } else {
            out[Choice.COMMA] = "$a0," to upper
        }
        return out
    }

    /**
     * What the language model scores for the boundary: each candidate on the window of the last [WORDS_BEFORE] words
     * of A and the first [WORDS_AFTER] of B, joined by a space (`lm_texts`). Only the candidates the decision for
     * [form] compares: all for "A. B" and "A. b", [Choice.JOIN] and [Choice.JOIN_KEEP] for "A B" and "A, B", none
     * for [Form.OTHER].
     */
    fun languageModelTexts(a: String, b: String, form: Form = form(a, b)): Map<Choice, String> {
        if (form == Form.OTHER) return emptyMap()
        val wa = words(a).takeLast(WORDS_BEFORE).joinToString(" ")
        val wb = words(b).take(WORDS_AFTER).joinToString(" ")
        val needed = when (form) {
            Form.UPPER -> setOf(Choice.JOIN, Choice.JOIN_KEEP)
            Form.COMMA_UPPER -> setOf(Choice.JOIN, Choice.JOIN_KEEP, Choice.UNCOMMA, Choice.UNCOMMA_KEEP)
            Form.COMMA_LOWER -> setOf(Choice.JOIN, Choice.UNCOMMA)
            else -> null
        }
        return candidates(wa, wb)
            .filterKeys { needed == null || it in needed }
            .mapValues { (_, ab) -> "${ab.first} ${ab.second}" }
    }

    /**
     * The decision for a boundary written as [form], from the language model's [scores] of its candidates
     * (`decide_class` with θ = [JOIN_MARGIN], [JOIN_MARGIN_AFTER_LOWER] for "A. b", δ = [CASE_MARGIN] and the
     * comma candidate). "A. B" and "A. b" are joined when the best joined candidate beats [Choice.SPLIT] by more
     * than θ; "A. b" left split gets a capital. "A B" and "A, B" only ever get a small letter: a period is never
     * added. A small letter only when [Choice.JOIN] beats [Choice.JOIN_KEEP] by more than δ ("A. b": a capital
     * only when [Choice.JOIN_KEEP] beats [Choice.JOIN] by more than δ). Without scores, [Choice.RAW].
     *
     * With [quiet] (the piece follows a cut on a quiet frame of continuous speech, not real silence) the decision leans
     * to going on for the period forms: the new sentence wins only when it is likelier than the BEST continuing candidate
     * (plain continuation or comma) by more than [QUIET_CUT_SPLIT_MARGIN] (plus, for "A. b", the shift above); the comma
     * replaces the plain continuation only by [COMMA_OVER_PLAIN_MARGIN], and a rejected comma leaves the plain continuation,
     * which is compared with the new sentence by the same margin; the first word loses its capital by [CASE_MARGIN_CONTINUED]
     * instead of [CASE_MARGIN] (a name the model prefers keeps it). After real silence the period forms ("A. B", "A. b",
     * "A B", "A, B") are the measured decision, unchanged; the ellipsis and the trailing comma that may go are decisions of
     * their own (see [Form.ELLIPSIS], [COMMA_REMOVE_MARGIN]), and [ellipsisPrior] says whether the prior to removing an ellipsis applies.
     */
    fun decide(
        form: Form, scores: Map<Choice, Double>?, quiet: Boolean = false, commaOverPlain: Double = COMMA_OVER_PLAIN_MARGIN,
        ellipsisPrior: Boolean = true,
    ): Choice {
        if (scores == null || form == Form.OTHER) return Choice.RAW
        if (form == Form.ELLIPSIS) return decideEllipsis(scores, ellipsisPrior)
        if (form == Form.COMMA_LOWER) {
            val join = scores[Choice.JOIN] ?: return Choice.RAW
            val uncomma = scores[Choice.UNCOMMA] ?: return Choice.RAW
            return if (uncomma - join > COMMA_REMOVE_MARGIN) Choice.UNCOMMA else Choice.RAW
        }
        val join = scores[Choice.JOIN]
        val keep = scores.getValue(Choice.JOIN_KEEP)
        val caseMargin = if (quiet) CASE_MARGIN_CONTINUED else CASE_MARGIN
        val lower = if (form == Form.PERIOD_LOWER) {
            !(join != null && keep - join > CASE_MARGIN)
        } else {
            join != null && join - keep > caseMargin
        }
        if (form == Form.COMMA_UPPER) {
            // the comma goes when the text without it is clearly likelier than the best with it
            val uncomma = scores[Choice.UNCOMMA]
            val uncommaKeep = scores[Choice.UNCOMMA_KEEP]
            val without = listOfNotNull(uncomma, uncommaKeep).maxOrNull()
            if (without != null && without - maxOf(keep, join ?: keep) > COMMA_REMOVE_MARGIN) {
                return if (uncomma != null && (uncommaKeep == null || uncomma - uncommaKeep > caseMargin)) Choice.UNCOMMA else Choice.UNCOMMA_KEEP
            }
        }
        if (form == Form.UPPER || form == Form.COMMA_UPPER) return if (lower) Choice.JOIN else Choice.RAW
        val theta = if (form == Form.PERIOD_LOWER) JOIN_MARGIN_AFTER_LOWER else JOIN_MARGIN
        if (quiet) {
            // leaning to going on: the new sentence is compared with the BEST continuing candidate (the comma among them) by the
            // margin; among the continuing ones the comma needs its margin over the plain continuation; a rejected comma leaves the
            // plain continuation, which is then compared with the new sentence by the same margin
            val splitMargin = QUIET_CUT_SPLIT_MARGIN + (JOIN_MARGIN - theta)
            val split = scores[Choice.SPLIT]
            var best = JOINS.filter { it in scores }.maxByOrNull { scores.getValue(it) }
            if (best == null || split == null) return Choice.SPLIT
            if (split - scores.getValue(best) > splitMargin) return Choice.SPLIT
            if (best == Choice.COMMA) {
                val plain = listOf(Choice.JOIN, Choice.JOIN_KEEP).filter { it in scores }.maxByOrNull { scores.getValue(it) }
                if (plain != null && scores.getValue(Choice.COMMA) - scores.getValue(plain) <= commaOverPlain) {
                    best = plain
                    if (split - scores.getValue(plain) > splitMargin) return Choice.SPLIT
                }
            }
            return if ((best == Choice.JOIN || best == Choice.COMMA) && !lower) Choice.JOIN_KEEP else best
        }
        val best = JOINS.filter { it in scores }.maxByOrNull { scores.getValue(it) }
        val margin = margin(scores)
        val choice = if (best != null && margin != null && margin > theta) best else Choice.SPLIT
        return if ((choice == Choice.JOIN || choice == Choice.COMMA) && !lower) Choice.JOIN_KEEP else choice
    }

    /**
     * After an ellipsis at the end of A, with [prior] (the pause is short, or it is a quiet cut): the plain continuation (the first
     * word lowered when [Choice.JOIN] beats [Choice.JOIN_KEEP] by more than [CASE_MARGIN_CONTINUED]) or the comma, which needs
     * [COMMA_OVER_PLAIN_MARGIN] over it, is the removal; the ellipsis kept ("A… b") replaces it when it is likelier by more than
     * [ELLIPSIS_KEEP_MARGIN]; the new sentence ("A… B") wins when it is likelier than that best continuing reading by more than
     * [ELLIPSIS_SPLIT_MARGIN]. Without [prior] the two margins are 0: the likelier reading wins.
     */
    private fun decideEllipsis(scores: Map<Choice, Double>, prior: Boolean): Choice {
        val join = scores[Choice.JOIN]
        val keep = scores[Choice.JOIN_KEEP] ?: return Choice.RAW
        val lower = join != null && join - keep > CASE_MARGIN_CONTINUED
        val plainScore = if (join != null) maxOf(join, keep) else keep
        var best = if (lower) Choice.JOIN else Choice.JOIN_KEEP
        var bestScore = plainScore
        val comma = scores[Choice.COMMA]
        if (comma != null && comma - plainScore > COMMA_OVER_PLAIN_MARGIN) {
            best = if (lower) Choice.COMMA else Choice.JOIN_KEEP
            bestScore = comma
        }
        val ellipsis = scores[Choice.ELLIPSIS]
        if (ellipsis != null && ellipsis - bestScore > (if (prior) ELLIPSIS_KEEP_MARGIN else 0.0)) {
            best = Choice.ELLIPSIS
            bestScore = ellipsis
        }
        val split = scores[Choice.SPLIT]
        if (split != null && split - bestScore > (if (prior) ELLIPSIS_SPLIT_MARGIN else 0.0)) return Choice.SPLIT
        return best
    }

    /**
     * log P of the best joined candidate less that of the end of the sentence (`margin`): positive when the language
     * model prefers to join. Null without the end of the sentence or a joined candidate among [scores].
     */
    fun margin(scores: Map<Choice, Double>): Double? {
        val split = scores[Choice.SPLIT] ?: return null
        val best = JOINS.mapNotNull { scores[it] }.maxOrNull() ?: return null
        return best - split
    }

    /**
     * Decides the boundary between [a] and [b], scoring with [score] (one log P per text, null when the language
     * model failed: then [Choice.RAW]). [quiet]: see [decide] by form.
     */
    fun decide(a: String, b: String, score: (List<String>) -> DoubleArray?): Choice = decide(a, b, score, false)

    fun decide(a: String, b: String, score: (List<String>) -> DoubleArray?, quiet: Boolean): Choice = decide(a, b, score, quiet, true)

    /** [ellipsisPrior]: whether the prior to removing an ellipsis applies (see [ELLIPSIS_PAUSE_LIMIT_MS]); it always does when [quiet]. */
    fun decide(a: String, b: String, score: (List<String>) -> DoubleArray?, quiet: Boolean, ellipsisPrior: Boolean): Choice {
        val form = form(a, b)
        val texts = languageModelTexts(a, b, form)
        if (texts.isEmpty()) return Choice.RAW
        val values = score(texts.values.toList()) ?: return Choice.RAW
        if (values.size != texts.size || values.any { !it.isFinite() }) return Choice.RAW
        return decide(form, texts.keys.zip(values.toList()).toMap(), quiet, ellipsisPrior = ellipsisPrior || quiet)
    }

    /**
     * For a piece that continues the sentence of [a] for sure (a cut on speech): the two texts the language model scores to
     * decide the first word's capital, [Choice.JOIN] (small letter) and [Choice.JOIN_KEEP] (as it is), on the window of
     * [decide] by text. Empty when there is nothing to decide: the first word of [b] has no capital that could be
     * lowered (a small letter already, a number, an acronym).
     */
    fun continuationTexts(a: String, b: String): Map<Choice, String> {
        val first = words(b).firstOrNull() ?: return emptyMap()
        if (!startsUpper(first) || lowerFirst(first) == first) return emptyMap()
        val wa = words(a).takeLast(WORDS_BEFORE).joinToString(" ")
        val wb = words(b).take(WORDS_AFTER).joinToString(" ")
        val candidates = candidates(wa, wb)
        val keep = candidates[Choice.JOIN_KEEP] ?: return emptyMap()
        val join = candidates[Choice.JOIN] ?: return emptyMap()
        return linkedMapOf(Choice.JOIN to "${join.first} ${join.second}", Choice.JOIN_KEEP to "${keep.first} ${keep.second}")
    }

    /**
     * Whether the first word of a piece that continues the sentence loses its capital, by the [scores] of
     * [continuationTexts]: when the small letter beats the capital by more than [CASE_MARGIN_CONTINUED]. Without scores
     * (no language model, it failed, nothing scored) it does, as it always did.
     */
    fun lowerAfterCut(scores: Map<Choice, Double>?): Boolean {
        val join = scores?.get(Choice.JOIN) ?: return true
        val keep = scores[Choice.JOIN_KEEP] ?: return true
        return join - keep > CASE_MARGIN_CONTINUED
    }

    /**
     * What the log says of [choice] at the boundary of [a]: which variant won, never any text. "join" for the plain continuation
     * (with or without the capital kept), and the new variants by name.
     */
    fun label(a: String, choice: Choice, quiet: Boolean): String {
        val ellipsis = ellipsisEnd(a)
        val name = when (choice) {
            Choice.JOIN, Choice.JOIN_KEEP -> if (ellipsis) "ellipsis removed" else if (quiet) "join" else choice.name.lowercase()
            Choice.COMMA -> if (ellipsis) "ellipsis to comma" else "comma"
            Choice.ELLIPSIS -> "ellipsis kept"
            Choice.UNCOMMA, Choice.UNCOMMA_KEEP -> "comma removed"
            else -> choice.name.lowercase()
        }
        return if (quiet) "$name (quiet cut)" else name
    }

    /**
     * What a client needs for [choice] at the boundary of [a] and [b]: null when B stays as Whisper wrote it (no
     * decision, or the end of a sentence already written "A. B").
     */
    fun sentenceJoin(a: String, b: String, choice: Choice): SentenceJoin? {
        val standalone = upperFirstWord(b)
        val ellipsis = ellipsisEnd(a)
        val previousEnd = when {
            periodEnd(a) -> PreviousEnd.REMOVE_PERIOD
            ellipsis -> PreviousEnd.REMOVE_ELLIPSIS
            else -> PreviousEnd.KEEP
        }
        return when (choice) {
            Choice.RAW -> null
            Choice.SPLIT -> if (standalone == b) null else SentenceJoin(false, PreviousEnd.KEEP, standalone, standalone)
            Choice.JOIN -> SentenceJoin(true, previousEnd, lowerFirstWord(b), standalone)
            Choice.JOIN_KEEP -> SentenceJoin(true, previousEnd, standalone, standalone)
            Choice.COMMA -> SentenceJoin(true, if (ellipsis) PreviousEnd.ELLIPSIS_TO_COMMA else PreviousEnd.PERIOD_TO_COMMA, lowerFirstWord(b), standalone)
            Choice.ELLIPSIS -> SentenceJoin(true, PreviousEnd.KEEP, lowerFirstWord(b), standalone)
            Choice.UNCOMMA -> SentenceJoin(true, PreviousEnd.REMOVE_COMMA, lowerFirstWord(b), standalone)
            Choice.UNCOMMA_KEEP -> SentenceJoin(true, PreviousEnd.REMOVE_COMMA, standalone, standalone)
        }
    }
}
