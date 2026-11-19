package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechModel

/**
 * The language rule of one recognition: [shares] are the shares of the languages of the list (rescaled to sum 1),
 * [thresholds] one per language with the first one not read, NaN for a language that is not a candidate, otherwise 0..1;
 * null means no rule: the most probable language, the first of equals. With thresholds the candidate is the most
 * probable language with a threshold other than the first, and it is chosen when its share reaches its threshold;
 * otherwise the first language. Returns the index.
 *
 * A copy of `cs_pick_language` in native/src/czechspeech_jni.c, which decides inside whisper_full; a test compares the
 * two on a grid, so change them together.
 */
internal fun pickLanguage(shares: FloatArray, thresholds: FloatArray?): Int {
    var best = 0
    if (thresholds == null) {
        for (i in 1 until shares.size) if (shares[i] > shares[best]) best = i
        return best
    }
    var candidate = -1
    for (i in 1 until shares.size) {
        if (!thresholds[i].isNaN() && (candidate < 0 || shares[i] > shares[candidate])) candidate = i
    }
    return if (candidate >= 0 && shares[candidate] >= thresholds[candidate]) candidate else 0
}

/**
 * The rule of one recognition call, which turns the state of the session (the language it is in, [current]) and the
 * length of the audio the recogniser gets into the thresholds of [pickLanguage]. It holds no state itself: the engine
 * keeps the session's language ([LanguageSession]) and the piece's, and asks this for every call.
 *
 * Without a [switch], or with one language, there are no thresholds: the recogniser takes the most probable language
 * of the list (the behaviour of the first published version), and nothing is rescued.
 *
 * With one: leaving the first language needs audio of at least `minLeaveMs`, returning to it `minReturnMs`; below the
 * minimum the language the session is in is certain. Between the two minimums a session in another language keeps
 * only that language as a candidate; from `minLeaveMs` all languages of the switch are candidates. A language that is
 * not in the switch is never a candidate.
 */
internal class LanguageRule(val languages: List<String>, private val switch: CzechSpeechModel.LanguageSwitch?) {
    val primary: String get() = languages[0]

    /** The rule is on: a switch and more than one language. */
    val active: Boolean get() = switch != null && languages.size > 1

    /** What the native layer gets as the languages: [forced] (one code, no recognition), or all of the list. */
    fun argument(forced: String?): String = forced ?: languages.joinToString(",")

    /**
     * The thresholds of a call over [audioMs] of audio in a session that is in [current], one per language of the list
     * (the first is not read): NaN is not a candidate, 0 is certain. Null without the rule.
     */
    fun thresholds(current: String, audioMs: Long): FloatArray? {
        val sw = switch?.takeIf { active } ?: return null
        val out = FloatArray(languages.size) { Float.NaN }
        val at = languages.indexOf(current)
        if (at <= 0) {
            // in the first language: others only from the minimum to leave
            if (audioMs >= sw.minLeaveMs) fill(out, sw)
            return out
        }
        when {
            audioMs < sw.minReturnMs -> out[at] = 0f
            audioMs < sw.minLeaveMs -> sw.thresholds[current]?.let { out[at] = it }
            else -> fill(out, sw)
        }
        return out
    }

    private fun fill(out: FloatArray, sw: CzechSpeechModel.LanguageSwitch) {
        for (i in 1 until languages.size) sw.thresholds[languages[i]]?.let { out[i] = it }
    }

    /** True when audio of [audioMs] lets the language of [current] change (so a pass of that length decides). */
    fun decisive(current: String, audioMs: Long): Boolean {
        val sw = switch?.takeIf { active } ?: return true
        return audioMs >= if (current == primary) sw.minLeaveMs else sw.minReturnMs
    }

    /**
     * The language to try once more after a run in [ran] gave nothing: the most probable language of the switch (the
     * first one, or one with a threshold) other than [ran], when its share in [shares] is at least [RESCUE_MIN_SHARE];
     * otherwise null, also without shares or without the rule.
     */
    fun rescue(ran: String, shares: FloatArray?): String? {
        val sw = switch?.takeIf { active } ?: return null
        if (shares == null || shares.size != languages.size) return null
        var best = -1
        for (i in languages.indices) {
            val language = languages[i]
            if (language == ran || (i > 0 && language !in sw.thresholds)) continue
            if (best < 0 || shares[i] > shares[best]) best = i
        }
        return if (best >= 0 && shares[best] >= RESCUE_MIN_SHARE) languages[best] else null
    }

    /** Why [rescue] names no language, for the log: language codes and numbers only. */
    fun noRescueReason(ran: String, shares: FloatArray?): String {
        val sw = switch?.takeIf { active } ?: return "no language rule"
        if (shares == null || shares.size != languages.size) return "no shares"
        var best = -1
        for (i in languages.indices) {
            if (languages[i] == ran || (i > 0 && languages[i] !in sw.thresholds)) continue
            if (best < 0 || shares[i] > shares[best]) best = i
        }
        if (best < 0) return "no other language in the switch"
        return "best other ${languages[best]} ${String.format(java.util.Locale.ROOT, "%.3f", shares[best])} < $RESCUE_MIN_SHARE"
    }

    companion object {
        /** The share another language needs for a rescue; a stretch without speech has about 0.3 at most. */
        const val RESCUE_MIN_SHARE = 0.5f
    }
}
