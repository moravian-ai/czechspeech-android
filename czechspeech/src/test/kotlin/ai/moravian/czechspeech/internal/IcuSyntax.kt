package ai.moravian.czechspeech.internal

/**
 * A check of a regular expression's source for what Android's regex engine (ICU) refuses or reads differently from the
 * desktop JVM, on which the unit tests run (the JVM took `\[[^\]]*]|\([^)]*\)|\{[^}]*}`, ICU threw
 * PatternSyntaxException, and the app crashed). Stricter than either engine on purpose: a pattern that passes is one
 * both read the same way.
 *
 * Refused: an unescaped `]`, `{` or `}` outside a character class except as a quantifier `{n}`, `{n,}`, `{n,m}`; inside a
 * class an unescaped `[`, `{`, `}`, `&&`, or a `]` right at its start; possessive quantifiers; atomic groups and named
 * groups and every group but `(?:`, `(?=`, `(?!`, `(?<=`, `(?<!`; `\R \X \h \H \v \V \k \e`; look-behind of unbounded
 * length; property names other than a general category (`\p{L}`, `\p{Nd}`).
 */
internal object IcuSyntax {
    private val QUANTIFIER = Regex("""\{\d+(,\d*)?\}""")
    private val CATEGORY = Regex("[A-Z][a-z&]?")
    private const val REFUSED_ESCAPES = "RXhHvVke"

    fun problems(source: String): List<String> {
        val found = mutableListOf<String>()
        fun at(i: Int, what: String) { found += "$what at index $i" }
        var i = 0
        var inClass = false
        var classStart = 0
        var afterQuantifier = false
        // one entry per open group: whether it is a look-behind; true marks it as having seen something unbounded
        val groups = ArrayList<BooleanArray>()     // [isLookBehind, sawUnbounded]
        fun lookBehindOpen() = groups.any { it[0] }
        while (i < source.length) {
            val c = source[i]
            if (c == '\\') {
                val next = source.getOrNull(i + 1)
                if (next == null) {
                    at(i, "trailing backslash")
                    break
                }
                if (next in REFUSED_ESCAPES) at(i, "escape \\$next differs between the engines")
                if (next == 'p' || next == 'P') {
                    val open = source.indexOf('{', i + 2).takeIf { source.getOrNull(i + 2) == '{' }
                    val close = if (open == null) -1 else source.indexOf('}', open)
                    if (open == null || close < 0) {
                        at(i, "property without braces")
                    } else {
                        val name = source.substring(open + 1, close)
                        if (!CATEGORY.matches(name)) at(i, "property name $name (only general categories are the same in both)")
                        i = close + 1
                        afterQuantifier = false
                        continue
                    }
                }
                i += if (next == 'u') 6 else 2
                afterQuantifier = false
                continue
            }
            if (inClass) {
                when {
                    c == ']' && i == classStart + 1 || c == ']' && i == classStart + 2 && source[classStart + 1] == '^' ->
                        at(i, "] at the start of a character class")
                    c == ']' -> inClass = false
                    c == '[' -> at(i, "unescaped [ in a character class (a nested set in ICU)")
                    c == '{' || c == '}' -> at(i, "unescaped $c in a character class")
                    c == '&' && source.getOrNull(i + 1) == '&' -> at(i, "&& in a character class")
                }
                i++
                continue
            }
            when (c) {
                '[' -> {
                    inClass = true
                    classStart = i
                    afterQuantifier = false
                }
                ']' -> {
                    at(i, "unescaped ] outside a character class")
                    afterQuantifier = false
                }
                '}' -> {
                    at(i, "unescaped } outside a character class")
                    afterQuantifier = false
                }
                '{' -> {
                    val quantifier = QUANTIFIER.find(source, i)?.takeIf { it.range.first == i }
                    if (quantifier == null) {
                        at(i, "unescaped { outside a character class that is not a quantifier")
                        afterQuantifier = false
                    } else {
                        if (afterQuantifier) at(i, "quantifier after a quantifier")
                        if (quantifier.value.endsWith(",}") && lookBehindOpen()) groups.filter { it[0] }.forEach { it[1] = true }
                        i += quantifier.value.length
                        afterQuantifier = true
                        continue
                    }
                }
                '*', '+', '?' -> {
                    if (afterQuantifier) {
                        if (c == '+') at(i, "possessive quantifier")
                        if (c == '*') at(i, "quantifier after a quantifier")
                        afterQuantifier = false       // `?` after a quantifier makes it lazy
                    } else {
                        afterQuantifier = true
                        if (c != '?' && lookBehindOpen()) groups.filter { it[0] }.forEach { it[1] = true }
                    }
                }
                '(' -> {
                    afterQuantifier = false
                    var kind = 1                 // the length of the "(" itself
                    var behind = false
                    if (source.getOrNull(i + 1) == '?') {
                        val rest = source.substring(i + 2, minOf(source.length, i + 4))
                        when {
                            rest.startsWith(":") || rest.startsWith("=") || rest.startsWith("!") -> kind = 3
                            rest == "<=" || rest == "<!" -> {
                                kind = 4
                                behind = true
                            }
                            else -> at(i, "group construct (?$rest is not the same in both engines")
                        }
                    }
                    groups += booleanArrayOf(behind, false)
                    i += kind
                    continue
                }
                ')' -> {
                    afterQuantifier = false
                    val group = groups.removeLastOrNull()
                    if (group == null) at(i, "unbalanced )") else if (group[0] && group[1]) at(i, "look-behind of unbounded length")
                }
                else -> afterQuantifier = false
            }
            i++
        }
        if (inClass) found += "unterminated character class"
        if (groups.isNotEmpty()) found += "unbalanced ("
        return found
    }
}
