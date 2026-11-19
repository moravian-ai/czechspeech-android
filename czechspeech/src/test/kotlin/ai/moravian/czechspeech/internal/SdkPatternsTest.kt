package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.lang.reflect.Modifier
import java.util.regex.Pattern

/**
 * The guard against a regular expression that the desktop JVM takes and Android's engine refuses (the crash of a final
 * recognition without a letter or a digit: an unescaped `}` in a pattern of the log diagnostic). The unit tests cannot
 * run Android's engine, so they hold every pattern of the SDK to the syntax both read the same way ([IcuSyntax]).
 */
class SdkPatternsTest {
    @Test
    fun everyPatternOfTheSdkIsInTheSyntaxBothEnginesRead() {
        for (entry in SdkPatterns.ALL) {
            assertEquals("${entry.name}: ${entry.source}", emptyList<String>(), IcuSyntax.problems(entry.source))
        }
    }

    @Test
    fun everyPatternCompilesOnThisJvmAndTheStartUpCheckSaysSo() {
        for (entry in SdkPatterns.ALL) Pattern.compile(entry.source)
        assertEquals(SdkPatterns.ALL.size, SdkPatterns.verify())
        assertEquals("names are unique", SdkPatterns.ALL.size, SdkPatterns.ALL.map { it.name }.toSet().size)
    }

    @Test
    fun everyStringConstantOfTheRegistryIsEitherListedOrAFragment() {
        val listed = SdkPatterns.ALL.map { it.source }.toSet()
        val fragments = setOf("WS")
        val unlisted = SdkPatterns::class.java.declaredFields
            .filter { Modifier.isStatic(it.modifiers) && it.type == String::class.java && it.name !in fragments }
            .filter { it.apply { isAccessible = true }.get(null) !in listed }
            .map { it.name }
        assertEquals("a pattern constant that is not in ALL is not checked by any test", emptyList<String>(), unlisted)
    }

    @Test
    fun noOtherMainSourceBuildsARegularExpressionOfItsOwn() {
        val root = File("src/main/kotlin")
        assertTrue("run from the module directory: ${root.absolutePath}", root.isDirectory)
        val stray = root.walkTopDown().filter { it.isFile && it.extension == "kt" && it.name != "SdkPatterns.kt" }.flatMap { file ->
            file.readLines().withIndex().filter { (_, line) -> builds(line) }.map { (n, line) -> "${file.name}:${n + 1}: ${line.trim()}" }
        }.toList()
        assertEquals("regular expressions belong in SdkPatterns (checked for Android's syntax)", emptyList<String>(), stray)
    }

    private fun builds(line: String): Boolean {
        val code = line.substringBefore("//")
        if (code.trimStart().startsWith("*") || code.trimStart().startsWith("/*")) return false
        val viaRegistry = Regex("""(Regex|Pattern\.compile)\(SdkPatterns\.[A-Z0-9_]+\)""")
        val rest = viaRegistry.replace(code, "")
        return listOf("Regex(", ".toRegex(", "Pattern.compile(", "Pattern.matches(", "replaceAll(", ".toPattern(").any { it in rest }
    }

    // --- the checker itself: it must refuse what Android refuses, and let through what the SDK uses ---

    @Test
    fun theCheckerRefusesThePatternThatCrashedTheApp() {
        val crashed = """\[[^\]]*]|\([^)]*\)|\{[^}]*}"""
        val problems = IcuSyntax.problems(crashed)
        assertTrue(problems.toString(), problems.any { it.contains("unescaped ] outside") })
        assertTrue(problems.toString(), problems.any { it.contains("unescaped } outside") })
        assertTrue(problems.toString(), problems.any { it.contains("unescaped } in a character class") })
    }

    @Test
    fun theCheckerRefusesTheOtherConstructsTheEnginesReadDifferently() {
        val refused = listOf(
            "a}", "a]", "a{", "a{x}", "[[a]]", "[]a]", "[a&&b]", "a*+", "a++", "a{2}+", "(?>a)", "(?<name>a)", "(?i)a", "\\Rx", "\\hx",
            "\\vx", "(?<=a+)b", "(?<=a{1,})b", "(?<!a*)b", "\\p{javaLowerCase}", "\\p{IsAlphabetic}", "\\p{L", "(a", "a)", "[a",
        )
        for (pattern in refused) assertTrue("$pattern is not refused", IcuSyntax.problems(pattern).isNotEmpty())
    }

    @Test
    fun theCheckerLetsThroughWhatBothEnginesRead() {
        val accepted = listOf(
            "\\d{2,3}", "\\d{4}", "x{1,}", "[a-z]+", "\\{x\\}", "\\]", "\\[", "[^\\]]", "[^\\p{L}\\p{N}]+", "(?<=[a-z])x", "(?<![^a])x",
            "(?<=\\p{Ll})(?=\\p{Lu})", "x+?", "x*?", "(?:a|b)+", "(?=a)", "(?!a)", "\\u00A0", "[\\u2000-\\u200A]", "\\.", "^\\(?\\d+\\.\$",
        )
        for (pattern in accepted) assertEquals(pattern, emptyList<String>(), IcuSyntax.problems(pattern))
    }
}
