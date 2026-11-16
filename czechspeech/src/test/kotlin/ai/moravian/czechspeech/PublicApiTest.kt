package ai.moravian.czechspeech

import java.io.File
import java.lang.reflect.Modifier
import java.util.zip.ZipFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Guards the public API: the public and protected members of every class outside `internal`, as the JVM sees them,
 * must equal the listing in `czechspeech/api/public-api.txt`. A change of the API is then a change of that file in
 * the same commit, where a reviewer sees it.
 *
 * To accept a change, run the tests once with the environment variable `CZECHSPEECH_UPDATE_API=1` and commit the
 * file.
 */
class PublicApiTest {
    @Test
    fun thePublicApiIsTheRecordedOne() {
        val actual = listing()
        val file = File("api/public-api.txt")
        if (System.getenv("CZECHSPEECH_UPDATE_API") == "1") {
            file.parentFile.mkdirs()
            file.writeText(actual)
        }
        assertTrue("${file.absolutePath} is missing; run the tests once with CZECHSPEECH_UPDATE_API=1", file.isFile)
        assertEquals(
            "the public API changed; if that is intended, run the tests with CZECHSPEECH_UPDATE_API=1 and commit api/public-api.txt",
            file.readText().replace("\r\n", "\n"),
            actual,
        )
    }

    private fun listing(): String {
        // the library's classes are a directory or a jar, depending on the variant and the Gradle plugin
        val root = File(CzechSpeech::class.java.protectionDomain.codeSource.location.toURI())
        val prefix = "ai/moravian/czechspeech/"
        val names = if (root.isDirectory) {
            root.walkTopDown().filter { it.isFile }.map { it.relativeTo(root).path.replace(File.separatorChar, '/') }.toList()
        } else {
            ZipFile(root).use { zip -> zip.entries().asSequence().map { it.name }.toList() }
        }
        val classes = names.asSequence()
            .filter { it.startsWith(prefix) && it.endsWith(".class") }
            .map { it.removeSuffix(".class") }
            .filter { "/internal/" !in it }
            .map { Class.forName(it.replace('/', '.'), false, CzechSpeech::class.java.classLoader) }
            .filter { visible(it.modifiers) && !it.isSynthetic && !it.isAnonymousClass && !it.isLocalClass }
            .filter { c -> generateSequence(c.enclosingClass) { it.enclosingClass }.all { visible(it.modifiers) } }
            .sortedBy { it.name }
            .toList()
        assertTrue("no public classes found in $root", classes.isNotEmpty())
        return buildString {
            for (c in classes) {
                append(c.toGenericString())
                c.genericSuperclass?.takeIf { it != Any::class.java }?.let { append(" extends ").append(it.typeName) }
                if (c.genericInterfaces.isNotEmpty()) {
                    append(" implements ").append(c.genericInterfaces.map { it.typeName }.sorted().joinToString(", "))
                }
                append('\n')
                val members = c.declaredConstructors.filter { visible(it.modifiers) && !it.isSynthetic }.map { it.toGenericString() } +
                    c.declaredFields.filter { visible(it.modifiers) && !it.isSynthetic }.map { it.toGenericString() } +
                    c.declaredMethods.filter { visible(it.modifiers) && !it.isSynthetic && !it.isBridge }.map { it.toGenericString() }
                for (m in members.sorted()) append("    ").append(m).append('\n')
            }
        }
    }

    private fun visible(modifiers: Int) = Modifier.isPublic(modifiers) || Modifier.isProtected(modifiers)
}
