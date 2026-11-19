package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * `native/source-stamp.sh`, which `checkNativeLibs` of the module's build runs before an AAR is packaged: the hash over the
 * native sources changes with each of them, and a stamp written beside the libraries is checked against it. The test runs the
 * script on a small copy of the tree (it needs bash and a SHA-256 tool, as the native build does).
 */
class NativeSourceStampTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val script = File("../native/source-stamp.sh").absoluteFile
    private lateinit var root: File

    @Before
    fun tree() {
        assertTrue("run from the module directory: $script", script.isFile)
        val hasTools = try {
            run(listOf("bash", "-c", "command -v sha256sum || command -v shasum"), tmp.root).first == 0
        } catch (e: java.io.IOException) {
            false       // no bash on the PATH
        }
        assumeTrue("needs bash and a SHA-256 tool", hasTools)
        root = tmp.newFolder("tree")
        write("native/src/a.c", "int a;")
        write("native/src/sub/b.h", "int b;")
        write("native/patches/p.patch", "patch")
        write("native/CMakeLists.txt", "project(x)")
        write("native/build-native.sh", "WHISPER_COMMIT=1111111111111111111111111111111111111111\n")
    }

    private fun write(path: String, text: String) {
        File(root, path).apply { parentFile.mkdirs() }.writeText(text)
    }

    private fun run(command: List<String>, dir: File): Pair<Int, String> {
        val builder = ProcessBuilder(command).directory(dir).redirectErrorStream(true)
        if (this::root.isInitialized) builder.environment()["NATIVE_STAMP_ROOT"] = root.absolutePath
        val process = builder.start()
        val out = process.inputStream.bufferedReader().readText()
        return process.waitFor() to out.trim()
    }

    private fun stamp() = run(listOf("bash", script.path), tmp.root).also { assertEquals(it.second, 0, it.first) }.second

    private fun check(file: File) = run(listOf("bash", script.path, "check", file.path), tmp.root)

    @Test
    fun theHashIsStableAndEachSourceChangesIt() {
        val base = stamp()
        assertTrue(base, Regex("[0-9a-f]{64}").matches(base))
        assertEquals("same tree, same hash", base, stamp())
        val changes = listOf<(Unit) -> Unit>(
            { write("native/src/a.c", "int a2;") },
            { write("native/src/sub/b.h", "int b2;") },
            { write("native/src/new.c", "int c;") },
            { write("native/patches/p.patch", "other patch") },
            { write("native/patches/q.patch", "one more") },
            { write("native/CMakeLists.txt", "project(y)") },
            { write("native/build-native.sh", "WHISPER_COMMIT=1111111111111111111111111111111111111111\n# a change\n") },
            { write("native/build-native.sh", "WHISPER_COMMIT=2222222222222222222222222222222222222222\n# a change\n") },    // another pinned whisper.cpp commit
        )
        val seen = mutableSetOf(base)
        for (change in changes) {
            change(Unit)
            assertTrue("a change left the hash as it was", seen.add(stamp()))
        }
    }

    @Test
    fun checkPassesForTheStampOfThisTreeAndNamesTheCommandOtherwise() {
        val file = File(tmp.root, "source-stamp")
        val missing = check(file)
        assertEquals(1, missing.first)
        assertTrue(missing.second, "native/build-native.sh android" in missing.second && "missing" in missing.second)

        assertEquals(0, run(listOf("bash", script.path, "write", file.path), tmp.root).first)
        assertEquals(stamp(), file.readText().trim())
        assertEquals(0, check(file).first)

        write("native/src/a.c", "int changed;")
        val stale = check(file)
        assertEquals(1, stale.first)
        assertTrue(stale.second, "native/build-native.sh android" in stale.second && "other sources" in stale.second)

        file.writeText("not a hash\n")
        assertEquals(1, check(file).first)
    }
}
