package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** Loading the native library: the environment ggml needs, the variant order, the fallback and the failures. */
class NativeLoaderTest {
    private val i8mmCpu = "processor\t: 0\nFeatures\t: fp asimd asimddp fphp asimdhp i8mm\n"

    @Test
    fun ggmlBacktracesAreSwitchedOffBeforeTheLibraryLoads() {
        // Without GGML_NO_BACKTRACE a GGML_ASSERT forks the app process and execs gdb/lldb (ggml_print_backtrace).
        val calls = mutableListOf<String>()
        val loaded = NativeLoader.load(
            android = true,
            arch = "aarch64",
            cpuInfo = { i8mmCpu },
            setEnv = { name, value -> calls += "setenv $name=$value" },
            loadLibrary = { calls += "load $it" },
        )
        assertEquals(NativeLoader.ARM_I8MM, loaded)
        assertEquals(listOf("setenv GGML_NO_BACKTRACE=1", "load ${NativeLoader.ARM_I8MM}"), calls)
    }

    @Test
    fun aRefusedSetenvDoesNotStopTheLibraryFromLoading() {
        val loaded = NativeLoader.load(
            android = true,
            arch = "aarch64",
            cpuInfo = { i8mmCpu },
            setEnv = { _, _ -> throw RuntimeException("EPERM") },
            loadLibrary = { },
        )
        assertEquals("JNI_OnLoad sets it as well", NativeLoader.ARM_I8MM, loaded)
    }

    @Test
    fun aLibraryThatCannotLoadFallsBackToTheNextLevelDown() {
        val tried = mutableListOf<String>()
        val loaded = NativeLoader.load(true, "aarch64", { i8mmCpu }, setEnv = { _, _ -> }) {
            tried += it
            if (it == NativeLoader.ARM_I8MM) throw UnsatisfiedLinkError("dlopen failed")
        }
        assertEquals(NativeLoader.ARM_DOTPROD, loaded)
        assertEquals(listOf(NativeLoader.ARM_I8MM, NativeLoader.ARM_DOTPROD), tried)
    }

    @Test
    fun noLoadableLibraryIsACzechSpeechExceptionWithTheLastError() {
        val error = UnsatisfiedLinkError("dlopen failed")
        try {
            NativeLoader.load(true, "aarch64", { i8mmCpu }, setEnv = { _, _ -> }) { throw error }
            fail("expected a CzechSpeechException")
        } catch (e: CzechSpeechException) {
            assertTrue(e.message, e.message!!.contains(NativeLoader.ARM_BASE))
            assertSame(error, e.cause)
        }
    }

    @Test
    fun anUnsupportedArchitectureIsRefusedBeforeAnythingIsTouched() {
        val calls = mutableListOf<String>()
        try {
            NativeLoader.load(true, "mips", { null }, setEnv = { name, _ -> calls += name }) { calls += it }
            fail("expected a CzechSpeechException")
        } catch (e: CzechSpeechException) {
            assertTrue(e.message, e.message!!.contains("mips"))
        }
        assertTrue(calls.isEmpty())
    }

    @Test
    fun onADesktopJvmOnlyTheHostLibraryLoadsAndNoEnvironmentIsSetFromJava() {
        // Java cannot set the process environment on a desktop JVM (JNI_OnLoad sets it there, untested here).
        val calls = mutableListOf<String>()
        val loaded = NativeLoader.load(false, "amd64", { null }, setEnv = { name, _ -> calls += name }) { calls += it }
        assertEquals(NativeLoader.HOST, loaded)
        assertEquals(listOf(NativeLoader.HOST), calls)
    }
}
