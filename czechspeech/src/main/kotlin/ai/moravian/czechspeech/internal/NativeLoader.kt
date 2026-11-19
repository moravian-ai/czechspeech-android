package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechException
import java.io.File

/**
 * Loads the native library variant that matches the CPU.
 *
 * The AAR carries whole-library builds of whisper.cpp + ggml for several instruction-set levels (see
 * native/build-native.sh). A library compiled for newer instructions than the CPU has would crash with SIGILL, so
 * the variant is chosen from the kernel's feature list in /proc/cpuinfo, best first, falling back to lower levels.
 * On a desktop JVM (tests) the host build is loaded from java.library.path.
 */
internal object NativeLoader {
    const val ARM_BASE = "czechspeech_armv8a"
    const val ARM_DOTPROD = "czechspeech_armv82_dotprod_fp16"
    const val ARM_I8MM = "czechspeech_armv82_dotprod_fp16_i8mm"
    const val X86_64 = "czechspeech_x86_64"
    const val HOST = "czechspeech_host"

    /**
     * Process environment for ggml. On GGML_ASSERT/GGML_ABORT, and on an uncaught C++ exception,
     * ggml_print_backtrace() forks the app process and execs gdb, then lldb, to print a backtrace
     * (third_party/whisper.cpp/ggml/src/ggml.c, under __linux__, which includes Android); it returns at once when
     * GGML_NO_BACKTRACE is set. Set on Android before the library loads; JNI_OnLoad sets it too, for the host build
     * (Java cannot set the environment on a desktop JVM).
     */
    val NATIVE_ENVIRONMENT: Map<String, String> = mapOf("GGML_NO_BACKTRACE" to "1")

    @Volatile
    var loadedLibrary: String? = null
        private set

    fun ensureLoaded(): String {
        loadedLibrary?.let { return it }
        synchronized(this) {
            loadedLibrary?.let { return it }
            val name = load(
                android = isAndroid(),
                arch = System.getProperty("os.arch").orEmpty(),
                cpuInfo = ::readCpuInfo,
                setEnv = ::setEnvAndroid,
                loadLibrary = System::loadLibrary,
            )
            loadedLibrary = name
            return name
        }
    }

    /** Loads the best library variant with [loadLibrary]; the platform calls are parameters for tests. */
    fun load(
        android: Boolean,
        arch: String,
        cpuInfo: () -> String?,
        setEnv: (name: String, value: String) -> Unit,
        loadLibrary: (name: String) -> Unit,
    ): String {
        val candidates = candidates(android, arch, cpuInfo)
        if (candidates.isEmpty()) {
            throw CzechSpeechException("Unsupported CPU architecture '$arch': CzechSpeech needs arm64-v8a or x86_64")
        }
        if (android) {
            for ((name, value) in NATIVE_ENVIRONMENT) {
                try {
                    setEnv(name, value)
                } catch (e: Exception) {
                    CsLog.w("could not set $name", e)       // JNI_OnLoad sets it too
                }
            }
        }
        var lastError: Throwable? = null
        for (name in candidates) {
            try {
                loadLibrary(name)
                CsLog.i("native library: lib$name.so (candidates: ${candidates.joinToString()})")
                return name
            } catch (e: UnsatisfiedLinkError) {
                CsLog.w("could not load lib$name.so", e)
                lastError = e
            }
        }
        throw CzechSpeechException("No CzechSpeech native library could be loaded (tried ${candidates.joinToString()})", lastError)
    }

    /** Library names to try, best first. [cpuInfo] is read only on arm64 Android. */
    fun candidates(android: Boolean, arch: String, cpuInfo: () -> String?): List<String> {
        if (!android) return listOf(HOST)
        return when (arch.lowercase()) {
            "aarch64", "arm64" -> {
                val features = CpuFeatures.parse(cpuInfo())
                buildList {
                    if (features.dotprod && features.fp16 && features.i8mm) add(ARM_I8MM)
                    if (features.dotprod && features.fp16) add(ARM_DOTPROD)
                    add(ARM_BASE)
                }
            }
            "x86_64", "amd64" -> listOf(X86_64)
            else -> emptyList()
        }
    }

    fun isAndroid(): Boolean = System.getProperty("java.vm.name").orEmpty().contains("Dalvik", ignoreCase = true)

    /** setenv(3) in this process, without overwriting; Android only (android.system.Os). */
    private fun setEnvAndroid(name: String, value: String) = android.system.Os.setenv(name, value, false)

    private fun readCpuInfo(): String? = try {
        File("/proc/cpuinfo").readText()
    } catch (e: Exception) {
        CsLog.w("cannot read /proc/cpuinfo, using the baseline library", e)
        null
    }
}

/**
 * CPU features from the "Features" lines of /proc/cpuinfo (arm64). Lines are intersected, so a feature counts only
 * when every core reports it.
 */
internal class CpuFeatures(val flags: Set<String>) {
    /** SDOT/UDOT (FEAT_DotProd). */
    val dotprod: Boolean get() = "asimddp" in flags

    /** Half-precision arithmetic, scalar and vector (FEAT_FP16). */
    val fp16: Boolean get() = "asimdhp" in flags && "fphp" in flags

    /** Int8 matrix multiply (FEAT_I8MM). */
    val i8mm: Boolean get() = "i8mm" in flags

    companion object {
        fun parse(cpuInfo: String?): CpuFeatures {
            if (cpuInfo == null) return CpuFeatures(emptySet())
            var common: Set<String>? = null
            for (line in cpuInfo.lineSequence()) {
                val key = line.substringBefore(':').trim()
                if (!key.equals("Features", ignoreCase = true)) continue
                val flags = line.substringAfter(':').trim().split(' ', '\t').filter { it.isNotEmpty() }.toSet()
                common = common?.intersect(flags) ?: flags
            }
            return CpuFeatures(common.orEmpty())
        }
    }
}
