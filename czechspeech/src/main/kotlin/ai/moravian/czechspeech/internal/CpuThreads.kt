package ai.moravian.czechspeech.internal

import java.io.File

/**
 * Number of recogniser threads: the big (high-frequency) cores of the device, at least 2 and at most 8. With thread
 * placement on ([ThreadPlanner]) it is the limit of each recognition; off, the count of every recognition.
 *
 * Adapted from WhisperCpuConfig in whisper.cpp examples/whisper.android (MIT licence, The ggml authors): cores are
 * binned by their maximum frequency and the slowest bin is dropped; when frequencies cannot be read, the "CPU
 * variant" fields of /proc/cpuinfo are used instead.
 */
internal object CpuThreads {
    fun defaultThreadCount(): Int {
        val big = try {
            highPerfCoreCount(File("/proc/cpuinfo").readLines()) { cpu ->
                File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }
        } catch (e: Exception) {
            CsLog.w("cannot read the CPU layout", e)
            Runtime.getRuntime().availableProcessors() - 4
        }
        return big.coerceIn(2, 8)
    }

    /** Cores faster than the slowest ones; all cores when they are all equal. */
    fun highPerfCoreCount(cpuInfoLines: List<String>, maxFrequency: (cpu: Int) -> Long): Int {
        val processors = valuesOf(cpuInfoLines, "processor") { it.toInt() }
        if (processors.isEmpty()) error("no processor entries")
        return try {
            countAboveMin(processors.map(maxFrequency))
        } catch (e: Exception) {
            val variants = valuesOf(cpuInfoLines, "CPU variant") { it.substringAfter("0x").toInt(16) }
            if (variants.isEmpty()) throw e
            // Same heuristic as WhisperCpuConfig: the cores sharing the lowest variant number are the big ones.
            val min = variants.min()
            variants.count { it == min }
        }
    }

    private fun countAboveMin(values: List<Long>): Int {
        val min = values.min()
        val above = values.count { it > min }
        return if (above == 0) values.size else above
    }

    private fun <T> valuesOf(lines: List<String>, key: String, parse: (String) -> T): List<T> = lines
        .filter { it.substringBefore(':').trim() == key }
        .map { parse(it.substringAfter(':').trim()) }
}
