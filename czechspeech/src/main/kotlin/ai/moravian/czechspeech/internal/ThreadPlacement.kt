package ai.moravian.czechspeech.internal

import java.io.Closeable
import java.io.File

/** Bit masks of CPUs (bit i = CPU i), as the SDK passes them to and from native code and logs them. */
internal object CpuMask {
    fun of(cpus: Iterable<Int>): Long = cpus.fold(0L) { mask, cpu -> mask or (1L shl cpu) }

    fun cpus(mask: Long): List<Int> = (0 until 64).filter { mask and (1L shl it) != 0L }

    fun hex(mask: Long): String = "0x" + java.lang.Long.toHexString(mask)

    /** A sysfs CPU list ("0-3,6"); null when it is empty or malformed. */
    fun parseList(text: String): List<Int>? {
        val cpus = mutableListOf<Int>()
        for (part in text.trim().split(',')) {
            val bounds = part.split('-').map { it.trim().toIntOrNull() ?: return null }
            when (bounds.size) {
                1 -> cpus += bounds[0]
                2 -> if (bounds[0] <= bounds[1]) cpus += bounds[0]..bounds[1] else return null
                else -> return null
            }
        }
        return cpus
    }

    /**
     * The Cpus_allowed mask of a /proc/.../status text (hex, in comma-separated 32-bit words); null when the line is
     * missing, malformed or empty, and when a CPU above 63 is in it.
     */
    fun parseCpusAllowed(status: String): Long? {
        val line = status.lineSequence().firstOrNull { it.startsWith("Cpus_allowed:") } ?: return null
        val hex = line.substringAfter(':').trim().replace(",", "").trimStart('0')
        if (hex.isEmpty() || hex.length > 16 || hex.any { Character.digit(it, 16) < 0 }) return null
        return java.lang.Long.parseUnsignedLong(hex, 16)
    }
}

/**
 * The relative speed of each CPU (core number to capacity) as the kernel reports it: cpu_capacity, which the
 * scheduler itself uses (the fastest core 1024, the others by their maximum performance), or else the maximum clock
 * (cpufreq/cpuinfo_max_freq), which cannot tell a big core from a little one at a similar clock. Never a mix of the two.
 * Read once per engine: neither changes while the device runs.
 */
internal class CpuTopology(val capacities: Map<Int, Long>, val source: String) {
    /** Numbers only: "capacity 191,191,741,1024 (cpu_capacity)", or "unknown". */
    fun describe(): String =
        if (capacities.isEmpty()) "unknown"
        else "capacity ${capacities.toSortedMap().values.joinToString(",")} ($source)"

    companion object {
        val UNKNOWN = CpuTopology(emptyMap(), "none")

        /**
         * Reads the possible CPUs of [root] (/sys/devices/system/cpu): cpu_capacity when every one has it, else
         * cpuinfo_max_freq when every one has it, else whichever covers more of them (the planner then falls back to
         * today's threads when a core it needs is missing). Never throws.
         */
        fun read(root: File = File("/sys/devices/system/cpu")): CpuTopology = try {
            readOrThrow(root)
        } catch (e: Exception) {
            CsLog.w("cannot read the CPU capacities", e)
            UNKNOWN
        }

        /** The number of a directory named `cpu<digits>` of the CPU tree, null for any other name (`cpufreq`, `cpuidle`). */
        private fun cpuNumber(name: String): Int? {
            val digits = name.removePrefix("cpu")
            return if (digits.length < name.length && digits.isNotEmpty() && digits.all { it in '0'..'9' }) digits.toIntOrNull() else null
        }

        private fun readOrThrow(root: File): CpuTopology {
            val cpus = readText(File(root, "possible"))?.let(CpuMask::parseList)
                ?: root.list().orEmpty().mapNotNull { cpuNumber(it) }.sorted()
            if (cpus.isEmpty()) return UNKNOWN
            val capacity = valuesOf(cpus) { File(root, "cpu$it/cpu_capacity") }
            val maxFreq = valuesOf(cpus) { File(root, "cpu$it/cpufreq/cpuinfo_max_freq") }
            val (values, source) = when {
                capacity.size == cpus.size -> capacity to CAPACITY
                maxFreq.size == cpus.size -> maxFreq to MAX_FREQ
                maxFreq.size > capacity.size -> maxFreq to MAX_FREQ
                else -> capacity to CAPACITY
            }
            return if (values.isEmpty()) UNKNOWN else CpuTopology(values, source)
        }

        private const val CAPACITY = "cpu_capacity"
        private const val MAX_FREQ = "cpuinfo_max_freq"

        private fun valuesOf(cpus: List<Int>, file: (Int) -> File): Map<Int, Long> =
            cpus.mapNotNull { cpu -> readText(file(cpu))?.trim()?.toLongOrNull()?.takeIf { it > 0 }?.let { cpu to it } }.toMap()

        private fun readText(file: File): String? = try {
            file.readText()
        } catch (e: Exception) {
            null
        }
    }
}

/**
 * The recogniser threads of one recognition and the cores to pin them to ([cpus]; only a [Rule.FAST] plan pins),
 * chosen from the cores the thread may use now ([allowed], its mask; 0 = unknown). [fallbackThreads] are the threads
 * of the plan without pinning, which a refused pin falls back to.
 */
internal class ThreadPlan(
    val threads: Int,
    val cpus: Long,
    val allowed: Long,
    val rule: Rule,
    val fallbackThreads: Int,
) {
    enum class Rule(val label: String) {
        /** [ai.moravian.czechspeech.CzechSpeechConfig.autoThreads] off: the limit's threads, not pinned. */
        OFF("automatic placement off"),

        /** The allowed cores or their capacities are unknown: the limit's threads, not pinned. */
        UNKNOWN("CPU layout unknown"),

        /** As before: the limit's threads, not pinned (no fast-core plan beats it by enough). */
        TODAY("as before"),

        /** As before, but fewer threads than the limit: never more threads than allowed cores. */
        BOUNDED("as before, one per allowed core"),

        /** The fast cores the thread may use, pinned: they beat the threads as before at least [ThreadPlanner] says. */
        FAST("fast cores"),

        /** The kernel refused the pin of a [FAST] plan: the threads as before, not pinned. */
        REFUSED("as before"),

        /** An earlier pin of this thread could not be undone: no more threads than its mask's cores, nothing pinned. */
        NOT_PUT_BACK("affinity not put back"),

        /** A pinned computation found fewer of its cores running than threads: as before for a while ([PinGuard]). */
        PAUSED("pinning paused"),
    }

    /** The thread is pinned to [cpus] for the call. */
    val narrows: Boolean get() = rule == Rule.FAST && cpus != 0L && cpus != allowed

    /** This plan when the kernel refuses its pin: the threads as before, the refused cores kept for the log. */
    fun refused() = ThreadPlan(fallbackThreads, cpus, allowed, Rule.REFUSED, fallbackThreads)

    /** Numbers only, for the log line of each recognition: "threads 3 on cpus 0x70 (allowed 0x7f, fast cores)". */
    fun describe(pinned: Boolean): String {
        val placed = when {
            pinned -> " on cpus ${CpuMask.hex(cpus)}"
            rule == Rule.FAST -> " on cpus ${CpuMask.hex(cpus)}, not pinned"
            rule == Rule.REFUSED -> ", not pinned: cpus ${CpuMask.hex(cpus)} refused"
            else -> ", not pinned"
        }
        val from = if (allowed != 0L) CpuMask.hex(allowed) else "unknown"
        return "threads $threads$placed (allowed $from, ${rule.label})"
    }

    companion object {
        fun off(maxThreads: Int, allowed: Long) = unpinned(maxThreads, allowed, Rule.OFF)

        /** Exactly as before while [PinGuard] pauses pinning: the limit's threads, not pinned. */
        fun paused(maxThreads: Int, allowed: Long) = unpinned(maxThreads, allowed, Rule.PAUSED)

        /** A thread still narrowed to [mask] (0 = unreadable) by a pin it could not undo: at most its cores' threads. */
        fun notPutBack(maxThreads: Int, mask: Long): ThreadPlan {
            val cores = java.lang.Long.bitCount(mask)
            val limit = maxThreads.coerceAtLeast(1)
            return unpinned(if (cores > 0) minOf(limit, cores) else limit, mask, Rule.NOT_PUT_BACK)
        }

        private fun unpinned(threads: Int, allowed: Long, rule: Rule): ThreadPlan {
            val n = threads.coerceAtLeast(1)
            return ThreadPlan(n, 0L, allowed, rule, n)
        }
    }
}

/**
 * Chooses the recogniser threads and cores ([plan]) from the cores the thread may use now (its mask, which Android's
 * cpuset of the process bounds) and the cores' capacities. Simple and conservative: it changes today's behaviour only
 * where the gain is large, since it cannot be measured on phones other than the Pixel 9a.
 *
 * ggml splits much of a recognition evenly between its threads and waits for all of them at each step, so a slow
 * thread holds the others back: on the Pixel 9a's floating button (a foreground service, cores 0-6) four threads on
 * three A720s and one A520 took 5.0 s for what 2.6-2.8 s took on screen. A plan is judged by threads x the capacity of
 * the slowest core a thread ends up on. Two candidates:
 * - as before: the limit's threads ([maxThreads]: CzechSpeechConfig.threads or the big core count), not pinned, which
 *   the scheduler puts on the fastest allowed cores first; but never more threads than allowed cores, since ggml's
 *   barriers wait spinning, without yielding, and a thread that shares a core stalls every step;
 * - the fast cores: the allowed cores faster than the device's little cores (the slowest ones and any at most a fifth
 *   faster, [LITTLE_NUM] / [LITTLE_DEN]), at most the limit's threads, pinned to them.
 * The fast cores are taken only when they do at least [PIN_GAIN_NUM] / [PIN_GAIN_DEN] (1.5) times better. Cores are
 * counted from the thread's mask, never from those running at the moment of planning: phones park idle big cores
 * (Qualcomm's core_ctl) and wake them under load, so the cores running at a quiet moment say little; [ThreadPlacement]
 * checks after a pinned computation instead ([PinGuard]).
 */
internal object ThreadPlanner {
    private const val LITTLE_NUM = 6L
    private const val LITTLE_DEN = 5L

    /**
     * Pin only for at least 1.5 times the estimate as before. ggml splits matrix products between its threads
     * dynamically, so the model "the slowest core sets the pace" overstates the delay a slow core causes: on the Tensor
     * G4 under the floating button the measured gap (5.0 s against 2.6-2.8 s, 1.9 times) lay between that model's 3.9
     * times and 1.35 times of a model that adds the cores' capacities. Close cases (2+6 under 0-6: 1.31 times; three
     * fastest cores against four: 1.2 times) stay as before.
     */
    private const val PIN_GAIN_NUM = 3L
    private const val PIN_GAIN_DEN = 2L

    fun plan(allowed: Long, topology: CpuTopology, maxThreads: Int): ThreadPlan {
        val limit = maxThreads.coerceAtLeast(1)
        val cpus = CpuMask.cpus(allowed)
        val capacity = topology.capacities
        if (cpus.isEmpty() || cpus.any { (capacity[it] ?: 0L) <= 0L }) {
            return ThreadPlan(limit, 0L, allowed, ThreadPlan.Rule.UNKNOWN, limit)
        }
        val spread = minOf(limit, cpus.size)
        val asBefore =
            ThreadPlan(spread, 0L, allowed, if (spread < limit) ThreadPlan.Rule.BOUNDED else ThreadPlan.Rule.TODAY, spread)
        val slowest = capacity.values.filter { it > 0 }.min()
        val fast = cpus.filterNot { capacity.getValue(it) * LITTLE_DEN <= slowest * LITTLE_NUM }
        if (fast.isEmpty()) return asBefore
        val threads = minOf(limit, fast.size)
        val gain = estimate(threads, fast, capacity) * PIN_GAIN_DEN >= estimate(spread, cpus, capacity) * PIN_GAIN_NUM
        return if (gain) ThreadPlan(threads, CpuMask.of(fast), allowed, ThreadPlan.Rule.FAST, spread) else asBefore
    }

    /** [threads] threads on the fastest of [cores], split evenly: their number times the slowest one's capacity. */
    private fun estimate(threads: Int, cores: List<Int>, capacity: Map<Int, Long>): Long =
        threads * cores.map { capacity.getValue(it) }.sortedDescending()[threads - 1]
}

/**
 * Pauses pinning for [PAUSE_MS] after a pinned computation in which fewer of the pinned cores ran than there were
 * threads, under load (cores isolated for heat, say, which no load wakes): then two threads shared a core, and ggml's
 * spinning barriers make that slower than before. Meanwhile placements behave exactly as before; then pinning is tried
 * again. One per engine.
 */
internal class PinGuard(private val nowMs: () -> Long = { System.nanoTime() / 1_000_000 }) {
    @Volatile
    private var pausedUntil: Long? = null

    val paused: Boolean get() = pausedUntil?.let { nowMs() < it } ?: false

    fun pause() {
        pausedUntil = nowMs() + PAUSE_MS
    }

    companion object {
        /** How long pinning stays off after cores did not run under load. */
        const val PAUSE_MS = 30_000L
    }
}

/**
 * A [ThreadPlan] in effect on the calling thread (the engine's whisper thread) for one recognition or run of the
 * language model. A [ThreadPlan.Rule.FAST] plan pins the thread; ggml's and whisper.cpp's worker threads, started by it
 * during the call, inherit the pinned cores (see native/src/czechspeech_jni.c). [close] unpins it: every CPU, which the
 * kernel bounds by the cpuset of the moment. It never puts back a mask read earlier: on Linux 6.2+ the kernel keeps
 * sched_setaffinity's mask as the thread's own request, intersects every later cpuset with it and gives it to every
 * thread the thread starts, so a mask read under the floating button (0-6) would keep the whisper thread, and the
 * threads it starts, off cpu7 after the app comes on screen. For the same reason a plan is never made from a mask this
 * thread was narrowed to: a thread left pinned by a failed unpin is unpinned first, and while that fails it runs no more
 * threads than its mask's cores, pinning nothing.
 *
 * After a pinned computation ([computed]), before unpinning, [close] reads the cores that ran (sched_getaffinity):
 * fewer than threads, and [PinGuard] pauses pinning.
 *
 * Nesting: a placement made on a thread that already has one open is nested. It has the outer plan, reads and pins
 * nothing, its [computed] counts for the outer one, and its [close] leaves the pin to the outer placement, which alone
 * unpins.
 *
 * Nothing here ever fails a recognition: an affinity that cannot be read gives the limit's threads, a pin the kernel
 * refuses gives the threads as before, unpinned, and a failed unpin is logged.
 */
internal class ThreadPlacement private constructor(
    private val native: NativeApi,
    val plan: ThreadPlan,
    /** This placement pinned the thread (a nested one never does). */
    val pinned: Boolean,
    private val outer: ThreadPlacement?,
    private val guard: PinGuard,
) : Closeable {
    private var closed = false
    private var ran = false

    val threads: Int get() = plan.threads

    fun describe(): String = outer?.describe() ?: plan.describe(pinned)

    /** A computation ran to its end under this placement (not one cancelled before or during it). */
    fun computed() {
        if (outer != null) outer.computed() else ran = true
    }

    override fun close() {
        if (closed) return
        closed = true
        if (outer != null) return                   // the outer placement owns the thread's affinity
        current.remove()
        if (!pinned) return
        if (ran) checkCores()
        if (!call { native.unpinThread() }) {
            narrowed.set(true)
            CsLog.w("could not unpin the recogniser thread; it keeps cpus ${CpuMask.hex(plan.cpus)}")
        }
    }

    /** The pinned cores that ran at the end of the computation; fewer than threads pauses pinning. */
    private fun checkCores() {
        val active = read { native.activeThreadCpus() }
        if (active == 0L) return
        val running = java.lang.Long.bitCount(active and plan.cpus)
        if (running < plan.threads) {
            guard.pause()
            CsLog.w(
                "only $running of the pinned cores ${CpuMask.hex(plan.cpus)} ran for ${plan.threads} threads; pinning " +
                    "paused for ${PinGuard.PAUSE_MS / 1000} s",
            )
        }
    }

    companion object {
        /** The placement open on each thread (for nesting) and threads left pinned by a failed unpin. */
        private val current = ThreadLocal<ThreadPlacement?>()
        private val narrowed = ThreadLocal<Boolean>()

        /**
         * Plans with [ThreadPlanner] ([auto]) or as before, and pins the calling thread when the plan says so; nested
         * in a placement already open on this thread, takes its plan; while [guard] pauses pinning, as before.
         */
        fun place(
            native: NativeApi,
            auto: Boolean,
            topology: CpuTopology,
            maxThreads: Int,
            guard: PinGuard,
        ): ThreadPlacement {
            current.get()?.let { outer ->
                return ThreadPlacement(native, outer.plan, pinned = false, outer = outer, guard = guard)
            }
            var pinned = false
            val plan = if (narrowed.get() == true && !call { native.unpinThread() }) {
                ThreadPlan.notPutBack(maxThreads, read { native.threadCpus() })
            } else {
                narrowed.remove()
                val allowed = read { native.threadCpus() }
                when {
                    !auto -> ThreadPlan.off(maxThreads, allowed)
                    guard.paused -> ThreadPlan.paused(maxThreads, allowed)
                    else -> ThreadPlanner.plan(allowed, topology, maxThreads).let { planned ->
                        when {
                            !planned.narrows -> planned
                            call { native.setThreadCpus(planned.cpus) } -> planned.also { pinned = true }
                            else -> planned.refused()
                        }
                    }
                }
            }
            return ThreadPlacement(native, plan, pinned, outer = null, guard = guard).also { current.set(it) }
        }

        private fun read(mask: () -> Long): Long = try {
            mask()
        } catch (e: Exception) {
            0L
        } catch (e: LinkageError) {
            0L
        }

        private fun call(affinity: () -> Boolean): Boolean = try {
            affinity()
        } catch (e: Exception) {
            false
        } catch (e: LinkageError) {
            false
        }
    }
}
