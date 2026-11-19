package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.internal.ThreadPlan.Rule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The choice of recogniser threads and cores from the cores the thread may use now (its mask) and their capacities, on
 * the CPU layouts of real phones (cpu_capacity as the kernel reports it, rounded) and of an emulator. A plan pins only
 * when it beats today's (the limit's threads, unpinned, at most one per allowed core) at least 1.5 times by the estimate
 * threads x capacity of the slowest core a thread ends up on.
 */
class ThreadPlannerTest {
    private fun layout(vararg capacities: Long) =
        CpuTopology(capacities.withIndex().associate { (cpu, c) -> cpu to c }, "cpu_capacity")

    /** Pixel 9a, Tensor G4: cpu0-3 Cortex-A520, cpu4-6 A720, cpu7 X4 (measured 14 Nov 2026). */
    private val tensorG4 = layout(191, 191, 191, 191, 741, 741, 741, 1024)

    // Android's cpusets on the Pixel 9a: top-app 0-7, foreground 0-6 (no X4), background 0-3 (the A520 only).
    private val topApp = 0xffL
    private val foreground = 0x7fL
    private val background = 0x0fL

    private fun plan(allowed: Long, layout: CpuTopology, max: Int) = ThreadPlanner.plan(allowed, layout, max)

    private fun assertPlan(threads: Int, cpus: Long, rule: Rule, plan: ThreadPlan) {
        assertEquals("rule", rule, plan.rule)
        assertEquals("threads", threads, plan.threads)
        assertEquals("cpus ${CpuMask.hex(plan.cpus)}", cpus, plan.cpus)
        assertEquals("pins exactly the fast-core plans", rule == Rule.FAST, plan.narrows)
    }

    /** Today's threads, not pinned. */
    private fun assertToday(threads: Int, plan: ThreadPlan, rule: Rule = Rule.TODAY) = assertPlan(threads, 0L, rule, plan)

    @Test
    fun tensorG4TopAppStaysAsTodayUnpinned() {
        // 4 threads on the X4 and the A720s: 4 x 741, the same as today's 4 threads where the scheduler puts them
        val plan = plan(topApp, tensorG4, 4)
        assertToday(4, plan)
        assertEquals(topApp, plan.allowed)
        assertEquals("a refused pin would have today's threads too", 4, plan.fallbackThreads)
    }

    @Test
    fun tensorG4ForegroundTakesThreeThreadsOnTheThreeA720s() {
        // the floating button: today 4 threads, one on an A520 (4 x 191); 3 on the A720s give 3 x 741, 2.9 times more
        val plan = plan(foreground, tensorG4, 4)
        assertPlan(3, 0x70, Rule.FAST, plan)
        assertEquals(4, plan.fallbackThreads)
    }

    @Test
    fun tensorG4BackgroundKeepsTodaysThreads() {
        assertToday(4, plan(background, tensorG4, 4))
    }

    @Test
    fun theLimitIsACeiling() {
        assertToday(2, plan(topApp, tensorG4, 2))
        assertToday(1, plan(foreground, tensorG4, 1))       // 1 x 741 either way: unpinned
        assertToday(3, plan(foreground, tensorG4, 3))       // the scheduler puts 3 threads on the A720s too
        // a limit above the fast cores: the 4 fast ones beat 8 threads ending on an A520 (2964 vs 1528, 1.9 times)
        assertPlan(4, 0xf0, Rule.FAST, plan(topApp, tensorG4, 8))
        assertEquals(1, plan(topApp, tensorG4, 0).threads)
    }

    @Test
    fun bigLittleFourPlusFour() {
        val layout = layout(382, 382, 382, 382, 1024, 1024, 1024, 1024)       // e.g. 4x A53 + 4x A73
        assertToday(4, plan(0xff, layout, 4))
        assertPlan(3, 0x70, Rule.FAST, plan(0x7f, layout, 4))                 // 3072 vs 1528: 2.0 times
        assertToday(4, plan(0x0f, layout, 4))
        // one fast core among stronger little ones (cpuset 0-4): 1 x 1024 loses to today's 4 x 382
        assertToday(4, plan(0x1f, layout, 4))
    }

    @Test
    fun oneThreeFour() {
        val layout = layout(220, 220, 220, 220, 855, 855, 855, 1024)          // e.g. Snapdragon 865
        assertToday(4, plan(0xff, layout, 4))
        assertPlan(3, 0x70, Rule.FAST, plan(0x7f, layout, 4))                 // 2565 vs 880: 2.9 times
        assertToday(4, plan(0x0f, layout, 4))
    }

    @Test
    fun twoPlusSix() {
        val layout = layout(391, 391, 391, 391, 391, 391, 1024, 1024)         // e.g. Helio G99: 6x A55 + 2x A76
        assertToday(2, plan(0xff, layout, 2))
        assertToday(2, plan(0x7f, layout, 2))                                 // 1 x 1024 vs 2 x 391: only 1.31 times
        assertToday(2, plan(0x0f, layout, 2))
    }

    @Test
    fun oneFiveTwoSnapdragon8Gen3() {
        // cpu0-1 A520, cpu2-4 A720 at 3.15 GHz, cpu5-6 A720 at 2.96 GHz, cpu7 X4
        val layout = layout(325, 325, 870, 870, 870, 820, 820, 1024)
        assertToday(6, plan(0xff, layout, 6))
        assertPlan(5, 0x7c, Rule.FAST, plan(0x7f, layout, 6))                 // 4100 vs 1950: 2.1 times
        // two cores for a limit of 6: today would put 3 threads on each, and ggml's barriers spin
        assertToday(2, plan(0x03, layout, 6), Rule.BOUNDED)
    }

    @Test
    fun dimensity9300WithStrongSlowestCores() {
        // 4x A720 as the slowest tier, 3x X4, 1x X4 prime: on the A720 alone no thread is taken away, and under 0-6
        // the X4s gain too little to pin (3 x 900 vs 4 x 560: 1.2 times)
        val layout = layout(560, 560, 560, 560, 900, 900, 900, 1024)
        assertToday(4, plan(0xff, layout, 4))
        assertToday(4, plan(0x0f, layout, 4))
        assertToday(4, plan(0x7f, layout, 4))
    }

    @Test
    fun aGainBelowOneAndAHalfStaysAsBefore() {
        // the three fastest cores against four threads, 1.2 times better by the estimate: as before
        assertToday(4, plan(0x7f, layout(600, 600, 600, 600, 960, 960, 960, 1024), 4))
        // 1.5 times exactly pins: 3 x 1000 against 4 x 500
        assertPlan(3, 0x70, Rule.FAST, plan(0x7f, layout(500, 500, 500, 500, 1000, 1000, 1000, 1024), 4))
    }

    @Test
    fun pixel6TwoTwoFour() {
        val layout = layout(160, 160, 160, 160, 498, 498, 1024, 1024)         // Tensor G1
        assertToday(4, plan(0xff, layout, 4))
        assertPlan(3, 0x70, Rule.FAST, plan(0x7f, layout, 4))                 // 1494 vs 640: 2.3 times
    }

    @Test
    fun eightAlikeCores() {
        val layout = layout(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)
        assertToday(8, plan(0xff, layout, 8))
        // a cpuset of four of them: today 8 threads on 4 cores; never more threads than allowed cores
        assertToday(4, plan(0x0f, layout, 8), Rule.BOUNDED)
        val close = layout(910, 910, 910, 910, 1024, 1024, 1024, 1024)       // within a fifth: no little cores
        assertToday(4, plan(0xff, close, 4))
    }

    @Test
    fun anEmulatorWithoutCapacitiesKeepsTodaysThreads() {
        assertToday(2, plan(0x0f, CpuTopology.UNKNOWN, 2), Rule.UNKNOWN)
    }

    @Test
    fun unknownCoresKeepTodaysThreads() {
        assertToday(4, plan(0L, tensorG4, 4), Rule.UNKNOWN)
        val partial = CpuTopology(mapOf(0 to 191L, 1 to 191L, 4 to 741L), "cpu_capacity")
        assertToday(4, plan(0x33, partial, 4), Rule.UNKNOWN)
        assertToday(4, plan(0x03, CpuTopology(mapOf(0 to 0L, 1 to 1024L), "cpu_capacity"), 4), Rule.UNKNOWN)
        // a core missing from the map but outside the allowed set does not matter: 1 x 741 vs 3 x 191 is 1.29 times
        assertToday(3, plan(0x13, partial, 4), Rule.BOUNDED)
    }

    @Test
    fun theFrequencyFallbackIsConservative() {
        // clocks do not show that an A720 does twice an A520's work per cycle: on Tensor G4's clocks
        // 3 x 2.6 GHz only ties 4 x 1.95 GHz, so the foreground keeps today's threads
        val g4 = CpuTopology(
            (0..7).associateWith { if (it < 4) 1_950_000L else if (it < 7) 2_600_000L else 3_100_000L },
            "cpuinfo_max_freq",
        )
        assertToday(4, plan(foreground, g4, 4))
        val wide = CpuTopology((0..7).associateWith { if (it < 4) 1_200_000L else 2_800_000L }, "cpuinfo_max_freq")
        assertPlan(3, 0x70, Rule.FAST, plan(foreground, wide, 4))             // 8.4 vs 4.8: 1.75 times
        val g99 = CpuTopology((0..7).associateWith { if (it < 6) 2_000_000L else 2_200_000L }, "cpuinfo_max_freq")
        assertToday(2, plan(0xff, g99, 2))
    }

    @Test
    fun turnedOffPausedOrNotPutBackMeanTodaysThreadsWithoutPinning() {
        val off = ThreadPlan.off(maxThreads = 4, allowed = foreground)
        assertToday(4, off, Rule.OFF)
        assertEquals(foreground, off.allowed)
        assertToday(4, ThreadPlan.paused(maxThreads = 4, allowed = foreground), Rule.PAUSED)
        // a thread left narrowed: no more threads than the cores of its actual mask
        assertToday(3, ThreadPlan.notPutBack(maxThreads = 4, mask = 0x70), Rule.NOT_PUT_BACK)
        assertToday(2, ThreadPlan.notPutBack(maxThreads = 2, mask = 0x70), Rule.NOT_PUT_BACK)
        assertToday(4, ThreadPlan.notPutBack(maxThreads = 4, mask = 0L), Rule.NOT_PUT_BACK)
    }

    @Test
    fun aRefusedPinFallsBackToTodaysThreads() {
        val refused = plan(foreground, tensorG4, 4).refused()
        assertEquals(Rule.REFUSED, refused.rule)
        assertEquals(4, refused.threads)
        assertFalse(refused.narrows)
        assertTrue(plan(foreground, tensorG4, 4).narrows)
    }

    @Test
    fun describesItselfInNumbers() {
        assertEquals(
            "threads 3 on cpus 0x70 (allowed 0x7f, fast cores)",
            plan(foreground, tensorG4, 4).describe(pinned = true),
        )
        assertEquals("threads 4, not pinned (allowed 0xff, as before)", plan(topApp, tensorG4, 4).describe(pinned = false))
        assertEquals(
            "threads 2, not pinned (allowed 0x3, as before, one per allowed core)",
            plan(0x03, tensorG4, 4).describe(pinned = false),
        )
        assertEquals(
            "threads 4, not pinned: cpus 0x70 refused (allowed 0x7f, as before)",
            plan(foreground, tensorG4, 4).refused().describe(pinned = false),
        )
        assertEquals(
            "threads 4, not pinned (allowed 0x7f, automatic placement off)",
            ThreadPlan.off(4, foreground).describe(pinned = false),
        )
        assertEquals(
            "threads 4, not pinned (allowed 0x7f, pinning paused)",
            ThreadPlan.paused(4, foreground).describe(pinned = false),
        )
        assertEquals(
            "threads 2, not pinned (allowed unknown, CPU layout unknown)",
            plan(0L, CpuTopology.UNKNOWN, 2).describe(pinned = false),
        )
        assertEquals(
            "threads 3, not pinned (allowed 0x70, affinity not put back)",
            ThreadPlan.notPutBack(4, 0x70).describe(pinned = false),
        )
    }
}
