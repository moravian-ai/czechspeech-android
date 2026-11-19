package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.internal.ThreadPlan.Rule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [ThreadPlacement] on one thread, with the fake native layer's kernel model: a placement inside another on the same
 * thread neither reads nor pins nor unpins (the outer one owns the pin until it closes); a thread left narrowed by a
 * failed unpin is never planned from until an unpin succeeds; and a pinned computation whose cores did not all run,
 * even under load, pauses pinning for [PinGuard.PAUSE_MS]. Each test runs on a thread of its own, so no thread-local
 * state leaks into another test.
 */
class ThreadPlacementTest {
    private val tensorG4 = CpuTopology(
        mapOf(0 to 191L, 1 to 191L, 2 to 191L, 3 to 191L, 4 to 741L, 5 to 741L, 6 to 741L, 7 to 1024L),
        "cpu_capacity",
    )

    private var now = 5_000_000L
    private val guard = PinGuard { now }

    private fun onNewThread(block: () -> Unit) {
        var failure: Throwable? = null
        val thread = Thread({ failure = runCatching(block).exceptionOrNull() }, "cs-placement")
        thread.start()
        thread.join()
        failure?.let { throw it }
    }

    private fun place(native: FakeNative, auto: Boolean = true) = ThreadPlacement.place(native, auto, tensorG4, 4, guard)

    /** A pinned placement with a computation under it, as the engine runs one; whether it pinned. */
    private fun computeOnce(native: FakeNative) = place(native).use {
        native.compute("greedy", it.threads)
        it.computed()
        it.pinned
    }

    @Test
    fun aNestedPlacementKeepsTheOuterPinUntilTheOuterCloses() = onNewThread {
        val native = FakeNative().apply { cpuset = 0x7f }
        val outer = place(native)
        assertTrue(outer.pinned)
        val inner = place(native)
        assertSame("the outer plan", outer.plan, inner.plan)
        assertEquals(3, inner.threads)
        assertFalse("the inner one owns no pin", inner.pinned)
        assertEquals("still in effect", "threads 3 on cpus 0x70 (allowed 0x7f, fast cores)", inner.describe())
        inner.close()
        assertEquals("the inner close leaves the pin", listOf("set 0x70"), native.cpuCallNames)
        assertEquals(0x70L, native.cpus)
        // even a placement that would plan differently (the cpuset changed meanwhile) nests without touching it
        native.cpuset = 0xff
        place(native, auto = false).use { assertSame(outer.plan, it.plan) }
        assertEquals(listOf("set 0x70"), native.cpuCallNames)
        outer.close()
        assertEquals(listOf("set 0x70", "unpin"), native.cpuCallNames)
        assertEquals(FakeNative.ALL_CPUS, native.request)
        // after the outer one, a placement plans afresh
        place(native).use { assertEquals(Rule.TODAY, it.plan.rule) }
    }

    @Test
    fun closingTwiceUnpinsOnce() = onNewThread {
        val native = FakeNative().apply { cpuset = 0x7f }
        val outer = place(native)
        outer.close()
        outer.close()
        assertEquals(listOf("set 0x70", "unpin"), native.cpuCallNames)
        place(native).use { assertTrue("a fresh placement pins again", it.pinned) }
    }

    @Test
    fun aThreadLeftNarrowedIsUnpinnedBeforeAnyPlan() = onNewThread {
        val native = FakeNative().apply {
            cpuset = 0x7f
            refusedCpuCalls = setOf(1, 2)
        }
        place(native).close()                                   // the unpin fails: narrowed by us
        place(native).use {
            assertEquals("the retry failed", Rule.NOT_PUT_BACK, it.plan.rule)
            assertEquals("no more threads than the cores of the narrowed mask", 3, it.threads)
            assertFalse(it.pinned)
        }
        place(native).use { assertTrue("the retry worked: planned and pinned again", it.pinned) }
        assertEquals(listOf("set 0x70", "unpin", "unpin", "unpin", "set 0x70", "unpin"), native.cpuCallNames)
        assertEquals(FakeNative.ALL_CPUS, native.request)
    }

    @Test
    fun aPlacementOffOrUnpinnedTouchesNothing() = onNewThread {
        val native = FakeNative().apply { cpuset = 0xff }
        place(native).use { assertFalse(it.pinned) }
        place(native, auto = false).use { assertEquals(Rule.OFF, it.plan.rule) }
        assertTrue(native.cpuCalls.isEmpty())
    }

    @Test
    fun coresIsolatedUnderLoadPausePinningForThirtySeconds() = onNewThread {
        val native = FakeNative().apply {
            cpuset = 0x7f
            isolated = 0x20
        }
        assertTrue(computeOnce(native))
        now += PinGuard.PAUSE_MS - 1
        place(native).use {
            assertEquals(Rule.PAUSED, it.plan.rule)
            assertEquals("today's threads", 4, it.threads)
            assertFalse(it.pinned)
        }
        now += 1
        assertTrue("tried again after the pause", computeOnce(native))
    }

    @Test
    fun parkedCoresWokenByTheLoadDoNotPause() = onNewThread {
        val native = FakeNative().apply {
            cpuset = 0x7f
            parked = 0x20
        }
        assertTrue(computeOnce(native))
        assertTrue("not paused", computeOnce(native))
    }

    @Test
    fun aPinClosedWithoutAComputationNeverPauses() = onNewThread {
        // a recognition cancelled before Whisper started: nothing ran, so the cores tell nothing
        val native = FakeNative().apply {
            cpuset = 0x7f
            isolated = 0x20
        }
        place(native).close()
        place(native).use { assertTrue("not paused", it.pinned) }
    }

    @Test
    fun aComputationUnderANestedPlacementCountsForTheOuterPin() = onNewThread {
        val native = FakeNative().apply {
            cpuset = 0x7f
            isolated = 0x20
        }
        val outer = place(native)
        place(native).use {
            native.compute("lm", it.threads)
            it.computed()
        }
        outer.close()
        place(native).use { assertEquals(Rule.PAUSED, it.plan.rule) }
    }

    @Test
    fun anUnreadableActiveMaskDecidesNothing() = onNewThread {
        val native = FakeNative().apply {
            cpuset = 0x7f
            isolated = 0x20
        }
        place(native).use {
            native.compute("greedy", it.threads)
            it.computed()
            native.threadCpusError = RuntimeException("/proc")
        }
        native.threadCpusError = null
        place(native).use { assertTrue("not paused", it.pinned) }
    }
}
