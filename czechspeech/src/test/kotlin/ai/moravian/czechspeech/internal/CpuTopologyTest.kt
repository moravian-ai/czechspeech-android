package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** Core capacities from sysfs (a fake /sys/devices/system/cpu) and the CPU masks of /proc and the native calls. */
class CpuTopologyTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private fun sysfs(possible: String?, vararg cores: Pair<Long?, Long?>): File {
        val root = tmp.newFolder("cpu")
        possible?.let { File(root, "possible").writeText("$it\n") }
        for ((cpu, values) in cores.withIndex()) {
            val (capacity, maxFreq) = values
            val dir = File(root, "cpu$cpu").apply { mkdirs() }
            capacity?.let { File(dir, "cpu_capacity").writeText("$it\n") }
            maxFreq?.let { File(dir, "cpufreq").mkdirs(); File(dir, "cpufreq/cpuinfo_max_freq").writeText("$it\n") }
        }
        return root
    }

    @Test
    fun capacitiesComeFromCpuCapacity() {
        val root = sysfs("0-3", 191L to 1_950_000L, 191L to 1_950_000L, 741L to 2_600_000L, 1024L to 3_100_000L)
        val t = CpuTopology.read(root)
        assertEquals("cpu_capacity", t.source)
        assertEquals(mapOf(0 to 191L, 1 to 191L, 2 to 741L, 3 to 1024L), t.capacities)
        assertEquals("capacity 191,191,741,1024 (cpu_capacity)", t.describe())
    }

    @Test
    fun maxFrequencyWhenThereIsNoCapacity() {
        val root = sysfs("0-1", null to 1_800_000L, null to 2_400_000L)
        val t = CpuTopology.read(root)
        assertEquals("cpuinfo_max_freq", t.source)
        assertEquals(mapOf(0 to 1_800_000L, 1 to 2_400_000L), t.capacities)
    }

    @Test
    fun aCompleteSourceWinsOverAPartialOne() {
        // capacity missing on one core, frequencies on all: the frequencies, never a mix of the two
        val root = sysfs("0-2", 191L to 1_950_000L, null to 1_950_000L, 1024L to 3_100_000L)
        assertEquals("cpuinfo_max_freq", CpuTopology.read(root).source)
    }

    @Test
    fun partialKnowledgeIsKeptForThePlannerToRefuse() {
        // offline cores may lack cpufreq; whatever is known is kept, and the planner falls back when it needs a gap
        val root = sysfs("0-2", null to 1_950_000L, null to null, null to 3_100_000L)
        val t = CpuTopology.read(root)
        assertEquals(mapOf(0 to 1_950_000L, 2 to 3_100_000L), t.capacities)
        assertEquals("cpuinfo_max_freq", t.source)
    }

    @Test
    fun nothingReadableIsUnknown() {
        val t = CpuTopology.read(sysfs("0-3", null to null, null to null))
        assertTrue(t.capacities.isEmpty())
        assertEquals("unknown", t.describe())
        assertTrue(CpuTopology.read(File(tmp.root, "missing")).capacities.isEmpty())
    }

    @Test
    fun aFailingSysfsIsUnknownAndLogged() {
        val logs = mutableListOf<String>()
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        val t = try {
            CpuTopology.read(object : File(tmp.root, "denied") {
                override fun list(): Array<String> = throw SecurityException("denied")
            })
        } finally {
            CsLog.sink = sink
        }
        assertTrue(t.capacities.isEmpty())
        assertEquals(listOf("cannot read the CPU capacities: SecurityException: denied"), logs)
    }

    @Test
    fun withoutAPossibleListTheCoreDirectoriesCount() {
        val root = sysfs(null, 300L to null, 1024L to null)
        File(root, "cpufreq").mkdirs()          // not a core
        File(root, "cpuidle").mkdirs()
        assertEquals(mapOf(0 to 300L, 1 to 1024L), CpuTopology.read(root).capacities)
    }

    @Test
    fun garbageIsIgnored() {
        val root = sysfs("0-1", 191L to null, null to null)
        File(root, "cpu1/cpu_capacity").writeText("fast\n")
        assertEquals(mapOf(0 to 191L), CpuTopology.read(root).capacities)
    }

    @Test
    fun cpuLists() {
        assertEquals(listOf(0, 1, 2, 3, 6, 8, 9), CpuMask.parseList("0-3,6,8-9\n"))
        assertEquals(listOf(0), CpuMask.parseList("0"))
        assertNull(CpuMask.parseList(""))
        assertNull(CpuMask.parseList("0-x"))
        assertNull(CpuMask.parseList("1-2-3"))
        assertNull(CpuMask.parseList("3-1"))
    }

    @Test
    fun cpusAllowedOfAThreadStatus() {
        val status = "Name:\tczechspeech-whi\nCpus_allowed:\t7f\nCpus_allowed_list:\t0-6\nMems_allowed:\t1\n"
        assertEquals(0x7fL, CpuMask.parseCpusAllowed(status))
        // the kernel groups the mask in 32-bit words
        assertEquals(0xf0L, CpuMask.parseCpusAllowed("Cpus_allowed:\t00000000,000000f0\n"))
        assertEquals(1L shl 40 or 1L, CpuMask.parseCpusAllowed("Cpus_allowed:\t00000100,00000001\n"))
        // a CPU above 63 does not fit a mask: unknown
        assertNull(CpuMask.parseCpusAllowed("Cpus_allowed:\t00000001,00000000,00000000\n"))
        assertEquals(0xffL, CpuMask.parseCpusAllowed("Cpus_allowed:\t00000000,00000000,000000ff\n"))
        assertNull(CpuMask.parseCpusAllowed("Cpus_allowed_list:\t0-6\n"))
        assertNull(CpuMask.parseCpusAllowed("Cpus_allowed:\tzz\n"))
        assertNull(CpuMask.parseCpusAllowed("Cpus_allowed:\t0\n"))
    }

    @Test
    fun masks() {
        assertEquals(0x70L, CpuMask.of(listOf(4, 5, 6)))
        assertEquals(listOf(4, 5, 6), CpuMask.cpus(0x70))
        assertEquals(listOf(63), CpuMask.cpus(Long.MIN_VALUE))
        assertEquals("0xf0", CpuMask.hex(0xf0))
        assertEquals("0x8000000000000000", CpuMask.hex(Long.MIN_VALUE))
    }
}
