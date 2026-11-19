package ai.moravian.czechspeech.internal

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The affinity calls of the host build (Linux): they pin and put back the calling thread only, and a mask the kernel
 * refuses changes nothing. With the model: whisper.cpp's and ggml's worker threads, which the pinned thread starts
 * during the call, inherit its cores, and the transcript does not depend on the number of threads or the cores.
 * The tests run on threads of their own, so the test runner's thread is never pinned.
 */
class NativeAffinityTest {
    @Before
    fun library() = TestEnv.assumeNativeLibrary()

    private fun <T> onNewThread(name: String = "cs-affinity", block: () -> T): T {
        var result: Result<T>? = null
        val thread = Thread({ result = runCatching(block) }, name)
        thread.start()
        thread.join()
        return result!!.getOrThrow()
    }

    @Test
    fun theCallingThreadIsPinnedAndUnpinned() = onNewThread {
        val allowed = JniNativeApi.threadCpus()
        assertNotEquals(0L, allowed)
        val one = java.lang.Long.lowestOneBit(allowed)
        assertTrue(JniNativeApi.setThreadCpus(one))
        assertEquals(one, JniNativeApi.threadCpus())
        assertEquals("sched_getaffinity agrees", one, JniNativeApi.activeThreadCpus())
        // unpinned: every CPU, which the kernel bounds by what the process may use, not a snapshot of the thread
        assertTrue(JniNativeApi.unpinThread())
        assertEquals(allowed, JniNativeApi.threadCpus())
    }

    @Test
    fun unpinningWidensEvenAMaskTheThreadStartedWith() {
        // a thread started by a narrowed thread inherits its mask; unpinning gives it every allowed core
        val allowed = onNewThread { JniNativeApi.threadCpus() }
        onNewThread {
            assertTrue(JniNativeApi.setThreadCpus(java.lang.Long.lowestOneBit(allowed)))
            onNewThread("cs-child") {
                assertEquals("inherited", java.lang.Long.lowestOneBit(allowed), JniNativeApi.threadCpus())
                assertTrue(JniNativeApi.unpinThread())
                assertEquals(allowed, JniNativeApi.threadCpus())
            }
        }
    }

    @Test
    fun otherThreadsKeepTheirCores() {
        val before = onNewThread { JniNativeApi.threadCpus() }
        val mine = JniNativeApi.threadCpus()
        onNewThread { assertTrue(JniNativeApi.setThreadCpus(java.lang.Long.lowestOneBit(JniNativeApi.threadCpus()))) }
        assertEquals("the caller's thread untouched", mine, JniNativeApi.threadCpus())
        assertEquals("a new thread starts unpinned", before, onNewThread { JniNativeApi.threadCpus() })
    }

    @Test
    fun aRefusedMaskChangesNothing() = onNewThread {
        val allowed = JniNativeApi.threadCpus()
        assertFalse("no cores", JniNativeApi.setThreadCpus(0L))
        val possible = runCatching { CpuMask.parseList(File("/sys/devices/system/cpu/possible").readText()) }.getOrNull()
        Assume.assumeTrue("fewer than 64 possible CPUs", possible != null && possible.max() < 63)
        assertFalse("a core that does not exist", JniNativeApi.setThreadCpus(1L shl 63))
        assertEquals(allowed, JniNativeApi.threadCpus())
    }

    @Test
    fun theStatusAndSchedGetaffinityAgreeWhileAllCoresAreOnline() = onNewThread {
        assertEquals(JniNativeApi.activeThreadCpus(), JniNativeApi.threadCpus())
    }

    @Test
    fun theStatusFileFirstThenSchedGetaffinity() = onNewThread {
        val status = File.createTempFile("status", "")
        try {
            status.writeText("Name:\tcs-affinity\nCpus_allowed:\t3\nCpus_allowed_list:\t0-1\n")
            assertEquals(3L, JniNativeApi.threadCpus(status))
            status.writeText("Name:\tcs-affinity\n")
            assertEquals("no mask in it", NativeBridge.nativeThreadCpus(), JniNativeApi.threadCpus(status))
        } finally {
            status.delete()
        }
        assertEquals("unreadable", NativeBridge.nativeThreadCpus(), JniNativeApi.threadCpus(File(status, "missing")))
    }

    /**
     * Every thread named [name] seen while [block] runs, with the Cpus_allowed of each. Linux copies the name of the
     * thread that starts a thread, so whisper.cpp's and ggml's workers carry the name of the thread that started them.
     * The watcher is started here, by the caller's thread, so it carries neither that name nor its cores.
     */
    private fun <T> watchThreads(name: String, block: () -> T): Pair<T, Map<Int, Long>> {
        val seen = ConcurrentHashMap<Int, Long>()
        val done = AtomicBoolean(false)
        val watcher = Thread({
            while (!done.get()) {
                for (tid in File("/proc/self/task").list().orEmpty()) {
                    val comm = runCatching { File("/proc/self/task/$tid/comm").readText().trim() }.getOrNull() ?: continue
                    if (comm != name) continue
                    val status = runCatching { File("/proc/self/task/$tid/status").readText() }.getOrNull() ?: continue
                    CpuMask.parseCpusAllowed(status)?.let { seen[tid.toInt()] = it }
                }
            }
        }, "cs-watcher")
        watcher.start()
        try {
            return block() to seen
        } finally {
            done.set(true)
            watcher.join()
        }
    }

    @Test
    fun whisperWorkersInheritThePinnedCoresAndTheTranscriptStaysTheSame() {
        TestEnv.assumeFile(TestEnv.model, "CzechSpeech model (CZECHSPEECH_MODEL)")
        val audio = TestEnv.audio("e-rec.wav")
        val handle = JniNativeApi.whisperInit(TestEnv.model.absolutePath)
        assertNotEquals(0L, handle)
        try {
            fun greedy(threads: Int): String =
                String(JniNativeApi.whisperTranscribe(handle, 1, audio, audio.size, threads, "cs", null)!!, Charsets.UTF_8)
            fun beams(threads: Int): ByteArray = JniNativeApi.whisperTranscribeNBest(handle, 1, audio, audio.size, threads, 5, "cs", null)!!
            val free = onNewThread { greedy(TestEnv.threads) }
            val freeBeams = onNewThread { beams(TestEnv.threads) }
            val allowed = onNewThread { JniNativeApi.threadCpus() }
            val cpus = CpuMask.cpus(allowed)
            Assume.assumeTrue("at least 3 cores", cpus.size >= 3)
            for ((threads, pinned) in listOf(4 to CpuMask.of(cpus.take(2)), 3 to CpuMask.of(cpus.takeLast(3)))) {
                // the recognising thread takes the watched name only once pinned; it ends pinned (a thread of its own)
                val (text, seen) = watchThreads("cs-run-$threads") {
                    onNewThread("cs-pin-$threads") {
                        assertTrue(JniNativeApi.setThreadCpus(pinned))
                        File("/proc/thread-self/comm").writeText("cs-run-$threads")
                        greedy(threads)
                    }
                }
                println("$threads threads pinned to ${CpuMask.hex(pinned)}: ${seen.size} threads seen, cpus ${seen.values.toSet().map(CpuMask::hex)}")
                assertEquals("$threads threads on ${CpuMask.hex(pinned)}", free, text)
                assertTrue("worker threads seen: ${seen.size}", seen.size >= 2)
                assertEquals("every worker on the pinned cores", setOf(pinned), seen.values.toSet())
                val pinnedBeams = onNewThread {
                    assertTrue(JniNativeApi.setThreadCpus(pinned))
                    beams(threads)
                }
                assertArrayEquals("beam search, $threads threads on ${CpuMask.hex(pinned)}", freeBeams, pinnedBeams)
            }
        } finally {
            JniNativeApi.whisperFree(handle)
        }
    }

    @Test
    fun languageModelScoresDoNotDependOnTheThreads() {
        val file = TestEnv.languageModel.takeIf { it.isFile } ?: TestEnv.tinyLm
        val lm = LanguageModel.load(JniNativeApi, file)
        try {
            val texts = listOf("Viděl jsem velký psy.", "Viděl jsem velké psy.", "pojišťování vývozu se státní podporou")
            val reference = onNewThread { lm.score(texts, 1)!! }
            for (threads in listOf(2, 3, 6)) {
                val pinned = onNewThread {
                    val allowed = JniNativeApi.threadCpus()
                    JniNativeApi.setThreadCpus(CpuMask.of(CpuMask.cpus(allowed).take(2)))
                    lm.score(texts, threads)!!
                }
                assertArrayEquals("${file.name}, $threads threads", reference, pinned, 0.0)
            }
        } finally {
            lm.free()
        }
    }
}
