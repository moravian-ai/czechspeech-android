package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechException
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechEvent
import ai.moravian.czechspeech.VariantSelection
import ai.moravian.czechspeech.internal.NBestBytes.Hyp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Thread placement in the pipeline, on the fake native layer, which models the kernel's affinity of the whisper thread
 * (Linux 6.2+: sched_setaffinity's mask stays the thread's own request across cpuset changes, and threads it starts
 * copy it): every native computation (final text, preview, beam search, language model) gets the planned threads with
 * the whisper thread pinned to the planned cores, and afterwards the thread is unpinned (every CPU, which the kernel
 * bounds by the cpuset), never put back to a snapshot; also when the recognition fails or is cancelled. A refused or
 * failing affinity call changes nothing else, and every affinity call is made on the whisper thread.
 */
class ThreadPlacementEngineTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    /** Pixel 9a (Tensor G4); its cpusets: top-app 0xff, foreground 0x7f, background 0x0f. */
    private val tensorG4 = CpuTopology(
        mapOf(0 to 191L, 1 to 191L, 2 to 191L, 3 to 191L, 4 to 741L, 5 to 741L, 6 to 741L, 7 to 1024L),
        "cpu_capacity",
    )
    private val speech = Signals.tone(1.0)

    /** An engine with today's limit of the Pixel 9a (4 threads, or [threads]) and its CPU layout. */
    private fun engine(
        native: FakeNative,
        config: CzechSpeechConfig = CzechSpeechConfig(),
        audio: FloatArray = Signals.concat(speech, Signals.zeros(2.0)),
        hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE,
        cpus: CpuTopology = tensorG4,
        threads: Int = 4,
        pinGuard: PinGuard = PinGuard(),
    ) = fakeEngine(native, hooks, config = config, threads = threads, cpus = cpus, pinGuard = pinGuard) {
        ArrayAudioSource(audio)
    }

    private fun FakeNative.compute(what: String) = computes.single { it.what == what }

    private fun withLogs(block: () -> Unit): List<String> {
        val logs = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            block()
        } finally {
            CsLog.sink = sink
        }
        return logs.toList()
    }

    /** Unpinned: the thread's own request is none, so threads it starts are not pinned either. */
    private fun FakeNative.assertUnpinned() = assertEquals("request after the call", FakeNative.ALL_CPUS, request)

    @Test
    fun onByDefault() {
        assertTrue(CzechSpeechConfig().autoThreads)
    }

    @Test
    fun theForegroundGetsThreeThreadsOnTheThreeBigCores() {
        val native = FakeNative().apply { cpuset = 0x7f }
        assertEquals("dobrý den", runBlocking { engine(native).transcribe(speech) })
        val greedy = native.compute("greedy")
        assertEquals(3, greedy.threads)
        assertEquals(0x70L, greedy.cpus)
        assertEquals("pinned for the call, then unpinned", listOf("set 0x70", "unpin"), native.cpuCallNames)
        native.assertUnpinned()
        assertEquals(0x7fL, native.cpus)
    }

    @Test
    fun afterTheForegroundTopAppPlansFourThreadsAgain() {
        // the floating button, then the keyboard on screen: no mask of the foreground may stay behind
        val native = FakeNative().apply { cpuset = 0x7f }
        val engine = engine(native)
        runBlocking { engine.transcribe(speech) }
        native.cpuset = 0xff
        runBlocking { engine.transcribe(speech) }
        assertEquals(listOf("greedy/3/0x70", "greedy/4/0xff"), native.computes.map { it.toString() })
        native.assertUnpinned()
    }

    @Test
    fun aCpusetChangeDuringTheComputationLeavesNoMaskBehind() {
        for (olderKernel in listOf(false, true)) {
            val native = FakeNative().apply {
                cpusetChangeDropsRequest = olderKernel
                cpuset = 0x7f
                onTranscribe = { if (it == 0) cpuset = 0xff }         // the keyboard comes on screen meanwhile
            }
            val engine = engine(native)
            runBlocking { engine.transcribe(speech) }
            native.assertUnpinned()
            assertEquals(0xffL, native.cpus)
            runBlocking { engine.transcribe(speech) }
            assertEquals("kernel before 6.2: $olderKernel", "greedy/4/0xff", native.computes.last().toString())
        }
    }

    @Test
    fun topAppKeepsTodaysFourThreadsUnpinned() {
        val native = FakeNative().apply { cpuset = 0xff }
        runBlocking { engine(native).transcribe(speech) }
        assertEquals("greedy/4/0xff", native.compute("greedy").toString())
        assertTrue("nothing to pin", native.cpuCalls.isEmpty())
    }

    @Test
    fun inTheBackgroundTodaysThreads() {
        val native = FakeNative().apply { cpuset = 0x0f }
        runBlocking { engine(native).transcribe(speech) }
        assertEquals("greedy/4/0xf", native.compute("greedy").toString())
        assertTrue(native.cpuCalls.isEmpty())
    }

    @Test
    fun coresParkedAtThePlanButWokenByTheLoadKeepThePlan() {
        // foreground with cpu5 parked while idle (core_ctl): the plan counts the thread's mask, the load wakes cpu5
        val native = FakeNative().apply {
            cpuset = 0x7f
            parked = 0x20
        }
        val engine = engine(native)
        repeat(2) { runBlocking { engine.transcribe(speech) } }
        assertEquals(listOf("greedy/3/0x70", "greedy/3/0x70"), native.computes.map { it.toString() })
        assertEquals("never paused", listOf("set 0x70", "unpin", "set 0x70", "unpin"), native.cpuCallNames)
    }

    @Test
    fun coresThatStayIsolatedUnderLoadPausePinningForThirtySeconds() {
        // cpu5 isolated (thermal): of the 3 pinned threads one had no core of its own. Today's behaviour for 30 s
        var now = 1_000_000L
        val native = FakeNative().apply {
            cpuset = 0x7f
            isolated = 0x20
        }
        val engine = engine(native, pinGuard = PinGuard { now })
        val logs = withLogs {
            runBlocking { engine.transcribe(speech) }
            now += PinGuard.PAUSE_MS - 1
            runBlocking { engine.transcribe(speech) }
            now += 1
            runBlocking { engine.transcribe(speech) }
        }
        assertEquals(listOf("greedy/3/0x70", "greedy/4/0x7f", "greedy/3/0x70"), native.computes.map { it.toString() })
        assertEquals(listOf("set 0x70", "unpin", "set 0x70", "unpin"), native.cpuCallNames)
        assertTrue(logs.joinToString("\n"), logs.any { it.startsWith("only 2 of the pinned cores 0x70 ran for 3 threads; pinning paused for 30 s") })
        assertTrue(logs.any { "threads 4, not pinned (allowed 0x7f, pinning paused)" in it })
    }

    @Test
    fun neverMoreThreadsThanAllowedCores() {
        // a cpuset of two cores and a limit of 4: 2 threads, not 4 on 2 cores (ggml's barriers spin)
        val native = FakeNative().apply { cpuset = 0x03 }
        runBlocking { engine(native).transcribe(speech) }
        assertEquals(2, native.compute("greedy").threads)
        assertTrue(native.cpuCalls.isEmpty())
    }

    @Test
    fun aPinOnSleepingCoresIsRefusedAndRunsAsBefore() {
        // every A720 parked when the pin is set: the kernel refuses it (EINVAL), today's 4 threads run unpinned
        val native = FakeNative().apply {
            cpuset = 0x7f
            parked = 0x70
        }
        runBlocking { engine(native).transcribe(speech) }
        assertEquals("greedy/4/0x7f", native.compute("greedy").toString())
        assertEquals(listOf("set 0x70"), native.cpuCallNames)
    }

    @Test
    fun unknownCoresOrLayoutKeepTodaysThreads() {
        val noCores = FakeNative()                     // cpuset 0: the thread's CPUs unknown
        runBlocking { engine(noCores).transcribe(speech) }
        assertEquals(4, noCores.compute("greedy").threads)
        assertTrue(noCores.cpuCalls.isEmpty())
        val noLayout = FakeNative().apply { cpuset = 0x7f }
        runBlocking { engine(noLayout, cpus = CpuTopology.UNKNOWN).transcribe(speech) }
        assertEquals(4, noLayout.compute("greedy").threads)
        assertTrue(noLayout.cpuCalls.isEmpty())
    }

    @Test
    fun turnedOffMakesTodaysCallsWithoutTouchingTheCores() {
        val native = FakeNative().apply { cpuset = 0x7f }
        runBlocking { engine(native, CzechSpeechConfig(autoThreads = false)).transcribe(speech) }
        assertEquals("greedy/4/0x7f", native.compute("greedy").toString())
        assertTrue(native.cpuCalls.isEmpty())
    }

    @Test
    fun turnedOffWithAForcedThreadCountUsesExactlyIt() {
        // the Benchmark's --ei threads 3 --ez auto_threads false, for comparing transcripts on 3 and 4 threads
        val native = FakeNative().apply { cpuset = 0xff }
        runBlocking { engine(native, CzechSpeechConfig(threads = 3, autoThreads = false), threads = 3).transcribe(speech) }
        assertEquals(3, native.compute("greedy").threads)
        assertTrue(native.cpuCalls.isEmpty())
    }

    @Test
    fun aRefusedPinRunsTodaysThreads() {
        val native = FakeNative().apply {
            cpuset = 0x7f
            refusedCpuCalls = setOf(0)
        }
        assertEquals("dobrý den", runBlocking { engine(native).transcribe(speech) })
        assertEquals("today's threads, not the plan's", "greedy/4/0x7f", native.compute("greedy").toString())
        assertEquals("nothing to undo", listOf("set 0x70"), native.cpuCallNames)
        native.assertUnpinned()
    }

    @Test
    fun aFailingAffinityCallBreaksNothing() {
        for (error in listOf(RuntimeException("sched_setaffinity"), UnsatisfiedLinkError("old library"))) {
            val native = FakeNative().apply {
                cpuset = 0x7f
                cpusError = error
            }
            assertEquals("dobrý den", runBlocking { engine(native).transcribe(speech) })
            assertEquals(4, native.compute("greedy").threads)
        }
        for (error in listOf(RuntimeException("/proc"), UnsatisfiedLinkError("old library"))) {
            val native = FakeNative().apply {
                cpuset = 0x7f
                threadCpusError = error
            }
            assertEquals("dobrý den", runBlocking { engine(native).transcribe(speech) })
            assertEquals("cores unknown: today's threads", 4, native.compute("greedy").threads)
            assertTrue(native.cpuCalls.isEmpty())
        }
    }

    @Test
    fun aFailedUnpinIsLoggedRetriedAndNeverPlannedFrom() {
        val native = FakeNative().apply {
            cpuset = 0x7f
            refusedCpuCalls = setOf(1, 2)              // the unpin after the first recognition, and its first retry
        }
        val engine = engine(native)
        val logs = withLogs {
            runBlocking { engine.transcribe(speech) }
            // still narrowed by us: the retry fails, so no more threads than the cores of that mask, nothing pinned
            runBlocking { engine.transcribe(speech) }
            // the next retry works: planned from the cpuset again
            runBlocking { engine.transcribe(speech) }
        }
        assertTrue(logs.joinToString("\n"), logs.any { it.startsWith("could not unpin the recogniser thread; it keeps cpus 0x70") })
        assertEquals(listOf("set 0x70", "unpin", "unpin", "unpin", "set 0x70", "unpin"), native.cpuCallNames)
        assertEquals(listOf("greedy/3/0x70", "greedy/3/0x70", "greedy/3/0x70"), native.computes.map { it.toString() })
        assertTrue(logs.any { "threads 3, not pinned (allowed 0x70, affinity not put back)" in it })
        native.assertUnpinned()
    }

    @Test
    fun theCoresAreUnpinnedWhenTheRecognitionFails() {
        val native = FakeNative().apply {
            cpuset = 0x7f
            failNext = 1
        }
        assertThrows(CzechSpeechException::class.java) { runBlocking { engine(native).transcribe(speech) } }
        assertEquals(listOf("set 0x70", "unpin"), native.cpuCallNames)
        native.assertUnpinned()
    }

    @Test
    fun theCoresAreUnpinnedWhenTheRecognitionIsCancelled() = runBlocking {
        val native = FakeNative().apply {
            cpuset = 0x7f
            blockUntilAborted = true
        }
        val engine = engine(native)
        val job = launch(Dispatchers.Default) { engine.transcribe(speech) }
        assertTrue(native.transcribeStarted.await(10, TimeUnit.SECONDS))
        assertEquals("pinned while it runs", 0x70L, native.cpus)
        job.cancel()
        job.join()
        engine.awaitWhisperThread()
        assertEquals(listOf("set 0x70", "unpin"), native.cpuCallNames)
        native.assertUnpinned()
    }

    @Test
    fun aRecognitionCancelledDuringTheModelLoadUnpinsWithoutComputing() = runBlocking {
        val load = CountDownLatch(1)
        val native = FakeNative().apply {
            cpuset = 0x7f
            holdInit = load
        }
        val engine = engine(native)
        val job = launch(Dispatchers.Default) { engine.transcribe(speech) }
        assertTrue(native.initStarted.await(10, TimeUnit.SECONDS))
        job.cancel()                    // the token is cancelled before Whisper could start
        job.join()
        load.countDown()
        engine.awaitWhisperThread()
        assertTrue(native.computes.isEmpty())
        assertEquals(listOf("set 0x70", "unpin"), native.cpuCallNames)
        native.assertUnpinned()
    }

    @Test
    fun variantSelectionScoresWithThePlacementOfItsRecognition() {
        val native = FakeNative().apply {
            cpuset = 0x7f
            hypotheses = listOf(
                listOf(Hyp(" Viděl jsem velký psy.", -2.0, 9, best = true), Hyp(" Viděl jsem velké psy.", -2.3, 9)),
            )
        }
        runBlocking { engine(native, CzechSpeechConfig(variantSelection = VariantSelection(TestEnv.tinyLm))).transcribe(speech) }
        assertEquals(listOf("beam/3/0x70", "lm/3/0x70"), native.computes.map { it.toString() })
        assertEquals("one pin for the whole recognition", listOf("set 0x70", "unpin"), native.cpuCallNames)
    }

    @Test
    fun previewsAndSentenceJoiningPinOnTheWhisperThreadOnly() = runBlocking {
        val native = FakeNative().apply {
            cpuset = 0x7f
            texts += listOf("dobrý", "dobrý den", "Jasně, ale můžeš nějak.", "s tebou spojit.")
        }
        // 12 s of speech (two previews), a pause, a second utterance: the boundary runs the language model
        val audio = Signals.concat(Signals.zeros(0.5), Signals.tone(12.0), Signals.zeros(1.0), speech, Signals.zeros(1.5))
        // capture waits at each request until the preview has FINISHED (its tracker is free again), so both previews run whatever the timing
        val hooks = object : SpeechEngine.Hooks {
            var requests = 0
            val finished = java.util.concurrent.atomic.AtomicInteger()
            override fun previewFinished() {
                finished.incrementAndGet()
            }

            override fun previewRequested(samples: FloatArray) {
                requests++
                val deadline = System.nanoTime() + 10_000_000_000L
                while (finished.get() < requests) {
                    check(System.nanoTime() < deadline) { "preview $requests never finished" }
                    Thread.sleep(1)
                }
            }
        }
        val config = CzechSpeechConfig(sentenceJoining = SentenceJoining(TestEnv.tinyLm))
        val events = withContext(Dispatchers.Default) {
            engine(native, config, audio, hooks).listenWithPreviews(PreviewConfig()).toList()
        }
        assertEquals(2, events.count { it is SpeechEvent.Transcript })
        assertEquals(listOf("preview", "preview", "greedy", "greedy", "lm"), native.computes.map { it.what })
        assertTrue(native.computes.joinToString(), native.computes.all { it.threads == 3 && it.cpus == 0x70L })
        assertEquals("one pin per computation", 5, native.cpuCalls.count { it.what == "set" })
        assertEquals(
            "never the capture thread nor the collecting coroutine",
            setOf(FakeNative.WHISPER_THREAD), native.cpuCalls.map { it.thread }.toSet(),
        )
        native.assertUnpinned()
    }

    @Test
    fun theLogShowsTheCoresOncePerSessionAndThreadsAndMaskPerRecognition() {
        val native = FakeNative().apply { cpuset = 0x7f }
        val engine = engine(native)
        val logs = withLogs {
            runBlocking { withContext(Dispatchers.Default) { engine.listen().toList() } }
            runBlocking { engine.transcribe(speech) }
        }
        val layout = "cpus: capacity 191,191,191,191,741,741,741,1024 (cpu_capacity), at most 4 threads, placed per recognition"
        assertEquals(logs.joinToString("\n"), 1, logs.count { it == layout })
        val recognitions = logs.filter { it.startsWith("utterance: ") || it.startsWith("transcribe: ") }
        assertEquals(2, recognitions.size)
        assertTrue(recognitions.joinToString("\n"), recognitions.all { "threads 3 on cpus 0x70 (allowed 0x7f, fast cores)" in it })
        assertTrue(logs.none { "dobrý" in it })
    }

    @Test
    fun withoutASessionTheCoresAreLoggedOnce() {
        val native = FakeNative().apply { cpuset = 0x7f }
        val engine = engine(native, CzechSpeechConfig(autoThreads = false))
        val logs = withLogs { repeat(2) { runBlocking { engine.transcribe(speech) } } }
        val layout = "cpus: capacity 191,191,191,191,741,741,741,1024 (cpu_capacity), 4 threads, automatic placement off"
        assertEquals(logs.joinToString("\n"), 1, logs.count { it == layout })
        assertTrue(logs.filter { it.startsWith("transcribe: ") }.all { "threads 4, not pinned (allowed 0x7f, automatic placement off)" in it })
    }
}
