package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * keepModelLoaded and releaseModel(): the model stays loaded from its first load until releaseModel() or close().
 * releaseModel() racing the start or the end of a session or call is held in a test hook, so each order is exact.
 * Every test also checks that each context was freed on its own thread, the model on the whisper thread
 * ([FreedOnOwnThreads]).
 */
class KeepModelLoadedTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0))

    /** Holds the first thread that reaches [here] once [armed] until [proceed] is released; lets others through. */
    private class Hold {
        val armed = AtomicBoolean(false)
        val reached = CountDownLatch(1)
        val proceed = CountDownLatch(1)

        fun here() {
            if (!armed.compareAndSet(true, false)) return
            reached.countDown()
            check(proceed.await(10, TimeUnit.SECONDS)) { "never released" }
        }

        fun reachedWithin(): Boolean = reached.await(10, TimeUnit.SECONDS)
    }

    private fun keeping(
        native: FakeNative,
        hooks: SpeechEngine.Hooks = SpeechEngine.Hooks.NONE,
        source: () -> AudioSource,
    ) = fakeEngine(native, hooks, keepModelLoaded = true, source = source)

    private fun noMicrophone(): AudioSource = error("no microphone in this test")

    /** An endless session of silence (paced), which never runs the recogniser; it ends with stop() or cancellation. */
    private fun endlessSilence() = ArrayAudioSource(FloatArray(0), tail = { 0f }, sleepPerFrameMs = 1)

    private suspend fun SpeechEngine.transcriptCount(): Int = listen().toList().count { it is SpeechEvent.Transcript }

    /**
     * Holds the next model free in [FakeNative.holdFree] and lets it go only after giving [ending] the chance to end
     * first: for an engine that does not wait for the free, 300 ms is plenty; one that does can never end before the
     * free, however long this waits. [ending] records its end in [FakeNative.events], so the log holds the order.
     */
    private class HeldFree(private val native: FakeNative) {
        private val free = CountDownLatch(1)

        init {
            native.holdFree = free
        }

        suspend fun letGoAfter(ending: Job) {
            try {
                assertTrue("the free has started", native.freeStarted.await(10, TimeUnit.SECONDS))
                withTimeoutOrNull(300) { ending.join() }
            } finally {
                free.countDown()
            }
        }
    }

    // keepModelLoaded

    @Test
    fun theModelIsLoadedOnceAndKeptAcrossSessionsAndCallsUntilClose() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())
        assertTrue("kept after the session", engine.isModelLoaded)
        assertEquals(1, engine.transcriptCount())
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertEquals("loaded once", 1, native.whisperInits.get())
        assertEquals("never freed in between", 0, native.whisperFrees.get())
        assertTrue(engine.isModelLoaded)

        engine.close()
        engine.awaitClosed()
        assertEquals("close() frees the kept model", 1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertEquals("the gate VAD goes with it", native.vadInits.get(), native.vadFrees.get())
    }

    @Test
    fun transcribeKeepsTheModelAndSilenceStillNeverLoadsIt() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { noMicrophone() }
        assertEquals("", engine.transcribe(Signals.zeros(1.0)))
        assertEquals("no speech: the model is not loaded", 0, native.whisperInits.get())
        assertFalse(engine.isModelLoaded)
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertTrue("kept after the call", engine.isModelLoaded)
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertEquals(1, native.whisperInits.get())
        assertEquals(0, native.whisperFrees.get())
    }

    @Test
    fun withoutKeepModelLoadedEverySessionAndCallStillFreesTheModelAndReleaseChangesNothing() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native) { ArrayAudioSource(utterance) }
        engine.releaseModel()
        assertEquals(1, engine.transcriptCount())
        assertFalse(engine.isModelLoaded)
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertFalse(engine.isModelLoaded)
        engine.releaseModel()
        engine.awaitWhisperThread()
        assertEquals(1, engine.transcriptCount())
        assertEquals(3, native.whisperInits.get())
        assertEquals(3, native.whisperFrees.get())
    }

    @Test
    fun cancellingASessionKeepsTheModel() = runBlocking {
        val native = FakeNative()
        val sources = ArrayDeque(
            listOf(ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1), ArrayAudioSource(utterance)),
        )
        val engine = keeping(native) { sources.removeFirst() }
        val firstTranscript = CompletableDeferred<Unit>()
        val job = launch(Dispatchers.Default) {
            engine.listen().collect { if (it is SpeechEvent.Transcript) firstTranscript.complete(Unit) }
        }
        firstTranscript.awaitWithin(10_000, "the first transcript")
        job.cancel()
        job.joinWithin(2_000, "the cancelled session")
        engine.awaitWhisperThread()                  // the cancelled session's own hold is dropped by now
        assertTrue("kept", engine.isModelLoaded)
        assertEquals(0, native.whisperFrees.get())
        assertEquals(1, engine.transcriptCount())
        assertEquals("the next session uses it", 1, native.whisperInits.get())
    }

    @Test
    fun aSessionCancelledDuringTheModelLoadKeepsTheModelOnceTheLoadEnds() = runBlocking {
        val load = CountDownLatch(1)
        val native = FakeNative().apply { holdInit = load }
        val engine = keeping(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        try {
            assertTrue("the model is loading", native.initStarted.await(10, TimeUnit.SECONDS))
            job.cancel()
            job.joinWithin(2_000, "the session cancelled during the model load")
        } finally {
            load.countDown()
        }
        engine.awaitWhisperThread()
        assertTrue("the load has ended and the model is kept", engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
        assertEquals(0, native.whisperFrees.get())
    }

    // releaseModel()

    @Test
    fun releaseFreesAnIdleModelAndTheNextSessionLoadsAndKeepsItAgain() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())
        engine.releaseModel()
        engine.awaitWhisperThread()
        assertEquals("freed", 1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertEquals("the gate VAD too", native.vadInits.get(), native.vadFrees.get())

        assertEquals(1, engine.transcriptCount())
        assertEquals("loaded again", 2, native.whisperInits.get())
        assertTrue("and kept again", engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun releaseWithNoModelLoadedAndNothingRunningDoesNothing() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { ArrayAudioSource(utterance) }
        engine.releaseModel()
        engine.releaseModel()
        engine.awaitWhisperThread()
        assertEquals(
            "no native call at all (model loads, frees, VAD inits, frees)",
            listOf(0, 0, 0, 0),
            listOf(native.whisperInits.get(), native.whisperFrees.get(), native.vadInits.get(), native.vadFrees.get()),
        )
        // keepModelLoaded still applies: the next session loads the model and keeps it.
        assertEquals(1, engine.transcriptCount())
        assertTrue(engine.isModelLoaded)
        engine.releaseModel()
        engine.awaitWhisperThread()
        engine.releaseModel()                        // freed already
        engine.awaitWhisperThread()
        assertEquals(1, native.whisperFrees.get())
        engine.close()
        engine.releaseModel()                        // after close(): nothing, and no exception
        engine.awaitClosed()
        assertEquals(1, native.whisperInits.get())
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun releaseDuringASessionFreesTheModelWhenTheSessionEndsBeforeItsFlowCompletes() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertTrue("kept after the call", engine.isModelLoaded)
        val held = HeldFree(native)
        val firstTranscript = CompletableDeferred<Unit>()
        val collected = async(Dispatchers.Default) {
            engine.listen().onEach { if (it is SpeechEvent.Transcript) firstTranscript.complete(Unit) }.toList()
                .also { native.events += "flow completed" }
        }
        firstTranscript.awaitWithin(10_000, "the first transcript")
        engine.releaseModel()
        engine.awaitWhisperThread()
        assertTrue("the session still holds the model", engine.isModelLoaded)
        assertEquals(0, native.whisperFrees.get())
        engine.stop()
        held.letGoAfter(collected)
        collected.awaitWithin(10_000, "the stopped session")
        assertEquals(listOf("whisperFree", "flow completed"), native.events)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun releaseWaitsForTheLastSessionOrCallToEnd() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun callRegistered() = hold.here()
        }) { endlessSilence() }
        val session = async(Dispatchers.Default) { engine.listen().toList() }
        awaitCondition("the session's model") { engine.isModelLoaded }
        val held = HeldFree(native)
        hold.armed.set(true)
        val call = async(Dispatchers.Default) {
            engine.transcribe(Signals.tone(1.0)).also { native.events += "call returned" }
        }
        try {
            assertTrue("the call is registered", hold.reachedWithin())
            engine.releaseModel()
            engine.stop()
            session.awaitWithin(10_000, "the stopped session")
            assertTrue("the call still running keeps the model", engine.isModelLoaded)
            assertEquals(0, native.whisperFrees.get())
        } finally {
            hold.proceed.countDown()
        }
        held.letGoAfter(call)
        assertEquals("dobrý den", call.awaitWithin(10_000, "the call"))
        assertEquals("freed before the call returned", listOf("whisperFree", "call returned"), native.events)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
    }

    @Test
    fun releaseNeverWaitsForTheWhisperThreadAndTheModelIsFreedThere() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative()
        val engine = keeping(native) { noMicrophone() }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))      // loaded and kept
        native.holdGate = gate
        val busy = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        try {
            awaitCondition("the second call in the speech gate") { native.gateCalls.get() == 2 }
            busy.cancel()                            // unregistered at once; its task stays in the gate
            busy.joinWithin(2_000, "the cancelled call")
            engine.releaseModel()                    // nothing is registered: due now, queued behind the gate
            assertEquals("releaseModel() has returned before the free", 0, native.whisperFrees.get())
            assertTrue(engine.isModelLoaded)
        } finally {
            gate.countDown()
        }
        engine.awaitWhisperThread()
        assertEquals(1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertEquals("freed on the whisper thread", listOf(FakeNative.WHISPER_THREAD), native.whisperFreeThreads)
    }

    @Test
    fun releaseDuringASessionThatIsThenCancelledFreesTheModelRightAfter() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { endlessSilence() }
        val job = launch(Dispatchers.Default) { engine.listen().collect { } }
        awaitCondition("the session's model") { engine.isModelLoaded }
        engine.releaseModel()                        // a session is running: due when it ends
        job.cancel()
        job.joinWithin(2_000, "the cancelled session")
        engine.awaitWhisperThread()                  // a cancelled session does not wait; the free follows it there
        assertEquals(1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun releaseDuringACallThatIsThenCancelledFreesTheModelOnceItsTaskHasEnded() = runBlocking {
        val gate = CountDownLatch(1)
        val native = FakeNative()
        val engine = keeping(native) { noMicrophone() }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))      // loaded and kept
        native.holdGate = gate
        val call = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        try {
            awaitCondition("the second call in the speech gate") { native.gateCalls.get() == 2 }
            engine.releaseModel()                    // the call is running: due when it ends
            call.cancel()                            // it ends now; its task stays in the gate
            call.joinWithin(2_000, "the cancelled call")
            assertEquals("not freed under the task still in the gate", 0, native.whisperFrees.get())
            assertTrue(engine.isModelLoaded)
        } finally {
            gate.countDown()
        }
        engine.awaitWhisperThread()
        assertEquals(1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertEquals("the cancelled call never reached the recogniser", 1, native.passedToWhisper.size)
    }

    // close()

    @Test
    fun closeDuringASessionFreesTheKeptModel() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val firstTranscript = CompletableDeferred<Unit>()
        val job = launch(Dispatchers.Default) {
            runCatching { engine.listen().collect { if (it is SpeechEvent.Transcript) firstTranscript.complete(Unit) } }
        }
        firstTranscript.awaitWithin(10_000, "the first transcript")
        engine.close()
        job.joinWithin(2_000, "the session cancelled by close()")
        engine.awaitClosed()
        assertEquals(1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
        assertEquals(native.vadInits.get(), native.vadFrees.get())
    }

    @Test
    fun closeDuringTheModelLoadFreesTheModelWhenTheLoadEnds() = runBlocking {
        val load = CountDownLatch(1)
        val native = FakeNative().apply { holdInit = load }
        val engine = keeping(native) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        val job = launch(Dispatchers.Default) { runCatching { engine.listen().collect { } } }
        try {
            assertTrue("the model is loading", native.initStarted.await(10, TimeUnit.SECONDS))
            engine.close()
            job.joinWithin(2_000, "the session cancelled by close()")
        } finally {
            load.countDown()
        }
        engine.awaitClosed()
        assertEquals(1, native.whisperInits.get())
        assertEquals("a load that ends after close() is not kept", 1, native.whisperFrees.get())
        assertFalse(engine.isModelLoaded)
    }

    // releaseModel() racing the start and the end of a session or a call

    @Test
    fun releaseJustBeforeASessionRegistersFreesTheIdleModelAndTheSessionLoadsAndKeepsItAgain() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun sessionStarting() = hold.here()
        }) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())    // loaded and kept
        hold.armed.set(true)
        val collected = async(Dispatchers.Default) { engine.transcriptCount() }
        try {
            assertTrue("the collection is starting, its session not registered", hold.reachedWithin())
            engine.releaseModel()
            engine.awaitWhisperThread()
            assertEquals("nothing is running: freed at once", 1, native.whisperFrees.get())
            assertFalse(engine.isModelLoaded)
        } finally {
            hold.proceed.countDown()
        }
        assertEquals(1, collected.awaitWithin(10_000, "the session that started after releaseModel()"))
        assertEquals("the session loads the model again", 2, native.whisperInits.get())
        assertTrue("and keeps it", engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun releaseJustAfterASessionRegisteredFreesTheModelWhenThatSessionEnds() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun sessionRegistered() = hold.here()
        }) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())
        val held = HeldFree(native)
        hold.armed.set(true)
        val collected = async(Dispatchers.Default) {
            engine.transcriptCount().also { native.events += "flow completed" }
        }
        try {
            assertTrue("the session is registered, nothing of it has run", hold.reachedWithin())
            engine.releaseModel()
            engine.awaitWhisperThread()
            assertTrue("a session is running: it keeps the model", engine.isModelLoaded)
            assertEquals(0, native.whisperFrees.get())
        } finally {
            hold.proceed.countDown()
        }
        held.letGoAfter(collected)
        assertEquals(1, collected.awaitWithin(10_000, "the session"))
        assertEquals("not loaded again", 1, native.whisperInits.get())
        assertEquals("freed before the flow completed", listOf("whisperFree", "flow completed"), native.events)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun releaseWhileASessionIsEndingIsNotLost() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun sessionEnding() = hold.here()
        }) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())
        val held = HeldFree(native)
        hold.armed.set(true)
        val collected = async(Dispatchers.Default) {
            engine.transcriptCount().also { native.events += "flow completed" }
        }
        try {
            assertTrue("the session has dropped its own hold and is still registered", hold.reachedWithin())
            engine.releaseModel()
            engine.awaitWhisperThread()
            assertTrue("the session is still registered: it keeps the model", engine.isModelLoaded)
        } finally {
            hold.proceed.countDown()
        }
        held.letGoAfter(collected)
        assertEquals(1, collected.awaitWithin(10_000, "the session"))
        assertEquals("freed as the session was unregistered", listOf("whisperFree", "flow completed"), native.events)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, native.whisperInits.get())
    }

    @Test
    fun releaseJustBeforeACallRegistersFreesTheIdleModelAndTheCallLoadsAndKeepsItAgain() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun callStarting() = hold.here()
        }) { noMicrophone() }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))      // loaded and kept
        hold.armed.set(true)
        val call = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        try {
            assertTrue("the call passed its first check, it is not registered", hold.reachedWithin())
            engine.releaseModel()
            engine.awaitWhisperThread()
            assertEquals("nothing is running: freed at once", 1, native.whisperFrees.get())
            assertFalse(engine.isModelLoaded)
        } finally {
            hold.proceed.countDown()
        }
        assertEquals("dobrý den", call.awaitWithin(10_000, "the call that started after releaseModel()"))
        assertEquals("the call loads the model again", 2, native.whisperInits.get())
        assertTrue("and keeps it", engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }

    @Test
    fun releaseJustAfterACallRegisteredFreesTheModelWhenThatCallEndsBeforeItReturns() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun callRegistered() = hold.here()
        }) { noMicrophone() }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        val held = HeldFree(native)
        hold.armed.set(true)
        val call = async(Dispatchers.Default) {
            engine.transcribe(Signals.tone(1.0)).also { native.events += "call returned" }
        }
        try {
            assertTrue("the call is registered, not queued on the whisper thread", hold.reachedWithin())
            engine.releaseModel()
            engine.awaitWhisperThread()
            assertTrue("a call is running: it keeps the model", engine.isModelLoaded)
            assertEquals(0, native.whisperFrees.get())
        } finally {
            hold.proceed.countDown()
        }
        held.letGoAfter(call)
        assertEquals("dobrý den", call.awaitWithin(10_000, "the call"))
        assertEquals("not loaded again", 1, native.whisperInits.get())
        assertEquals("freed before the call returned", listOf("whisperFree", "call returned"), native.events)
        assertFalse(engine.isModelLoaded)
    }

    @Test
    fun aReleaseStillWaitingDoesNotApplyToASessionStartedAfterIt() = runBlocking {
        val hold = Hold()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun callRegistered() = hold.here()
        }) { ArrayAudioSource(utterance, tail = { 0f }, sleepPerFrameMs = 1) }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))      // loaded and kept
        hold.armed.set(true)
        val call = async(Dispatchers.Default) { engine.transcribe(Signals.tone(1.0)) }
        val firstTranscript = CompletableDeferred<Unit>()
        val session = try {
            assertTrue("1. the call is registered", hold.reachedWithin())
            engine.releaseModel()                    // 2. it waits for the call
            val started = async(Dispatchers.Default) {
                engine.listen().onEach { if (it is SpeechEvent.Transcript) firstTranscript.complete(Unit) }.toList()
            }
            firstTranscript.awaitWithin(10_000, "3. the session's first transcript")
            started
        } finally {
            hold.proceed.countDown()                 // 4. the call finishes
        }
        assertEquals("dobrý den", call.awaitWithin(10_000, "the call"))
        engine.stop()                                // 5.
        session.awaitWithin(10_000, "the stopped session")
        engine.awaitWhisperThread()
        assertTrue("6. the session started after releaseModel() keeps the model", engine.isModelLoaded)
        assertEquals(0, native.whisperFrees.get())
        assertEquals(1, native.whisperInits.get())
    }

    @Test
    fun aReleaseStillWaitingDoesNotApplyToACallStartedAfterIt() = runBlocking {
        val native = FakeNative()
        val engine = keeping(native) { endlessSilence() }
        val session = async(Dispatchers.Default) { engine.listen().toList() }
        awaitCondition("the session's model") { engine.isModelLoaded }
        engine.releaseModel()                        // it waits for the session
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))      // a call started after it
        engine.stop()
        session.awaitWithin(10_000, "the stopped session")
        engine.awaitWhisperThread()
        assertTrue("the call started after releaseModel() keeps the model", engine.isModelLoaded)
        assertEquals(0, native.whisperFrees.get())
    }

    @Test
    fun releaseModelDecidesUnderTheLockThatRegistersSessions() = runBlocking {
        val hold = Hold()
        val starter = AtomicReference<Thread?>()
        val native = FakeNative()
        val engine = keeping(native, object : SpeechEngine.Hooks {
            override fun releaseDeciding() = hold.here()
            override fun sessionStarting() = starter.set(Thread.currentThread())
        }) { ArrayAudioSource(utterance) }
        assertEquals(1, engine.transcriptCount())    // loaded and kept
        starter.set(null)
        hold.armed.set(true)
        val releasing = async(Dispatchers.Default) { engine.releaseModel() }
        val collected = try {
            assertTrue("releaseModel() holds the lock and has not decided", hold.reachedWithin())
            val started = async(Dispatchers.Default) { engine.transcriptCount() }
            // Within 5 s, before the hold's own 10 s give up: without the lock this is what fails.
            awaitCondition("the session's registration blocked on the lock", timeoutMs = 5_000) {
                starter.get()?.state == Thread.State.BLOCKED
            }
            started
        } finally {
            hold.proceed.countDown()
        }
        releasing.awaitWithin(10_000, "releaseModel()")
        assertEquals(1, collected.awaitWithin(10_000, "the session"))
        assertEquals("freed, then loaded by the session that waited for the lock", 2, native.whisperInits.get())
        assertTrue("and kept", engine.isModelLoaded)
        assertEquals(1, native.whisperFrees.get())
    }
}
