package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.Dispatchers
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

/**
 * The floor of the encoder window ([CzechSpeechConfig.minEncoderWindow]) belongs to the model: the configuration
 * checks it, the engine sets it on the model's context when it loads, so the final text, a preview and an early
 * recognition all get it, and the log says which floor and which window were used.
 */
class EncoderWindowTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    @Test
    fun theDefaultIs512FramesAsBefore() {
        assertEquals(512, CzechSpeechConfig().minEncoderWindow)
        assertEquals(512, EncoderWindow.DEFAULT_MIN_FRAMES)
    }

    @Test
    fun theWindowIsTheUtteranceInFramesPlusAMarginRoundedUpAndNeverUnderTheFloor() {
        assertEquals(512, EncoderWindow.audioCtx(0, 512))
        assertEquals(512, EncoderWindow.audioCtx(4 * 16_000, 512))
        assertEquals(544, EncoderWindow.audioCtx(10 * 16_000, 512))
        assertEquals(384, EncoderWindow.audioCtx(4 * 16_000, 384))
        assertEquals(432, EncoderWindow.audioCtx(8 * 16_000, 384))
        assertEquals("the whole window", 0, EncoderWindow.audioCtx(30 * 16_000, 384))
        assertEquals("in the log, the whole window is 1500 frames", 1500, EncoderWindow.frames(30 * 16_000, 384))
        assertEquals(384, EncoderWindow.frames(4 * 16_000, 384))
    }

    @Test
    fun onlyAMultipleOf16From256To1500IsAFloor() {
        for (frames in listOf(256, 272, 384, 512, 1024, 1488)) CzechSpeechConfig(minEncoderWindow = frames)
        for (frames in listOf(0, 16, 240, 255, 257, 392, 1500, 1504, 3000, -384)) {
            val e = assertThrows("$frames", IllegalArgumentException::class.java) { CzechSpeechConfig(minEncoderWindow = frames) }
            assertTrue(e.message, e.message!!.contains("minEncoderWindow"))
        }
    }

    private fun speech(seconds: Double) = Signals.concat(Signals.zeros(0.5), Signals.tone(seconds), Signals.zeros(1.5))

    private fun withLogs(block: suspend () -> Unit): List<String> {
        val logs = Collections.synchronizedList(mutableListOf<String>())
        val sink = CsLog.sink
        CsLog.sink = { _, message -> logs += message }
        try {
            runBlocking { block() }
        } finally {
            CsLog.sink = sink
        }
        return synchronized(logs) { logs.toList() }
    }

    @Test
    fun theEngineSetsTheFloorOnTheModelWhenItLoadsAndTheLogHasTheFloorAndEachWindow() {
        for (floor in listOf(512, 384)) {
            val native = FakeNative()
            val config = if (floor == 512) CzechSpeechConfig() else CzechSpeechConfig(minEncoderWindow = floor)
            val engine = fakeEngine(native, config = config) { ArrayAudioSource(speech(1.0)) }
            val logs = withLogs { withContext(Dispatchers.Default) { engine.listen().collect { } } }
            assertEquals("set once, on the context it loaded", listOf(floor), native.minAudioCtxSet.toList())
            assertTrue(
                logs.joinToString("\n"),
                logs.any { it.startsWith("model loaded in ") && it.endsWith(", encoder window at least $floor frames)") },
            )
            // 1952 ms of audio: 98 frames + 32, far under either floor
            val line = logs.single { it.startsWith("utterance: ") }
            assertTrue(line, ", recogniser " in line && ", window $floor frames, " in line)
        }
    }

    @Test
    fun aPreviewAndAnEarlyRecognitionRunOnTheSameContextSoWithTheSameFloor() {
        // 12 s of speech: its previews are longer than the floor and are trimmed to their own length
        val native = FakeNative()
        val config = CzechSpeechConfig(minEncoderWindow = 384, earlyStart = ai.moravian.czechspeech.EarlyStart())
        val engine = fakeEngine(native, config = config) { ArrayAudioSource(speech(12.0)) }
        val logs = withLogs { withContext(Dispatchers.Default) { engine.listenWithPreviews(PreviewConfig()).collect { } } }
        assertEquals("one context, one floor, for every kind of recognition", listOf(384), native.minAudioCtxSet.toList())
        assertEquals(1, native.whisperInits.get())
        assertTrue("previews ran", native.previewCalls.get() >= 1)
        val windows = logs.filter { it.startsWith("preview: ") || it.startsWith("utterance: ") }
            .map { Regex("""window (\d+) frames""").find(it)?.groupValues?.get(1)?.toInt() }
        assertTrue("every recognition's line has its window: $windows", windows.isNotEmpty() && windows.none { it == null })
        // a preview of 5 s is under the floor and gets it; the utterance of 12 s is trimmed to its own length
        assertTrue("none under the floor: $windows", windows.all { it!! >= 384 })
        assertTrue("the floor itself and longer ones: $windows", 384 in windows && windows.any { it!! > 512 })
    }

    @Test
    fun transcribeGetsTheFloorToo() = runBlocking {
        val native = FakeNative()
        val engine = fakeEngine(native, config = CzechSpeechConfig(minEncoderWindow = 384)) { ArrayAudioSource(FloatArray(0)) }
        assertEquals("dobrý den", engine.transcribe(Signals.tone(1.0)))
        assertEquals(listOf(384), native.minAudioCtxSet.toList())
    }

    @Test
    fun aSessionsEventsDoNotDependOnTheFloorInTheFakeLayer() = runBlocking {
        fun events(config: CzechSpeechConfig) = runBlocking {
            val collected = mutableListOf<SpeechEvent>()
            withContext(Dispatchers.Default) {
                fakeEngine(FakeNative(), config = config) { ArrayAudioSource(speech(1.0)) }.listen().collect { collected += it }
            }
            collected.map { if (it is SpeechEvent.Transcript) SpeechEvent.Transcript(it.transcript.copy(processingMs = 0)) else it }
        }
        assertEquals(events(CzechSpeechConfig()), events(CzechSpeechConfig(minEncoderWindow = 384)))
    }
}
