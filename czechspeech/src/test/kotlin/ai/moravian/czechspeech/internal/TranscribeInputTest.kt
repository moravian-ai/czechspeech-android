package ai.moravian.czechspeech.internal

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule

/** The hint transcribe() gives for samples out of range: unscaled 16-bit PCM, or float audio that clips. */
class TranscribeInputTest {
    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val native = FakeNative()
    private val engine = fakeEngine(native) { error("no microphone in this test") }

    private fun rejected(value: Float): String = runBlocking {
        val samples = Signals.tone(0.5).also { it[100] = value }
        failure<IllegalArgumentException> { engine.transcribe(samples) }.message!!
    }

    @Test
    fun aSampleOf256OrMoreGetsTheHintToScale16BitPcm() {
        for (value in listOf(256f, -256f, 300f, 32_767f, -32_768f)) {
            val message = rejected(value)
            assertTrue(message, message.contains("samples[100] is $value") && message.contains("divide by 32768"))
        }
        assertEquals("rejected before the speech gate", 0, native.gateCalls.get())
    }

    @Test
    fun aSampleOver2AndUnder256GetsTheHintToClipOrNormalise() {
        // Float audio that overshoots (gain, mixing, a resampler) is not 16-bit PCM; dividing by 32768 would be wrong.
        for (value in listOf(Math.nextUp(2f), 3.5f, -17f, Math.nextDown(256f))) {
            val message = rejected(value)
            assertTrue(message, message.contains("samples[100] is $value") && message.contains("clip or normalise"))
            assertFalse(message, message.contains("32768"))
        }
        assertEquals("rejected before the speech gate", 0, native.gateCalls.get())
    }
}
