package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * The encoder window whisper.cpp gets for an utterance (host build of the JNI library, no model). The model was
 * never trained on a window shorter than 512 frames (about 10 s); trimmed shorter, it invents and repeats
 * sentences on short utterances. So the window is trimmed to the utterance only above that floor.
 */
class NativeAudioCtxTest {
    @Before
    fun load() = TestEnv.assumeNativeLibrary()

    private fun ctx(seconds: Double) = NativeBridge.nativeAudioCtx((seconds * 16_000).toInt())

    @Test
    fun shortUtterancesGetTheFloor() {
        for (s in listOf(0.0, 0.5, 1.0, 2.0, 3.9, 4.0, 8.0, 9.5)) {
            assertEquals("$s s", 512, ctx(s))
        }
    }

    @Test
    fun longerUtterancesAreTrimmedToTheirLength() {
        assertEquals(544, ctx(10.0))   // 500 + 32 frames, up to a multiple of 16
        assertEquals(1008, ctx(19.5))
        assertEquals(1488, ctx(29.1))
    }

    @Test
    fun fullWindowFromThirtySecondsUp() {
        assertEquals(0, ctx(29.5))
        assertEquals(0, ctx(30.0))
        assertEquals(0, ctx(120.0))
    }
}
