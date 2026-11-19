package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test

/**
 * The encoder window whisper.cpp gets for an utterance (host build of the JNI library, no model). A model invents and
 * repeats sentences on short utterances when its window is trimmed below what it bears, so the window is trimmed to
 * the utterance only above a floor, which belongs to the model: 512 frames (about 10 s) by default, as before the
 * floor could be set.
 */
class NativeAudioCtxTest {
    @Before
    fun load() = TestEnv.assumeNativeLibrary()

    private fun ctx(seconds: Double, floor: Int = 512) = NativeBridge.nativeAudioCtx((seconds * 16_000).toInt(), floor)

    /** The window as the library computed it while the floor was the constant 512 (czechspeech_jni.c, cs_audio_ctx). */
    private fun before(samples: Int): Int {
        var frames = (samples + 319) / 320 + 32
        frames = (frames + 15) / 16 * 16
        if (frames < 512) frames = 512
        return if (frames < 1500) frames else 0
    }

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

    @Test
    fun theDefaultFloorGivesExactlyTheWindowsOfBefore() {
        // every 32 ms frame of audio up to 31 s, and the samples around each step of the window
        var samples = 0
        while (samples <= 31 * 16_000) {
            assertEquals("$samples samples", before(samples), NativeBridge.nativeAudioCtx(samples, 512))
            samples += 512
        }
        for (frames in 480..1500 step 16) {
            for (d in -2..2) {
                val n = (frames - 32) * 320 + d
                assertEquals("$n samples", before(n), NativeBridge.nativeAudioCtx(n, 512))
            }
        }
    }

    @Test
    fun aLowerFloorTrimsShortUtterancesFurtherAndChangesNothingAboveTheOldOne() {
        for (s in listOf(0.0, 1.0, 4.0, 7.0)) assertEquals("$s s", 384, ctx(s, 384))
        assertEquals("7.04 s is 352 frames + 32", 384, NativeBridge.nativeAudioCtx(352 * 320, 384))
        assertEquals("one sample more: the next multiple of 16", 400, NativeBridge.nativeAudioCtx(352 * 320 + 1, 384))
        assertEquals(432, ctx(8.0, 384))
        assertEquals(512, ctx(9.5, 384))
        for (s in listOf(10.0, 19.5, 29.1, 29.5, 30.0)) assertEquals("$s s", ctx(s), ctx(s, 384))
        assertEquals(256, ctx(1.0, 256))
        assertEquals("a floor of the whole window", 0, ctx(1.0, 1500))
        assertEquals(1488, ctx(1.0, 1488))
    }

    @Test
    fun theKotlinFormulaForTheLogIsTheLibrarys() {
        for (floor in listOf(256, 384, 512, 768, 1488, 1500)) {
            var samples = 0
            while (samples <= 31 * 16_000) {
                assertEquals("$samples samples, floor $floor", NativeBridge.nativeAudioCtx(samples, floor), EncoderWindow.audioCtx(samples, floor))
                samples += 1_117
            }
        }
    }

    @Test
    fun aFloorOutOfRangeIsRefusedBeforeTheContextIsLookedAt() {
        for (floor in listOf(0, 255, 240, 392, 1501, 1504, -512)) {
            val e = failure<IllegalArgumentException> { NativeBridge.nativeWhisperSetMinAudioCtx(0, floor) }
            assertEquals("floor $floor", "minimum encoder window out of range", e.message)
        }
        val e = failure<IllegalStateException> { NativeBridge.nativeWhisperSetMinAudioCtx(0, 384) }
        assertEquals("whisper context not loaded", e.message)
    }
}
