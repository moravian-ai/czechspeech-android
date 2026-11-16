package ai.moravian.czechspeech.internal

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.sin

/**
 * Silero VAD through the JNI library (host build), without the speech model. Confirms what the pipeline relies
 * on: one probability per 512 samples, state carried between calls, reset between utterances.
 */
class NativeVadTest {
    private var vad = 0L

    @Before
    fun load() {
        TestEnv.assumeNativeLibrary()
        vad = JniNativeApi.vadInit(TestEnv.vadAsset.readBytes(), 1)
        assertTrue("VAD context", vad != 0L)
    }

    @After
    fun free() {
        if (vad != 0L) JniNativeApi.vadFree(vad)
    }

    @Test
    fun bundledAssetIsSileroV620() {
        val sha = MessageDigest.getInstance("SHA-256").digest(TestEnv.vadAsset.readBytes()).joinToString("") { "%02x".format(it) }
        assertEquals("2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987", sha)
    }

    @Test
    fun oneProbabilityPer512Samples() {
        val out = FloatArray(8)
        val noise = Signals.whiteNoise(1.0)
        JniNativeApi.vadReset(vad)
        assertEquals(1, JniNativeApi.vadProbs(vad, noise, 512, out))
        assertEquals(2, JniNativeApi.vadProbs(vad, noise, 513, out))      // last window zero-padded
        assertEquals(2, JniNativeApi.vadProbs(vad, noise, 1024, out))
        assertEquals(1, JniNativeApi.vadProbs(vad, noise, 1, out))
        assertTrue(out.take(2).all { it in 0f..1f })
    }

    @Test
    fun framewiseCallsEqualOneBatchCall() {
        val audio = speechOrSynthetic()
        val n = audio.size / FRAME_SAMPLES * FRAME_SAMPLES
        val frames = n / FRAME_SAMPLES

        JniNativeApi.vadReset(vad)
        val batch = FloatArray(frames)
        assertEquals(frames, JniNativeApi.vadProbs(vad, audio, n, batch))

        JniNativeApi.vadReset(vad)
        val streamed = FloatArray(frames)
        val frame = FloatArray(FRAME_SAMPLES)
        val one = FloatArray(1)
        val t0 = System.nanoTime()
        for (i in 0 until frames) {
            System.arraycopy(audio, i * FRAME_SAMPLES, frame, 0, FRAME_SAMPLES)
            assertEquals(1, JniNativeApi.vadProbs(vad, frame, FRAME_SAMPLES, one))
            streamed[i] = one[0]
        }
        val us = (System.nanoTime() - t0) / 1000 / frames
        println("VAD: $frames frames, $us us per 32 ms frame on this host")
        assertArrayEquals("only the LSTM state carries over between calls", batch, streamed, 1e-5f)
        println("VAD probabilities (every 4th frame): " + batch.filterIndexed { i, _ -> i % 4 == 0 }.joinToString(" ") { "%.2f".format(it) })
    }

    @Test
    fun resetRestoresTheInitialState() {
        val audio = speechOrSynthetic()
        val n = 40 * FRAME_SAMPLES
        val fresh = FloatArray(40)
        JniNativeApi.vadReset(vad)
        JniNativeApi.vadProbs(vad, audio, n, fresh)
        val again = FloatArray(40)
        JniNativeApi.vadProbs(vad, audio, n, FloatArray(40))   // advance the state
        JniNativeApi.vadReset(vad)
        JniNativeApi.vadProbs(vad, audio, n, again)
        assertArrayEquals(fresh, again, 1e-6f)
    }

    @Test
    fun gateFindsNoSpeechInSyntheticNonSpeech() {
        val hum = FloatArray(3 * SAMPLE_RATE) { (0.2 * sin(2 * PI * 50.0 * it / SAMPLE_RATE)).toFloat() }
        val clicks = FloatArray(3 * SAMPLE_RATE).also { a -> for (i in a.indices step 4_000) a[i] = 0.9f }
        val cases = mapOf(
            "digital zeros 10 s" to Signals.zeros(10.0),
            "white noise" to Signals.whiteNoise(3.0, 0.2f),
            "50 Hz hum" to hum,
            "clicks" to clicks,
        )
        for ((name, samples) in cases) {
            val segments = JniNativeApi.vadSpeechSegments(vad, samples, samples.size, 0.25f)
            println("gate: $name -> $segments speech segment(s)")
            assertEquals(name, 0, segments)
        }
    }

    @Test
    fun gateFindsSpeechInTheParliamentRecording() {
        val speech = TestEnv.audio("e-rec.wav")
        val segments = JniNativeApi.vadSpeechSegments(vad, speech, speech.size, 0.25f)
        println("gate: e-rec.wav -> $segments speech segment(s)")
        assertTrue(segments >= 1)
    }

    private fun speechOrSynthetic(): FloatArray {
        val file = File(TestEnv.audioDir, "e-rec.wav")
        return if (file.exists()) Wav.read(file) else Signals.concat(Signals.whiteNoise(1.0, 0.05f), Signals.tone(2.0), Signals.zeros(1.0))
    }
}
