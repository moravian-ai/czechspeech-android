package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule

/** The fake refuses the thresholds the real library refuses (`cs_thresholds`), so an engine bug fails in the tests. */
class FakeNativeThresholdsTest {
    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val nan = Float.NaN
    private val audio = FloatArray(16_000)

    private fun calls(native: FakeNative, handle: Long, languages: String, thresholds: FloatArray?): List<() -> Any?> = listOf(
        { native.whisperTranscribe(handle, 1, audio, audio.size, 1, languages, thresholds) },
        { native.whisperPreview(handle, 1, audio, audio.size, 1, languages, thresholds) },
        { native.whisperTranscribeNBest(handle, 1, audio, audio.size, 1, 3, languages, thresholds) },
    )

    private fun refused(languages: String, thresholds: FloatArray?) {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        for (call in calls(native, handle, languages, thresholds)) {
            try {
                call()
                throw AssertionError("$languages ${thresholds?.toList()} was accepted")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!.isNotEmpty())
            }
        }
        assertTrue("a refused call is not a recognition", native.languagesPassed.isEmpty())
    }

    @Test
    fun badThresholdsAreRefusedOnEveryRecognitionPath() {
        refused("cs", floatArrayOf(nan))                            // a single language takes no thresholds
        refused("en", floatArrayOf(.9f))                            // a forced language neither
        refused("cs,sk,en", floatArrayOf(nan, .9f))                 // wrong length
        refused("cs,sk,en", floatArrayOf(nan, nan, 1.5f))
        refused("cs,sk,en", floatArrayOf(nan, nan, -.1f))
        refused("cs,sk,en", floatArrayOf(nan, nan, Float.POSITIVE_INFINITY))
        refused("cs,sk,en", floatArrayOf(Float.NEGATIVE_INFINITY, nan, nan))
    }

    @Test
    fun validThresholdsAreAccepted() {
        val native = FakeNative()
        val handle = native.whisperInit("model")
        for (thresholds in listOf(null, floatArrayOf(nan, nan, nan), floatArrayOf(nan, nan, 0f), floatArrayOf(nan, nan, 1f))) {
            for (call in calls(native, handle, "cs,sk,en", thresholds)) assertNotNull(call())
        }
        assertEquals(12, native.languagesPassed.size)
    }
}
