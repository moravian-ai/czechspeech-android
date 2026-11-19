package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CutSeam
import ai.moravian.czechspeech.StreamingConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamingConfigTest {
    @Test
    fun streamingIsOffByDefaultAndItsLagLimitIsFifteenSeconds() {
        assertNull(CzechSpeechConfig().streaming)
        assertEquals(15_000, StreamingConfig().maxLagMs)
        assertEquals(StreamingConfig(), CzechSpeechConfig(streaming = StreamingConfig()).streaming)
    }

    @Test
    fun theCutSeamDefaultsToContextAndHeldWordAndKeepsItsValueInACopy() {
        assertEquals(CutSeam.CONTEXT_AND_HELD_WORD, StreamingConfig().cutSeam)
        assertEquals(CutSeam.CONTEXT_AND_HELD_WORD, StreamingConfig(maxLagMs = 20_000).cutSeam)
        assertEquals(CutSeam.CONTEXT_AND_HELD_WORD, StreamingConfig().copy(maxLagMs = 20_000).cutSeam)
        assertEquals(CutSeam.CONTEXT, StreamingConfig(cutSeam = CutSeam.CONTEXT).copy(maxLagMs = 20_000).cutSeam)
        assertEquals(CutSeam.NONE, StreamingConfig(cutSeam = CutSeam.NONE).cutSeam)
        assertEquals(CutSeam.NONE, StreamingConfig(12_000, CutSeam.NONE).copy(maxLagMs = 20_000).cutSeam)
        assertEquals(setOf("NONE", "CONTEXT", "CONTEXT_AND_HELD_WORD"), CutSeam.values().map { it.name }.toSet())
    }

    @Test
    fun aLagLimitOutsideTenToThirtySecondsIsRefused() {
        for (bad in listOf(9_999, 30_001, 0, -1)) {
            val e = assertThrows(IllegalArgumentException::class.java) { StreamingConfig(maxLagMs = bad) }
            assertEquals("maxLagMs must be in 10000..30000", e.message)
        }
        StreamingConfig(maxLagMs = 10_000)
        StreamingConfig(maxLagMs = 30_000)
    }

    @Test
    fun theLargestFloorOfTheEncoderWindowIsNamedAs1488() {
        val e = assertThrows(IllegalArgumentException::class.java) { CzechSpeechConfig(minEncoderWindow = 1504) }
        assertTrue(e.message!!.contains("from 256 to 1488"))
        CzechSpeechConfig(minEncoderWindow = 1488)
    }
}
