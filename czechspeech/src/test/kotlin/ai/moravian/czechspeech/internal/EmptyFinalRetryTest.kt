package ai.moravian.czechspeech.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Which encoder window a final recognition that came out empty is repeated with: pure rules. */
class EmptyFinalRetryTest {
    /** 2.9 s of audio: on the lowest window of a floor up to 384 (about 150 frames of audio and 32). */
    private val shortAudio = 46_400

    private fun next(
        first: Int,
        last: Int = first,
        retriesDone: Int = 0,
        pending: Int = 1,
        othersQueued: Boolean = false,
        audio: Int = shortAudio,
    ) = EmptyFinalRetry.nextFloor(first, last, audio, retriesDone, pending, othersQueued)

    @Test fun belowTheFloorOf512TheFirstRetryIsAt512AndThenTheFullWindow() {
        assertEquals(512, next(first = 384))
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 384, last = 512, retriesDone = 1))
        assertEquals(512, next(first = 256))
    }

    @Test fun fromTheFloorOf512OrMoreOnlyTheFullWindowFollows() {
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 512))
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 640, audio = 200_000))
    }

    @Test fun thereIsNoRetryAtTheFullWindowOrAfterTwoRetries() {
        assertNull(next(first = 1500, audio = 480_000))
        assertNull("the highest floor the library takes", next(first = 384, last = EmptyFinalRetry.FULL_WINDOW_FLOOR, retriesDone = 1))
        assertNull(next(first = 384, retriesDone = EmptyFinalRetry.MAX_RETRIES))
    }

    @Test fun twoPendingFinalsOrAWaitingQueueKeepTheFullWindowRetryAwayButNotThe512One() {
        assertEquals(512, next(first = 384, pending = 2))
        assertEquals(512, next(first = 384, othersQueued = true))
        assertNull(next(first = 384, last = 512, retriesDone = 1, pending = 2))
        assertNull(next(first = 384, last = 512, retriesDone = 1, pending = 3))
        assertNull(next(first = 384, last = 512, retriesDone = 1, othersQueued = true))
        assertNull(next(first = 512, othersQueued = true))
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 384, last = 512, retriesDone = 1, pending = 1))
    }

    @Test fun aRetryWhoseWindowGrowsByLessThanAQuarterIsSkipped() {
        assertEquals(1.25, EmptyFinalRetry.MIN_WINDOW_GROWTH, 0.0)
        // a first run on 1440 frames: the full window (1488) is 3 % more, pointless
        assertNull(next(first = 1440, audio = 440_000))
        // 1200 frames: 1488 is 24 % more, skipped; 1184 frames: 25.7 %, allowed
        assertNull(next(first = 1200, audio = 360_000))
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 1184, audio = 355_000))
        // 448 frames (an audio of 8.5 s under a floor of 384): 512 is 14 % more, skipped, the full window is tried at once
        assertEquals(EmptyFinalRetry.FULL_WINDOW_FLOOR, next(first = 448, audio = 133_000))
        // 384 -> 512 is 33 %: allowed
        assertEquals(512, next(first = 384))
    }

    @Test fun clearSpeechIsOneSecondOfFramesAboveTheThreshold() {
        assertEquals(1_000L, EmptyFinalRetry.CLEAR_SPEECH_MS)
    }
}
