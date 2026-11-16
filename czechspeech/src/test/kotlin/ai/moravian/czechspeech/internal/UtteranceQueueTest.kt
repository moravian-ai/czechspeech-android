package ai.moravian.czechspeech.internal

import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** The bound of the utterance queue, exactly, and the zeroing of an utterance lost to cancellation. */
class UtteranceQueueTest {
    private fun utterance(ms: Int) = Utterance(FloatArray(ms * SAMPLE_RATE / 1000) { 0.1f }, 0, ms.toLong())

    @Test
    fun anUtteranceHandedToAConsumerCancelledBeforeItRunsIsZeroed() = runBlocking {
        // The channel hands an utterance to a suspended consumer; if the consumer is cancelled before its dispatcher
        // runs it, the utterance is neither processed nor left in the queue for the drain, so only the channel's
        // onUndeliveredElement can zero it.
        val executor = Executors.newSingleThreadExecutor()
        val dispatcher = executor.asCoroutineDispatcher()
        try {
            val queue = UtteranceQueue(maxSamples = 10L * SAMPLE_RATE)
            val processed = mutableListOf<Utterance>()          // touched on the consumer's thread only
            val consumer = launch(dispatcher) { queue.forEach { processed += it } }
            // One task at a time: when this one runs, the consumer has suspended waiting for an utterance.
            val parked = CountDownLatch(1)
            val release = CountDownLatch(1)
            executor.execute {
                parked.countDown()
                release.await(10, TimeUnit.SECONDS)
            }
            assertTrue(parked.await(10, TimeUnit.SECONDS))
            val lost = utterance(500)
            assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(lost))   // handed over; the resumption waits
            consumer.cancel()
            release.countDown()
            consumer.joinWithin(5_000, "the cancelled consumer")
            assertTrue("never processed", processed.isEmpty())
            assertNull("not left for the drain either", queue.poll())
            assertTrue("an utterance lost to cancellation is zeroed", lost.samples.all { it == 0f })
        } finally {
            dispatcher.close()
        }
    }

    @Test
    fun anEmptyQueueTakesAnUtteranceLongerThanTheBound() {
        // Skipping speech while nothing waits would gain nothing; the bound only limits a backlog.
        val queue = UtteranceQueue(maxSamples = 1_000L * SAMPLE_RATE / 1000)
        assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(utterance(2_000)))
        assertEquals("with something waiting, the bound holds", UtteranceQueue.Offer.FULL, queue.offer(utterance(500)))
        queue.poll()
        assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(utterance(500)))
        assertEquals(2_000, queue.peakMs)
    }

    @Test
    fun anUtteranceThatFillsTheQueueExactlyFitsOneSampleMoreDoesNot() {
        val queue = UtteranceQueue(maxSamples = 3_000L * SAMPLE_RATE / 1000)
        assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(utterance(1_000)))
        assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(utterance(2_000)))
        assertEquals(3_000, queue.queuedMs)
        val oneSample = Utterance(FloatArray(1), 0, 0)
        assertEquals(UtteranceQueue.Offer.FULL, queue.offer(oneSample))
        assertEquals("a rejected utterance takes no room", 3_000, queue.queuedMs)
        assertEquals(3_000, queue.peakMs)
    }

    @Test
    fun takingAnUtteranceMakesRoomAndThePeakStays() = runBlocking {
        val queue = UtteranceQueue(maxSamples = 2_000L * SAMPLE_RATE / 1000)
        val first = utterance(1_500)
        queue.offer(first)
        assertEquals(UtteranceQueue.Offer.FULL, queue.offer(utterance(1_000)))
        assertSame(first, queue.poll())
        assertEquals(0, queue.queuedMs)
        assertEquals(UtteranceQueue.Offer.QUEUED, queue.offer(utterance(1_000)))
        queue.close()
        val taken = mutableListOf<Utterance>()
        queue.forEach { taken += it }
        assertEquals("the consumer gets what was queued, then the closed queue ends", 1, taken.size)
        assertEquals(0, queue.queuedMs)
        assertEquals(1_500, queue.peakMs)
    }

    @Test
    fun aClosedQueueRefusesAndTakesNoRoom() {
        val queue = UtteranceQueue(maxSamples = 2_000L * SAMPLE_RATE / 1000)
        queue.close()
        assertEquals(UtteranceQueue.Offer.CLOSED, queue.offer(utterance(500)))
        assertEquals(0, queue.queuedMs)
        assertEquals(0, queue.peakMs)
        assertNull(queue.poll())
    }
}
