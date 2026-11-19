package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.VariantSelection
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test

/**
 * Variant selection with the real CzechSpeech model on the JVM (host JNI library). Skips without the model and the
 * audio (CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO); the language model tests also need CZECHSPEECH_LM. Prints only
 * counts and times, never text.
 */
class VariantSelectionIntegrationTest {
    companion object {
        private const val SPEECH_ID = "test_0000-0000000333"

        @BeforeClass
        @JvmStatic
        fun requirements() {
            TestEnv.assumeNativeLibrary()
            TestEnv.assumeFile(TestEnv.model, "CzechSpeech model (CZECHSPEECH_MODEL)")
            TestEnv.assumeFile(TestEnv.audioDir, "test audio directory (CZECHSPEECH_TEST_AUDIO)")
        }
    }

    @Test
    fun beamSearchProposesDifferentVariants() {
        // whisper.cpp 1.9.5 drew the beams' candidates at random and started every beam with the same token: on a
        // confident model all beams came out the same. Patched, the beams take the most likely tokens.
        val audio = TestEnv.audio("e-rec.wav")
        val handle = JniNativeApi.whisperInit(TestEnv.model.absolutePath)
        assertNotEquals(0L, handle)
        try {
            val bytes = JniNativeApi.whisperTranscribeNBest(handle, 1, audio, audio.size, TestEnv.threads, 5, "cs", null)!!
            val windows = try {
                NBest.parse(bytes)
            } finally {
                bytes.fill(0)
            }
            assertEquals(1, windows.size)
            val hyps = windows[0]
            val distinct = hyps.map { it.text.trim() }.distinct().size
            val firstWords = hyps.map { it.text.trim().substringBefore(' ') }.distinct().size
            println("beam 5: ${hyps.size} hypotheses, $distinct different texts, $firstWords different first words")
            assertEquals(5, hyps.size)
            assertEquals(1, hyps.count { it.best })
            assertTrue("$distinct different texts", distinct >= 3)
            assertTrue(hyps.all { it.sumLogprob < 0 && it.length > 0 })
        } finally {
            JniNativeApi.whisperFree(handle)
        }
    }

    private fun engine(config: CzechSpeechConfig) = SpeechEngine(
        modelFile = TestEnv.model,
        config = config,
        native = JniNativeApi,
        vadModel = { TestEnv.vadAsset.readBytes() },
        audioSourceFactory = { ArrayAudioSource(FloatArray(0)) },
        threads = { TestEnv.threads },
        keepModelLoaded = true,
    )

    /** Median ms of [rounds] transcriptions after one to warm up, and the last text. */
    private fun time(engine: SpeechEngine, audio: FloatArray, rounds: Int = 3): Pair<Long, String> = runBlocking {
        engine.transcribe(audio)
        val ms = (1..rounds).map {
            val t0 = System.nanoTime()
            engine.transcribe(audio)
            (System.nanoTime() - t0) / 1_000_000
        }.sorted()
        ms[ms.size / 2] to engine.transcribe(audio)
    }

    @Test
    fun selectedTranscriptIsRightAndTimed() {
        TestEnv.assumeFile(TestEnv.languageModel, "Czech GPT-2 language model (CZECHSPEECH_LM)")
        val reference = TestEnv.reference(SPEECH_ID)
        val audio = TestEnv.audio("e-rec.wav")
        val lm = TestEnv.languageModel
        val modes = listOf(
            "greedy" to CzechSpeechConfig(),
            "beam 5" to CzechSpeechConfig(variantSelection = VariantSelection(lm, languageModelWeight = 0f)),
            "beam 5 + LM" to CzechSpeechConfig(variantSelection = VariantSelection(lm)),
            "beam 3 + LM" to CzechSpeechConfig(variantSelection = VariantSelection(lm, beamSize = 3, lengthBonus = 0.75f)),
        )
        for ((name, config) in modes) {
            val engine = engine(config)
            try {
                val (ms, text) = time(engine, audio)
                val recall = wordRecall(reference, text)
                println("%-12s %5d ms per %.1f s utterance, word recall %.2f".format(name, ms, audio.size / 16_000.0, recall))
                assertTrue("$name: word recall $recall", recall >= 0.8)
            } finally {
                engine.close()
                runBlocking { engine.awaitClosed() }
            }
        }
    }
}
