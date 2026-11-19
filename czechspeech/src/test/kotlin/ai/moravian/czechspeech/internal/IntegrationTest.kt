package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test

/**
 * The whole pipeline on the JVM: host JNI library, real Silero VAD, real CzechSpeech model, recordings fed through
 * [ArrayAudioSource] in place of the microphone. Skips when the library, the model or the audio is missing.
 *
 * Env: CZECHSPEECH_MODEL, CZECHSPEECH_TEST_AUDIO, CZECHSPEECH_TEST_MANIFEST, CZECHSPEECH_TEST_THREADS.
 *
 * The test prints transcripts of the (public) parliament recording to check them by eye; the SDK itself never
 * logs text.
 */
class IntegrationTest {
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

    private fun engine(audio: FloatArray, languages: List<String> = listOf("cs")) = SpeechEngine(
        modelFile = TestEnv.model,
        config = CzechSpeechConfig(),
        languages = languages,
        native = JniNativeApi,
        vadModel = { TestEnv.vadAsset.readBytes() },
        audioSourceFactory = { ArrayAudioSource(audio) },
        threads = { TestEnv.threads },
    )

    private fun listen(name: String, audio: FloatArray, languages: List<String> = listOf("cs")): List<SpeechEvent> = runBlocking {
        val engine = engine(audio, languages)
        val t0 = System.nanoTime()
        val events = engine.listen().toList()
        val ms = (System.nanoTime() - t0) / 1_000_000
        println("listen($name, ${audio.size * 1000L / SAMPLE_RATE} ms of audio) took $ms ms:")
        for (e in events) {
            when (e) {
                is SpeechEvent.Transcript -> println("  Transcript ${e.transcript.startMs}-${e.transcript.endMs} ms, " +
                    "${e.transcript.processingMs} ms: \"${e.transcript.text}\"")
                else -> println("  $e")
            }
        }
        assertFalse("model freed after the session", engine.isModelLoaded)
        events
    }

    private fun transcripts(events: List<SpeechEvent>) =
        events.filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }

    @Test
    fun nonSpeechRecordingsGiveNoTranscript() {
        val cases = listOf("a-silence-5s.wav", "b-pink-noise-2s.wav", "c-hum-2s.wav", "d-clicks-6s.wav")
            .map { it to TestEnv.audio(it) } + ("digital zeros 10 s" to Signals.zeros(10.0))
        for ((name, audio) in cases) {
            val events = listen(name, audio)
            assertTrue("$name: no transcript", transcripts(events).isEmpty())
            assertTrue("$name: no error", events.none { it is SpeechEvent.Error })
            val engine = engine(audio)
            val text = runBlocking { engine.transcribe(audio) }
            assertEquals("$name: transcribe() gives \"\"", "", text)
            assertEquals("$name: the gate keeps the model unloaded", 0, engine.modelLoads)
        }
    }

    @Test
    fun parliamentSpeechIsTranscribed() {
        val reference = TestEnv.reference(SPEECH_ID)
        val audio = TestEnv.audio("e-rec.wav")
        val copy = audio.copyOf()
        val engine = engine(audio)
        val t0 = System.nanoTime()
        val text = runBlocking { engine.transcribe(audio) }
        println("transcribe(e-rec.wav) in ${(System.nanoTime() - t0) / 1_000_000} ms: \"$text\"")
        println("reference: \"$reference\"")
        val recall = wordRecall(reference, text)
        println("word recall %.2f".format(recall))
        assertTrue("word recall $recall", recall >= 0.8)
        assertArrayEquals("the caller's samples are not modified", copy, audio, 0f)
        assertFalse(engine.isModelLoaded)
        assertEquals(1, engine.modelLoads)
    }

    @Test
    fun sessionsAreDeterministic() {
        // Regression: the streaming VAD state must start from zero in every session, not from leftover heap memory.
        val speech = TestEnv.audio("e-rec.wav")
        val audio = Signals.concat(Signals.whiteNoise(0.5, 0.01f), speech, Signals.zeros(2.0), speech)
        fun timeline(events: List<SpeechEvent>) = events.mapNotNull {
            when (it) {
                is SpeechEvent.SpeechStarted -> "S${it.atMs}"
                is SpeechEvent.SpeechEnded -> "E${it.atMs}"
                is SpeechEvent.Transcript -> "T${it.transcript.startMs}-${it.transcript.endMs}"
                is SpeechEvent.Error -> "error"
                is SpeechEvent.UtteranceDone -> "D${it.startMs}"
                is SpeechEvent.UtteranceNotWritten -> "N${it.startMs}"
            }
        }
        val first = timeline(listen("determinism run 1", audio))
        repeat(2) { run ->
            // Leave non-zero LSTM state behind in freed heap memory, on a short-lived thread like the capture
            // threads, so that the next session's VAD context is likely to get that memory.
            val dirtier = Thread {
                repeat(3) {
                    val dirty = JniNativeApi.vadInit(TestEnv.vadAsset.readBytes(), 1)
                    JniNativeApi.vadProbs(dirty, speech, speech.size, FloatArray(1))
                    JniNativeApi.vadFree(dirty)
                }
            }
            dirtier.start()
            dirtier.join()
            assertEquals("run ${run + 2}", first, timeline(listen("determinism run ${run + 2}", audio)))
        }
        assertEquals(2, first.count { it.startsWith("T") })
    }

    @Test
    fun cancellingATranscriptionAbortsWhisper() {
        val speech = TestEnv.audio("e-rec.wav")
        val long = Signals.concat(speech, speech, speech, speech)      // ~28 s, several seconds of work
        val engine = engine(long)
        runBlocking {
            var t0 = System.nanoTime()
            val job = launch(Dispatchers.Default) { engine.transcribe(long) }
            withTimeout(60_000) { while (!engine.isModelLoaded) delay(5) }
            delay(300)                                                // whisper is encoding now
            val loadedAfter = (System.nanoTime() - t0) / 1_000_000
            t0 = System.nanoTime()
            job.cancel()
            withTimeout(30_000) { while (engine.isModelLoaded) delay(5) }
            val ms = (System.nanoTime() - t0) / 1_000_000
            println("cancelled ${loadedAfter} ms into a 28 s transcription; model freed $ms ms after cancel")
            assertEquals("whisper stopped by the abort callback", 1, engine.abortedRecognitions)

            t0 = System.nanoTime()
            val text = engine.transcribe(long)
            val fullMs = (System.nanoTime() - t0) / 1_000_000
            println("the same transcription uncancelled: $fullMs ms, ${text.length} chars")
            assertTrue("cancelled run ($ms ms) shorter than a full one ($fullMs ms)", ms + loadedAfter < fullMs)
        }
        // the engine still works afterwards
        assertTrue(wordRecall(TestEnv.reference(SPEECH_ID), runBlocking { engine.transcribe(speech) }) >= 0.8)
    }

    @Test
    fun speechFollowedBySilenceGivesExactlyOneTranscript() {
        val reference = TestEnv.reference(SPEECH_ID)
        val events = listen("f-rec-plus-silence15s.wav", TestEnv.audio("f-rec-plus-silence15s.wav"))
        val t = transcripts(events)
        assertEquals(1, t.size)
        assertEquals(1, events.count { it is SpeechEvent.SpeechStarted })
        assertEquals(1, events.count { it is SpeechEvent.SpeechEnded })
        assertTrue(wordRecall(reference, t[0].text) >= 0.8)
        assertTrue(t[0].startMs < t[0].endMs && t[0].endMs <= 7_200)
    }

    @Test
    fun twoCopiesTwoSecondsApartGiveTwoTranscripts() {
        val reference = TestEnv.reference(SPEECH_ID)
        val speech = TestEnv.audio("e-rec.wav")
        val audio = Signals.concat(speech, Signals.zeros(2.0), speech)
        val t = transcripts(listen("e-rec + 2 s zeros + e-rec", audio))
        assertEquals(2, t.size)
        for (x in t) assertTrue(wordRecall(reference, x.text) >= 0.8)
        assertTrue(t[1].startMs >= t[0].endMs + 1_500)
    }

    @Test
    fun forcedCzechReportsCsAndTheSameTextAsAListWithEnglish() {
        val speech = TestEnv.audio("e-rec.wav")
        val forced = transcripts(listen("forced cs", speech))
        val detected = transcripts(listen("cs,en", speech, listOf("cs", "en")))
        assertEquals(1, forced.size)
        assertEquals("cs", forced[0].language)
        assertTrue(wordRecall(TestEnv.reference(SPEECH_ID), forced[0].text) >= 0.8)
        assertEquals(1, detected.size)
        assertEquals("cs", detected[0].language)
        assertEquals("the same language gives the same text", forced[0].text, detected[0].text)
        println("processing: forced cs ${forced[0].processingMs} ms, cs,en with detection ${detected[0].processingMs} ms")
    }

    @Test
    fun aCallCancelledBeforeItStartsReturnsCancelledAndTheNextOneDecodes() {
        val speech = TestEnv.audio("e-rec.wav")
        val api = JniNativeApi
        val handle = api.whisperInit(TestEnv.model.path)
        try {
            api.whisperAbort(handle, 7)
            val cancelled = api.whisperTranscribe(handle, 7, speech, speech.size, TestEnv.threads, "cs,en", null)
            assertEquals("cancelled", null, cancelled)
            assertTrue("no language", api.whisperLanguage(handle).isNullOrEmpty())

            val next = api.whisperTranscribe(handle, 8, speech, speech.size, TestEnv.threads, "cs,en", null)
            assertTrue("decodes normally", next != null && next.isNotEmpty())
            assertEquals("cs", api.whisperLanguage(handle))
            assertTrue(wordRecall(TestEnv.reference(SPEECH_ID), String(next!!, Charsets.UTF_8)) >= 0.8)

            // a forced language has no detection: it is reported as given
            api.whisperTranscribe(handle, 9, speech, speech.size, TestEnv.threads, "cs", null)
            assertEquals("cs", api.whisperLanguage(handle))
        } finally {
            api.whisperFree(handle)
        }
    }

    @Test
    fun malformedLanguageListsAreRejectedBeforeAnyWork() {
        val api = JniNativeApi
        val handle = api.whisperInit(TestEnv.model.path)
        try {
            val audio = FloatArray(16_000)
            for (bad in listOf("", ",", "cs,", ",cs", "cs,,en", "xx", "cs,xx", "csxxxxxxxx", " cs")) {
                val e = runCatching { api.whisperTranscribe(handle, 1, audio, audio.size, 1, bad, null) }.exceptionOrNull()
                assertTrue("\"$bad\" rejected: $e", e is IllegalArgumentException)
            }
            assertTrue(api.whisperLanguage(handle).isNullOrEmpty())
        } finally {
            api.whisperFree(handle)
        }
    }
}
