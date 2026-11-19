package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.EarlyStart
import ai.moravian.czechspeech.SpeechEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TestRule
import org.junit.rules.Timeout
import java.util.Collections

/**
 * The rescue after an empty final result ([LanguageRule.rescue]): one run forced in another language of the list, on the
 * same window, before the retries on a longer window, which then run in the language of the last attempt. As in
 * [LanguageSessionTest], lengths are the audio the recogniser gets, and the fake's shares are given by the index of the
 * call (a forced call has an index too, its entry is not read).
 */
class LanguageRescueTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(60)

    @get:Rule
    val freedOnOwnThreads: TestRule = FreedOnOwnThreads()

    private val logs = Collections.synchronizedList(mutableListOf<String>())
    private val sink = CsLog.sink

    @Before
    fun captureLogs() {
        CsLog.sink = { _, message -> logs += message }
    }

    @After
    fun restoreLogs() {
        CsLog.sink = sink
    }

    private val nan = Float.NaN
    private val three = listOf("cs", "sk", "en")
    private val switch = CzechSpeechModel.LanguageSwitch(mapOf("en" to 0.9f), minLeaveMs = 2_000, minReturnMs = 1_000)

    /** No pre-roll and 100 ms of silence: an utterance's audio is its speech and about 130 ms. */
    private val noRetry = CzechSpeechConfig(minEncoderWindow = 384, preRollMs = 0, minSilenceMs = 100, retryEmptyFinals = false)
    private val retry = noRetry.copy(retryEmptyFinals = true)

    private val czech = floatArrayOf(0.9999f, 0.0001f, 0f)
    private val english = floatArrayOf(0.002f, 0.001f, 0.997f)
    private val silence = floatArrayOf(0.70f, 0f, 0.30f)
    private val slovak = floatArrayOf(0.2f, 0.8f, 0f)
    private val unused = floatArrayOf(1f, 0f, 0f)

    private fun utterance(audioMs: Int) = Signals.tone((audioMs - 130) / 1000.0)

    private fun audio(vararg audioMs: Int): FloatArray = Signals.concat(
        Signals.zeros(0.5),
        *audioMs.map { Signals.concat(utterance(it), Signals.zeros(0.6)) }.toTypedArray(),
    )

    private fun FakeNative.sharesInOrder(vararg shares: FloatArray) {
        sharesOf = { index, _, _ -> shares[index] }
    }

    private fun th(vararg values: Float) = values

    private fun session(native: FakeNative, audio: FloatArray, config: CzechSpeechConfig): List<SpeechEvent> = runBlocking {
        val engine = fakeEngine(native, config = config, languages = three, languageSwitch = switch) { ArrayAudioSource(audio) }
        withContext(Dispatchers.Default) { engine.listen().toList() }
    }

    private fun List<SpeechEvent>.transcripts() = filterIsInstance<SpeechEvent.Transcript>().map { it.transcript }

    @Test
    fun anEmptyCzechResultOverEnglishIsRescuedInEnglishOnceWithTheSameWindow() {
        val native = FakeNative().apply {
            sharesInOrder(english, unused, english)
            texts += listOf("", "rescued words")
        }
        val events = session(native, audio(1_500, 1_500), retry)
        assertEquals(listOf("cs,sk,en", "en", "cs,sk,en"), native.languagesPassed.toList())
        assertEquals(3, native.thresholdsPassed.size)
        assertArrayEquals(th(nan, nan, nan), native.thresholdsPassed[0], 0f)
        assertNull(native.thresholdsPassed[1])
        assertTrue("no longer window: ${native.minAudioCtxSet}", native.minAudioCtxSet.none { it == 512 || it == 1_488 })
        val heard = events.transcripts()
        assertEquals(listOf("rescued words", "dobrý den"), heard.map { it.text })
        assertEquals(listOf("en", "cs"), heard.map { it.language })
        // the session is still Czech: the second utterance had no candidate
        assertArrayEquals(th(nan, nan, nan), native.thresholdsPassed[2], 0f)
        assertTrue("the rescue is logged without text: $logs", logs.any { it.contains("language en (rescue)") })
        assertTrue(logs.none { it.contains("rescued words") })
        assertTrue(logs.contains("language: 0 switches, 1 rescues"))
    }

    @Test
    fun aDroppedFinalLogsWhatTheFirstRunDecidedFromAndWhyNoRescueRan() {
        val native = FakeNative().apply {
            sharesInOrder(floatArrayOf(0.688f, 0f, 0.312f))
            texts += ""
        }
        session(native, audio(2_600), noRetry)
        assertEquals(1, native.languagesPassed.size)
        val line = logs.singleOrNull { it.startsWith("utterance: language decision of the first run:") }
        assertTrue(logs.toString(), line != null)
        assertTrue(line!!, line.contains("chose cs, current cs, audio 2"))
        assertTrue(line, line.endsWith("shares cs 0.688 sk 0.000 en 0.312, thresholds en 0.900, no rescue: best other en 0.312 < 0.5"))
    }

    @Test
    fun aShortFinalLogsThatItWasTooShortToDecide() {
        val native = FakeNative().apply {
            sharesInOrder(czech)
            texts += ""
        }
        session(native, audio(1_500), noRetry)
        assertTrue(logs.toString(), logs.any {
            it.startsWith("utterance: language decision of the first run:") && it.contains("audio 1") &&
                it.endsWith("thresholds too short to decide, no candidate, no rescue: best other en 0.000 < 0.5")
        })
    }

    @Test
    fun aRescuedFinalLogsTheDecisionAndTheRescue() {
        val native = FakeNative().apply {
            sharesInOrder(english, unused)
            texts += listOf("", "rescued words")
        }
        session(native, audio(1_500), noRetry)
        assertTrue(logs.toString(), logs.any { it.startsWith("utterance: language decision of the first run:") && it.endsWith("shares cs 0.002 sk 0.001 en 0.997, thresholds too short to decide, no candidate, rescue in en") })
        assertTrue(logs.none { it.contains("rescued words") })
    }

    @Test
    fun noRescueOnNonSpeech() {
        val native = FakeNative().apply {
            sharesInOrder(silence)
            texts += ""
        }
        val events = session(native, audio(1_500), noRetry)
        assertEquals(1, native.languagesPassed.size)
        assertTrue(events.transcripts().isEmpty())
    }

    @Test
    fun noRescueIntoSlovak() {
        val native = FakeNative().apply {
            sharesInOrder(slovak)
            texts += ""
        }
        val events = session(native, audio(1_500), noRetry)
        assertEquals(1, native.languagesPassed.size)
        assertTrue(events.transcripts().isEmpty())
    }

    @Test
    fun afterAFailedRescueTheWindowRetriesForceTheRescueLanguage() {
        val native = FakeNative().apply {
            sharesInOrder(english)
            texts += listOf("", "", "retried words")
        }
        val events = session(native, audio(1_500), retry)
        assertEquals(listOf("cs,sk,en", "en", "en"), native.languagesPassed.toList())
        assertTrue("the second retry window, 512, after the rescue: ${native.minAudioCtxSet}", 512 in native.minAudioCtxSet)
        val heard = events.transcripts()
        assertEquals(listOf("retried words"), heard.map { it.text })
        assertEquals(listOf("en"), heard.map { it.language })
    }

    @Test
    fun aCzechEmptyResultKeepsTodaysRetriesInCzech() {
        val native = FakeNative().apply {
            sharesInOrder(czech)
            texts += listOf("", "", "")
        }
        val events = session(native, audio(1_500), retry)
        assertEquals(listOf("cs,sk,en", "cs", "cs"), native.languagesPassed.toList())
        assertEquals(listOf(384, 512, 384, 1_488, 384), native.minAudioCtxSet.toList())
        assertTrue(events.transcripts().isEmpty())
        assertEquals(1, events.filterIsInstance<SpeechEvent.UtteranceNotWritten>().size)
    }

    @Test
    fun inAnEnglishSessionAShortCzechEmptyResultIsRescuedInCzech() {
        val native = FakeNative().apply {
            sharesInOrder(english, czech, unused)
            texts += listOf("hello there", "", "ahoj")
        }
        val events = session(native, audio(3_000, 800), noRetry)
        assertEquals(listOf("cs,sk,en", "cs,sk,en", "cs"), native.languagesPassed.toList())
        assertArrayEquals("the session is in English and the utterance short", th(nan, nan, 0f), native.thresholdsPassed[1], 0f)
        assertEquals(listOf("en", "cs"), events.transcripts().map { it.language })
    }

    @Test
    fun theRescueDoesNotDependOnRetryEmptyFinalsOrOnASecondOfClearSpeech() {
        val native = FakeNative().apply {
            sharesInOrder(english, unused)
            texts += listOf("", "rescued")
        }
        val config = retry.copy(retryEmptyFinals = false)
        val events = session(native, audio(1_200), config)
        assertEquals(listOf("cs,sk,en", "en"), native.languagesPassed.toList())
        assertEquals(listOf("rescued"), events.transcripts().map { it.text })
        assertTrue(native.minAudioCtxSet.none { it == 512 || it == 1_488 })
        // with the retries on (under a second of clear speech none is allowed) the rescue is the same
        val short = FakeNative().apply {
            sharesInOrder(english, unused)
            texts += listOf("", "rescued")
        }
        session(short, audio(800), retry)
        assertEquals(listOf("cs,sk,en", "en"), short.languagesPassed.toList())
    }

    @Test
    fun transcribeRescuesToo() {
        val native = FakeNative().apply {
            sharesInOrder(english, unused)
            texts += listOf("", "english text")
        }
        val engine = fakeEngine(native, config = noRetry, languages = three, languageSwitch = switch) { ArrayAudioSource(FloatArray(0)) }
        val text = runBlocking { engine.transcribe(Signals.tone(1.5)) }
        assertEquals("english text", text)
        assertEquals(listOf("cs,sk,en", "en"), native.languagesPassed.toList())
    }

    @Test
    fun earlyStartRecognitionsAreNotRescued() {
        val native = FakeNative().apply {
            sharesInOrder(english, english, english)
            texts += listOf("", "")
        }
        val config = noRetry.copy(preRollMs = 300, minSilenceMs = 600, earlyStart = EarlyStart())
        val events = session(native, Signals.concat(Signals.zeros(0.5), Signals.tone(0.8), Signals.zeros(1.5)), config)
        assertFalse("no run forced in English: ${native.languagesPassed}", native.languagesPassed.contains("en"))
        assertTrue(native.languagesPassed.isNotEmpty())
        assertTrue(events.transcripts().isEmpty())
    }
}
