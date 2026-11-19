package ai.moravian.czechspeech.sample

import android.content.Context
import android.media.AudioFormat
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.withTimeout
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sin

/** Invented Czech sentences for the benchmark, from about 1.5 to 12 s of speech. */
object Sentences {
    val ALL = listOf(
        "Ahoj, jak se máš?",
        "Zítra ráno pojedu vlakem do Brna.",
        "Prosím tě, kup cestou domů chleba, mléko a dvě rajčata.",
        "Schůzka se přesouvá na čtvrtek v deset hodin, sejdeme se v malé zasedačce ve druhém patře.",
        "Včera večer jsme s dětmi hráli deskové hry a pak jsme se dívali na starý film o cestě kolem světa.",
        "Petře, zdraví tě Jana a vzkazuje, že ten balík z pošty vyzvedne až v pondělí odpoledne.",
        "Kdybys měl čas, mohl bys mi prosím poslat ty fotky z výletu? Moc by mi to pomohlo.",
        "Podle předpovědi má být celý týden teplo, ale na víkend hlásí silný vítr a místy bouřky, takže výlet na " +
            "hory raději odložíme a domluvíme se znovu příští středu.",
        "Děkuji, to je všechno.",
        "Nezapomeň, že paní doktorka ordinuje jen dopoledne a k objednání je potřeba zavolat nejpozději den předem.",
    )
}

/**
 * The phone's text-to-speech into memory: synthesizeToFile with /dev/null as the file, the audio taken from
 * [UtteranceProgressListener.onAudioAvailable], so nothing is written to disk; then 16 kHz mono floats. When the
 * engine delivers no audio that way, it speaks each sentence at volume 0 instead (real time) and the audio comes from
 * the same callback.
 */
object SpeechSynthesis {
    /** Whether the last [speak] had a Czech voice and whether it needed the network (no voice name is logged). */
    @Volatile var lastVoice: String = "-"
        private set

    suspend fun speak(context: Context, sentences: List<String>): List<FloatArray> {
        val ready = CompletableDeferred<Int>()
        val tts = TextToSpeech(context.applicationContext) { ready.complete(it) }
        try {
            check(withTimeout(30_000) { ready.await() } == TextToSpeech.SUCCESS) { "text-to-speech unavailable" }
            val czech = tts.setLanguage(Locale("cs", "CZ")) >= TextToSpeech.LANG_AVAILABLE
            // A Czech voice installed on the phone first: a network voice sends the sentences to a server and its
            // timing depends on the connection.
            val installed = tts.voices.orEmpty()
                .filter { it.locale.language == "cs" && !it.isNetworkConnectionRequired }
                .filterNot { TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED in it.features.orEmpty() }
                .maxByOrNull { it.quality }
            if (installed != null) tts.voice = installed
            lastVoice = "czech_voice=$czech network_voice=${tts.voice?.isNetworkConnectionRequired ?: false}"
            val listener = Collector()
            tts.setOnUtteranceProgressListener(listener)
            return sentences.mapIndexed { i, text ->
                var pcm = listener.run("f$i") { tts.synthesizeToFile(text, Bundle(), File("/dev/null"), it) }
                if (pcm == null || pcm.samples.isEmpty()) {
                    val quiet = Bundle().apply { putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, 0f) }
                    pcm = listener.run("s$i") { tts.speak(text, TextToSpeech.QUEUE_ADD, quiet, it) }
                }
                checkNotNull(pcm) { "no audio for sentence $i" }
                resample(pcm.samples, pcm.rate, 16_000).also { pcm.samples.fill(0f) }
            }
        } finally {
            tts.shutdown()
        }
    }

    private class Pcm(val samples: FloatArray, val rate: Int)

    private class Collector : UtteranceProgressListener() {
        private var id = ""
        private var rate = 0
        private var format = AudioFormat.ENCODING_PCM_16BIT
        private val bytes = ByteArrayOutputStream()
        private var done = CompletableDeferred<Boolean>()

        suspend fun run(utterance: String, start: (String) -> Int): Pcm? {
            synchronized(this) {
                id = utterance
                rate = 0
                bytes.reset()
                done = CompletableDeferred()
            }
            if (start(utterance) != TextToSpeech.SUCCESS) return null
            val ok = withTimeout(120_000) { done.await() }
            synchronized(this) {
                val raw = bytes.toByteArray()
                bytes.reset()
                if (!ok || rate <= 0) return null
                val buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
                val samples = when (format) {
                    AudioFormat.ENCODING_PCM_FLOAT -> FloatArray(raw.size / 4) { buf.getFloat(4 * it) }
                    AudioFormat.ENCODING_PCM_8BIT -> FloatArray(raw.size) { ((raw[it].toInt() and 0xff) - 128) / 128f }
                    else -> FloatArray(raw.size / 2) { buf.getShort(2 * it) / 32768f }
                }
                raw.fill(0)
                return Pcm(samples, rate)
            }
        }

        @Synchronized
        override fun onBeginSynthesis(utteranceId: String, sampleRateInHz: Int, audioFormat: Int, channelCount: Int) {
            if (utteranceId == id) {
                rate = sampleRateInHz
                format = audioFormat
            }
        }

        @Synchronized
        override fun onAudioAvailable(utteranceId: String, audio: ByteArray) {
            if (utteranceId == id) bytes.write(audio)
        }

        override fun onStart(utteranceId: String) {}

        override fun onDone(utteranceId: String) {
            finish(utteranceId, true)
        }

        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String) {
            finish(utteranceId, false)
        }

        override fun onError(utteranceId: String, errorCode: Int) {
            finish(utteranceId, false)
        }

        @Synchronized
        private fun finish(utteranceId: String, ok: Boolean) {
            if (utteranceId == id) done.complete(ok)
        }
    }

    /** Windowed-sinc resampling (32 taps, Hann window, cut-off just below the lower Nyquist frequency). */
    fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to) return x.copyOf()
        val ratio = from.toDouble() / to
        val cutoff = 0.95 * min(1.0, 1.0 / ratio)              // in cycles per input sample, times 2
        val half = 16
        return FloatArray((x.size / ratio).toInt()) { n ->
            val t = n * ratio
            val center = floor(t).toInt()
            var acc = 0.0
            for (k in center - half + 1..center + half) {
                if (k < 0 || k >= x.size) continue
                val d = t - k
                val sinc = if (d == 0.0) cutoff else sin(PI * cutoff * d) / (PI * d)
                val w = 0.5 + 0.5 * cos(PI * d / half)
                acc += x[k] * sinc * w
            }
            acc.toFloat().coerceIn(-1f, 1f)
        }
    }
}
