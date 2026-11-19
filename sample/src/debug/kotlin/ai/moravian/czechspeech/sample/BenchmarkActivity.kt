package ai.moravian.czechspeech.sample

import ai.moravian.czechspeech.CzechSpeech
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.VariantSelection
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.math.BigInteger
import java.security.MessageDigest

/**
 * Debug screen (debug builds only, src/debug): the time per utterance and the peak memory of recognition on this phone, for greedy decoding (today),
 * beam search, and beam search with variant selection's language model. Only numbers are shown and logged (logcat tag
 * CzechSpeechBench), never text.
 *
 * Audio: invented Czech sentences ([Sentences]) spoken by the phone's text-to-speech into memory and resampled to
 * 16 kHz ([SpeechSynthesis]); nothing is written to disk. The models are the ones MainActivity uses, pushed to
 * getExternalFilesDir("models"): the CzechSpeech model (*.bin) and, for variant selection, the language model (*.gguf).
 *
 * One mode per process gives clean peak memory (VmHWM):
 *
 *   adb shell am force-stop ai.moravian.czechspeech.sample
 *   adb shell am start -n ai.moravian.czechspeech.sample/.BenchmarkActivity --es mode greedy|beam|lm \
 *       [--ei rounds 3] [--ei beam 5] [--ei threads 0] [--ez auto_threads false] [--ei min_audio_ctx 384]
 *   adb logcat -s CzechSpeechBench      # ends with "done"
 *
 * auto_threads false turns off the SDK's thread placement (CzechSpeechConfig.autoThreads): the recogniser then runs on
 * `threads` threads (0 = the big core count) wherever Android puts them, as before it existed. For A/B runs the RESULT
 * line says what Linux saw of the recogniser's threads, sampled every 25 ms from /proc (the SDK's whisper thread and the
 * worker threads it starts all carry its name): rec_threads_max (most at once), rec_cpus (their affinity masks, ORed)
 * and ran_cpus (the cores they were seen running on). proc_cpus and proc_cpus_end are the cores Android allowed the app
 * at the start and at the end (its cpuset: 0-7 on screen on a Pixel 9a), screen_off counts the times the screen went
 * off during the run: a run with screen_off above 0 or a changed proc_cpus measured something else. The screen is kept
 * on while the benchmark runs, and the screen turns on and shows over the lock screen when adb starts it (with mode).
 *
 * min_audio_ctx is the floor of the encoder window in frames (CzechSpeechConfig.minEncoderWindow, 512 by default): 384
 * only with a model that bears it (cs+sk+en). A value the SDK would refuse (not a multiple of 16 from 256 to 1500) is
 * shown as an error and nothing runs. RESULT repeats it, and the SDK's own lines (adb logcat -s CzechSpeech) give the
 * floor with "model loaded" and the window of each recognition with "utterance:".
 *
 * To compare transcripts between runs (3 against 4 threads: --ei threads 3 --ez auto_threads false), RESULT gives
 * text_hash, the first 16 hex digits of the SHA-256 of the first round's transcripts (each followed by a zero byte), and
 * text_rounds_same, whether every round gave the same; the text itself is never shown, logged or kept.
 */
class BenchmarkActivity : ComponentActivity() {
    private lateinit var output: TextView
    private var running: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val mode = intent.getStringExtra("mode")
        // A measurement started with adb (the mode extra) on a dark or locked phone: the screen would stay off and the
        // app would not be in front, so Android would run it in its background cpuset (the little cores), and the run
        // would measure that. Opened by hand, the screen behaves as any other.
        if (mode != null) wakeForMeasurement()
        output = TextView(this).apply { setPadding(32, 16, 32, 16); setTextIsSelectable(true) }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            for ((label, mode) in listOf("Greedy" to "greedy", "Beam" to "beam", "Beam + LM" to "lm")) {
                addView(Button(this@BenchmarkActivity).apply {
                    text = label
                    setOnClickListener { start(mode, rounds = 3, beam = 5, threads = 0, autoThreads = true) }
                })
            }
        }
        val screen = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(buttons)
            addView(ScrollView(this@BenchmarkActivity).apply { addView(output) },
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        ViewCompat.setOnApplyWindowInsetsListener(screen) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        setContentView(screen)
        if (mode != null && savedInstanceState == null) {
            // bounded: a typo in an adb extra must not start a run of hours or an invalid configuration
            start(
                mode,
                intent.getIntExtra("rounds", 3).coerceIn(1, 20),
                intent.getIntExtra("beam", 5).coerceIn(2, 8),
                intent.getIntExtra("threads", 0).coerceIn(0, 16),
                intent.getBooleanExtra("auto_threads", true),
                intent.getIntExtra("min_audio_ctx", 512),
            )
        } else if (mode == null) {
            output.setText(R.string.benchmark_hint)
        }
    }

    private fun wakeForMeasurement() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
        }
    }

    override fun onDestroy() {
        running?.cancel()
        super.onDestroy()
    }

    private fun show(line: String) {
        Log.i(TAG, line)
        runOnUiThread { output.append(line + "\n") }
    }

    private fun start(mode: String, rounds: Int, beam: Int, threads: Int, autoThreads: Boolean, minAudioCtx: Int = 512) {
        if (running?.isActive == true) return
        output.text = ""
        // not coerced: a run with another floor than the one asked for would measure something else
        if (minAudioCtx !in 256..1500 || minAudioCtx % 16 != 0) {
            show("error: min_audio_ctx=$minAudioCtx must be a multiple of 16 from 256 to 1500")
            show("done")
            return
        }
        // the screen stays on while the benchmark runs: a dark screen sends the app to the little cores
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val screenOffs = ScreenOffCounter().also { it.register(this) }
        running = lifecycleScope.launch {
            try {
                withContext(Dispatchers.Default) { run(mode, rounds, beam, threads, autoThreads, minAudioCtx, screenOffs) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                show("error: ${e.javaClass.simpleName}: ${e.message}")
            } finally {
                screenOffs.unregister(this@BenchmarkActivity)
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
            show("done")
        }
    }

    private suspend fun run(
        mode: String,
        rounds: Int,
        beam: Int,
        threads: Int,
        autoThreads: Boolean,
        minAudioCtx: Int,
        screenOffs: ScreenOffCounter,
    ) {
        val models = getExternalFilesDir("models")?.listFiles().orEmpty().filter { it.isFile }
        val model = models.filter { it.name.endsWith(".bin") }.minByOrNull { it.name }
            ?: error("no CzechSpeech model (*.bin) in ${getExternalFilesDir("models")}")
        val lm = models.filter { it.name.endsWith(".gguf") }.minByOrNull { it.name }
        val selection = when (mode) {
            "greedy" -> null
            // λ = 0: beam search alone, no language model
            "beam" -> VariantSelection(beamSize = beam, languageModelWeight = 0f)
            "lm" -> VariantSelection(lm ?: error("no language model (*.gguf) in ${getExternalFilesDir("models")}"),
                beamSize = beam, lengthBonus = if (beam == 3) 0.75f else 0.3f)
            else -> error("unknown mode '$mode' (greedy, beam, lm)")
        }
        val procCpus = ThreadObserver.processCpus()
        show("mode=$mode beam=${if (selection == null) 1 else beam} rounds=$rounds threads=$threads " +
            "auto_threads=$autoThreads min_audio_ctx=$minAudioCtx proc_cpus=$procCpus " +
            "model_mb=${model.length() / 1_000_000} lm_mb=${if (mode == "lm") lm!!.length() / 1_000_000 else 0}")

        val t0 = SystemClock.elapsedRealtime()
        val audio = SpeechSynthesis.speak(this, Sentences.ALL)
        show("audio: ${audio.size} utterances, %.1f s in all, synthesised in %d ms, %s".format(
            audio.sumOf { it.size } / 16_000.0, SystemClock.elapsedRealtime() - t0, SpeechSynthesis.lastVoice))

        val memory = MemorySampler().apply { start() }
        val baseline = memory.now()
        show("baseline: rss_mb=${baseline.rssMb} native_mb=${baseline.nativeMb}")
        // Self-check on the real regex engine of this device: every regular expression of the SDK compiles here (the desktop
        // unit tests cannot tell; a pattern Android refuses once crashed the app). The engine does the same when it is
        // created; this puts the result into the run's output. Reflection: the registry is internal to the SDK.
        try {
            val registry = Class.forName("ai.moravian.czechspeech.internal.SdkPatterns")
            show("regex self-check: ${registry.getMethod("verify").invoke(null)} pattern(s) compile on this device")
        } catch (e: java.lang.reflect.InvocationTargetException) {
            show("regex self-check FAILED: ${e.cause?.message}")
            throw e.cause ?: e
        }
        val speech = CzechSpeech(
            this, model,
            CzechSpeechConfig(
                threads = threads, variantSelection = selection, autoThreads = autoThreads, minEncoderWindow = minAudioCtx,
            ),
            keepModelLoaded = true,
        )
        val times = Array(audio.size) { LongArray(rounds) }
        val observer = ThreadObserver()
        // a fingerprint of each round's transcripts, to compare runs (3 vs 4 threads, say) without keeping or logging text
        val prints = Array(rounds) { MessageDigest.getInstance("SHA-256") }
        try {
            var t = SystemClock.elapsedRealtime()
            speech.transcribe(audio[0])                       // loads the model(s); not counted
            show("warm-up (model load + first utterance): ${SystemClock.elapsedRealtime() - t} ms")
            observer.start()                                  // the timed utterances only
            for (r in 0 until rounds) {
                for ((i, samples) in audio.withIndex()) {
                    t = SystemClock.elapsedRealtime()
                    val text = speech.transcribe(samples)
                    times[i][r] = SystemClock.elapsedRealtime() - t
                    prints[r].update(text.toByteArray(Charsets.UTF_8))       // the text itself is not kept
                    prints[r].update(0.toByte())
                }
            }
        } finally {
            observer.stop()
            memory.stop()                                     // stops even when the run fails or is cancelled
            speech.close()
            speech.awaitClosed()
        }

        for ((i, samples) in audio.withIndex()) {
            show("utt=$i audio_ms=${samples.size / 16} ms=${times[i].joinToString(",")}")
        }
        val all = times.flatMap { it.toList() }.sorted()
        val audioMs = audio.sumOf { it.size / 16L } * rounds
        val peak = memory.peak()
        val textHashes = prints.map { digest -> digest.digest().take(8).joinToString("") { "%02x".format(it) } }
        show(
            "RESULT mode=$mode beam=${if (selection == null) 1 else beam} n=${all.size} " +
                "median_ms=${all[all.size / 2]} mean_ms=${all.sum() / all.size} p90_ms=${all[(all.size * 9) / 10]} " +
                "max_ms=${all.last()} rtf=%.3f ".format(all.sum().toDouble() / audioMs) +
                "peak_rss_mb=${peak.rssMb} peak_native_mb=${peak.nativeMb} vmhwm_mb=${MemorySampler.vmHwmMb()} " +
                "above_baseline_rss_mb=${peak.rssMb - baseline.rssMb} " +
                "auto_threads=$autoThreads min_audio_ctx=$minAudioCtx rec_threads_max=${observer.maxThreads} rec_cpus=${observer.allowedHex()} " +
                "ran_cpus=${observer.ranHex()} proc_cpus=$procCpus proc_cpus_end=${ThreadObserver.processCpus()} " +
                "screen_off=${screenOffs.count} threads=$threads " +
                "text_hash=${textHashes.first()} text_rounds_same=${textHashes.distinct().size == 1}",
        )
        for (a in audio) a.fill(0f)
    }

    /** Counts the times the screen goes off while registered (a run that saw one measured the background). */
    private class ScreenOffCounter : BroadcastReceiver() {
        @Volatile var count = 0
            private set

        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == Intent.ACTION_SCREEN_OFF) count++
        }

        // On the application context, so that a run cancelled by onDestroy() can still unregister afterwards. Only the
        // system sends ACTION_SCREEN_OFF, which reaches a receiver that is not exported.
        fun register(context: Context) {
            ContextCompat.registerReceiver(
                context.applicationContext, this, IntentFilter(Intent.ACTION_SCREEN_OFF),
                ContextCompat.RECEIVER_NOT_EXPORTED,
            )
        }

        fun unregister(context: Context) = context.applicationContext.unregisterReceiver(this)
    }

    /**
     * The recogniser's threads as Linux sees them, every 25 ms: the SDK's whisper thread ("czechspeech-whisper", in
     * /proc its first 15 characters) and the threads whisper.cpp and ggml start from it, which inherit its name and its
     * CPU affinity. Masks and cores count only while at least two of them exist (a computation is running); outside one,
     * the whisper thread waits unpinned.
     */
    private class ThreadObserver {
        @Volatile private var stopped = false
        @Volatile var maxThreads = 0
            private set
        private var allowed = BigInteger.ZERO
        private var ran = 0L
        private val thread = Thread {
            while (!stopped) {
                sample()
                Thread.sleep(25)
            }
        }.apply { isDaemon = true }

        fun start() = thread.start()

        fun stop() {
            stopped = true
            if (thread.isAlive) thread.join()
        }

        @Synchronized
        fun allowedHex(): String = "0x" + allowed.toString(16)

        @Synchronized
        fun ranHex(): String = "0x" + java.lang.Long.toHexString(ran)

        private fun sample() {
            val ours = File("/proc/self/task").list().orEmpty().mapNotNull { tid ->
                val stat = readOrNull(File("/proc/self/task/$tid/stat")) ?: return@mapNotNull null
                val comm = stat.substringAfter('(').substringBeforeLast(')')
                if (comm != NAME) null else tid to stat.substringAfterLast(") ").split(' ')
            }
            if (ours.size > maxThreads) maxThreads = ours.size
            if (ours.size < 2) return
            synchronized(this) {
                for ((tid, fields) in ours) {
                    // fields[0] is the state (stat field 3), fields[36] the core it last ran on (field 39)
                    if (fields.getOrNull(0) == "R") fields.getOrNull(36)?.toIntOrNull()?.let { ran = ran or (1L shl it) }
                    readOrNull(File("/proc/self/task/$tid/status"))?.lineSequence()
                        ?.firstOrNull { it.startsWith("Cpus_allowed:") }
                        ?.substringAfter(':')?.trim()?.replace(",", "")
                        ?.toBigIntegerOrNull(16)?.let { allowed = allowed.or(it) }
                }
            }
        }

        companion object {
            const val NAME = "czechspeech-whi"

            /** The cores Android allows the app now (Cpus_allowed_list of its main thread: its cpuset). */
            fun processCpus(): String = readOrNull(File("/proc/self/status"))?.lineSequence()
                ?.firstOrNull { it.startsWith("Cpus_allowed_list:") }?.substringAfter(':')?.trim() ?: "?"

            private fun readOrNull(file: File): String? = try {
                file.readText()
            } catch (e: Exception) {
                null                                              // the thread ended meanwhile
            }
        }
    }

    /** Samples the process's resident memory and native heap every 10 ms; [peak] is the highest seen. */
    private class MemorySampler {
        class Snapshot(val rssMb: Long, val nativeMb: Long)

        @Volatile private var stopped = false
        private var peakRss = 0L
        private var peakNative = 0L
        private val thread = Thread {
            while (!stopped) {
                sample()
                Thread.sleep(10)
            }
        }.apply { isDaemon = true }

        fun start() = thread.start()

        fun stop() {
            stopped = true
            thread.join()
            sample()
        }

        @Synchronized
        private fun sample() {
            peakRss = maxOf(peakRss, statusKb("VmRSS"))
            peakNative = maxOf(peakNative, Debug.getNativeHeapAllocatedSize())
        }

        fun now() = Snapshot(statusKb("VmRSS") / 1024, Debug.getNativeHeapAllocatedSize() / 1_048_576)

        @Synchronized
        fun peak() = Snapshot(peakRss / 1024, peakNative / 1_048_576)

        companion object {
            fun statusKb(key: String): Long = File("/proc/self/status").useLines { lines ->
                lines.firstOrNull { it.startsWith("$key:") }?.filter { it.isDigit() }?.toLongOrNull() ?: 0L
            }

            /** Peak resident memory of the process since it started (one mode per process: the mode's peak). */
            fun vmHwmMb() = statusKb("VmHWM") / 1024
        }
    }

    private companion object {
        const val TAG = "CzechSpeechBench"
    }
}
