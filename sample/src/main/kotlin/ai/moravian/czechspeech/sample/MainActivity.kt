package ai.moravian.czechspeech.sample

import ai.moravian.czechspeech.CzechSpeech
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.SpeechEvent
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File

/**
 * Minimal CzechSpeech demo: Start/Stop and a list of transcripts.
 *
 * Model: a .bin pushed to getExternalFilesDir("models") wins, e.g.
 *   adb push ggml-czechspeech-q4_0.bin /sdcard/Android/data/ai.moravian.czechspeech.sample/files/models/
 * otherwise the "Download model" button fetches it with CzechSpeechModel.download. Debug builds add "Benchmark", the
 * screen that times recognition per decoding mode (BenchmarkActivity, src/debug).
 */
class MainActivity : ComponentActivity() {
    private var speech: CzechSpeech? = null
    private var listening: Job? = null
    private var finishing = false

    private lateinit var status: TextView
    private lateinit var startStop: Button
    private lateinit var download: Button
    private lateinit var transcripts: ArrayAdapter<String>

    private val askMicrophone = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startListening() else status.text = "Microphone permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { setPadding(32, 32, 32, 16) }
        startStop = Button(this).apply {
            text = "Start"
            setOnClickListener { toggle() }
        }
        download = Button(this).apply {
            text = "Download model (474 MB)"
            setOnClickListener { downloadModel() }
        }
        transcripts = ArrayAdapter(this, android.R.layout.simple_list_item_1, mutableListOf())
        val list = ListView(this).apply { adapter = transcripts }
        val screen = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status)
            addView(startStop)
            addView(download)
            DebugScreens.addTo(this@MainActivity, this)
            addView(list, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        // Edge to edge (Android 15+ for apps targeting API 35+): the action bar, the navigation bar and a cutout lie
        // over the screen, which gets their size as insets. Where the window keeps clear of them itself, they are zero.
        ViewCompat.setOnApplyWindowInsetsListener(screen) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        setContentView(screen)
    }

    override fun onStart() {
        super.onStart()
        refreshModel()
    }

    override fun onStop() {
        // Never keep the microphone open in the background.
        stopListening()
        super.onStop()
    }

    override fun onDestroy() {
        speech?.close()
        super.onDestroy()
    }

    private fun findModel(): File? {
        val pushed = getExternalFilesDir("models")
            ?.listFiles()
            ?.filter { it.isFile && it.name.endsWith(".bin") }
            ?.minByOrNull { it.name }
        return pushed ?: CzechSpeechModel.file(this).takeIf { CzechSpeechModel.isDownloaded(this) }
    }

    private fun refreshModel() {
        val model = findModel()
        if (model != null && speech?.modelFile != model) {
            speech?.close()
            speech = CzechSpeech(this, model)
        }
        startStop.isEnabled = model != null
        download.visibility = if (model == null) View.VISIBLE else View.GONE
        status.text = if (model == null) {
            "No model. Download it, or adb push a .bin to ${getExternalFilesDir("models")}"
        } else {
            "Model: ${model.name}"
        }
    }

    private fun toggle() {
        when {
            listening?.isActive == true -> finishListening()
            checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> startListening()
            else -> askMicrophone.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startListening() {
        val speech = speech ?: return
        startStop.text = "Stop"
        status.text = "Listening…"
        finishing = false
        listening = lifecycleScope.launch {
            try {
                speech.listen().collect { event ->
                    when (event) {
                        is SpeechEvent.SpeechStarted -> status.text = "Speech…"
                        is SpeechEvent.SpeechEnded -> status.text = "Recognising…"
                        is SpeechEvent.Transcript -> {
                            val t = event.transcript
                            val seconds = (t.endMs - t.startMs) / 1000.0
                            transcripts.insert("${t.text}\n%.1f s of speech, recognised in %d ms".format(seconds, t.processingMs), 0)
                            if (!finishing) status.text = "Listening…"
                        }
                        is SpeechEvent.UtteranceDone -> Unit
                        is SpeechEvent.UtteranceNotWritten -> status.text = "A stretch of speech could not be written"
                        is SpeechEvent.Error -> status.text = "Error: ${event.cause.message}"
                    }
                }
                status.text = "Stopped"
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                status.text = "Stopped: ${e.message}"
            } finally {
                startStop.text = "Start"
                startStop.isEnabled = true
            }
        }
    }

    /** The Stop button: the microphone closes, the last utterance is still transcribed, then the session ends. */
    private fun finishListening() {
        finishing = true
        speech?.stop()
        startStop.isEnabled = false
        status.text = "Finishing…"
    }

    /** Leaving the screen: cancel, which discards speech not yet transcribed. */
    private fun stopListening() {
        if (listening?.isActive == true) status.text = "Stopped"
        listening?.cancel()
        listening = null
        startStop.text = "Start"
    }

    private fun downloadModel() {
        download.isEnabled = false
        lifecycleScope.launch {
            try {
                CzechSpeechModel.download(this@MainActivity) { done, total ->
                    runOnUiThread { status.text = "Downloading… ${done * 100 / total} %" }
                }
                refreshModel()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                status.text = "Download failed: ${e.message}"
            } finally {
                download.isEnabled = true
            }
        }
    }
}
