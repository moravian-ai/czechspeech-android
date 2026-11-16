package ai.moravian.czechspeech.internal

import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull
import org.junit.Assume
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.min
import kotlin.math.sin

/**
 * Test inputs come only from the environment (no defaults, nothing machine-specific in the repo); integration tests
 * skip, with a message naming the variable, when one is unset or points nowhere.
 */
object TestEnv {
    val model: File get() = fromEnv("CZECHSPEECH_MODEL")
    val audioDir: File get() = fromEnv("CZECHSPEECH_TEST_AUDIO")
    val manifest: File get() = fromEnv("CZECHSPEECH_TEST_MANIFEST")
    val threads: Int get() = System.getenv("CZECHSPEECH_TEST_THREADS")?.toIntOrNull()?.coerceIn(1, 8) ?: 6

    /** The Silero VAD asset bundled in the AAR. */
    val vadAsset: File get() = File("src/main/assets/czechspeech/ggml-silero-v6.2.0.bin")

    /** An unset variable gives a path that does not exist, so [assumeFile] skips and names the variable. */
    private fun fromEnv(name: String): File =
        System.getenv(name)?.takeIf { it.isNotBlank() }?.let(::localPath) ?: File("<$name not set>")

    /** Windows paths ("C:\...") become /mnt/c/... when the tests run in WSL. */
    fun localPath(path: String): File {
        val windows = Regex("""^([A-Za-z]):[\\/](.*)$""").find(path)
        val isWindows = System.getProperty("os.name").orEmpty().startsWith("Windows")
        if (windows == null || isWindows) return File(path)
        val (drive, rest) = windows.destructured
        return File("/mnt/${drive.lowercase()}/${rest.replace('\\', '/')}")
    }

    fun assumeNativeLibrary() {
        val loaded = try {
            NativeLoader.ensureLoaded()
            true
        } catch (e: Throwable) {
            System.err.println("SKIP: host native library not available (${e.message}); run native/build-native.sh host")
            false
        }
        Assume.assumeTrue("host native library", loaded)
    }

    fun assumeFile(file: File, what: String) {
        if (!file.exists()) System.err.println("SKIP: $what not found at $file")
        Assume.assumeTrue("$what at $file", file.exists())
    }

    fun audio(name: String): FloatArray {
        val file = File(audioDir, name)
        assumeFile(file, "test audio $name")
        return Wav.read(file)
    }

    /** The reference transcript of a ParCzech segment, from the manifest. */
    fun reference(id: String): String {
        assumeFile(manifest, "ParCzech manifest")
        val line = manifest.useLines { lines -> lines.firstOrNull { it.contains("\"id\": \"$id\"") || it.contains("\"id\":\"$id\"") } }
        Assume.assumeTrue("manifest entry $id", line != null)
        return Regex(""""ref":\s*"((?:[^"\\]|\\.)*)"""").find(line!!)!!.groupValues[1]
    }
}

/** Minimal PCM WAV reader: 16-bit or 32-bit float, mono, 16 kHz. */
object Wav {
    fun read(file: File): FloatArray {
        val bytes = file.readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" && String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") {
            "$file is not a WAV file"
        }
        var pos = 12
        var format = 0
        var channels = 0
        var rate = 0
        var bits = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = buf.getInt(pos + 4)
            val body = pos + 8
            when (id) {
                "fmt " -> {
                    format = buf.getShort(body).toInt()
                    channels = buf.getShort(body + 2).toInt()
                    rate = buf.getInt(body + 4)
                    bits = buf.getShort(body + 14).toInt()
                }
                "data" -> {
                    require(channels == 1 && rate == SAMPLE_RATE) { "$file: need 16 kHz mono, got $rate Hz x $channels" }
                    val n = min(size, bytes.size - body)
                    return when {
                        format == 1 && bits == 16 -> FloatArray(n / 2) { buf.getShort(body + 2 * it) / 32768f }
                        format == 3 && bits == 32 -> FloatArray(n / 4) { buf.getFloat(body + 4 * it) }
                        else -> error("$file: unsupported WAV format $format/$bits bit")
                    }
                }
            }
            pos = body + size + (size and 1)
        }
        error("$file: no data chunk")
    }
}

/**
 * Plays a float array as audio, frame by frame, then ends (or repeats [tail] forever). [onFrame] runs on the capture
 * thread after each frame is read, with the number of frames read so far (e.g. to call stop() at an exact frame).
 */
class ArrayAudioSource(
    private val samples: FloatArray,
    private val tail: ((index: Long) -> Float)? = null,
    private val sleepPerFrameMs: Long = 0,
    private val onFrame: ((framesRead: Int) -> Unit)? = null,
) : AudioSource {
    private var pos = 0
    private var tailIndex = 0L

    @Volatile
    var opened = false
        private set

    @Volatile
    var closed = false
        private set

    @Volatile
    var framesRead = 0
        private set

    override fun open() {
        opened = true
    }

    override fun read(frame: FloatArray): Int {
        if (sleepPerFrameMs > 0) Thread.sleep(sleepPerFrameMs)
        val n = if (pos >= samples.size) {
            val generator = tail ?: return -1
            for (i in frame.indices) frame[i] = generator(tailIndex++)
            frame.size
        } else {
            val copied = min(frame.size, samples.size - pos)
            System.arraycopy(samples, pos, frame, 0, copied)
            frame.fill(0f, copied, frame.size)
            pos += copied
            copied
        }
        framesRead++
        onFrame?.invoke(framesRead)
        return n
    }

    override fun close() {
        closed = true
    }
}

object Signals {
    fun zeros(seconds: Double) = FloatArray((seconds * SAMPLE_RATE).toInt())

    fun tone(seconds: Double, hz: Double = 220.0, amplitude: Float = 0.3f) =
        toneSamples((seconds * SAMPLE_RATE).toInt(), hz, amplitude)

    /** A tone of exactly [n] samples (for tests that count frames). */
    fun toneSamples(n: Int, hz: Double = 220.0, amplitude: Float = 0.3f) =
        FloatArray(n) { (amplitude * sin(2 * PI * hz * it / SAMPLE_RATE)).toFloat() }

    /** Sample [i] of an endless tone, for [ArrayAudioSource]'s tail. */
    fun toneAt(i: Long, hz: Double = 220.0, amplitude: Float = 0.3f) =
        (amplitude * sin(2 * PI * hz * i / SAMPLE_RATE)).toFloat()

    fun whiteNoise(seconds: Double, amplitude: Float = 0.1f, seed: Long = 1): FloatArray {
        val random = java.util.Random(seed)
        return FloatArray((seconds * SAMPLE_RATE).toInt()) { (amplitude * (random.nextFloat() * 2 - 1)) }
    }

    fun concat(vararg parts: FloatArray): FloatArray {
        val out = FloatArray(parts.sumOf { it.size })
        var pos = 0
        for (part in parts) {
            System.arraycopy(part, 0, out, pos, part.size)
            pos += part.size
        }
        return out
    }
}

/**
 * Waits until [condition] holds and fails the test after [timeoutMs]. It polls with delay(), so the timeout really
 * fires: withTimeout cannot interrupt Thread.sleep() or any other blocking call inside it.
 */
suspend fun awaitCondition(what: String, timeoutMs: Long = 10_000, condition: () -> Boolean) {
    val met = withTimeoutOrNull(timeoutMs) {
        while (!condition()) delay(2)
        true
    }
    if (met == null) throw AssertionError("timed out after $timeoutMs ms waiting for $what")
}

/** Joins and fails the test (instead of hanging) when [what] has not completed within [timeoutMs]. */
suspend fun Job.joinWithin(timeoutMs: Long, what: String) {
    withTimeoutOrNull(timeoutMs) { join() } ?: throw AssertionError("$what did not complete within $timeoutMs ms")
}

/** The result of [Deferred.await], or an AssertionError when [what] has not completed within [timeoutMs]. */
suspend fun <T> Deferred<T>.awaitWithin(timeoutMs: Long, what: String): T {
    val outcome = withTimeoutOrNull(timeoutMs) { runCatching { await() } }
        ?: throw AssertionError("$what did not complete within $timeoutMs ms")
    return outcome.getOrThrow()
}

/** The [T] thrown by [block] (which may suspend: the lambda is inlined), or an AssertionError. */
inline fun <reified T : Throwable> failure(block: () -> Unit): T {
    try {
        block()
    } catch (e: Throwable) {
        if (e is T) return e
        throw AssertionError("expected ${T::class.java.simpleName}, got $e", e)
    }
    throw AssertionError("expected ${T::class.java.simpleName}, nothing was thrown")
}

/** [outcome] failed with the exception of a closed engine. */
fun assertClosed(outcome: Result<*>) {
    val error = outcome.exceptionOrNull()
    if (!(error is IllegalStateException && error.message == "CzechSpeech is closed")) {
        throw AssertionError("expected IllegalStateException(\"CzechSpeech is closed\"), got $outcome", error)
    }
}

/** Share of reference words found in the hypothesis (case and punctuation ignored, multiset). */
fun wordRecall(reference: String, hypothesis: String): Double {
    fun words(s: String) = s.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.isNotEmpty() }
    val ref = words(reference)
    val hyp = words(hypothesis).groupingBy { it }.eachCount().toMutableMap()
    var hit = 0
    for (w in ref) {
        val c = hyp[w] ?: 0
        if (c > 0) {
            hit++
            hyp[w] = c - 1
        }
    }
    return if (ref.isEmpty()) 1.0 else hit.toDouble() / ref.size
}
