package ai.moravian.czechspeech

import ai.moravian.czechspeech.internal.AndroidLog
import ai.moravian.czechspeech.internal.ModelDownloader
import ai.moravian.czechspeech.internal.requireLanguageCode
import android.content.Context
import android.os.storage.StorageManager
import java.io.File

/**
 * One-time download of the CzechSpeech model (~474 MB).
 *
 * This is the only network code in the SDK; the SDK manifest does not declare INTERNET, so an app that calls
 * [download] must declare `android.permission.INTERNET` itself. Apps that ship the model another way skip this
 * object and pass their own [File] to [CzechSpeech].
 *
 * The file goes to `context.noBackupFilesDir/czechspeech/`, which is private to the app and excluded from backups.
 */
public object CzechSpeechModel {
    /**
     * Where to download the model from. Override it to serve the model from your own mirror; the size and SHA-256
     * are always checked.
     */
    public data class Source(
        public val url: String,
        public val sizeBytes: Long,
        public val sha256: String,
        public val fileName: String = "ggml-czechspeech-q4_0.bin",
    ) {
        init {
            require(url.startsWith("https://") || url.startsWith("http://")) { "url must be http(s)" }
            require(sizeBytes > 0) { "sizeBytes must be positive" }
            require(sha256.matches(Regex("[0-9a-fA-F]{64}"))) { "sha256 must be 64 hex digits" }
            require(fileName.isNotBlank() && '/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..") {
                "fileName must be a plain file name"
            }
        }
    }

    /**
     * One model of the [CATALOG].
     *
     * @property id Short name, what [entry] looks up.
     * @property languages The languages the model recognises, two- or three-letter lowercase whisper language codes
     *   (such as `cs`, `pl`, `yue`); the first is the default.
     * @property tokensPerSecond Typical speech rate in tokens per second. The catalog carries it; the recogniser does
     *   not use it yet (there is no token cap on Android).
     */
    // Speaker separation is not part of the SDK yet; an entry will say which models support it once it is.
    public data class Entry(
        public val id: String,
        public val source: Source,
        public val languages: List<String>,
        public val tokensPerSecond: Int,
    ) {
        init {
            require(id.isNotBlank()) { "id must not be blank" }
            require(languages.isNotEmpty()) { "languages must not be empty" }
            languages.forEach(::requireLanguageCode)
            require(tokensPerSecond > 0) { "tokensPerSecond must be positive" }
        }
    }

    /**
     * The id of the model used when none is given: [DEFAULT], and the languages [CzechSpeech] assumes for a model file
     * passed without an entry or languages. This is the one place that chooses the default model.
     */
    public const val DEFAULT_ID: String = "cs"

    /** The models the SDK knows by id. Today only `"cs"`: CzechSpeech q4_0 from Hugging Face, pinned to a revision. */
    public val CATALOG: Map<String, Entry> = listOf(
        Entry(
            id = "cs",
            source = Source(
                url = "https://huggingface.co/moravianai/CzechSpeech/resolve/a4e21bbf9c143d3cfa42488472fed79e9eeba8a9/ggml-czechspeech-q4_0.bin",
                sizeBytes = 473_992_235L,
                sha256 = "04ead6662325354947015bba8604c0e3fbfb9fee598bca1d98340834cb8330a3",
            ),
            languages = listOf("cs"),
            tokensPerSecond = 25,
        ),
    ).associateBy { it.id }

    /** The source of the default model, the catalog entry [DEFAULT_ID]. */
    public val DEFAULT: Source = CATALOG.getValue(DEFAULT_ID).source

    /** The catalog entry for [id]. @throws IllegalArgumentException "unknown model: <id>" for an id not in [CATALOG]. */
    public fun entry(id: String): Entry = requireNotNull(CATALOG[id]) { "unknown model: $id" }

    /** Directory the model is downloaded to. */
    public fun directory(context: Context): File = File(context.noBackupFilesDir, "czechspeech")

    /** Location of the model file, whether or not it has been downloaded. */
    public fun file(context: Context, source: Source = DEFAULT): File = ModelDownloader.target(directory(context), source)

    /** True when the model was downloaded and verified. Cheap: does not hash the file. */
    public fun isDownloaded(context: Context, source: Source = DEFAULT): Boolean =
        ModelDownloader.isComplete(directory(context), source)

    /**
     * Downloads the model unless it is already there and returns its file.
     *
     * Idempotent and resumable: an interrupted or cancelled download continues from where it stopped on the next
     * call. The SHA-256 of the result is verified before the file is published under its final name. Follows
     * redirects (Hugging Face redirects to its CDN), never from HTTPS to HTTP.
     *
     * Before anything is downloaded, the free storage is checked: what is left to download plus 100 MB must be
     * available, so that the model never fills the device. The figure is StorageManager's allocatable bytes, which
     * also count cache the system can clear, or the usable space; when the device cannot tell, the download goes
     * ahead.
     *
     * @param onProgress called on a background thread with bytes done and total, about every MiB.
     * @throws java.io.IOException on network errors, a hash mismatch, or too little free storage (the message says
     *   how much is needed and how much is free; nothing has been downloaded, and a partial download is kept).
     */
    public suspend fun download(
        context: Context,
        source: Source = DEFAULT,
        onProgress: (done: Long, total: Long) -> Unit = { _, _ -> },
    ): File {
        AndroidLog.install()
        return ModelDownloader.download(
            directory(context),
            source,
            freeBytes = { dir -> ModelDownloader.availableBytes({ allocatableBytes(context, dir) }, { dir.usableSpace }) },
            onProgress = onProgress,
        )
    }

    /** StorageManager.getAllocatableBytes for the volume [dir] is on (API 26, the SDK's minSdk); -1 without a manager. */
    private fun allocatableBytes(context: Context, dir: File): Long {
        val storage = context.getSystemService(StorageManager::class.java) ?: return -1
        return storage.getAllocatableBytes(storage.getUuidForPath(dir))
    }

    /** Deletes the model and any partial download. Returns true when a model file existed. */
    public suspend fun delete(context: Context, source: Source = DEFAULT): Boolean =
        ModelDownloader.delete(directory(context), source)

    /**
     * Location of the model file of the catalog model [id], whether or not it has been downloaded.
     *
     * @throws IllegalArgumentException "unknown model: <id>" for an id not in [CATALOG], before any file is touched.
     */
    public fun file(context: Context, id: String): File = file(context, entry(id).source)

    /** [isDownloaded] for the catalog model [id]. @throws IllegalArgumentException for an id not in [CATALOG]. */
    public fun isDownloaded(context: Context, id: String): Boolean = isDownloaded(context, entry(id).source)

    /**
     * [download] for the catalog model [id]: the app passes the id, the source comes from [entry]. Pass the file and
     * the entry to [CzechSpeech], and its languages follow from the entry:
     * `CzechSpeech(context, CzechSpeechModel.download(context, "cs"), CzechSpeechModel.entry("cs"))`.
     *
     * @throws IllegalArgumentException "unknown model: <id>" for an id not in [CATALOG], before anything is
     *   downloaded or created.
     */
    public suspend fun download(
        context: Context,
        id: String,
        onProgress: (done: Long, total: Long) -> Unit = { _, _ -> },
    ): File = download(context, entry(id).source, onProgress)

    /** [delete] for the catalog model [id]. @throws IllegalArgumentException for an id not in [CATALOG]. */
    public suspend fun delete(context: Context, id: String): Boolean = delete(context, entry(id).source)
}
