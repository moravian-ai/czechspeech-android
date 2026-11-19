package ai.moravian.czechspeech

import ai.moravian.czechspeech.internal.AndroidLog
import ai.moravian.czechspeech.internal.ModelDownloader
import ai.moravian.czechspeech.internal.requireLanguageCode
import ai.moravian.czechspeech.internal.requireSwitchFits
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
     *
     * @property sha256 SHA-256 of the model as the recogniser loads it, that is of the plaintext when the hosted file
     *   is [encryption]-encrypted.
     * @property encryption How the hosted file is encrypted, or null (the default) for a plain file. The download is
     *   decrypted on the way and the file on the device is the plaintext. This keeps a found URL useless; it does not
     *   protect the model on the device, where the key is in your app.
     */
    public data class Source @JvmOverloads constructor(
        public val url: String,
        public val sizeBytes: Long,
        public val sha256: String,
        public val fileName: String = "ggml-czechspeech-q4_0.bin",
        public val encryption: Encryption? = null,
    ) {
        init {
            require(url.startsWith("https://") || url.startsWith("http://")) { "url must be http(s)" }
            require(sizeBytes > 0) { "sizeBytes must be positive" }
            require(sha256.length == 64 && sha256.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) { "sha256 must be 64 hex digits" }
            require(fileName.isNotBlank() && '/' !in fileName && '\\' !in fileName && fileName != "." && fileName != "..") {
                "fileName must be a plain file name"
            }
        }
    }

    /**
     * AES-256-CTR encryption of a hosted model file: the whole file is encrypted, the ciphertext is as long as the
     * plaintext. The IV is the initial 16-byte counter block, incremented as one 128-bit big-endian integer per 16-byte
     * block, as `openssl enc -aes-256-ctr -K <key> -iv <iv>` and `Cipher.getInstance("AES/CTR/NoPadding")` do. There is
     * no MAC: the plaintext [Source.sha256] is the integrity check. The key is never part of [toString].
     *
     * Never encrypt two different files with the same key and IV (a new version of a model gets a new key, or at least
     * a new IV): the two ciphertexts would reveal the XOR of the two plaintexts and a found URL would no longer be
     * useless. This protects the hosted file only, not the model on the device, where the key is part of your app.
     */
    public class Encryption private constructor(
        @Transient private val key: javax.crypto.spec.SecretKeySpec,
        @Transient private val iv: ByteArray,
    ) {
        public companion object {
            /** @param keyHex 64 hex digits (32 bytes). @param ivHex 32 hex digits (16 bytes). */
            @JvmStatic
            public fun aes256Ctr(keyHex: String, ivHex: String): Encryption {
                require(keyHex.length == 64) { "keyHex must be 64 hex digits" }
                require(ivHex.length == 32) { "ivHex must be 32 hex digits" }
                return Encryption(javax.crypto.spec.SecretKeySpec(parseHex(keyHex, "keyHex"), "AES"), parseHex(ivHex, "ivHex"))
            }

            private fun parseHex(hex: String, name: String): ByteArray {
                require(hex.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) { "$name must be hex digits" }
                return ByteArray(hex.length / 2) { ((Character.digit(hex[2 * it], 16) shl 4) or Character.digit(hex[2 * it + 1], 16)).toByte() }
            }
        }

        /** A cipher whose keystream is at byte [offset] of the file: counter = IV + offset / 16, offset % 16 bytes skipped. */
        @JvmSynthetic
        internal fun cipherAt(offset: Long): javax.crypto.Cipher {
            val counter = java.math.BigInteger(1, iv).add(java.math.BigInteger.valueOf(offset / 16))
                .and(java.math.BigInteger.ONE.shiftLeft(128).subtract(java.math.BigInteger.ONE))
            val raw = counter.toByteArray()
            val block = ByteArray(16)
            val n = minOf(raw.size, 16)
            System.arraycopy(raw, raw.size - n, block, 16 - n, n)
            val cipher = javax.crypto.Cipher.getInstance("AES/CTR/NoPadding")
            cipher.init(
                javax.crypto.Cipher.DECRYPT_MODE,
                key,
                javax.crypto.spec.IvParameterSpec(block),
            )
            val skip = (offset % 16).toInt()
            // the keystream of a block that was started earlier is spent up to here; a provider that buffers
            // (returns less than it was given) cannot be used for a stream, so that is refused, not worked around
            if (skip > 0 && cipher.update(ByteArray(skip))?.size != skip) {
                throw java.io.IOException("The AES/CTR provider ${cipher.provider.name} buffers its output; it cannot decrypt a stream")
            }
            return cipher
        }

        override fun equals(other: Any?): Boolean =
            other is Encryption && java.security.MessageDigest.isEqual(key.encoded, other.key.encoded) && java.security.MessageDigest.isEqual(iv, other.iv)

        override fun hashCode(): Int = 0x41455332

        override fun toString(): String = "Encryption(AES-256-CTR)"
    }

    /**
     * The numbers of the rule that lets a model of several languages leave its first language (the preferred one,
     * which is what the recogniser falls back to) automatically, and only with enough certainty. A language that is not
     * in [thresholds] is never chosen automatically, only when the app asks for it (`listen("sk")`).
     *
     * The numbers belong to one model file: they are measured on it. They are checked at construction.
     *
     * @property thresholds For each language that may be chosen automatically, the share of the recogniser's
     *   probability it must reach, from 0.5 up to but not including 1. Keys are language codes of the model other than
     *   its first language (checked by [Entry] and [CzechSpeech]). The map is copied and read-only.
     * @property minLeaveMs The shortest audio, in ms, from which the first language may be left (0 to 30,000). Streaming
     *   transcription cuts a piece when it reaches 12 s of audio (a cut the endpointer refuses is asked for again 2 s
     *   later), and a piece settles in its language only with a pass of at least this length: keep it well below that
     *   (the default 2,000 is), or a Czech piece never settles and streaming degrades to one transcript per cut.
     * @property minReturnMs The shortest audio, in ms, from which a session in another language returns to the first
     *   language (0 to 30,000); a shorter utterance keeps the session's language.
     */
    public class LanguageSwitch @JvmOverloads public constructor(
        thresholds: Map<String, Float>,
        public val minLeaveMs: Int = 2_000,
        public val minReturnMs: Int = 1_000,
    ) {
        public val thresholds: Map<String, Float> = java.util.Collections.unmodifiableMap(LinkedHashMap(thresholds))

        init {
            this.thresholds.forEach { (code, threshold) ->
                requireLanguageCode(code)
                require(threshold >= 0.5f && threshold < 1f) { "threshold of \"$code\" must be from 0.5 to below 1: $threshold" }
            }
            require(minLeaveMs in 0..30_000) { "minLeaveMs must be from 0 to 30000: $minLeaveMs" }
            require(minReturnMs in 0..30_000) { "minReturnMs must be from 0 to 30000: $minReturnMs" }
        }

        override fun equals(other: Any?): Boolean =
            other is LanguageSwitch && thresholds == other.thresholds && minLeaveMs == other.minLeaveMs && minReturnMs == other.minReturnMs

        override fun hashCode(): Int = (thresholds.hashCode() * 31 + minLeaveMs) * 31 + minReturnMs

        override fun toString(): String = "LanguageSwitch(thresholds=$thresholds, minLeaveMs=$minLeaveMs, minReturnMs=$minReturnMs)"
    }

    /**
     * One model of the [CATALOG].
     *
     * @property id Short name, what [entry] looks up.
     * @property languages The languages the model recognises, two- or three-letter lowercase whisper language codes
     *   (such as `cs`, `pl`, `yue`); the first is the default.
     * @property tokensPerSecond Typical speech rate in tokens per second. The catalog carries it; the recogniser does
     *   not use it yet (there is no token cap on Android).
     * @property languageSwitch The rule for leaving the first language automatically, or null: then a model of several
     *   languages takes the most probable of them. Needs at least two languages and names only the ones after the first.
     */
    // Speaker separation is not part of the SDK yet; an entry will say which models support it once it is.
    public data class Entry @JvmOverloads constructor(
        public val id: String,
        public val source: Source,
        public val languages: List<String>,
        public val tokensPerSecond: Int,
        public val languageSwitch: LanguageSwitch? = null,
    ) {
        init {
            require(id.isNotBlank()) { "id must not be blank" }
            require(languages.isNotEmpty()) { "languages must not be empty" }
            languages.forEach(::requireLanguageCode)
            require(tokensPerSecond > 0) { "tokensPerSecond must be positive" }
            requireSwitchFits(languages, languageSwitch)
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
