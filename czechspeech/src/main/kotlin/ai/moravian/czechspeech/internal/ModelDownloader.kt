package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Resumable, verified model download over [HttpURLConnection] (no other HTTP stack). The only network code in the
 * SDK. Downloads to `<name>.part`, resumes it with a Range request, checks size and SHA-256, then renames it.
 * A small `<name>.verified` marker (hash, size, mtime) lets later calls return without hashing 0.5 GB again.
 */
internal object ModelDownloader {
    const val USER_AGENT = "czechspeech-android/0.2.0"
    private const val MAX_REDIRECTS = 10
    private const val BUFFER_BYTES = 256 * 1024
    private const val PROGRESS_STEP_BYTES = 1L shl 20

    /** Storage left free after the download, so the model never fills the device to the last byte. */
    const val FREE_SPACE_MARGIN_BYTES = 100_000_000L

    private val mutex = Mutex()

    fun target(dir: File, source: CzechSpeechModel.Source) = File(dir, source.fileName)

    /** True when the file is complete and was verified earlier (no hashing). */
    fun isComplete(dir: File, source: CzechSpeechModel.Source): Boolean {
        val file = target(dir, source)
        return file.isFile && file.length() == source.sizeBytes && markerMatches(dir, source, file)
    }

    suspend fun delete(dir: File, source: CzechSpeechModel.Source): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val file = target(dir, source)
            val existed = file.exists()
            listOf(file, partFile(dir, source), markerFile(dir, source)).forEach { it.delete() }
            existed
        }
    }

    /**
     * Downloads [source] into [dir] unless it is there already. [freeBytes] gives the bytes the app may write in
     * [dir], 0 or less when unknown: [availableBytes] on Android, which asks StorageManager first.
     */
    suspend fun download(
        dir: File,
        source: CzechSpeechModel.Source,
        freeBytes: (File) -> Long,
        onProgress: (done: Long, total: Long) -> Unit,
    ): File = withContext(Dispatchers.IO) {
        mutex.withLock {
            val total = source.sizeBytes
            val expected = source.sha256.lowercase()
            val file = target(dir, source)
            val part = partFile(dir, source)
            val marker = markerFile(dir, source)
            if (!dir.isDirectory && !dir.mkdirs()) throw IOException("Cannot create $dir")

            if (file.exists()) {
                if (file.length() == total && markerMatches(dir, source, file)) {
                    onProgress(total, total)
                    return@withLock file
                }
                if (file.length() == total && sha256Hex(file) == expected) {
                    writeMarker(dir, source, file)
                    onProgress(total, total)
                    return@withLock file
                }
                file.delete()
                marker.delete()
            }

            var have = if (part.isFile) part.length() else 0L
            if (have > total) {
                part.delete()
                have = 0L
            }
            val digest = MessageDigest.getInstance("SHA-256")
            if (have > 0) feed(part, digest)
            onProgress(have, total)

            if (have < total) {
                // Before any request: the rest of the file must fit with the margin to spare. Should the server
                // ignore the Range request, the .part is rewritten from the start, which frees what it held.
                requireFreeSpace(dir, total - have, freeBytes(dir))
                val conn = open(source.url, have)
                try {
                    val code = conn.responseCode
                    val append = when {
                        code == HttpURLConnection.HTTP_PARTIAL && have > 0 -> {
                            checkContentRange(conn.getHeaderField("Content-Range"), have, total)
                            true
                        }
                        code == HttpURLConnection.HTTP_OK -> {
                            val length = conn.getHeaderFieldLong("Content-Length", -1L)
                            if (length >= 0 && length != total) {
                                throw IOException("Server offers $length bytes, expected $total")
                            }
                            if (have > 0) CsLog.i("download: server ignored the range request, restarting")
                            digest.reset()
                            have = 0L
                            false
                        }
                        code == 416 -> {
                            part.delete()
                            throw IOException("Server rejected the resume request (HTTP 416); try again")
                        }
                        else -> throw IOException("HTTP $code from ${conn.url.host}")
                    }
                    val buffer = ByteArray(BUFFER_BYTES)
                    var reported = have
                    conn.inputStream.use { input ->
                        FileOutputStream(part, append).use { out ->
                            while (true) {
                                ensureActive()
                                val n = input.read(buffer)
                                if (n < 0) break
                                if (have + n > total) throw IOException("Server sent more than $total bytes")
                                out.write(buffer, 0, n)
                                digest.update(buffer, 0, n)
                                have += n
                                if (have - reported >= PROGRESS_STEP_BYTES) {
                                    reported = have
                                    onProgress(have, total)
                                }
                            }
                            out.fd.sync()
                        }
                    }
                } finally {
                    conn.disconnect()
                }
            }
            if (have != total) throw IOException("Download incomplete: $have of $total bytes; call download() again to resume")

            val actual = digest.digest().toHex()
            if (actual != expected) {
                part.delete()
                throw IOException("SHA-256 mismatch: got $actual, expected $expected")
            }
            if (!part.renameTo(file)) throw IOException("Cannot rename ${part.name} to ${file.name}")
            writeMarker(dir, source, file)
            onProgress(total, total)
            CsLog.i("download: ${file.name} verified ($total bytes)")
            file
        }
    }

    /** GET with an optional Range, following redirects by hand so the Range header survives them. */
    private fun open(url: String, from: Long): HttpURLConnection {
        var current = URL(url)
        repeat(MAX_REDIRECTS + 1) {
            val conn = current.openConnection() as HttpURLConnection
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 30_000
            conn.readTimeout = 60_000
            conn.setRequestProperty("User-Agent", USER_AGENT)
            conn.setRequestProperty("Accept-Encoding", "identity")
            if (from > 0) conn.setRequestProperty("Range", "bytes=$from-")
            when (conn.responseCode) {
                301, 302, 303, 307, 308 -> {
                    val location = conn.getHeaderField("Location")
                    conn.disconnect()
                    if (location.isNullOrBlank()) throw IOException("Redirect without Location from ${current.host}")
                    val next = URL(current, location)
                    if (next.protocol != "https" && (current.protocol == "https" || next.protocol != "http")) {
                        throw IOException("Refusing redirect from ${current.protocol} to ${next.protocol}")
                    }
                    current = next
                }
                else -> return conn
            }
        }
        throw IOException("Too many redirects")
    }

    /**
     * The bytes an app may write in a directory: [allocatable] (StorageManager.getAllocatableBytes, which also counts
     * cache the system can clear) when it gives a figure, else [usable] (File.getUsableSpace). 0 or less: unknown.
     * Never throws.
     */
    fun availableBytes(allocatable: () -> Long, usable: () -> Long): Long {
        val fromStorageManager = try {
            allocatable()
        } catch (e: Exception) {
            CsLog.w("download: allocatable bytes unavailable", e)
            -1L
        }
        if (fromStorageManager > 0) return fromStorageManager
        return try {
            usable()
        } catch (e: Exception) {
            CsLog.w("download: usable space unavailable", e)
            -1L
        }
    }

    private fun requireFreeSpace(dir: File, left: Long, free: Long) {
        if (free <= 0) {
            // Unknown, not none: refusing would block every download on a device that cannot tell. A real lack of
            // space then shows as an IOException while writing, with the .part kept for a later call.
            CsLog.w("download: free storage unknown, not checked")
            return
        }
        val needed = left + FREE_SPACE_MARGIN_BYTES
        if (free >= needed) return
        fun mbUp(bytes: Long) = (bytes + 999_999) / 1_000_000
        throw IOException(
            "Not enough free storage for the CzechSpeech model: ${mbUp(needed)} MB needed (${mbUp(left)} MB to " +
                "download and ${FREE_SPACE_MARGIN_BYTES / 1_000_000} MB to leave free), ${free / 1_000_000} MB " +
                "free in $dir",
        )
    }

    private fun checkContentRange(header: String?, from: Long, total: Long) {
        // "bytes 1000-473992234/473992235"
        val match = Regex("""bytes\s+(\d+)-(\d+)/(\d+|\*)""").find(header.orEmpty())
            ?: throw IOException("Missing or invalid Content-Range: $header")
        val start = match.groupValues[1].toLong()
        val size = match.groupValues[3].toLongOrNull()
        if (start != from) throw IOException("Server resumed at $start, expected $from")
        if (size != null && size != total) throw IOException("Server file has $size bytes, expected $total")
    }

    private fun feed(file: File, digest: MessageDigest) {
        val buffer = ByteArray(BUFFER_BYTES)
        file.inputStream().use { input ->
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
    }

    private fun sha256Hex(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        feed(file, digest)
        return digest.digest().toHex()
    }

    private fun partFile(dir: File, source: CzechSpeechModel.Source) = File(dir, source.fileName + ".part")

    private fun markerFile(dir: File, source: CzechSpeechModel.Source) = File(dir, source.fileName + ".verified")

    private fun markerText(source: CzechSpeechModel.Source, file: File) =
        "sha256=${source.sha256.lowercase()}\nsize=${source.sizeBytes}\nmtime=${file.lastModified()}\n"

    private fun markerMatches(dir: File, source: CzechSpeechModel.Source, file: File): Boolean = try {
        markerFile(dir, source).readText() == markerText(source, file)
    } catch (e: IOException) {
        false
    }

    private fun writeMarker(dir: File, source: CzechSpeechModel.Source, file: File) {
        markerFile(dir, source).writeText(markerText(source, file))
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }
}
