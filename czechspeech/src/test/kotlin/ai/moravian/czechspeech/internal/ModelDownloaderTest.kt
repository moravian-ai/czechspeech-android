package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechModel
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.Collections
import java.util.Random

/** The download helper against a local HTTP server with a redirect and Range support. */
class ModelDownloaderTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val content = ByteArray(3 * 1024 * 1024 + 123).also { Random(7).nextBytes(it) }
    private val sha = MessageDigest.getInstance("SHA-256").digest(content).joinToString("") { "%02x".format(it) }
    private val requests = Collections.synchronizedList(mutableListOf<String>())
    private var honourRange = true
    private lateinit var server: TinyHttpServer
    private lateinit var base: String

    @Before
    fun start() {
        server = TinyHttpServer { path, headers ->
            when {
                path == "/resolve/model.bin" -> {
                    requests += "redirect"
                    // relative, like a CDN redirect
                    TinyHttpServer.Response(302, mapOf("Location" to "/cdn/model.bin?sig=abc"))
                }
                path.startsWith("/cdn/model.bin") -> serveFile(headers["range"])
                else -> TinyHttpServer.Response(404)
            }
        }
        base = "http://127.0.0.1:${server.port}"
    }

    @After
    fun stop() = server.close()

    private fun serveFile(range: String?): TinyHttpServer.Response {
        requests += "file range=${range ?: "-"}"
        val from = if (honourRange && range != null) Regex("""bytes=(\d+)-""").find(range)!!.groupValues[1].toInt() else 0
        return if (from > 0) {
            TinyHttpServer.Response(206, mapOf("Content-Range" to "bytes $from-${content.size - 1}/${content.size}"), content, from)
        } else {
            TinyHttpServer.Response(200, emptyMap(), content)
        }
    }

    private fun source(sha256: String = sha) =
        CzechSpeechModel.Source("$base/resolve/model.bin", content.size.toLong(), sha256, "model.bin")

    private fun dir() = File(tmp.root, "czechspeech")

    /** A download given the usable space of the temporary folder (CzechSpeechModel passes availableBytes). */
    private suspend fun downloadWithUsableSpace(source: CzechSpeechModel.Source, onProgress: (Long, Long) -> Unit) =
        ModelDownloader.download(dir(), source, { it.usableSpace }, onProgress)

    @Test
    fun downloadsThroughARedirectAndVerifies() = runBlocking {
        var last = 0L
        val file = downloadWithUsableSpace(source()) { done, total ->
            assertEquals(content.size.toLong(), total)
            assertTrue(done >= last)
            last = done
        }
        assertArrayEquals(content, file.readBytes())
        assertEquals(content.size.toLong(), last)
        assertTrue(ModelDownloader.isComplete(dir(), source()))
        assertFalse(File(dir(), "model.bin.part").exists())
        assertEquals(listOf("redirect", "file range=-"), requests.toList())
    }

    @Test
    fun secondCallIsIdempotentAndOffline() = runBlocking {
        downloadWithUsableSpace(source()) { _, _ -> }
        requests.clear()
        val file = downloadWithUsableSpace(source()) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
        assertTrue("no network for a verified file", requests.isEmpty())
    }

    @Test
    fun resumesAPartialDownloadWithRange() = runBlocking {
        dir().mkdirs()
        File(dir(), "model.bin.part").writeBytes(content.copyOf(1_000_000))
        val file = downloadWithUsableSpace(source()) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
        assertEquals(listOf("redirect", "file range=bytes=1000000-"), requests.toList())
    }

    @Test
    fun restartsWhenTheServerIgnoresRange() = runBlocking {
        honourRange = false
        dir().mkdirs()
        File(dir(), "model.bin.part").writeBytes(ByteArray(500_000) { 1 })  // garbage that must not survive
        val file = downloadWithUsableSpace(source()) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
    }

    @Test
    fun hashMismatchFailsAndLeavesNothingBehind() = runBlocking {
        val wrong = "0".repeat(64)
        try {
            downloadWithUsableSpace(source(wrong)) { _, _ -> }
            fail("expected a hash mismatch")
        } catch (e: IOException) {
            assertTrue(e.message!!.contains("SHA-256"))
        }
        assertFalse(File(dir(), "model.bin").exists())
        assertFalse(File(dir(), "model.bin.part").exists())
    }

    @Test
    fun aCorruptedFinalFileIsReplaced() = runBlocking {
        downloadWithUsableSpace(source()) { _, _ -> }
        val file = File(dir(), "model.bin")
        val bytes = file.readBytes().also { it[10] = (it[10] + 1).toByte() }
        file.writeBytes(bytes)                           // changes mtime, so the marker no longer matches
        file.setLastModified(file.lastModified() + 5_000)
        assertFalse(ModelDownloader.isComplete(dir(), source()))
        downloadWithUsableSpace(source()) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
    }

    @Test
    fun wrongSizeOnTheServerIsRejected() = runBlocking {
        val bigger = CzechSpeechModel.Source("$base/resolve/model.bin", content.size + 1L, sha, "model.bin")
        try {
            downloadWithUsableSpace(bigger) { _, _ -> }
            fail("expected a size error")
        } catch (e: IOException) {
            assertTrue(e.message, e.message!!.contains("bytes"))
        }
    }

    @Test
    fun deleteRemovesEverything() = runBlocking {
        downloadWithUsableSpace(source()) { _, _ -> }
        assertTrue(ModelDownloader.delete(dir(), source()))
        assertFalse(ModelDownloader.isComplete(dir(), source()))
        assertTrue(dir().listFiles()!!.isEmpty())
    }

    private val margin = ModelDownloader.FREE_SPACE_MARGIN_BYTES

    private suspend fun downloadFailsForSpace(freeBytes: (File) -> Long): IOException {
        try {
            ModelDownloader.download(dir(), source(), freeBytes) { _, _ -> }
        } catch (e: IOException) {
            return e
        }
        throw AssertionError("expected an IOException for too little free storage")
    }

    @Test
    fun tooLittleFreeStorageFailsBeforeAnyRequestOrWrite() = runBlocking {
        val asked = mutableListOf<File>()
        val e = downloadFailsForSpace { asked += it; content.size + margin - 1 }
        assertTrue(e.message, e.message!!.startsWith("Not enough free storage for the CzechSpeech model"))
        assertTrue(e.message, e.message!!.contains("MB free in ${dir()}"))
        assertEquals("the model's own directory is checked", listOf(dir()), asked)
        assertTrue("no request was made", requests.isEmpty())
        assertFalse(File(dir(), "model.bin.part").exists())
    }

    @Test
    fun theModelSizePlusTheMarginIsEnough() = runBlocking {
        val file = ModelDownloader.download(dir(), source(), freeBytes = { content.size + margin }) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
    }

    @Test
    fun aResumedDownloadNeedsRoomOnlyForWhatIsLeft() = runBlocking {
        dir().mkdirs()
        File(dir(), "model.bin.part").writeBytes(content.copyOf(1_000_000))
        val left = content.size - 1_000_000L
        downloadFailsForSpace { left + margin - 1 }
        assertEquals("the partial download is kept for later", 1_000_000L, File(dir(), "model.bin.part").length())
        assertTrue("no request was made", requests.isEmpty())
        val file = ModelDownloader.download(dir(), source(), freeBytes = { left + margin }) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
    }

    @Test
    fun aVerifiedModelNeedsNoFreeStorage() = runBlocking {
        downloadWithUsableSpace(source()) { _, _ -> }
        val file = ModelDownloader.download(dir(), source(), freeBytes = { 0L }) { _, _ -> }
        assertArrayEquals(content, file.readBytes())
    }

    @Test
    fun anUnknownFreeSpaceIsNotARefusal() = runBlocking {
        // 0 and below mean "unknown" (File.getUsableSpace gives 0 when it cannot tell): download, and let a real
        // ENOSPC, if any, surface as an IOException with the .part kept for a later call.
        for (unknown in listOf(0L, -1L)) {
            ModelDownloader.delete(dir(), source())
            val file = ModelDownloader.download(dir(), source(), freeBytes = { unknown }) { _, _ -> }
            assertArrayEquals(content, file.readBytes())
        }
    }

    @Test
    fun allocatableBytesArePreferredAndUsableSpaceIsTheFallback() {
        val usable = { 1_000L }
        assertEquals(
            "StorageManager's figure counts cache the system can clear", 5_000L,
            ModelDownloader.availableBytes(allocatable = { 5_000L }, usable = usable),
        )
        assertEquals(
            "allocatable bytes unavailable", 1_000L,
            ModelDownloader.availableBytes(allocatable = { throw IOException("no volume for this path") }, usable = usable),
        )
        assertEquals(
            "allocatable bytes 0: unknown, so the usable space", 1_000L,
            ModelDownloader.availableBytes(allocatable = { 0L }, usable = usable),
        )
        assertTrue(
            "neither can tell: unknown",
            ModelDownloader.availableBytes(allocatable = { 0L }, usable = { 0L }) <= 0,
        )
        assertTrue(
            "neither works: unknown, never an exception",
            ModelDownloader.availableBytes(
                allocatable = { throw IOException("x") },
                usable = { throw SecurityException("y") },
            ) <= 0,
        )
    }

    @Test
    fun defaultSourceIsThePinnedHuggingFaceFile() {
        val d = CzechSpeechModel.DEFAULT
        assertTrue(d.url.startsWith("https://huggingface.co/moravianai/CzechSpeech/resolve/a4e21bbf9c143d3cfa42488472fed79e9eeba8a9/"))
        assertEquals(473_992_235L, d.sizeBytes)
        assertEquals("04ead6662325354947015bba8604c0e3fbfb9fee598bca1d98340834cb8330a3", d.sha256)
        assertEquals("ggml-czechspeech-q4_0.bin", d.fileName)
    }
}
