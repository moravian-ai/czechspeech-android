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
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/** The download of an AES-256-CTR encrypted file: decrypted on the way, resumed at any offset. */
class EncryptedDownloadTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val keyHex = "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"
    private val ivHex = "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
    private val plain = ByteArray(3 * 1024 * 1024 + 123).also { Random(11).nextBytes(it) }
    private val sha = MessageDigest.getInstance("SHA-256").digest(plain).joinToString("") { "%02x".format(it) }
    private val cipherText = Cipher.getInstance("AES/CTR/NoPadding").run {
        init(Cipher.ENCRYPT_MODE, SecretKeySpec(hex(keyHex), "AES"), IvParameterSpec(hex(ivHex)))
        doFinal(plain)
    }
    private val requests = Collections.synchronizedList(mutableListOf<String>())
    private var honourRange = true
    private lateinit var server: TinyHttpServer

    private fun hex(s: String) = ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }

    @Before
    fun start() {
        server = TinyHttpServer { _, headers ->
            val range = headers["range"]
            requests += "range=${range ?: "-"}"
            val from = if (honourRange && range != null) Regex("""bytes=(\d+)-""").find(range)!!.groupValues[1].toInt() else 0
            if (from > 0) {
                TinyHttpServer.Response(206, mapOf("Content-Range" to "bytes $from-${cipherText.size - 1}/${cipherText.size}"), cipherText, from)
            } else {
                TinyHttpServer.Response(200, emptyMap(), cipherText)
            }
        }
    }

    @After
    fun stop() = server.close()

    private fun source(key: String = keyHex) = CzechSpeechModel.Source(
        "http://127.0.0.1:${server.port}/m.bin", plain.size.toLong(), sha, "model.bin",
        CzechSpeechModel.Encryption.aes256Ctr(key, ivHex),
    )

    private fun dir() = File(tmp.root, "czechspeech")

    private fun download(source: CzechSpeechModel.Source) = runBlocking {
        ModelDownloader.download(dir(), source, { it.usableSpace }) { _, _ -> }
    }

    /** The .part holds plaintext of [have] bytes, as an interrupted download leaves it. */
    private fun partial(have: Int) {
        dir().mkdirs()
        File(dir(), "model.bin.part").writeBytes(plain.copyOf(have))
    }

    @Test
    fun theFileOnTheDeviceIsThePlaintextOfAMultiChunkDownload() {
        val file = download(source())
        assertArrayEquals(plain, file.readBytes())
        assertEquals("one request, no range", listOf("range=-"), requests.toList())
        assertFalse("the server file really is encrypted", cipherText.copyOf(1000).contentEquals(plain.copyOf(1000)))
    }

    @Test
    fun resumesAtAnyOffsetWithRange() {
        val chunk = 256 * 1024
        for (have in listOf(1, 15, 16, 17, 31, 1_000_001, chunk, chunk + 1, 2 * chunk - 1)) {
            File(dir(), "model.bin").delete()
            File(dir(), "model.bin.verified").delete()
            requests.clear()
            partial(have)
            val file = download(source())
            assertArrayEquals("resume at $have", plain, file.readBytes())
            assertEquals(listOf("range=bytes=$have-"), requests.toList())
        }
    }

    @Test
    fun serverIgnoringRangeRestartsFromTheBeginning() {
        honourRange = false
        partial(17)
        val file = download(source())
        assertArrayEquals(plain, file.readBytes())
        assertEquals(listOf("range=bytes=17-"), requests.toList())
    }

    @Test
    fun aWrongKeyIsAShaMismatchAndNothingIsKept() {
        val wrong = keyHex.replaceRange(0, 2, "00")
        try {
            download(source(key = wrong))
            fail("accepted")
        } catch (e: IOException) {
            assertTrue(e.message, e.message!!.contains("SHA-256 mismatch"))
            assertFalse(e.message!!.contains(keyHex.take(16)))
            assertFalse(e.message!!.contains(wrong.take(16)))
        }
        assertEquals(emptyList<String>(), dir().list()!!.toList())
    }
}
