package ai.moravian.czechspeech

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.file.Files
import java.nio.file.attribute.PosixFilePermission
import java.security.MessageDigest
import java.util.Random

/** scripts/encrypt-model.sh against openssl and the SDK's own decryption. Skipped where bash or openssl are missing. */
class EncryptModelScriptTest {
    @get:Rule
    val tmp = TemporaryFolder()

    private val script = File("../scripts/encrypt-model.sh").absoluteFile
    private val data = ByteArray(100_003).also { Random(5).nextBytes(it) }

    private class Result(val code: Int, val out: ByteArray, val err: String)

    private fun run(vararg args: String, stdin: ByteArray? = null, env: Map<String, String> = emptyMap()): Result {
        val pb = ProcessBuilder(listOf("bash", script.path) + args)
        pb.environment().putAll(env)
        val p = pb.start()
        val errBuf = StringBuilder()
        val errThread = Thread { errBuf.append(p.errorStream.readBytes().toString(Charsets.UTF_8)) }.also { it.start() }
        val outBox = arrayOfNulls<ByteArray>(1)
        val outThread = Thread { outBox[0] = p.inputStream.readBytes() }.also { it.start() }
        try {
            p.outputStream.use { if (stdin != null) it.write(stdin) }
        } catch (e: java.io.IOException) {
            // the script refused before reading its input
        }
        val code = p.waitFor()
        outThread.join()
        errThread.join()
        return Result(code, outBox[0]!!, errBuf.toString())
    }

    private fun tools() = try {
        run("--help-not-an-option").code != 127 && ProcessBuilder("openssl", "version").start().waitFor() == 0
    } catch (e: Exception) {
        false
    }

    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    private fun decrypt(file: File, keyFile: File): ByteArray {
        val lines = keyFile.readLines().associate { it.substringBefore('=') to it.substringAfter('=') }
        val p = ProcessBuilder("openssl", "enc", "-d", "-aes-256-ctr", "-K", lines.getValue("key"), "-iv", lines.getValue("iv"), "-in", file.path).start()
        val bytes = p.inputStream.readBytes()
        assertEquals(0, p.waitFor())
        return bytes
    }

    @Test
    fun filesRoundTripThroughOpensslAndTheKeyStaysInTheKeyFile() {
        assumeTrue(tools())
        val input = tmp.newFile("in.bin").also { it.writeBytes(data) }
        val output = File(tmp.root, "out.bin")
        val keyFile = File(tmp.root, "key.txt")
        val r = run(input.path, output.path, keyFile.path)
        assertEquals(r.err, 0, r.code)
        assertEquals(data.size.toLong(), output.length())
        assertArrayEquals(data, decrypt(output, keyFile))
        val text = r.out.toString(Charsets.UTF_8)
        assertTrue(text, text.contains("plaintext sha256:  ${sha(data)}"))
        assertTrue(text, text.contains("ciphertext sha256: ${sha(output.readBytes())}"))
        assertTrue(text, text.contains("size:              ${data.size}"))
        val key = keyFile.readLines().first { it.startsWith("key=") }.substringAfter('=')
        assertEquals(64, key.length)
        assertFalse("the key must not be printed", text.contains(key) || r.err.contains(key))
        assertEquals(
            setOf(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE),
            Files.getPosixFilePermissions(keyFile.toPath()),
        )
        // the SDK reads what the script wrote
        val e = CzechSpeechModel.Encryption.aes256Ctr(key, keyFile.readLines().first { it.startsWith("iv=") }.substringAfter('='))
        assertArrayEquals(data, e.cipherAt(0).doFinal(output.readBytes()))
    }

    @Test
    fun worksAsAPipeAndTakesAGivenKey() {
        assumeTrue(tools())
        val keyFile = File(tmp.root, "key.txt")
        val key = "11".repeat(32)
        val iv = "22".repeat(16)
        val r = run("--print-key", "--reuse-iv-i-know", "-", "-", keyFile.path, stdin = data, env = mapOf("ENC_KEY" to key, "ENC_IV" to iv))
        assertEquals(r.err, 0, r.code)
        val cipher = r.out
        assertEquals(data.size, cipher.size)
        assertTrue(r.err, r.err.contains("plaintext sha256:  ${sha(data)}"))
        assertTrue(r.err, r.err.contains("ciphertext sha256: ${sha(cipher)}"))
        assertTrue("--print-key prints it, to stderr in pipe mode", r.err.contains("key=$key"))
        assertFalse(String(cipher, Charsets.ISO_8859_1).contains("key="))
        assertEquals(listOf("key=$key", "iv=$iv"), keyFile.readLines())
        assertArrayEquals(data, CzechSpeechModel.Encryption.aes256Ctr(key, iv).cipherAt(0).doFinal(cipher))
    }

    @Test
    fun badKeyFromTheEnvironmentIsRefused() {
        assumeTrue(tools())
        val r = run("-", "-", File(tmp.root, "k").path, stdin = data, env = mapOf("ENC_KEY" to "abc"))
        assertEquals(2, r.code)
        assertEquals(0, r.out.size)
    }

    @Test
    fun aBrokenOffInputWithTheKnownHashFailsAndLeavesNothing() {
        assumeTrue(tools())
        // pipe: the upstream command died after 1000 bytes, the script sees a shorter file
        val keyFile = File(tmp.root, "key.txt")
        val r = run("--expect-sha256", sha(data), "--expect-size", data.size.toString(), "-", "-", keyFile.path, stdin = data.copyOf(1000))
        assertEquals(r.err, 3, r.code)
        assertTrue(r.err, r.err.contains("FAILED"))
        assertFalse("no key file for a failed run", keyFile.exists())
        assertFalse("no hash is printed as the model's", r.err.contains("ciphertext sha256"))
        // file: the partial output is removed too
        val input = tmp.newFile("short.bin").also { it.writeBytes(data.copyOf(1000)) }
        val output = File(tmp.root, "short.enc")
        val r2 = run("--expect-size", data.size.toString(), input.path, output.path, keyFile.path)
        assertEquals(r2.err, 3, r2.code)
        assertFalse(output.exists())
        assertFalse(File(tmp.root, "short.enc.part").exists())
        assertFalse(keyFile.exists())
    }

    @Test
    fun theKnownHashAndSizeAreAccepted() {
        assumeTrue(tools())
        val keyFile = File(tmp.root, "key.txt")
        val r = run("--expect-sha256", sha(data).uppercase(), "--expect-size", data.size.toString(), "-", "-", keyFile.path, stdin = data)
        assertEquals(r.err, 0, r.code)
        assertEquals(data.size, r.out.size)
        assertTrue(keyFile.exists())
    }

    @Test
    fun aMissingInputFailsWithoutLeavingFiles() {
        assumeTrue(tools())
        val output = File(tmp.root, "out.enc")
        val keyFile = File(tmp.root, "key.txt")
        val r = run(File(tmp.root, "nope.bin").path, output.path, keyFile.path)
        assertTrue(r.err, r.code != 0)
        assertFalse(output.exists())
        assertFalse(File(tmp.root, "out.enc.part").exists())
        assertFalse(keyFile.exists())
    }

    @Test
    fun anExistingKeyFileOrOutputIsNeverOverwrittenWithoutForce() {
        assumeTrue(tools())
        val input = tmp.newFile("in.bin").also { it.writeBytes(data) }
        val keyFile = tmp.newFile("key.txt").also { it.writeText("old key") }
        val r = run(input.path, File(tmp.root, "o.enc").path, keyFile.path)
        assertEquals(2, r.code)
        assertEquals("old key", keyFile.readText())
        val output = tmp.newFile("o2.enc").also { it.writeText("old") }
        val r2 = run(input.path, output.path, File(tmp.root, "k2.txt").path)
        assertEquals(2, r2.code)
        assertEquals("old", output.readText())
        val r3 = run("--force", input.path, output.path, keyFile.path)
        assertEquals(r3.err, 0, r3.code)
        assertEquals(data.size.toLong(), output.length())
        assertTrue(keyFile.readText().startsWith("key="))
        assertEquals(2, run(input.path, input.path, File(tmp.root, "k3.txt").path).code)
    }

    @Test
    fun anIvFromTheEnvironmentNeedsTheExplicitFlag() {
        assumeTrue(tools())
        val r = run("-", "-", File(tmp.root, "k.txt").path, stdin = data, env = mapOf("ENC_IV" to "22".repeat(16)))
        assertEquals(2, r.code)
        assertTrue(r.err, r.err.contains("--reuse-iv-i-know"))
        assertEquals(0, r.out.size)
        assertFalse(File(tmp.root, "k.txt").exists())
    }

    @Test
    fun everyRunDrawsAFreshIv() {
        assumeTrue(tools())
        val ivs = (1..2).map { i ->
            val keyFile = File(tmp.root, "k$i.txt")
            assertEquals(0, run("-", "-", keyFile.path, stdin = data).code)
            keyFile.readLines().first { it.startsWith("iv=") }
        }
        assertNotEquals(ivs[0], ivs[1])
    }
}
