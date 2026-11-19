package ai.moravian.czechspeech

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.fail
import org.junit.Test
import java.util.Random
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/** The AES-256-CTR option of a hosted model file: known answer, reference round trip, no key in text. */
class EncryptionTest {
    private val key = "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"
    private val iv = "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"

    private fun hex(s: String) = ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }

    private fun reference(keyHex: String, ivHex: String, data: ByteArray): ByteArray {
        val c = Cipher.getInstance("AES/CTR/NoPadding")
        c.init(Cipher.ENCRYPT_MODE, SecretKeySpec(hex(keyHex), "AES"), IvParameterSpec(hex(ivHex)))
        return c.doFinal(data)
    }

    @Test
    fun nistSp80038aF55KnownAnswer() {
        val plain = hex(
            "6bc1bee22e409f96e93d7e117393172a" + "ae2d8a571e03ac9c9eb76fac45af8e51" +
                "30c81c46a35ce411e5fbc1191a0a52ef" + "f69f2445df4f9b17ad2b417be66c3710",
        )
        val cipher = hex(
            "601ec313775789a5b7a7f504bbf3d228" + "f443e3ca4d62b59aca84e990cacaf5c5" +
                "2b0930daa23de94ce87017ba2d84988d" + "dfc9c58db67aada613c2dd08457941a6",
        )
        assertArrayEquals(plain, CzechSpeechModel.Encryption.aes256Ctr(key, iv).cipherAt(0).doFinal(cipher))
    }

    @Test
    fun cipherAtAnyOffsetContinuesTheKeystream() {
        val data = ByteArray(200).also { Random(3).nextBytes(it) }
        for (ivHex in listOf(iv, "00000000000000000000000000000000", "ffffffffffffffffffffffffffffffff", "0000000000000000ffffffffffffffff")) {
            val enc = reference(key, ivHex, data)
            val e = CzechSpeechModel.Encryption.aes256Ctr(key, ivHex)
            for (offset in listOf(0, 1, 15, 16, 17, 31, 32, 100, 199)) {
                val tail = e.cipherAt(offset.toLong()).doFinal(enc, offset, enc.size - offset)
                assertArrayEquals("iv $ivHex offset $offset", data.copyOfRange(offset, data.size), tail)
            }
        }
    }

    @Test
    fun textNeverContainsTheKey() {
        val e = CzechSpeechModel.Encryption.aes256Ctr(key, iv)
        assertEquals("Encryption(AES-256-CTR)", e.toString())
        val source = CzechSpeechModel.Source("https://x.example/m.bin", 10, "0".repeat(64), "m.bin", e)
        for (text in listOf(e.toString(), source.toString(), CzechSpeechModel.Entry("m", source, listOf("cs"), 25).toString())) {
            assertFalse(text, text.contains(key.take(16), ignoreCase = true))
            assertFalse(text, text.contains(iv, ignoreCase = true))
        }
        assertEquals(e.hashCode(), CzechSpeechModel.Encryption.aes256Ctr(key, iv).hashCode())
    }

    @Test
    fun equalityFollowsKeyAndIv() {
        val e = CzechSpeechModel.Encryption.aes256Ctr(key, iv)
        assertEquals(e, CzechSpeechModel.Encryption.aes256Ctr(key.uppercase(), iv))
        assertNotEquals(e, CzechSpeechModel.Encryption.aes256Ctr(key.replaceRange(0, 2, "00"), iv))
        assertNotEquals(e, CzechSpeechModel.Encryption.aes256Ctr(key, iv.replaceRange(0, 2, "00")))
    }

    @Test
    fun invalidHexIsRejected() {
        fun rejects(k: String, i: String) {
            try {
                CzechSpeechModel.Encryption.aes256Ctr(k, i)
                fail("accepted")
            } catch (e: IllegalArgumentException) {
                // expected
            }
        }
        rejects(key.dropLast(2), iv)
        rejects(key + "00", iv)
        rejects(key, iv.dropLast(2))
        rejects(key, iv + "00")
        rejects("z".repeat(64), iv)
        rejects(key, "g".repeat(32))
        rejects("", "")
        assertNull(CzechSpeechModel.Source("https://x.example/m.bin", 1, "0".repeat(64)).encryption)
    }

    /** A cipher that, like a buffering provider, returns one byte less than it was given. */
    private class BufferingSpi : javax.crypto.CipherSpi() {
        override fun engineSetMode(mode: String?) = Unit
        override fun engineSetPadding(padding: String?) = Unit
        override fun engineGetBlockSize() = 16
        override fun engineGetOutputSize(inputLen: Int) = inputLen
        override fun engineGetIV(): ByteArray? = null
        override fun engineGetParameters(): java.security.AlgorithmParameters? = null
        override fun engineInit(opmode: Int, key: java.security.Key?, random: java.security.SecureRandom?) = Unit
        override fun engineInit(opmode: Int, key: java.security.Key?, params: java.security.spec.AlgorithmParameterSpec?, random: java.security.SecureRandom?) = Unit
        override fun engineInit(opmode: Int, key: java.security.Key?, params: java.security.AlgorithmParameters?, random: java.security.SecureRandom?) = Unit
        override fun engineUpdate(input: ByteArray, inputOffset: Int, inputLen: Int): ByteArray = input.copyOfRange(inputOffset, inputOffset + inputLen - 1)
        override fun engineUpdate(input: ByteArray, inputOffset: Int, inputLen: Int, output: ByteArray, outputOffset: Int): Int {
            System.arraycopy(input, inputOffset, output, outputOffset, inputLen - 1)
            return inputLen - 1
        }
        override fun engineDoFinal(input: ByteArray?, inputOffset: Int, inputLen: Int): ByteArray = ByteArray(0)
        override fun engineDoFinal(input: ByteArray?, inputOffset: Int, inputLen: Int, output: ByteArray?, outputOffset: Int) = 0
    }

    @Test
    fun aBufferingProviderIsRefusedNotSilentlyShifted() {
        val fake = object : Cipher(BufferingSpi(), Cipher.getInstance("AES/CTR/NoPadding").provider, "AES/CTR/NoPadding") {}
        fake.init(Cipher.DECRYPT_MODE, SecretKeySpec(ByteArray(32), "AES"))
        val buffer = ByteArray(100)
        try {
            ai.moravian.czechspeech.internal.ModelDownloader.decryptInto(fake, buffer, 100, ByteArray(100))
            fail("accepted")
        } catch (e: java.io.IOException) {
            assertEquals(true, e.message!!.contains("buffers its output"))
        }
        // and the real cipher passes the check
        val real = CzechSpeechModel.Encryption.aes256Ctr(key, iv).cipherAt(17)
        val out = ByteArray(100)
        ai.moravian.czechspeech.internal.ModelDownloader.decryptInto(real, buffer, 100, out)
    }
}
