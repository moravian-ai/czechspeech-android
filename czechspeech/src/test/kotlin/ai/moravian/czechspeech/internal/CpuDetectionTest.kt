package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeechConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CpuDetectionTest {
    // Pixel 9a class (Tensor G4: Cortex-X4 / A720 / A520), abbreviated.
    private val armv9 = """
        processor	: 0
        BogoMIPS	: 49.15
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
        CPU implementer	: 0x41
        CPU variant	: 0x2
        CPU part	: 0xd80

        processor	: 7
        BogoMIPS	: 49.15
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti
        CPU implementer	: 0x41
        CPU variant	: 0x0
        CPU part	: 0xd82
    """.trimIndent()

    // Cortex-A55/A75 class: dotprod and fp16, no i8mm.
    private val armv82 = """
        processor	: 0
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp
    """.trimIndent()

    // Cortex-A53 class: plain ARMv8.0.
    private val armv80 = """
        processor	: 0
        Features	: fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid
    """.trimIndent()

    @Test
    fun featuresAreParsed() {
        val f = CpuFeatures.parse(armv9)
        assertTrue(f.dotprod && f.fp16 && f.i8mm)
        val g = CpuFeatures.parse(armv82)
        assertTrue(g.dotprod && g.fp16)
        assertFalse(g.i8mm)
        val h = CpuFeatures.parse(armv80)
        assertFalse(h.dotprod || h.fp16 || h.i8mm)
    }

    @Test
    fun featureMissingOnOneCoreDoesNotCount() {
        val mixed = "Features\t: fp asimd asimddp fphp asimdhp i8mm\nFeatures\t: fp asimd asimddp fphp asimdhp\n"
        assertFalse(CpuFeatures.parse(mixed).i8mm)
        assertTrue(CpuFeatures.parse(mixed).dotprod)
    }

    @Test
    fun fp16NeedsBothScalarAndVector() {
        assertFalse(CpuFeatures.parse("Features\t: fp asimd asimddp asimdhp i8mm").fp16)
    }

    @Test
    fun androidArmVariantsBestFirst() {
        assertEquals(
            listOf(NativeLoader.ARM_I8MM, NativeLoader.ARM_DOTPROD, NativeLoader.ARM_BASE),
            NativeLoader.candidates(android = true, arch = "aarch64") { armv9 },
        )
        assertEquals(
            listOf(NativeLoader.ARM_DOTPROD, NativeLoader.ARM_BASE),
            NativeLoader.candidates(android = true, arch = "aarch64") { armv82 },
        )
        assertEquals(listOf(NativeLoader.ARM_BASE), NativeLoader.candidates(android = true, arch = "aarch64") { armv80 })
        assertEquals(listOf(NativeLoader.ARM_BASE), NativeLoader.candidates(android = true, arch = "aarch64") { null })
    }

    @Test
    fun otherPlatforms() {
        assertEquals(listOf(NativeLoader.X86_64), NativeLoader.candidates(android = true, arch = "x86_64") { null })
        assertEquals(emptyList<String>(), NativeLoader.candidates(android = true, arch = "armv7l") { null })
        assertEquals(listOf(NativeLoader.HOST), NativeLoader.candidates(android = false, arch = "amd64") { null })
        assertFalse(NativeLoader.isAndroid())
    }

    @Test
    fun bigCoresByFrequency() {
        val lines = (0..7).map { "processor\t: $it" }
        val freq = mapOf(0 to 1_950_000L, 1 to 1_950_000L, 2 to 1_950_000L, 3 to 1_950_000L,
            4 to 2_600_000L, 5 to 2_600_000L, 6 to 2_600_000L, 7 to 3_100_000L)
        assertEquals(4, CpuThreads.highPerfCoreCount(lines) { freq.getValue(it) })
        assertEquals(8, CpuThreads.highPerfCoreCount(lines) { 2_000_000L })   // all equal: all cores
    }

    @Test
    fun bigCoresByVariantWhenFrequenciesAreUnreadable() {
        val lines = armv9.lines()
        assertEquals(1, CpuThreads.highPerfCoreCount(lines) { error("no cpufreq") })
    }

    @Test
    fun configDefaultsAreTheValidatedOnes() {
        val c = CzechSpeechConfig()
        assertEquals(0.25f, c.threshold)
        assertEquals(0.10f, c.negThreshold)
        assertEquals(600, c.minSilenceMs)
        assertEquals(300, c.preRollMs)
        assertEquals(250, c.minSpeechMs)
        assertEquals(28_000, c.maxUtteranceMs)
    }

    @Test(expected = IllegalArgumentException::class)
    fun configRejectsNegThresholdAboveThreshold() {
        CzechSpeechConfig(threshold = 0.3f, negThreshold = 0.4f)
    }
}
