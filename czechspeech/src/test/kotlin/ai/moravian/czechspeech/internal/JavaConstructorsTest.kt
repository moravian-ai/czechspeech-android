package ai.moravian.czechspeech.internal

import ai.moravian.czechspeech.CzechSpeech
import ai.moravian.czechspeech.CzechSpeechConfig
import ai.moravian.czechspeech.CzechSpeechModel
import ai.moravian.czechspeech.EarlyStart
import ai.moravian.czechspeech.PreviewConfig
import ai.moravian.czechspeech.SentenceJoining
import ai.moravian.czechspeech.SpeechPreview
import ai.moravian.czechspeech.StreamingConfig
import ai.moravian.czechspeech.VariantSelection
import android.content.Context
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The classes that gained a parameter keep the constructors Java code was compiled against (Kotlin callers use the
 * defaults; a Java caller, or Kotlin code compiled against the older class, calls a constructor by its exact types).
 */
class JavaConstructorsTest {
    private fun has(type: Class<*>, vararg parameters: Class<*>) =
        type.constructors.any { it.parameterTypes.toList() == parameters.toList() }

    private val int = Int::class.javaPrimitiveType!!
    private val long = Long::class.javaPrimitiveType!!
    private val float = Float::class.javaPrimitiveType!!
    private val boolean = Boolean::class.javaPrimitiveType!!

    @Test
    fun speechPreviewKeepsTheConstructorWithoutTheStableLength() {
        assertTrue(has(SpeechPreview::class.java, long, String::class.java))
        assertTrue(has(SpeechPreview::class.java, long, String::class.java, int))
        val old = SpeechPreview::class.java.getConstructor(long, String::class.java).newInstance(480L, "dobrý den")
        assertEquals(SpeechPreview(480, "dobrý den", 0), old)
    }

    @Test
    fun previewConfigKeepsTheConstructorsWithoutFromPauses() {
        assertTrue(has(PreviewConfig::class.java))
        assertTrue(has(PreviewConfig::class.java, int, int))
        assertTrue(has(PreviewConfig::class.java, int, int, boolean))
        assertEquals(PreviewConfig(), PreviewConfig::class.java.getConstructor(int, int).newInstance(5_000, 5_000))
    }

    @Test
    fun czechSpeechConfigKeepsTheConstructorsWithoutItsLastParameters() {
        val upToAutoThreads = arrayOf(
            float, float, int, int, int, int, int, VariantSelection::class.java, SentenceJoining::class.java, boolean,
        )
        assertTrue(has(CzechSpeechConfig::class.java))
        assertTrue("before earlyStart", has(CzechSpeechConfig::class.java, *upToAutoThreads))
        assertTrue("before minEncoderWindow", has(CzechSpeechConfig::class.java, *upToAutoThreads, EarlyStart::class.java))
        assertTrue("before streaming", has(CzechSpeechConfig::class.java, *upToAutoThreads, EarlyStart::class.java, int))
        assertTrue(
            has(CzechSpeechConfig::class.java, *upToAutoThreads, EarlyStart::class.java, int, StreamingConfig::class.java),
        )
        assertTrue("before autoThreads", has(CzechSpeechConfig::class.java, *upToAutoThreads.take(9).toTypedArray()))
        assertTrue("the settings alone", has(CzechSpeechConfig::class.java, *upToAutoThreads.take(7).toTypedArray()))
    }

    @Test
    fun entryAndCzechSpeechKeepTheirOlderConstructorsAndListenWithPreviewsItsOlderMethod() {
        val source = CzechSpeechModel.Source::class.java
        val switch = CzechSpeechModel.LanguageSwitch::class.java
        assertTrue(has(CzechSpeechModel.Entry::class.java, String::class.java, source, List::class.java, int))
        assertTrue(has(CzechSpeechModel.Entry::class.java, String::class.java, source, List::class.java, int, switch))
        val plain = arrayOf(Context::class.java, File::class.java, CzechSpeechConfig::class.java, boolean, List::class.java)
        assertTrue(has(CzechSpeech::class.java, *plain))
        assertTrue(has(CzechSpeech::class.java, *plain, switch))
        val methods = CzechSpeech::class.java.methods.filter { it.name == "listenWithPreviews" }.map { it.parameterTypes.toList() }
        assertTrue(methods.contains(listOf(PreviewConfig::class.java)))
        assertTrue(methods.contains(listOf(PreviewConfig::class.java, String::class.java)))
    }

    @Test
    fun streamingConfigHasJavaConstructors() {
        assertTrue(has(StreamingConfig::class.java))
        assertTrue(has(StreamingConfig::class.java, int))
        assertTrue(has(StreamingConfig::class.java, int, ai.moravian.czechspeech.CutSeam::class.java))
    }
}
