package ai.moravian.czechspeech.internal

import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * The whisper_full parameters live in C and cannot be read back without a model, so this guards the source: whisper.cpp
 * keeps `suppress_blank` on by default, which forbids an empty transcript as the first token. A model that learned to
 * answer a segment without speech with an empty transcript then has to write something into silence.
 */
class NativeParamsTest {
    private val source = File("../native/src/czechspeech_jni.c").takeIf { it.exists() }?.readText()

    @Test
    fun suppressBlankIsOff() {
        val text = source ?: error("native/src/czechspeech_jni.c not found from ${File(".").absolutePath}")
        assertTrue("params.suppress_blank = false;", Regex("""params\.suppress_blank\s*=\s*false\s*;""").containsMatchIn(text))
    }
}
