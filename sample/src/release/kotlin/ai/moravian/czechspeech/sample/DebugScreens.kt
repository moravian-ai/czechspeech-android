package ai.moravian.czechspeech.sample

import android.app.Activity
import android.widget.LinearLayout

/** Release builds have no debug screens (the Benchmark screen is in the debug source set). */
internal object DebugScreens {
    @Suppress("UNUSED_PARAMETER")
    fun addTo(activity: Activity, screen: LinearLayout) = Unit
}
