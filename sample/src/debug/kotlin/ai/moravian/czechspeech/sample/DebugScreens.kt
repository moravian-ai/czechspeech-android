package ai.moravian.czechspeech.sample

import android.app.Activity
import android.content.Intent
import android.widget.Button
import android.widget.LinearLayout

/** Debug builds: MainActivity gets a button to the Benchmark screen. Release builds have an empty twin. */
internal object DebugScreens {
    fun addTo(activity: Activity, screen: LinearLayout) {
        screen.addView(Button(activity).apply {
            setText(R.string.benchmark)
            setOnClickListener { activity.startActivity(Intent(activity, BenchmarkActivity::class.java)) }
        })
    }
}
