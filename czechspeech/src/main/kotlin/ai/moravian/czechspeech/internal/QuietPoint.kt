package ai.moravian.czechspeech.internal

/**
 * Finds the quietest place in a range of audio: the cut of streaming transcription falls there, so that a word is
 * unlikely to be split. A port of `quiet_point` of the measurement's simulation.
 */
internal object QuietPoint {
    /** Window of 30 ms at 16 kHz. */
    private const val WINDOW = 480

    /** Step of 10 ms. */
    private const val STEP = 160

    /** The window's centre, as an offset from its start. */
    private const val CENTRE = 240

    /**
     * The centre of the 30 ms window with the least energy (mean of squares) among those that lie wholly in
     * `from until to`, windows 10 ms apart, as an index into [samples]: so the result is at least 15 ms after [from]
     * and at least 15 ms before [to]. The range is clamped to the array; the first of equally quiet windows wins. A
     * range shorter than one window gives its start.
     */
    fun find(samples: FloatArray, from: Int, to: Int): Int {
        val lo = maxOf(0, from)
        val hi = minOf(samples.size, to)
        if (hi - lo < WINDOW) return lo
        var best = lo
        var bestEnergy = Double.MAX_VALUE
        var start = lo
        while (start <= hi - WINDOW) {
            var sum = 0.0
            for (i in start until start + WINDOW) {
                val v = samples[i].toDouble()
                sum += v * v
            }
            val energy = sum / WINDOW
            if (energy < bestEnergy) {
                bestEnergy = energy
                best = start
            }
            start += STEP
        }
        return best + CENTRE
    }
}
