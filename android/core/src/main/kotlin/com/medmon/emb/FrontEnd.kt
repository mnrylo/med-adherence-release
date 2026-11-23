package com.medmon.emb

/**
 * Causal 0.3 Hz high-pass on the accelerometer (scipy sosfilt, direct form II transposed), starting
 * in the steady state of the first sample. Channels 3..5 (gyroscope) pass unchanged.
 */
class CausalHighPass(private val sos: Array<DoubleArray>, private val zi: Array<DoubleArray>) {
    private val z = Array(3) { Array(sos.size) { DoubleArray(2) } }
    private var started = false

    fun reset() {
        started = false
    }

    /** Filters one sample in place (acc in g, gyro in deg/s). */
    fun process(x: DoubleArray) {
        if (!started) {
            for (c in 0 until 3) for (s in sos.indices) {
                z[c][s][0] = zi[s][0] * x[c]
                z[c][s][1] = zi[s][1] * x[c]
            }
            started = true
        }
        for (c in 0 until 3) {
            var v = x[c]
            for (s in sos.indices) {
                val q = sos[s]
                val zs = z[c][s]
                val y = q[0] * v + zs[0]
                zs[0] = q[1] * v - q[4] * y + zs[1]
                zs[1] = q[2] * v - q[5] * y
                v = y
            }
            x[c] = v
        }
    }
}

/** Window classifier: input (n, 6, window) channels-first float32, output (n, classes). */
interface WindowClassifier : AutoCloseable {
    fun classify(batch: FloatArray, n: Int): Array<FloatArray>
    override fun close() {}
}

/**
 * Filter + windows + classifier. Window j is centred on sample j * stride and spans
 * [c - window/2, c + window/2); samples before the start repeat the first one. A window is
 * classified when its last sample arrives; flush() classifies the remaining centres
 * (j * stride < T) with the last sample repeated.
 */
class FrontEnd(
    private val cfg: PipelineConfig,
    private val classifier: WindowClassifier,
    private val clock: StageClock = StageClock(),
    private val onWindow: (index: Int, probs: FloatArray) -> Unit,
) {
    private val filter = CausalHighPass(cfg.sos, cfg.zi)
    private val half = cfg.window / 2
    private val ring = Array(cfg.window) { FloatArray(6) }
    private val first = FloatArray(6)
    private val buf = DoubleArray(6)
    private val input = FloatArray(6 * cfg.window)
    var samples = 0L
        private set
    var windows = 0
        private set

    fun reset() {
        filter.reset()
        samples = 0
        windows = 0
    }

    /** One raw sample: acc x, y, z in g and gyro x, y, z in deg/s. */
    fun push(x: FloatArray) {
        clock.start(Stage.FILTER)
        for (c in 0 until 6) buf[c] = x[c].toDouble()
        filter.process(buf)
        val slot = ring[(samples % cfg.window).toInt()]
        for (c in 0 until 6) slot[c] = buf[c].toFloat()
        if (samples == 0L) slot.copyInto(first)
        samples++
        clock.stop(Stage.FILTER)
        val needed = windows.toLong() * cfg.stride + (cfg.window - half - 1)
        if (samples - 1 == needed) emit(windows, samples - 1)
    }

    /** End of stream: windows whose centre is inside the stream but whose end is not. */
    fun flush() {
        if (samples == 0L) return
        while (windows.toLong() * cfg.stride < samples) emit(windows, samples - 1)
    }

    private fun emit(j: Int, last: Long) {
        clock.start(Stage.WINDOW)
        val c0 = j.toLong() * cfg.stride - half
        for (k in 0 until cfg.window) {
            val idx = (c0 + k).coerceAtMost(last)
            val row = if (idx < 0) first else ring[(idx % cfg.window).toInt()]
            for (c in 0 until 6) input[c * cfg.window + k] = row[c]
        }
        clock.stop(Stage.WINDOW)
        clock.start(Stage.INFERENCE)
        val p = classifier.classify(input, 1)[0]
        clock.stop(Stage.INFERENCE)
        windows++
        onWindow(j, p)
    }
}
