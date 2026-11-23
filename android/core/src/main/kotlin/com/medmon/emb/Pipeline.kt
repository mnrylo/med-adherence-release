package com.medmon.emb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

enum class Stage { FILTER, WINDOW, INFERENCE, TIMELINE, VITERBI, LLR, TRANSPORT }

/** Cumulative CPU-side wall time per pipeline stage (single thread), plus every inference time. */
class StageClock {
    val nanos = LongArray(Stage.entries.size)
    val counts = LongArray(Stage.entries.size)
    private val t0 = LongArray(Stage.entries.size)
    private var inference = LongArray(1024)
    var inferenceCount = 0
        private set

    fun start(s: Stage) {
        t0[s.ordinal] = System.nanoTime()
    }

    fun stop(s: Stage) {
        val d = System.nanoTime() - t0[s.ordinal]
        nanos[s.ordinal] += d
        counts[s.ordinal]++
        if (s == Stage.INFERENCE) {
            if (inferenceCount == inference.size) inference = inference.copyOf(inference.size * 2)
            inference[inferenceCount++] = d
        }
    }

    /** Percentile (0..100) of the inference times, in microseconds. */
    fun inferencePercentileUs(q: Double): Double {
        if (inferenceCount == 0) return Double.NaN
        val s = inference.copyOf(inferenceCount).also { it.sort() }
        return s[((q / 100.0) * (inferenceCount - 1)).toInt()] / 1000.0
    }

    fun reset() {
        nanos.fill(0); counts.fill(0); inferenceCount = 0
    }
}

/** ONNX Runtime classifier; the model input is "imu" (n, 6, window), the output the probabilities. */
class OnnxWindowClassifier(model: ByteArray, threads: Int = 1, private val window: Int = 200) : WindowClassifier {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(model, OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    })
    private val inputName = session.inputNames.first()

    override fun classify(batch: FloatArray, n: Int): Array<FloatArray> {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(batch, 0, n * 6 * window), longArrayOf(n.toLong(), 6, window.toLong()))
            .use { t ->
                session.run(mapOf(inputName to t)).use { r ->
                    @Suppress("UNCHECKED_CAST")
                    return (r[0].value as Array<FloatArray>).map { it.copyOf() }.toTypedArray()
                }
            }
    }

    override fun close() {
        session.close()
    }
}

/** The whole chain on one device (placements P1 on the phone and P3 on the watch). */
class Pipeline(
    cfg: PipelineConfig,
    classifier: WindowClassifier,
    steps: List<Int>,
    listener: BackEnd.Listener,
    val clock: StageClock = StageClock(),
    /** optional tap on every window probability (logging) */
    private val onWindow: ((Int, FloatArray) -> Unit)? = null,
) {
    val back = BackEnd(cfg, steps, clock, listener)
    val front = FrontEnd(cfg, classifier, clock) { j, p ->
        onWindow?.invoke(j, p)
        back.onWindow(j, p)
    }

    fun push(sample: FloatArray) = front.push(sample)

    /** End of a replayed stream: flush, then ready for the next one. */
    fun endStream() {
        front.flush()
        back.flush(front.samples)
        front.reset()
        back.reset()
    }
}

/** One stream of a replay file. Times in seconds of stream time; null when not applicable. */
data class ReplayStream(
    val sid: String,
    val offset: Long,
    val length: Int,
    val intake: Int,
    val sessionStartS: Double?,
    val intakeEndS: Double?,
    val g3EndS: Double?,
)

/**
 * Replay files written by src/embedded/phase1_export.py: <set>.f32 holds float32 little-endian (T, 6) samples
 * (acc g, gyro deg/s, 100 Hz, not high-pass filtered) and <set>.json the stream index.
 */
class ReplaySet(dir: File, val name: String) : AutoCloseable {
    val streams: List<ReplayStream>
    private val file = RandomAccessFile(File(dir, "$name.f32"), "r")

    init {
        val o = org.json.JSONObject(File(dir, "$name.json").readText())
        val a = o.getJSONArray("streams")
        fun org.json.JSONObject.optD(k: String) = if (has(k) && !isNull(k)) getDouble(k) else null
        streams = List(a.length()) {
            val s = a.getJSONObject(it)
            ReplayStream(s.getString("sid"), s.getLong("offset"), s.getInt("length"), s.getInt("intake"),
                s.optD("session_start_s"), s.optD("intake_end_s"), s.optD("g3_end_s"))
        }
    }

    val totalSamples: Long get() = streams.sumOf { it.length.toLong() }

    /** Samples of one stream, row-major (length * 6). */
    fun read(s: ReplayStream): FloatArray {
        val buf = file.channel.map(FileChannel.MapMode.READ_ONLY, s.offset * 24, s.length * 24L)
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(s.length * 6).also { buf.get(it) }
    }

    override fun close() = file.close()
}
