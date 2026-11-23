package com.medmon.emb.bench

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import com.medmon.emb.ReplaySet
import com.medmon.emb.ReplayStream
import java.io.File
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/** Receiver of the samples of a source (always called on the engine thread). */
interface SampleSink {
    fun onStreamStart(index: Int, stream: ReplayStream?)
    /** n samples, row-major (n * 6): acc g, gyro deg/s */
    fun onSamples(x: FloatArray, n: Int)
    fun onStreamEnd(index: Int, nSamples: Long)
    fun onSourceDone()
}

interface SampleSource {
    fun start()
    fun stop()
}

/**
 * Replay of a blind set (src/embedded/phase1_export.py files in <internal files>/replay/, see embedded.phase1_bench push). Realtime pacing
 * delivers fs * batchMs / 1000 samples every batchMs on the engine executor; fast pacing delivers
 * them as fast as the sink consumes them. Streams follow each other without gaps.
 * The moment the last sample of the intake (G3/G4) and of the G3 is delivered is logged as the
 * events "intake_end" / "g3_end": the alarm latency at the wrist is measured from them.
 */
class ReplaySource(
    ctx: Context,
    private val cfg: BenchConfig,
    private val exec: ScheduledExecutorService,
    private val sink: SampleSink,
    private val logger: RunLogger,
) : SampleSource {
    val set = ReplaySet(File(ctx.filesDir, "replay"), cfg.set)
    val streams: List<ReplayStream> = set.streams.drop(cfg.firstStream)
        .let { if (cfg.maxStreams > 0) it.take(cfg.maxStreams) else it }
    private val perBatch = (100 * cfg.batchMs / 1000).coerceAtLeast(1)
    private var si = -1
    private var x = FloatArray(0)
    private var pos = 0
    private var task: ScheduledFuture<*>? = null
    @Volatile private var stopped = false

    override fun start() {
        if (cfg.pacing == "fast") exec.execute { while (!stopped && tick()) Unit }
        else task = exec.scheduleAtFixedRate({ if (!stopped) tick() }, 0, cfg.batchMs.toLong(), TimeUnit.MILLISECONDS)
    }

    override fun stop() {
        stopped = true
        task?.cancel(false)
        set.close()
    }

    /** One batch; false when the set is finished. */
    private fun tick(): Boolean {
        var left = perBatch
        while (left > 0) {
            if (si < 0 || pos >= streams[si].length) {
                if (si >= 0) sink.onStreamEnd(si + cfg.firstStream, streams[si].length.toLong())
                si++
                if (si >= streams.size) {
                    task?.cancel(false)
                    stopped = true
                    sink.onSourceDone()
                    return false
                }
                x = set.read(streams[si]); pos = 0
                sink.onStreamStart(si + cfg.firstStream, streams[si])
            }
            val s = streams[si]
            val n = minOf(left, s.length - pos)
            val idx = si + cfg.firstStream
            s.intakeEndS?.let { t -> mark("intake_end", idx, Math.round(t * 100).toInt(), n) }
            s.g3EndS?.let { t -> mark("g3_end", idx, Math.round(t * 100).toInt(), n) }
            sink.onSamples(x.copyOfRange(pos * 6, (pos + n) * 6), n)
            pos += n
            left -= n
        }
        return true
    }

    private fun mark(kind: String, stream: Int, sample: Int, n: Int) {
        if (sample - 1 in pos until pos + n) logger.event(kind, stream)
    }
}

/**
 * The watch IMU at 100 Hz: accelerometer events, each with the latest gyroscope value (as the
 * production wear app does), linearly interpolated onto a 10 ms grid. m/s^2 -> g, rad/s -> deg/s.
 * With a null sink the samples are discarded (condition C1, or sensorsOn during a replay).
 */
class LiveSource(
    ctx: Context,
    private val batchMs: Int,
    private val exec: ScheduledExecutorService,
    private val sink: SampleSink?,
) : SampleSource, SensorEventListener {
    private val sm = ctx.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread = HandlerThread("imu").apply { start() }
    private val gyr = FloatArray(3)
    private var lastT = -1L
    private val lastAcc = FloatArray(3)
    private var nextT = -1L
    private val out = ArrayList<Float>(6 * 128)
    private var n = 0
    var events = 0L
        private set

    override fun start() {
        val h = Handler(thread.looper)
        val lat = batchMs * 1000
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let { sm.registerListener(this, it, 10_000, lat, h) }
        sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { sm.registerListener(this, it, 10_000, lat, h) }
        if (sink != null) exec.execute { sink.onStreamStart(0, null) }
    }

    override fun stop() {
        sm.unregisterListener(this)
        thread.quitSafely()
        if (sink != null) exec.execute { sink.onStreamEnd(0, -1) }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(e: SensorEvent) {
        events++
        if (sink == null) return
        if (e.sensor.type == Sensor.TYPE_GYROSCOPE) {
            for (c in 0 until 3) gyr[c] = (e.values[c] * 180.0 / Math.PI).toFloat()
            return
        }
        val acc = FloatArray(3) { (e.values[it] / 9.80665).toFloat() }
        if (lastT < 0) { nextT = e.timestamp }
        else while (nextT <= e.timestamp) {
            val f = ((nextT - lastT).toDouble() / (e.timestamp - lastT).coerceAtLeast(1)).toFloat()
            for (c in 0 until 3) out.add(lastAcc[c] + f * (acc[c] - lastAcc[c]))
            for (c in 0 until 3) out.add(gyr[c])
            n++
            nextT += 10_000_000L
        }
        lastT = e.timestamp
        acc.copyInto(lastAcc)
        if (n >= 100 * batchMs / 1000) {
            val x = out.toFloatArray()
            val k = n
            out.clear(); n = 0
            exec.execute { sink.onSamples(x, k) }
        }
    }
}
