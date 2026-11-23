package com.medmon.emb.bench

import android.content.Context
import com.google.android.gms.wearable.MessageEvent
import com.medmon.emb.BackEnd
import com.medmon.emb.FrontEnd
import com.medmon.emb.OnnxWindowClassifier
import com.medmon.emb.Pipeline
import com.medmon.emb.PipelineConfig
import com.medmon.emb.ReplayStream
import com.medmon.emb.Stage
import com.medmon.emb.StageClock
import com.medmon.emb.WindowClassifier
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

enum class Role { WATCH, PHONE }

/**
 * One run of one condition on one device. Every pipeline call happens on a single engine thread;
 * link messages and sensor batches are posted to it.
 *
 * | condition | watch                                   | phone                          |
 * |-----------|-----------------------------------------|--------------------------------|
 * | C0        | metrics                                 | metrics                        |
 * | C1        | IMU at 100 Hz, discarded                | metrics                        |
 * | P1        | source -> raw batches                   | raw -> Pipeline, alarms back   |
 * | P2        | source -> FrontEnd -> probability batches | probabilities -> BackEnd, alarms back |
 * | P3        | source -> Pipeline, alarms to the phone | alarms logged                  |
 * | LOCAL     | replay -> Pipeline (if started here)    | replay -> Pipeline (if started here) |
 */
class BenchEngine(private val ctx: Context, val role: Role, private val onFinished: () -> Unit) : SampleSink {
    private val exec: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor { r ->
        Thread(r, "bench-engine").apply { priority = Thread.NORM_PRIORITY }
    }
    private val metricsExec = Executors.newSingleThreadScheduledExecutor { r -> Thread(r, "bench-metrics") }
    lateinit var cfg: BenchConfig
        private set
    lateinit var logger: RunLogger
        private set
    private lateinit var pc: PipelineConfig
    private val clock = StageClock()
    private var classifier: WindowClassifier? = null
    private var pipeline: Pipeline? = null
    private var front: FrontEnd? = null
    private var back: BackEnd? = null
    private var source: SampleSource? = null
    private var sensors: LiveSource? = null
    private var replay: ReplaySource? = null
    private val link = Link(ctx) { e -> exec.execute { onLink(e) } }
    private val reorder = Reorder { b -> onBatch(b) }

    @Volatile var status = "idle"
        private set
    @Volatile var running = false
        private set

    private var stream = -1
    private var sid = ""
    private var seq = 0
    private var outVals = FloatArray(0)
    private var outN = 0
    private var outFirst = 0L
    private var produced = 0L
    private var labelRun = intArrayOf(-1, -1, -1)
    private val intakeEnd = HashMap<Int, Pair<Double, Double?>>()

    fun start(config: BenchConfig) = exec.execute {
        cfg = config
        logger = RunLogger(ctx, cfg, role.name.lowercase())
        link.logger = logger
        try {
            pc = PipelineConfig.parse(ctx.assets.open("bundle/pipeline.json").bufferedReader().readText())
            link.open()
            logger.event("start", detail = "link=${link.connected}")
            setup()
            running = true
            status = "running ${cfg.condition} ${cfg.source}/${cfg.set}"
            metricsExec.scheduleAtFixedRate({ runCatching { logger.sampleMetrics() } }, 0,
                cfg.metricsPeriodS.toLong(), TimeUnit.SECONDS)
            if (cfg.durationS > 0) exec.schedule({ stop("duration") }, cfg.durationS.toLong(), TimeUnit.SECONDS)
            source?.start()
            sensors?.start()
        } catch (t: Throwable) {
            logger.event("error", detail = t.toString())
            status = "error: $t"
            stop("error")
        }
    }

    private fun model(): WindowClassifier {
        val file = pc.modelFiles.getValue(cfg.precision)
        return OnnxWindowClassifier(ctx.assets.open("bundle/$file").readBytes(), cfg.threads, pc.window)
            .also { classifier = it }
    }

    private val listener = object : BackEnd.Listener {
        override fun onLabel(frame: Int, label: Int) {
            if (label == labelRun[2] && frame == labelRun[1] + 1) labelRun[1] = frame
            else {
                flushLabels()
                labelRun = intArrayOf(frame, frame, label)
            }
        }

        override fun onAlarm(stepS: Int, budget: String, streamTimeS: Double) = alarm(stepS, budget, streamTimeS)
    }

    private fun flushLabels() {
        if (labelRun[2] >= 0) logger.labelRun(stream, labelRun[0], labelRun[1], labelRun[2])
        labelRun = intArrayOf(-1, -1, -1)
    }

    private fun setup() {
        val c = cfg.condition
        val watch = role == Role.WATCH
        val needsSource = (watch && c in setOf(Condition.P1, Condition.P2, Condition.P3, Condition.LOCAL)) ||
            (!watch && c == Condition.LOCAL)
        when {
            c == Condition.LOCAL || (watch && c == Condition.P3) || (!watch && c == Condition.P1) ->
                pipeline = Pipeline(pc, model(), cfg.steps, listener, clock) { _, p ->
                    logger.windows++
                    if (cfg.logProbs) logger.probs(stream, sid, p)
                }
            watch && c == Condition.P2 -> front = FrontEnd(pc, model(), clock) { j, p -> onWindowWatchP2(j, p) }
            !watch && c == Condition.P2 -> back = BackEnd(pc, cfg.steps, clock, listener)
        }
        if (needsSource) source = if (cfg.source == "live" && watch && c != Condition.LOCAL)
            LiveSource(ctx, cfg.batchMs, exec, this)
        else ReplaySource(ctx, cfg, exec, this, logger).also { replay = it }
        if (watch && (c == Condition.C1 || (cfg.sensorsOn && cfg.source == "replay")))
            sensors = LiveSource(ctx, cfg.batchMs, exec, null)
    }

    override fun onStreamStart(index: Int, stream: ReplayStream?) {
        this.stream = index
        sid = stream?.sid ?: "live"
        produced = 0
        stream?.intakeEndS?.let { intakeEnd[index] = it to stream.g3EndS }
        logger.event("stream_start", index, sid)
    }

    override fun onSamples(x: FloatArray, n: Int) {
        logger.samples += n
        when {
            pipeline != null -> {
                val p = pipeline!!
                val row = FloatArray(6)
                for (i in 0 until n) {
                    System.arraycopy(x, i * 6, row, 0, 6)
                    p.push(row)
                }
            }
            front != null -> {
                val row = FloatArray(6)
                for (i in 0 until n) {
                    System.arraycopy(x, i * 6, row, 0, 6)
                    front!!.push(row)
                }
            }
            cfg.condition == Condition.P1 -> {
                for (i in 0 until n) appendOut(x, i * 6, 6)
                produced += n
                if (outN >= 100 * cfg.linkBatchMs / 1000) sendOut(Paths.RAW, 6, end = false)
            }
        }
    }

    override fun onStreamEnd(index: Int, nSamples: Long) {
        when {
            pipeline != null -> {
                pipeline!!.endStream()
                flushLabels()
            }
            front != null -> {
                front!!.flush()
                sendOut(Paths.PROBS, 5, end = true, total = front!!.samples)
                front!!.reset()
            }
            cfg.condition == Condition.P1 -> sendOut(Paths.RAW, 6, end = true, total = produced)
        }
        logger.closeProbStream()
        logger.event("stream_end", index, "$nSamples")
    }

    override fun onSourceDone() {
        logger.event("source_done")
        exec.schedule({ stop("done") }, if (cfg.condition in setOf(Condition.P1, Condition.P2)) 20 else 1, TimeUnit.SECONDS)
    }

    private fun onWindowWatchP2(j: Int, p: FloatArray) {
        logger.windows++
        if (cfg.logProbs) logger.probs(stream, sid, p)
        if (outN == 0) outFirst = j.toLong()
        appendOut(p, 0, 5)
        if (outN * pc.stride * 1000 / pc.fs >= cfg.linkBatchMs) sendOut(Paths.PROBS, 5, end = false)
    }

    private fun appendOut(src: FloatArray, off: Int, width: Int) {
        if (outVals.size < (outN + 1) * width) outVals = outVals.copyOf(maxOf(64 * width, outVals.size * 2))
        System.arraycopy(src, off, outVals, outN * width, width)
        if (outN == 0 && width == 6) outFirst = produced
        outN++
    }

    private fun sendOut(path: String, width: Int, end: Boolean, total: Long = -1) {
        if (outN == 0 && !end) return
        clock.start(Stage.TRANSPORT)
        val b = Batch(seq++, stream, outFirst, outN, end, total, outVals.copyOf(outN * width))
        link.send(path, b.encode())
        clock.stop(Stage.TRANSPORT)
        outN = 0
    }

    private fun onLink(e: MessageEvent) {
        if (!running) return
        when (e.path) {
            Paths.RAW, Paths.PROBS -> reorder.offer(Batch.decode(e.data))
            Paths.STOP -> stop("peer_done")
            Paths.ALARM -> {
                val a = AlarmMsg.decode(e.data)
                logger.alarm(a.stream, "", a.step, a.budget, a.streamTimeS, "rx_from_${if (role == Role.WATCH) "phone" else "watch"}",
                    null, null, null)
            }
        }
    }

    private fun onBatch(b: Batch) {
        if (b.stream != stream) {
            stream = b.stream
            sid = ""
            logger.event("stream_start", stream, "link")
        }
        if (cfg.condition == Condition.P1 && pipeline != null) {
            onSamples(b.values, b.n)
            if (b.end) onStreamEnd(b.stream, b.totalSamples)
        } else if (cfg.condition == Condition.P2 && back != null) {
            for (i in 0 until b.n) {
                val p = b.values.copyOfRange(i * 5, i * 5 + 5)
                if (cfg.logProbs) logger.probs(stream, sid, p)
                logger.windows++
                back!!.onWindow((b.first + i).toInt(), p)
            }
            if (b.end) {
                back!!.flush(b.totalSamples)
                back!!.reset()
                flushLabels()
                logger.event("stream_end", b.stream, "${b.totalSamples}")
            }
        }
    }

    private fun alarm(stepS: Int, budget: String, streamTimeS: Double) {
        val ie = intakeEnd[stream]
        val alg = ie?.let { streamTimeS + pc.lookaheadS - it.first }
        val algG3 = ie?.second?.let { streamTimeS + pc.lookaheadS - it }
        logger.alarm(stream, sid, stepS, budget, streamTimeS, role.name.lowercase(), alg, null, algG3)
        if (cfg.condition in setOf(Condition.P1, Condition.P2, Condition.P3))
            link.send(Paths.ALARM, AlarmMsg(stream, stepS, budget, streamTimeS).encode())
    }

    fun stop(reason: String) = exec.execute {
        if (!::logger.isInitialized) return@execute
        if (!running && reason != "error") return@execute
        running = false
        status = "stopping ($reason)"
        source?.stop()
        sensors?.stop()
        logger.event("stop", detail = reason)
        if (role == Role.WATCH && reason == "done" && cfg.condition != Condition.LOCAL)
            link.send(Paths.STOP, cfg.runId.toByteArray())
        metricsExec.shutdown()
        val extra = JSONObject().put("reason", reason).put("reorder_max_pending", reorder.maxPending)
            .put("imu_events", sensors?.events ?: 0)
        logger.finish(clock, extra)
        classifier?.close()
        link.close()
        status = "finished ($reason): ${logger.dir}"
        onFinished()
    }

    fun shutdown() {
        exec.shutdown()
    }
}
