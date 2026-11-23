package com.medmon.emb.bench

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.Process
import android.os.SystemClock
import com.medmon.emb.Stage
import com.medmon.emb.StageClock
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream

/**
 * Files of one run on one device, under <external files>/runs/<run id>/<role>/:
 *  config.json, device.json, summary.json,
 *  metrics.csv  every metricsPeriodS: battery, process CPU time, memory, link bytes, progress,
 *  events.csv   stream boundaries, link problems, errors,
 *  alarms.csv   every alarm (and, on the watch, its latency at the wrist),
 *  labels.csv   decided Gesture Timeline as runs (stream, first frame, last frame, label),
 *  probs.f32    (optional) every window probability, float32 LE, with probs_index.csv.
 */
class RunLogger(private val ctx: Context, val config: BenchConfig, val role: String) {
    val dir: File = File(ctx.getExternalFilesDir(null), "runs/${config.runId}/$role").apply { mkdirs() }
    private val t0 = SystemClock.elapsedRealtimeNanos()
    private val cpu0 = Process.getElapsedCpuTime()
    private val metrics = writer("metrics.csv",
        "elapsed_s,wall_ms,battery_pct,charge_uah,current_now_ua,current_avg_ua,voltage_mv,temp_dc,plugged," +
            "cpu_ms,pss_kb,java_heap_kb,native_heap_kb,tx_bytes,rx_bytes,tx_msgs,rx_msgs,samples,windows,alarms")
    private val events = writer("events.csv", "elapsed_s,wall_ms,type,stream,detail")
    private val alarms = writer("alarms.csv",
        "elapsed_s,wall_ms,stream,sid,step_s,budget,stream_time_s,where,alg_latency_s,wall_latency_s,alg_latency_g3_s")
    private val labels = writer("labels.csv", "stream,first_frame,last_frame,label")
    private var probs: DataOutputStream? = null
    private var probsIndex: BufferedWriter? = null
    private val battery = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    @Volatile var txBytes = 0L
    @Volatile var rxBytes = 0L
    @Volatile var txMsgs = 0L
    @Volatile var rxMsgs = 0L
    @Volatile var samples = 0L
    @Volatile var windows = 0L
    @Volatile var nAlarms = 0L

    init {
        File(dir, "config.json").writeText(config.toJson().toString(1))
        File(dir, "device.json").writeText(deviceInfo().toString(1))
        if (config.logProbs) {
            probs = DataOutputStream(FileOutputStream(File(dir, "probs.f32")).buffered(1 shl 16))
            probsIndex = writer("probs_index.csv", "stream,sid,first_row,n_windows")
        }
    }

    private fun writer(name: String, header: String) =
        File(dir, name).bufferedWriter().apply { write(header); newLine() }

    fun elapsedS() = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9

    private fun prefix() = "%.3f,%d".format(elapsedS(), System.currentTimeMillis())

    @Synchronized
    fun event(type: String, stream: Int = -1, detail: String = "") {
        events.write("${prefix()},$type,$stream,\"${detail.replace("\"", "'")}\""); events.newLine()
        events.flush()
    }

    @Synchronized
    fun alarm(stream: Int, sid: String, step: Int, budget: String, streamTimeS: Double, where: String,
              alg: Double?, wall: Double?, algG3: Double?) {
        nAlarms++
        alarms.write("${prefix()},$stream,$sid,$step,$budget,$streamTimeS,$where,${alg ?: ""},${wall ?: ""},${algG3 ?: ""}")
        alarms.newLine(); alarms.flush()
    }

    @Synchronized
    fun labelRun(stream: Int, first: Int, last: Int, label: Int) {
        labels.write("$stream,$first,$last,$label"); labels.newLine()
    }

    private var probRows = 0L
    private var probStreamRows = 0L
    private var probStream = -1
    private var probSid = ""

    @Synchronized
    fun probs(stream: Int, sid: String, p: FloatArray) {
        val out = probs ?: return
        if (stream != probStream) closeProbStream()
        if (probStream < 0) { probStream = stream; probSid = sid; probStreamRows = 0 }
        for (v in p) out.writeInt(Integer.reverseBytes(java.lang.Float.floatToRawIntBits(v)))
        probStreamRows++
    }

    @Synchronized
    fun closeProbStream() {
        if (probStream >= 0) {
            probsIndex?.apply { write("$probStream,$probSid,$probRows,$probStreamRows"); newLine() }
            probRows += probStreamRows
        }
        probStream = -1
    }

    @Synchronized
    fun sampleMetrics() {
        val st = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val mi = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        val rt = Runtime.getRuntime()
        metrics.write(listOf(
            prefix(),
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
            battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE),
            st?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1,
            st?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1,
            st?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1,
            Process.getElapsedCpuTime() - cpu0,
            mi.totalPss,
            (rt.totalMemory() - rt.freeMemory()) / 1024,
            Debug.getNativeHeapAllocatedSize() / 1024,
            txBytes, rxBytes, txMsgs, rxMsgs, samples, windows, nAlarms,
        ).joinToString(","))
        metrics.newLine(); metrics.flush()
    }

    @Synchronized
    fun finish(clock: StageClock?, extra: JSONObject = JSONObject()) {
        sampleMetrics()
        closeProbStream()
        val stages = JSONObject()
        if (clock != null) for (s in Stage.entries) stages.put(s.name.lowercase(),
            JSONObject().put("total_ms", clock.nanos[s.ordinal] / 1e6).put("count", clock.counts[s.ordinal])
                .put("mean_us", if (clock.counts[s.ordinal] > 0) clock.nanos[s.ordinal] / 1e3 / clock.counts[s.ordinal] else 0.0))
        val inf = JSONObject()
        if (clock != null && clock.inferenceCount > 0) for (q in listOf(50.0, 90.0, 95.0, 99.0, 100.0))
            inf.put("p${q.toInt()}_us", clock.inferencePercentileUs(q))
        File(dir, "summary.json").writeText(JSONObject()
            .put("run_id", config.runId).put("role", role).put("elapsed_s", elapsedS())
            .put("cpu_ms", Process.getElapsedCpuTime() - cpu0)
            .put("samples", samples).put("windows", windows).put("alarms", nAlarms)
            .put("tx_bytes", txBytes).put("rx_bytes", rxBytes).put("tx_msgs", txMsgs).put("rx_msgs", rxMsgs)
            .put("stages", stages).put("inference", inf).put("extra", extra).toString(1))
        listOf(metrics, events, alarms, labels).forEach { it.flush(); it.close() }
        probs?.close(); probsIndex?.close()
    }

    private fun deviceInfo() = JSONObject()
        .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("device", Build.DEVICE)
        .put("hardware", Build.HARDWARE).put("board", Build.BOARD)
        .put("soc_manufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else "")
        .put("soc_model", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "")
        .put("sdk", Build.VERSION.SDK_INT).put("release", Build.VERSION.RELEASE)
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
        .put("cores", Runtime.getRuntime().availableProcessors())
        .put("max_memory_mb", Runtime.getRuntime().maxMemory() / 1048576)
}
