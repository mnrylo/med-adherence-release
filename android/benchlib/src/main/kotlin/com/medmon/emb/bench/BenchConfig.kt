package com.medmon.emb.bench

import org.json.JSONArray
import org.json.JSONObject

/**
 * Experimental conditions (README, phase 2):
 *  * C0 idle: no sensors, no processing (metrics only).
 *  * C1 sensors on: the watch samples the IMU at 100 Hz and discards it.
 *  * P1 all on the phone: the watch streams raw samples; the phone runs the whole pipeline.
 *  * P2 hybrid: the watch filters and classifies, streams window probabilities; the phone runs the
 *    timeline, the Viterbi and the LLR.
 *  * P3 all on the watch: the phone only receives the alarms.
 *  * LOCAL: the whole pipeline on the device that receives the command, reading the replay file
 *    there (fidelity against the PC reference and throughput, usually with pacing = fast).
 */
enum class Condition { C0, C1, P1, P2, P3, LOCAL }

data class BenchConfig(
    val runId: String,
    val condition: Condition,
    /** "replay" (blind files injected as sensor data) or "live" (the watch IMU) */
    val source: String = "replay",
    val set: String = "test_wild",
    /** "realtime" (100 Hz) or "fast" (as fast as possible; LOCAL only) */
    val pacing: String = "realtime",
    val precision: String = "int8",
    /** LLR evaluation steps (s); the first one is the deployed step */
    val steps: List<Int> = listOf(2),
    /** samples are delivered in batches of this many ms (replay pacing, sensor max report latency) */
    val batchMs: Int = 100,
    /** Bluetooth messages carry this many ms of data (P1, P2) */
    val linkBatchMs: Int = 1000,
    /** replay: also sample the IMU (discarded) so the sensor energy is included */
    val sensorsOn: Boolean = false,
    /** stop after this many seconds of wall time (0 = whole set) */
    val durationS: Int = 0,
    val firstStream: Int = 0,
    /** 0 = every stream of the set */
    val maxStreams: Int = 0,
    val threads: Int = 1,
    /** write every window probability (fidelity runs) */
    val logProbs: Boolean = false,
    val metricsPeriodS: Int = 10,
    val wakeLock: Boolean = true,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("run_id", runId).put("condition", condition.name).put("source", source).put("set", set)
        .put("pacing", pacing).put("precision", precision).put("steps", JSONArray(steps))
        .put("batch_ms", batchMs).put("link_batch_ms", linkBatchMs).put("sensors_on", sensorsOn)
        .put("duration_s", durationS).put("first_stream", firstStream).put("max_streams", maxStreams)
        .put("threads", threads).put("log_probs", logProbs).put("metrics_period_s", metricsPeriodS)
        .put("wake_lock", wakeLock)

    companion object {
        fun fromJson(o: JSONObject) = BenchConfig(
            runId = o.getString("run_id"),
            condition = Condition.valueOf(o.getString("condition")),
            source = o.optString("source", "replay"),
            set = o.optString("set", "test_wild"),
            pacing = o.optString("pacing", "realtime"),
            precision = o.optString("precision", "int8"),
            steps = o.optJSONArray("steps")?.let { a -> List(a.length()) { a.getInt(it) } } ?: listOf(2),
            batchMs = o.optInt("batch_ms", 100),
            linkBatchMs = o.optInt("link_batch_ms", 1000),
            sensorsOn = o.optBoolean("sensors_on", false),
            durationS = o.optInt("duration_s", 0),
            firstStream = o.optInt("first_stream", 0),
            maxStreams = o.optInt("max_streams", 0),
            threads = o.optInt("threads", 1),
            logProbs = o.optBoolean("log_probs", false),
            metricsPeriodS = o.optInt("metrics_period_s", 10),
            wakeLock = o.optBoolean("wake_lock", true),
        )

        fun parse(s: String) = fromJson(JSONObject(s))
    }
}
