package com.medmon.emb

import org.json.JSONArray
import org.json.JSONObject

/**
 * Expanded HMM with sparse incoming transitions, as exported by devicepipe.HMM.to_json.
 * inFrom[j] lists the source states of j in increasing order, so a strict ">" scan returns the
 * first maximum, as numpy argmax does.
 */
class SparseHmm(
    val stateClass: IntArray,
    val logStart: DoubleArray,
    val final: BooleanArray,
    val inFrom: Array<IntArray>,
    val inLogP: Array<DoubleArray>,
) {
    val size: Int get() = stateClass.size

    companion object {
        fun fromJson(o: JSONObject): SparseHmm {
            val sc = o.getJSONArray("state_class").toIntArray()
            val n = sc.size
            val start = DoubleArray(n) { Double.NEGATIVE_INFINITY }
            val st = o.getJSONArray("start")
            for (k in 0 until st.length()) {
                val e = st.getJSONArray(k)
                start[e.getInt(0)] = e.getDouble(1)
            }
            val fin = o.getJSONArray("final").let { a -> BooleanArray(n) { a.getBoolean(it) } }
            val incoming = Array(n) { mutableListOf<Pair<Int, Double>>() }
            val tr = o.getJSONArray("transitions")
            for (k in 0 until tr.length()) {
                val e = tr.getJSONArray(k)
                incoming[e.getInt(1)].add(e.getInt(0) to e.getDouble(2))
            }
            val sorted = incoming.map { l -> l.sortedBy { it.first } }
            return SparseHmm(
                sc, start, fin,
                Array(n) { j -> IntArray(sorted[j].size) { sorted[j][it].first } },
                Array(n) { j -> DoubleArray(sorted[j].size) { sorted[j][it].second } },
            )
        }
    }
}

/** Every parameter of the device pipeline (bundle/pipeline.json, written by src/embedded/phase1_export.py). */
class PipelineConfig(
    val fs: Double,
    val window: Int,
    val stride: Int,
    val timelineHz: Int,
    val sos: Array<DoubleArray>,
    val zi: Array<DoubleArray>,
    val prior: DoubleArray,
    val emissionFloor: Double,
    val alpha: Double,
    val segmentation: SparseHmm,
    val lagFrames: Int,
    val llrHz: Int,
    val llrAlpha: Double,
    val episodeBlocks: Int,
    val intake: SparseHmm,
    val noIntake: SparseHmm,
    /** step (s) -> budget name -> threshold */
    val thresholds: Map<Int, Map<String, Double>>,
    val labels: List<String>,
    val modelFiles: Map<String, String>,
) {
    val samplesPerFrame: Int get() = (fs / timelineHz).toInt()
    val framesPerBlock: Int get() = timelineHz / llrHz
    val lookaheadS: Double get() = (window / 2 + stride) / fs

    companion object {
        fun parse(text: String): PipelineConfig {
            val o = JSONObject(text)
            val f = o.getJSONObject("filter")
            val llr = o.getJSONObject("llr")
            val th = llr.getJSONObject("thresholds")
            val files = o.getJSONObject("model").getJSONObject("files")
            return PipelineConfig(
                fs = o.getDouble("fs"),
                window = o.getInt("window"),
                stride = o.getInt("stride"),
                timelineHz = o.getInt("timeline_hz"),
                sos = f.getJSONArray("sos").toMatrix(),
                zi = f.getJSONArray("zi").toMatrix(),
                prior = o.getJSONArray("prior").toDoubleArray(),
                emissionFloor = o.getDouble("emission_floor"),
                alpha = o.getDouble("alpha"),
                segmentation = SparseHmm.fromJson(o.getJSONObject("segmentation_hmm")),
                lagFrames = o.getInt("viterbi_lag_frames"),
                llrHz = llr.getInt("hz"),
                llrAlpha = llr.getDouble("alpha"),
                episodeBlocks = llr.getInt("episode_blocks"),
                intake = SparseHmm.fromJson(llr.getJSONObject("intake_hmm")),
                noIntake = SparseHmm.fromJson(llr.getJSONObject("no_intake_hmm")),
                thresholds = th.keys().asSequence().associate { step ->
                    val b = th.getJSONObject(step)
                    step.toInt() to b.keys().asSequence().associateWith { b.getDouble(it) }
                },
                labels = o.getJSONArray("labels").let { a -> List(a.length()) { a.getString(it) } },
                modelFiles = files.keys().asSequence().associateWith { files.getString(it) },
            )
        }
    }
}

internal fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }
internal fun JSONArray.toDoubleArray() = DoubleArray(length()) { getDouble(it) }
internal fun JSONArray.toMatrix() = Array(length()) { getJSONArray(it).toDoubleArray() }
