package com.medmon.emb

import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Whole replay (test_wild + test_external, 19 h) through the Kotlin pipeline with the INT8 model,
 * against the alarms of src/embedded/phase1_reference.py. Opt-in (about a minute):
 *
 *     ./gradlew :core:test --tests '*ReplayParityTest*' -Preplay
 */
class ReplayParityTest {
    @Test
    fun alarmsMatchPythonReference() {
        val replay = System.getProperty("replay.dir")
        assumeTrue("set -Preplay", replay != null && File(replay).isDirectory)
        val bundle = File(System.getProperty("bundle.dir"))
        val cfg = PipelineConfig.parse(File(bundle, "pipeline.json").readText())
        val ref = JSONObject(File(System.getProperty("reference.dir"), "int8/alarms.json").readText())
        val steps = listOf(2, 5, 10)
        var same = 0
        var total = 0
        var windows = 0L
        var nanos = 0L
        OnnxWindowClassifier(File(bundle, cfg.modelFiles.getValue("int8")).readBytes(), threads = 1).use { clf ->
            for (name in listOf("test_wild", "test_external")) {
                ReplaySet(File(replay), name).use { set ->
                    val r = ref.getJSONObject(name)
                    for (s in set.streams) {
                        val alarms = HashMap<String, MutableList<Double>>()
                        val listener = object : BackEnd.Listener {
                            override fun onAlarm(stepS: Int, budget: String, streamTimeS: Double) {
                                alarms.getOrPut("$stepS/$budget") { mutableListOf() }.add(streamTimeS)
                            }
                        }
                        val p = Pipeline(cfg, clf, steps, listener)
                        val x = set.read(s)
                        for (i in 0 until s.length) p.push(x.copyOfRange(i * 6, i * 6 + 6))
                        p.endStream()
                        windows += p.clock.counts[Stage.INFERENCE.ordinal]
                        nanos += p.clock.nanos[Stage.INFERENCE.ordinal]
                        val want = r.getJSONObject(s.sid)
                        for (st in steps) for (b in cfg.thresholds.getValue(st).keys) {
                            val w = want.getJSONObject("$st").getJSONArray(b).let { a -> List(a.length()) { a.getDouble(it) } }
                            total++
                            if (w == (alarms["$st/$b"] ?: emptyList<Double>())) same++
                            else println("${s.sid} $st/$b: python $w, kotlin ${alarms["$st/$b"]}")
                        }
                    }
                }
            }
        }
        println("alarm lists identical: $same / $total; inference ${nanos / 1e3 / windows} us/window over $windows windows")
        assert(same >= total * 0.99) { "only $same of $total alarm lists identical" }
    }
}
