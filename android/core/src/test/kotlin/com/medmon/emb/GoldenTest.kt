package com.medmon.emb

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.abs

/**
 * Parity with the Python reference (src/embedded/devicepipe.py) on the golden streams of embedded.phase1_export.
 *
 * * Filter: same arithmetic in double precision -> 1e-9.
 * * Classifier (ONNX Runtime JVM vs Python, same version and model): 1e-4 and the same argmax.
 * * Back end fed with the reference probabilities: timeline 1e-9, Gesture Timeline identical,
 *   episode scores 1e-6, alarms identical.
 * * End to end (our own probabilities): Gesture Timeline >= 99.5 % identical, alarms identical.
 */
class GoldenTest {
    private val bundleDir = File(System.getProperty("bundle.dir"))
    private val cfg = PipelineConfig.parse(File(bundleDir, "pipeline.json").readText())
    private val steps = listOf(2, 5, 10)

    private fun goldens(): List<JSONObject> =
        File(javaClass.getResource("/golden")!!.toURI()).listFiles { f -> f.name.endsWith(".json.gz") }!!
            .sortedBy { it.name }
            .map { f -> JSONObject(GZIPInputStream(f.inputStream()).bufferedReader().readText()) }

    private fun JSONArray.rows() = Array(length()) { getJSONArray(it).toDoubleArray() }

    private class Collect : BackEnd.Listener {
        val frames = ArrayList<DoubleArray>()
        val labels = HashMap<Int, Int>()
        val scores = HashMap<Int, ArrayList<Pair<Int, Double>>>()
        val alarms = HashMap<String, ArrayList<Double>>()
        override fun onFrame(frame: Int, probs: DoubleArray) {
            frames.add(probs.copyOf())
        }

        override fun onLabel(frame: Int, label: Int) {
            check(labels.put(frame, label) == null) { "frame $frame decided twice" }
        }

        override fun onEpisode(stepS: Int, endBlock: Int, score: Double) {
            scores.getOrPut(stepS) { ArrayList() }.add(endBlock to score)
        }

        override fun onAlarm(stepS: Int, budget: String, streamTimeS: Double) {
            alarms.getOrPut("$stepS/$budget") { ArrayList() }.add(streamTimeS)
        }
    }

    private fun checkBack(g: JSONObject, c: Collect, exactPath: Boolean) {
        val sid = g.getString("sid")
        val path = g.getJSONArray("path").toIntArray()
        assertEquals("$sid frames", path.size, c.labels.size)
        val same = path.indices.count { c.labels[it] == path[it] }
        if (exactPath) assertEquals("$sid path", path.size, same)
        else assertTrue("$sid path agreement ${same.toDouble() / path.size}", same >= 0.995 * path.size)
        val llr = g.getJSONObject("llr")
        for (st in steps) {
            val ref = llr.getJSONObject("$st")
            val sc = ref.getJSONArray("scores").toDoubleArray()
            val ends = ref.getJSONArray("ends").toIntArray()
            val got = c.scores.getValue(st)
            assertEquals("$sid step $st episodes", sc.size, got.size)
            for (i in sc.indices) {
                assertEquals("$sid step $st end $i", ends[i], got[i].first)
                if (exactPath) assertEquals("$sid step $st score $i", sc[i], got[i].second, 1e-6)
            }
            val al = ref.getJSONObject("alarms")
            for (b in al.keys()) {
                val want = al.getJSONArray(b).toDoubleArray().toList()
                assertEquals("$sid alarms $st/$b", want, c.alarms["$st/$b"] ?: emptyList<Double>())
            }
        }
    }

    @Test
    fun filterMatchesScipy() {
        for (g in goldens()) {
            val x = g.getJSONArray("x").rows()
            val want = g.getJSONArray("filtered").rows()
            val f = CausalHighPass(cfg.sos, cfg.zi)
            var worst = 0.0
            for (i in x.indices) {
                val v = x[i].copyOf()
                f.process(v)
                for (c in 0 until 6) worst = maxOf(worst, abs(v[c] - want[i][c]))
            }
            println("${g.getString("sid")}: filter max |diff| $worst")
            assertTrue(worst < 1e-9)
        }
    }

    @Test
    fun backEndMatchesReferenceOnReferenceProbabilities() {
        for (g in goldens()) {
            val probs = g.getJSONArray("probs").rows()
            val c = Collect()
            val back = BackEnd(cfg, steps, StageClock(), c)
            probs.forEachIndexed { j, p -> back.onWindow(j, FloatArray(p.size) { p[it].toFloat() }) }
            back.flush(g.getLong("n_samples"))
            val tl = g.getJSONArray("timeline").rows()
            assertEquals(tl.size, c.frames.size)
            var worst = 0.0
            for (u in tl.indices) for (k in tl[u].indices) worst = maxOf(worst, abs(tl[u][k] - c.frames[u][k]))
            println("${g.getString("sid")}: timeline max |diff| $worst")
            assertTrue(worst < 1e-6)
            checkBack(g, c, exactPath = true)
        }
    }

    @Test
    fun endToEndWithOnnxRuntime() {
        val model = File(bundleDir, cfg.modelFiles.getValue("fp32")).readBytes()
        OnnxWindowClassifier(model, threads = 4).use { clf ->
            for (g in goldens()) {
                val x = g.getJSONArray("x").rows()
                val want = g.getJSONArray("probs").rows()
                val got = ArrayList<FloatArray>()
                val c = Collect()
                val back = BackEnd(cfg, steps, StageClock(), c)
                val front = FrontEnd(cfg, clf) { j, p -> got.add(p); back.onWindow(j, p) }
                for (row in x) front.push(FloatArray(6) { row[it].toFloat() })
                front.flush()
                back.flush(front.samples)
                assertEquals(want.size, got.size)
                var worst = 0.0
                var agree = 0
                for (j in want.indices) {
                    for (k in want[j].indices) worst = maxOf(worst, abs(want[j][k] - got[j][k]))
                    if (argmax(want[j]) == argmax(DoubleArray(5) { got[j][it].toDouble() })) agree++
                }
                println("${g.getString("sid")}: ${want.size} windows, probs max |diff| $worst, argmax agree $agree")
                assertTrue(worst < 1e-4)
                assertEquals(want.size, agree)
                checkBack(g, c, exactPath = false)
            }
        }
    }
}
