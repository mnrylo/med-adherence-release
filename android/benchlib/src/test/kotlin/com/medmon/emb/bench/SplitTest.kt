package com.medmon.emb.bench

import com.medmon.emb.BackEnd
import com.medmon.emb.FrontEnd
import com.medmon.emb.Pipeline
import com.medmon.emb.PipelineConfig
import com.medmon.emb.WindowClassifier
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

/**
 * P1 and P2 split the pipeline over the Bluetooth link. Their result must equal the whole pipeline
 * on one device, even when the Data Layer delivers the batches out of order.
 */
class SplitTest {
    private val cfg = PipelineConfig.parse(File(System.getProperty("bundle.dir"), "pipeline.json").readText())
    private val steps = listOf(2, 5, 10)

    /** Deterministic stand-in for the network: softmax of per-channel means (no native ONNX on the host). */
    private class FakeClassifier(private val win: Int) : WindowClassifier {
        override fun classify(batch: FloatArray, n: Int): Array<FloatArray> {
            val z = DoubleArray(5) { k ->
                var s = 0.0
                for (t in 0 until win) s += batch[k * win + t] * (k + 1) + batch[5 * win + t] * 0.01 * k
                3 * s / win
            }
            val m = z.max()
            val e = z.map { exp(it - m) }
            return arrayOf(FloatArray(5) { (e[it] / e.sum()).toFloat() })
        }
    }

    private class Collect : BackEnd.Listener {
        val labels = ArrayList<Pair<Int, Int>>()
        val alarms = ArrayList<String>()
        override fun onLabel(frame: Int, label: Int) {
            labels.add(frame to label)
        }

        override fun onAlarm(stepS: Int, budget: String, streamTimeS: Double) {
            alarms.add("$stepS/$budget@$streamTimeS")
        }
    }

    /** Two synthetic streams with slow orientation changes and bursts of movement. */
    private fun streams(): List<Array<FloatArray>> = listOf(7_345, 12_010).mapIndexed { si, n ->
        val r = Random(si)
        Array(n) { t ->
            val burst = if ((t / 700) % 3 == 1) 1.0 else 0.1
            FloatArray(6) { c ->
                (sin(t * (0.002 + 0.001 * c) + si) + burst * sin(t * 0.3 * (c + 1)) * 0.5 + r.nextDouble(-0.05, 0.05)).toFloat()
            }
        }
    }

    private fun whole(): List<Collect> = streams().map { x ->
        val c = Collect()
        val p = Pipeline(cfg, FakeClassifier(cfg.window), steps, c)
        x.forEach(p::push)
        p.endStream()
        c
    }

    /** Deliver encoded batches in a shuffled order (bounded displacement, like a real link). */
    private fun <T> shuffled(items: List<T>, r: Random): List<T> {
        val out = items.toMutableList()
        for (i in out.indices) {
            val j = (i + r.nextInt(0, 4)).coerceAtMost(out.size - 1)
            out[i] = out[j].also { out[j] = out[i] }
        }
        return out
    }

    @Test
    fun p2SplitEqualsWholePipeline() {
        val want = whole()
        val r = Random(42)
        streams().forEachIndexed { si, x ->
            val msgs = ArrayList<ByteArray>()
            var seq = 0
            val pend = ArrayList<FloatArray>()
            var first = 0
            val front = FrontEnd(cfg, FakeClassifier(cfg.window)) { j, p ->
                if (pend.isEmpty()) first = j
                pend.add(p)
                if (pend.size == 2) {
                    msgs.add(Batch(seq++, si, first.toLong(), 2, false, -1, pend.flatMap { it.toList() }.toFloatArray()).encode())
                    pend.clear()
                }
            }
            x.forEach(front::push)
            front.flush()
            msgs.add(Batch(seq++, si, first.toLong(), pend.size, true, front.samples,
                pend.flatMap { it.toList() }.toFloatArray()).encode())
            val c = Collect()
            val back = BackEnd(cfg, steps, listener = c)
            val ro = Reorder { b ->
                for (i in 0 until b.n) back.onWindow((b.first + i).toInt(), b.values.copyOfRange(i * 5, i * 5 + 5))
                if (b.end) back.flush(b.totalSamples)
            }
            shuffled(msgs, r).forEach { ro.offer(Batch.decode(it)) }
            assertTrue("messages were reordered", ro.maxPending > 1)
            assertEquals(want[si].labels, c.labels)
            assertEquals(want[si].alarms, c.alarms)
        }
    }

    @Test
    fun p1SplitEqualsWholePipeline() {
        val want = whole()
        val r = Random(7)
        streams().forEachIndexed { si, x ->
            val msgs = x.toList().chunked(100).mapIndexed { k, rows ->
                val last = (k + 1) * 100 >= x.size
                Batch(k, si, k * 100L, rows.size, last, if (last) x.size.toLong() else -1,
                    rows.flatMap { it.toList() }.toFloatArray()).encode()
            }
            val c = Collect()
            val p = Pipeline(cfg, FakeClassifier(cfg.window), steps, c)
            val ro = Reorder { b ->
                for (i in 0 until b.n) p.push(b.values.copyOfRange(i * 6, i * 6 + 6))
                if (b.end) p.endStream()
            }
            shuffled(msgs, r).forEach { ro.offer(Batch.decode(it)) }
            assertEquals(want[si].labels, c.labels)
            assertEquals(want[si].alarms, c.alarms)
        }
    }

    @Test
    fun alarmMessageRoundTrip() {
        val a = AlarmMsg.decode(AlarmMsg(12, 2, "fa10", 61.5).encode())
        assertEquals(listOf(12, 2), listOf(a.stream, a.step))
        assertEquals("fa10", a.budget)
        assertEquals(61.5, a.streamTimeS, 0.0)
    }
}
