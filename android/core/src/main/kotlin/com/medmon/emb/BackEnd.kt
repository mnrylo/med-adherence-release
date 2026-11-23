package com.medmon.emb

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/** Index of the first maximum (numpy argmax). */
internal fun argmax(v: DoubleArray, mask: BooleanArray? = null): Int {
    var best = Double.NEGATIVE_INFINITY
    var arg = -1
    for (i in v.indices) {
        val x = if (mask == null || mask[i]) v[i] else Double.NEGATIVE_INFINITY
        if (arg < 0 || x > best) {
            best = x
            arg = i
        }
    }
    return arg
}

internal fun logSumExp(v: DoubleArray, n: Int = v.size): Double {
    var m = Double.NEGATIVE_INFINITY
    for (i in 0 until n) if (v[i] > m) m = v[i]
    if (m == Double.NEGATIVE_INFINITY) return m
    var s = 0.0
    for (i in 0 until n) s += exp(v[i] - m)
    return m + ln(s)
}

/** (log clip(p, floor, 1) - log prior) * alpha, per class. */
class Emissions(private val prior: DoubleArray, private val floor: Double, private val alpha: Double) {
    private val logPrior = DoubleArray(prior.size) { ln(prior[it]) }
    fun of(p: DoubleArray, out: DoubleArray = DoubleArray(p.size)): DoubleArray {
        for (k in p.indices) out[k] = (ln(p[k].coerceIn(floor, 1.0)) - logPrior[k]) * alpha
        return out
    }
}

/**
 * Fixed-lag Viterbi: frame u is decided when frame u + lag arrives, by backtracking from the best
 * state at u + lag. flush() decides the remaining frames with the full path (final states only).
 */
class FixedLagViterbi(private val hmm: SparseHmm, private val lag: Int) {
    private val n = hmm.size
    private var delta = DoubleArray(n)
    private var next = DoubleArray(n)
    private val back = Array(lag + 1) { IntArray(n) }
    var t = -1
        private set

    fun reset() {
        t = -1
    }

    /** Pushes the class log-emissions of frame t; returns the class decided for frame t - lag, or -1. */
    fun push(e: DoubleArray): Int {
        t++
        if (t == 0) {
            for (j in 0 until n) delta[j] = hmm.logStart[j] + e[hmm.stateClass[j]]
        } else {
            val bp = back[t % (lag + 1)]
            for (j in 0 until n) {
                val from = hmm.inFrom[j]
                val lp = hmm.inLogP[j]
                var best = Double.NEGATIVE_INFINITY
                var arg = 0
                for (k in from.indices) {
                    val v = delta[from[k]] + lp[k]
                    if (v > best) {
                        best = v
                        arg = from[k]
                    }
                }
                bp[j] = arg
                next[j] = best + e[hmm.stateClass[j]]
            }
            val tmp = delta; delta = next; next = tmp
        }
        if (t < lag) return -1
        var cur = argmax(delta)
        for (j in 0 until lag) cur = back[(t - j) % (lag + 1)][cur]
        return hmm.stateClass[cur]
    }

    /** Classes of the frames not decided yet (max(0, t - lag + 1) .. t). */
    fun flush(): IntArray {
        if (t < 0) return IntArray(0)
        val first = max(0, t - lag + 1)
        val out = IntArray(t - first + 1)
        var cur = argmax(delta, hmm.final)
        out[out.size - 1] = hmm.stateClass[cur]
        for (k in t downTo first + 1) {
            cur = back[k % (lag + 1)][cur]
            out[k - 1 - first] = hmm.stateClass[cur]
        }
        return out
    }
}

/** Forward log-likelihood of a sequence of class log-emissions (rows offset until offset + len). */
class Forward(private val hmm: SparseHmm) {
    private var a = DoubleArray(hmm.size)
    private var b = DoubleArray(hmm.size)
    private val tmp = DoubleArray(hmm.inFrom.maxOf { it.size }.coerceAtLeast(1))
    private val fin = DoubleArray(hmm.size)

    fun logLik(e: Array<DoubleArray>, rows: IntArray): Double {
        val n = hmm.size
        for (j in 0 until n) a[j] = hmm.logStart[j] + e[rows[0]][hmm.stateClass[j]]
        for (r in 1 until rows.size) {
            val er = e[rows[r]]
            for (j in 0 until n) {
                val from = hmm.inFrom[j]
                val lp = hmm.inLogP[j]
                for (k in from.indices) tmp[k] = a[from[k]] + lp[k]
                b[j] = logSumExp(tmp, from.size) + er[hmm.stateClass[j]]
            }
            val s = a; a = b; b = s
        }
        var m = 0
        for (j in 0 until n) if (hmm.final[j]) fin[m++] = a[j]
        return logSumExp(fin, m)
    }
}

/**
 * Window probabilities -> 10 Hz timeline -> fixed-lag Gesture Timeline and streaming intake LLR.
 *
 * Frame u (sample u * samplesPerFrame) averages windows lo..hi,
 * lo = ceil((u * spf - window/2) / stride), hi = floor((u * spf + window/2) / stride).
 * Block k (2 Hz) is the mean of frames 5k..5k+4. Every `step` blocks, the episode of the last
 * episodeBlocks blocks is scored; an alarm is the first positive episode of a run.
 */
class BackEnd(
    private val cfg: PipelineConfig,
    /** steps (s) to score; each with every budget of cfg.thresholds */
    private val steps: List<Int>,
    private val clock: StageClock = StageClock(),
    private val listener: Listener,
) {
    interface Listener {
        fun onFrame(frame: Int, probs: DoubleArray) {}
        fun onLabel(frame: Int, label: Int) {}
        fun onEpisode(stepS: Int, endBlock: Int, score: Double) {}
        fun onAlarm(stepS: Int, budget: String, streamTimeS: Double) {}
    }

    private val k = cfg.prior.size
    private val half = cfg.window / 2
    private val spf = cfg.samplesPerFrame
    private val probs = ArrayDeque<FloatArray>()
    private var probsBase = 0
    private var nWindows = 0
    private var nFrames = 0
    private val emis = Emissions(cfg.prior, cfg.emissionFloor, cfg.alpha)
    private val llrEmis = Emissions(cfg.prior, cfg.emissionFloor, cfg.llrAlpha)
    private val viterbi = FixedLagViterbi(cfg.segmentation, cfg.lagFrames)
    private val fwdIn = Forward(cfg.intake)
    private val fwdOut = Forward(cfg.noIntake)
    private val e10 = DoubleArray(k)
    private val blockSum = DoubleArray(k)
    private var blockFrames = 0
    private var nBlocks = 0
    private val blockE = Array(cfg.episodeBlocks) { DoubleArray(k) }
    private val stepBlocks = steps.associateWith { (it * cfg.llrHz) }
    private val positive = HashMap<String, Boolean>()
    private val scored = HashSet<Int>()

    fun reset() {
        probs.clear(); probsBase = 0; nWindows = 0; nFrames = 0
        viterbi.reset(); blockFrames = 0; nBlocks = 0; positive.clear(); scored.clear()
        blockSum.fill(0.0)
    }

    private fun lo(u: Int) = Math.floorDiv(u * spf - half + cfg.stride - 1, cfg.stride).coerceAtLeast(0)
    private fun hi(u: Int) = Math.floorDiv(u * spf + half, cfg.stride)

    fun onWindow(index: Int, p: FloatArray) {
        check(index == nWindows) { "windows must arrive in order: got $index, expected $nWindows" }
        probs.addLast(p)
        nWindows++
        while (hi(nFrames) <= nWindows - 1) frame(nWindows - 1)
    }

    /** End of stream with nSamples samples (after FrontEnd.flush()). */
    fun flush(nSamples: Long) {
        val total = ((nSamples + spf - 1) / spf).toInt()
        if (nWindows > 0) while (nFrames < total) frame(nWindows - 1)
        clock.start(Stage.VITERBI)
        val tail = viterbi.flush()
        clock.stop(Stage.VITERBI)
        val first = nFrames - tail.size
        for (i in tail.indices) listener.onLabel(first + i, tail[i])
        if (nFrames > 0 && steps.any { it !in scored }) {
            if (nBlocks == 0) closeBlock(blockFrames)
            for (st in steps) if (st !in scored) score(st)
        }
    }

    private fun frame(lastWindow: Int) {
        clock.start(Stage.TIMELINE)
        val u = nFrames
        val l = lo(u).coerceAtMost(lastWindow)
        val h = hi(u).coerceAtMost(lastWindow).coerceAtLeast(l)
        val mean = DoubleArray(k)
        for (j in l..h) {
            val p = probs[j - probsBase]
            for (c in 0 until k) mean[c] += p[c].toDouble()
        }
        for (c in 0 until k) mean[c] /= (h - l + 1).toDouble()
        nFrames++
        while (probs.size > 1 && probsBase < lo(nFrames).coerceAtMost(lastWindow)) {
            probs.removeFirst(); probsBase++
        }
        clock.stop(Stage.TIMELINE)
        listener.onFrame(u, mean)

        clock.start(Stage.VITERBI)
        val lab = viterbi.push(emis.of(mean, e10))
        clock.stop(Stage.VITERBI)
        if (lab >= 0) listener.onLabel(u - cfg.lagFrames, lab)

        clock.start(Stage.LLR)
        for (c in 0 until k) blockSum[c] += mean[c]
        blockFrames++
        clock.stop(Stage.LLR)
        if (blockFrames == cfg.framesPerBlock) {
            closeBlock(blockFrames)
            for (st in steps) if (nBlocks % stepBlocks.getValue(st) == 0) score(st)
        }
    }

    private fun closeBlock(n: Int) {
        clock.start(Stage.LLR)
        val m = DoubleArray(k) { blockSum[it] / n }
        llrEmis.of(m, blockE[nBlocks % cfg.episodeBlocks])
        nBlocks++
        blockSum.fill(0.0)
        blockFrames = 0
        clock.stop(Stage.LLR)
    }

    private fun score(stepS: Int) {
        clock.start(Stage.LLR)
        val w = minOf(nBlocks, cfg.episodeBlocks)
        val rows = IntArray(w) { (nBlocks - w + it) % cfg.episodeBlocks }
        val s = fwdIn.logLik(blockE, rows) - fwdOut.logLik(blockE, rows)
        scored.add(stepS)
        clock.stop(Stage.LLR)
        listener.onEpisode(stepS, nBlocks, s)
        for ((budget, tau) in cfg.thresholds.getValue(stepS)) {
            val key = "$stepS/$budget"
            val pos = s > tau
            if (pos && positive[key] != true) listener.onAlarm(stepS, budget, nBlocks.toDouble() / cfg.llrHz)
            positive[key] = pos
        }
    }
}
