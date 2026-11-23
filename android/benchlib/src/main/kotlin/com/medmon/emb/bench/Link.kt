package com.medmon.emb.bench

import android.content.Context
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

object Paths {
    const val START = "/emb/start"
    const val STOP = "/emb/stop"
    const val RAW = "/emb/raw"
    const val PROBS = "/emb/probs"
    const val ALARM = "/emb/alarm"
    const val STATUS = "/emb/status"
}

/**
 * Batches sent over the Bluetooth link. Every batch has a sequence number, so the receiver can
 * restore the order, and END marks the last batch of a replayed stream.
 *
 * Raw (P1):   seq, stream, firstSample (long), n, flags, totalSamples (long), n * 6 float32
 * Probs (P2): seq, stream, firstWindow (long), n, flags, totalSamples (long), n * 5 float32
 */
class Batch(
    val seq: Int,
    val stream: Int,
    val first: Long,
    val n: Int,
    val end: Boolean,
    val totalSamples: Long,
    val values: FloatArray,
) {
    fun encode(): ByteArray {
        val b = ByteBuffer.allocate(HEADER + values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        b.putInt(seq).putInt(stream).putLong(first).putInt(n).putInt(if (end) 1 else 0).putLong(totalSamples)
        b.asFloatBuffer().put(values)
        return b.array()
    }

    companion object {
        const val HEADER = 32

        fun decode(bytes: ByteArray): Batch {
            val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val seq = b.int
            val stream = b.int
            val first = b.long
            val n = b.int
            val flags = b.int
            val total = b.long
            val v = FloatArray((bytes.size - HEADER) / 4)
            b.asFloatBuffer().get(v)
            return Batch(seq, stream, first, n, flags and 1 != 0, total, v)
        }
    }
}

/** Alarm raised on one device, sent to the other: stream, step, budget, stream time. */
class AlarmMsg(val stream: Int, val step: Int, val budget: String, val streamTimeS: Double) {
    fun encode(): ByteArray = "$stream,$step,$budget,$streamTimeS".toByteArray()

    companion object {
        fun decode(b: ByteArray) = String(b).split(",").let { AlarmMsg(it[0].toInt(), it[1].toInt(), it[2], it[3].toDouble()) }
    }
}

/** Data Layer messages to the other device of the pair, with byte counters for the run logger. */
class Link(ctx: Context, private val onMessage: (MessageEvent) -> Unit) : MessageClient.OnMessageReceivedListener {
    private val client = Wearable.getMessageClient(ctx)
    private val nodes = Wearable.getNodeClient(ctx)
    @Volatile private var peer: String? = null
    var logger: RunLogger? = null

    fun open() {
        client.addListener(this)
        peer = runCatching { Tasks.await(nodes.connectedNodes, 10, TimeUnit.SECONDS).firstOrNull()?.id }.getOrNull()
    }

    fun close() {
        client.removeListener(this)
    }

    val connected: Boolean get() = peer != null

    /** Fire and forget; failures are logged. Returns false when no peer is connected. */
    fun send(path: String, data: ByteArray): Boolean {
        val id = peer ?: return false
        logger?.let { it.txBytes += data.size; it.txMsgs++ }
        client.sendMessage(id, path, data).addOnFailureListener { e ->
            logger?.event("link_error", detail = "$path: ${e.message}")
        }
        return true
    }

    override fun onMessageReceived(e: MessageEvent) {
        logger?.let { it.rxBytes += e.data.size; it.rxMsgs++ }
        onMessage(e)
    }
}

/** Restores the sequence order of batches (Data Layer messages may overtake each other). */
class Reorder(private val deliver: (Batch) -> Unit) {
    private var next = 0
    private val pending = java.util.TreeMap<Int, Batch>()
    var maxPending = 0
        private set

    fun reset() {
        next = 0; pending.clear()
    }

    fun offer(b: Batch) {
        pending[b.seq] = b
        maxPending = maxOf(maxPending, pending.size)
        while (pending.isNotEmpty() && pending.firstKey() == next) {
            deliver(pending.pollFirstEntry()!!.value)
            next++
        }
    }
}
