package com.medmon.emb.bench

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import java.util.concurrent.TimeUnit
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import com.google.android.gms.wearable.WearableListenerService

/**
 * Foreground service that owns the engine of a run. Started by the app UI, by a /emb/start message
 * from the other device (BenchListenerService) or by adb through ControlActivity:
 *
 *   adb shell am start -n com.medmon.embbench/com.medmon.emb.bench.ControlActivity \
 *       --es action start --es config_b64 <base64 of {"run_id":"r1","condition":"P3",...}>
 *   adb shell am start -n com.medmon.embbench/com.medmon.emb.bench.ControlActivity --es action stop
 */
class BenchService : Service() {
    companion object {
        const val ACTION_START = "com.medmon.emb.START"
        const val ACTION_STOP = "com.medmon.emb.STOP"
        const val EXTRA_CONFIG = "config"
        private const val CHANNEL = "bench"

        /** Set by the app module (watch or phone). */
        @Volatile var role: Role = Role.PHONE
        @Volatile var engine: BenchEngine? = null
            private set

        fun start(ctx: Context, config: String) = ctx.startForegroundService(
            Intent(ctx, BenchService::class.java).setAction(ACTION_START).putExtra(EXTRA_CONFIG, config))

        fun stop(ctx: Context) = ctx.startService(Intent(ctx, BenchService::class.java).setAction(ACTION_STOP))
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                foreground()
                if (engine?.running == true) return START_NOT_STICKY
                val cfg = BenchConfig.parse(intent.getStringExtra(EXTRA_CONFIG) ?: return START_NOT_STICKY)
                if (cfg.wakeLock) wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "EmbBench:run").apply { acquire() }
                engine = BenchEngine(applicationContext, role) { finished() }.also { it.start(cfg) }
            }
            ACTION_STOP -> engine?.stop("command") ?: finished()
        }
        return START_NOT_STICKY
    }

    private fun finished() {
        wakeLock?.takeIf { it.isHeld }?.release()
        wakeLock = null
        engine?.shutdown()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun foreground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Benchmark", NotificationManager.IMPORTANCE_LOW))
        val n = Notification.Builder(this, CHANNEL).setContentTitle("Embedded benchmark running")
            .setSmallIcon(android.R.drawable.ic_media_play).setOngoing(true).build()
        if (Build.VERSION.SDK_INT >= 34) startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        else startForeground(1, n)
    }

    override fun onDestroy() {
        wakeLock?.takeIf { it.isHeld }?.release()
        super.onDestroy()
    }
}

/** Starts and stops runs on request of the other device of the pair. */
class BenchListenerService : WearableListenerService() {
    override fun onMessageReceived(e: MessageEvent) {
        when (e.path) {
            Paths.START -> BenchService.start(this, String(e.data))
            Paths.STOP -> if (BenchService.engine?.running == true) BenchService.stop(this)
        }
    }
}

/**
 * adb entry point (an activity may start a foreground service; a broadcast may not).
 * With `--ez pair true` the run is also started on the other device through the Data Layer, so the
 * PC needs adb to one device only (the watch drops Wi-Fi adb when it is off the charger).
 */
class ControlActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pair = intent.getBooleanExtra("pair", false)
        when (intent.getStringExtra("action")) {
            "start" -> (intent.getStringExtra("config_b64")?.let { String(android.util.Base64.decode(it, 0)) }
                ?: intent.getStringExtra("config"))?.let { cfg ->
                BenchService.start(this, cfg)
                if (pair) toPeer(Paths.START, cfg.toByteArray())
            }
            "stop" -> {
                BenchService.stop(this)
                if (pair) toPeer(Paths.STOP, ByteArray(0))
            }
        }
        finish()
    }

    private fun toPeer(path: String, data: ByteArray) {
        val app = applicationContext
        Thread {
            val r = runCatching {
                val nodes = Tasks.await(Wearable.getNodeClient(app).connectedNodes, 10, TimeUnit.SECONDS)
                nodes.forEach { Tasks.await(Wearable.getMessageClient(app).sendMessage(it.id, path, data), 10, TimeUnit.SECONDS) }
                nodes.size
            }
            android.util.Log.i("EmbBench", "pair $path -> ${r.getOrElse { "failed: $it" }} node(s)")
        }.start()
    }
}
