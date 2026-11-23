package com.medmon.emb.wear

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.medmon.emb.bench.BenchService
import com.medmon.emb.bench.Role

class WatchApp : Application() {
    override fun onCreate() {
        super.onCreate()
        BenchService.role = Role.WATCH
    }
}

/** Status of the current run; runs are started from the phone app or by adb (ControlActivity). */
class MainActivity : Activity() {
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply { gravity = Gravity.CENTER; textSize = 12f; setPadding(24, 48, 24, 8) }
        val stop = Button(this).apply { text = "Stop"; setOnClickListener { BenchService.stop(this@MainActivity) } }
        setContentView(ScrollView(this).apply {
            addView(LinearLayout(this@MainActivity).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                addView(status)
                addView(stop)
            })
        })
    }

    private val tick = object : Runnable {
        override fun run() {
            val e = BenchService.engine
            status.text = if (e == null) "EmbBench (watch)\nidle" else buildString {
                append("EmbBench (watch)\n${e.status}\n")
                if (e.running) append("samples ${e.logger.samples}\nwindows ${e.logger.windows}\nalarms ${e.logger.nAlarms}\n" +
                    "tx ${e.logger.txBytes / 1024} KiB")
            }
            handler.postDelayed(this, 1000)
        }
    }

    override fun onResume() {
        super.onResume(); handler.post(tick)
    }

    override fun onPause() {
        handler.removeCallbacks(tick); super.onPause()
    }
}
