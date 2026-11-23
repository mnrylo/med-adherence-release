package com.medmon.emb.phone

import android.app.Activity
import android.app.Application
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.google.android.gms.wearable.Wearable
import com.medmon.emb.bench.BenchConfig
import com.medmon.emb.bench.BenchService
import com.medmon.emb.bench.Condition
import com.medmon.emb.bench.Paths
import com.medmon.emb.bench.Role
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PhoneApp : Application() {
    override fun onCreate() {
        super.onCreate()
        BenchService.role = Role.PHONE
    }
}

/**
 * Starts runs on both devices. "Start" sends the configuration to the watch (/emb/start) and starts
 * the phone side; for LOCAL, "Start here" runs on the phone and "Start on watch" on the watch only.
 */
class MainActivity : Activity() {
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var status: TextView
    private val fields = LinkedHashMap<String, Spinner>()
    private lateinit var sensorsOn: CheckBox
    private lateinit var logProbs: CheckBox
    private lateinit var duration: EditText
    private lateinit var maxStreams: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        fun spinner(label: String, values: List<String>, default: String) {
            col.addView(TextView(this).apply { text = label })
            val s = Spinner(this).apply {
                adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, values)
                setSelection(values.indexOf(default))
            }
            fields[label] = s
            col.addView(s)
        }
        spinner("condition", Condition.entries.map { it.name }, "P3")
        spinner("source", listOf("replay", "live"), "replay")
        spinner("set", listOf("test_wild", "test_external"), "test_wild")
        spinner("pacing", listOf("realtime", "fast"), "realtime")
        spinner("precision", listOf("int8", "fp16", "fp32"), "int8")
        spinner("llr step (s)", listOf("2", "5", "10", "2,5,10"), "2")
        sensorsOn = CheckBox(this).apply { text = "sensors on during replay" }
        logProbs = CheckBox(this).apply { text = "log window probabilities" }
        duration = EditText(this).apply { hint = "duration (s), 0 = whole set"; inputType = InputType.TYPE_CLASS_NUMBER }
        maxStreams = EditText(this).apply { hint = "max streams, 0 = all"; inputType = InputType.TYPE_CLASS_NUMBER }
        listOf(sensorsOn, logProbs, duration, maxStreams).forEach(col::addView)
        col.addView(Button(this).apply { text = "Start (watch + phone)"; setOnClickListener { start(watch = true, phone = true) } })
        col.addView(Button(this).apply { text = "Start here (phone only)"; setOnClickListener { start(watch = false, phone = true) } })
        col.addView(Button(this).apply { text = "Start on watch only"; setOnClickListener { start(watch = true, phone = false) } })
        col.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener {
                BenchService.stop(this@MainActivity)
                send(Paths.STOP, ByteArray(0))
            }
        })
        status = TextView(this).apply { setPadding(0, 24, 0, 0) }
        col.addView(status)
        setContentView(ScrollView(this).apply { addView(col) })
    }

    private fun value(k: String) = fields.getValue(k).selectedItem as String

    private fun config(): BenchConfig {
        val c = Condition.valueOf(value("condition"))
        val id = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date()) + "_" + c.name
        return BenchConfig(
            runId = id, condition = c, source = value("source"), set = value("set"), pacing = value("pacing"),
            precision = value("precision"), steps = value("llr step (s)").split(",").map { it.trim().toInt() },
            sensorsOn = sensorsOn.isChecked, logProbs = logProbs.isChecked,
            durationS = duration.text.toString().toIntOrNull() ?: 0,
            maxStreams = maxStreams.text.toString().toIntOrNull() ?: 0,
        )
    }

    private fun start(watch: Boolean, phone: Boolean) {
        val cfg = config().toJson().toString()
        if (watch) send(Paths.START, cfg.toByteArray())
        if (phone) BenchService.start(this, cfg)
    }

    private fun send(path: String, data: ByteArray) {
        Wearable.getNodeClient(this).connectedNodes.addOnSuccessListener { nodes ->
            if (nodes.isEmpty()) status.append("\nno watch connected")
            nodes.forEach { Wearable.getMessageClient(this).sendMessage(it.id, path, data) }
        }
    }

    private val tick = object : Runnable {
        override fun run() {
            val e = BenchService.engine
            status.text = if (e == null) "idle" else buildString {
                append(e.status)
                if (e.running) append("\nsamples ${e.logger.samples}  windows ${e.logger.windows}  alarms ${e.logger.nAlarms}" +
                    "\nrx ${e.logger.rxBytes / 1024} KiB in ${e.logger.rxMsgs} messages")
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
