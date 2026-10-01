package ru.tgdigest

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*

object C {
    const val BG = 0xFF0E1621.toInt()
    const val CARD = 0xFF17212B.toInt()
    const val CARD2 = 0xFF1F2C3A.toInt()
    const val TEXT = 0xFFE8EEF4.toInt()
    const val DIM = 0xFF8A9BAD.toInt()
    const val ACC = 0xFF2AABEE.toInt()
    val TOPIC = intArrayOf(0xFF2AABEE.toInt(), 0xFFE5894F.toInt(), 0xFF6CC15A.toInt(), 0xFFB36AE2.toInt(),
        0xFFE05A6D.toInt(), 0xFF3CC4B5.toInt(), 0xFFE0B341.toInt(), 0xFF7A8CFF.toInt())
}

fun Context.dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

fun Context.round(color: Int, r: Int = 12): GradientDrawable =
    GradientDrawable().apply { setColor(color); cornerRadius = dp(r).toFloat() }

fun Context.label(text: String, size: Float = 15f, color: Int = C.TEXT, bold: Boolean = false): TextView =
    TextView(this).apply {
        this.text = text; textSize = size; setTextColor(color)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

fun Context.button(text: String, filled: Boolean = true, onClick: () -> Unit): Button =
    Button(this).apply {
        this.text = text; isAllCaps = false; textSize = 14f
        setTextColor(if (filled) Color.WHITE else C.ACC)
        background = round(if (filled) C.ACC else C.CARD2, 10)
        setPadding(dp(14), dp(8), dp(14), dp(8))
        minHeight = 0; minimumHeight = dp(40)
        setOnClickListener { onClick() }
    }

fun Context.edit(hint: String, value: String = "", multiline: Boolean = false): EditText =
    EditText(this).apply {
        this.hint = hint; setText(value); setTextColor(C.TEXT); setHintTextColor(C.DIM); textSize = 15f
        background = round(C.CARD2, 10)
        setPadding(dp(12), dp(10), dp(12), dp(10))
        inputType = if (multiline) InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        else InputType.TYPE_CLASS_TEXT
    }

fun Context.chip(text: String, color: Int): TextView =
    TextView(this).apply {
        this.text = text; textSize = 11f; setTextColor(Color.WHITE)
        background = round(color, 8)
        setPadding(dp(8), dp(3), dp(8), dp(3))
    }

fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT,
       l: Int = 0, t: Int = 0, r: Int = 0, b: Int = 0, weight: Float = 0f): LinearLayout.LayoutParams =
    LinearLayout.LayoutParams(w, h, weight).apply { setMargins(l, t, r, b) }

fun Context.vbox(pad: Int = 0): LinearLayout =
    LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(pad), dp(pad), dp(pad), dp(pad)) }

fun Context.hbox(): LinearLayout =
    LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }

fun topicColor(name: String, order: List<String>): Int {
    val i = order.indexOf(name).let { if (it < 0) name.hashCode().let { h -> Math.abs(h) } else it }
    return C.TOPIC[i % C.TOPIC.size]
}

fun View.setPadAll(px: Int) = setPadding(px, px, px, px)
