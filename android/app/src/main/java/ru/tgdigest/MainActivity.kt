package ru.tgdigest

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONArray
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val ui = Handler(Looper.getMainLooper())
    private val bg = Executors.newSingleThreadExecutor()
    private val bgDl = Executors.newSingleThreadExecutor()
    private lateinit var store: Store
    private lateinit var core: PyObject
    private lateinit var helper: Helper
    private lateinit var content: FrameLayout
    private lateinit var tabs: LinearLayout
    private val logBuf = StringBuilder()
    private var logView: TextView? = null
    private var statusView: TextView? = null
    private var scanning = false
    private var loggedIn = false

    private lateinit var feedPage: View
    private lateinit var chanPage: View
    private lateinit var setPage: View
    private lateinit var logPage: View

    // feed state
    private var feed: List<FeedPost> = emptyList()
        private lateinit var feedAdapter: FeedAdapter
    private lateinit var feedEmpty: TextView
    // channels state
    private var channels: List<Chan> = emptyList()
    private var blacklist = mutableSetOf<Long>()
    private var chanFilter = ""
    private lateinit var chanAdapter: ChanAdapter
    private lateinit var chanCount: TextView

    private fun log(msg: String) {
        ui.post {
            val t = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
            logBuf.append(t).append("  ").append(msg).append('\n')
            if (logBuf.length > 60000) logBuf.delete(0, 20000)
            logView?.text = logBuf
            statusView?.text = msg
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = Store(this)
        helper = Helper(applicationContext) { log(it) }
        if (!Python.isStarted()) Python.start(AndroidPlatform(this))
        core = Python.getInstance().getModule("core")

        val root = vbox()
        root.setBackgroundColor(C.BG)
        content = FrameLayout(this)
        root.addView(content, lp(h = 0, weight = 1f))
        tabs = hbox().apply { setBackgroundColor(C.CARD) }
        root.addView(tabs, lp())
        setContentView(root)

        feedPage = buildFeed(); chanPage = buildChannels(); setPage = buildSettings(); logPage = buildLog()
        content.addView(label("Подключаюсь к Telegram…", 16f, C.DIM).apply { gravity = Gravity.CENTER }, FrameLayout.LayoutParams(-1, -1))

        bg.execute {
            try {
                val ok = core.callAttr("init", File(filesDir, "tg").absolutePath).toBoolean()
                ui.post { if (ok) enterMain() else showLogin() }
            } catch (t: Throwable) {
                log("Ошибка запуска: ${t.message}")
                ui.post { Toast.makeText(this, "Ошибка: ${t.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun show(v: View) {
        content.removeAllViews()
        (v.parent as? ViewGroup)?.removeView(v)
        content.addView(v, FrameLayout.LayoutParams(-1, -1))
    }

    // ------------------------------------------------------------ login
    private fun showLogin() {
        tabs.visibility = View.GONE
        val box = vbox(24).apply { gravity = Gravity.CENTER_VERTICAL }
        box.addView(label("Telegram Digest", 26f, C.TEXT, true))
        box.addView(label("Вход в ваш аккаунт Telegram. Всё работает на телефоне.", 14f, C.DIM), lp(t = dp(6), b = dp(20)))
        val phone = edit("Телефон, например +79991234567")
        phone.inputType = android.text.InputType.TYPE_CLASS_PHONE
        val code = edit("Код из Telegram").apply { visibility = View.GONE; inputType = android.text.InputType.TYPE_CLASS_NUMBER }
        val pass = edit("Пароль двухфакторной защиты").apply {
            visibility = View.GONE
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val status = label("", 13f, C.DIM)
        var step = 0
        val go = Button(this)
        go.text = "Получить код"; go.isAllCaps = false
        go.setTextColor(0xFFFFFFFF.toInt()); go.background = round(C.ACC, 10)
        go.setOnClickListener {
            go.isEnabled = false
            status.text = "Подождите…"
            bg.execute {
                try {
                    when (step) {
                        0 -> {
                            core.callAttr("send_code", phone.text.toString())
                            ui.post { step = 1; code.visibility = View.VISIBLE; go.text = "Войти"; status.text = "Код отправлен в Telegram (в чат «Telegram» на другом устройстве)." }
                        }
                        1 -> {
                            val r = core.callAttr("sign_in", code.text.toString()).toString()
                            ui.post {
                                if (r == "ok") enterMain()
                                else { step = 2; pass.visibility = View.VISIBLE; go.text = "Подтвердить пароль"; status.text = "Нужен облачный пароль." }
                            }
                        }
                        else -> {
                            core.callAttr("sign_in_password", pass.text.toString())
                            ui.post { enterMain() }
                        }
                    }
                } catch (t: Throwable) {
                    log("Ошибка входа: ${t.message}")
                    ui.post { status.text = "Ошибка: ${t.message}" }
                }
                ui.post { go.isEnabled = true }
            }
        }
        box.addView(phone, lp(b = dp(10))); box.addView(code, lp(b = dp(10))); box.addView(pass, lp(b = dp(10)))
        box.addView(go, lp()); box.addView(status, lp(t = dp(12)))
        show(box)
    }

    private fun enterMain() {
        loggedIn = true
        tabs.visibility = View.VISIBLE
        tabs.removeAllViews()
        val pages = listOf("Лента" to feedPage, "Каналы" to chanPage, "Настройки" to setPage, "Лог" to logPage)
        for ((name, page) in pages) {
            val b = TextView(this).apply {
                text = name; gravity = Gravity.CENTER; setTextColor(C.TEXT); textSize = 14f
                setPadding(0, dp(14), 0, dp(14))
                setOnClickListener { show(page) }
            }
            tabs.addView(b, lp(h = -2, weight = 1f))
        }
        channels = store.channels(); blacklist = store.blacklist
        feed = store.feed(); renderFeed()
        refreshChannelList()
        show(feedPage)
        if (channels.isEmpty()) loadChannels()
    }

    // ------------------------------------------------------------ feed
    private sealed class Row {
        data class Head(val topic: String, val n: Int, val closed: Boolean) : Row()
        data class Item(val p: FeedPost) : Row()
        data class More(val topic: String, val left: Int) : Row()
    }

    private val expanded = HashSet<String>()
    private val groupLimit = HashMap<String, Int>()
    private val PAGE = 20
    private val bmpCache = object : android.util.LruCache<String, android.graphics.Bitmap>(24) {}

    private fun renderFeed() {
        val rows = ArrayList<Row>()
        val order = store.topics.map { it.name.trim() }
        val groups = feed.groupBy { it.topic }
        val keys = groups.keys.sortedWith(compareBy({ it.startsWith("Прочее") }, { order.indexOf(it).let { i -> if (i < 0) 99 else i } }))
        for (k in keys) {
            val l = groups[k]!!.sortedByDescending { it.date }
            val closed = if (k in collapsedSet) true else (k.startsWith("Прочее") && k !in openedOther)
            rows.add(Row.Head(k, l.size, closed))
            if (!closed) {
                val lim = groupLimit[k] ?: PAGE
                l.take(lim).forEach { rows.add(Row.Item(it)) }
                if (l.size > lim) rows.add(Row.More(k, l.size - lim))
            }
        }
        feedAdapter.rows = rows
        feedAdapter.order = order
        feedAdapter.notifyDataSetChanged()
        feedEmpty.visibility = if (feed.isEmpty()) View.VISIBLE else View.GONE
    }

    private val collapsedSet = HashSet<String>()
    private val openedOther = HashSet<String>()

    private fun buildFeed(): View {
        val root = vbox()
        val top = hbox().apply { setPadding(dp(14), dp(10), dp(10), dp(6)) }
        top.addView(label("Дайджест", 20f, C.TEXT, true), lp(w = 0, weight = 1f))
        top.addView(button("Обновить") { startScan() })
        root.addView(top, lp())
        statusView = label("", 12f, C.DIM).apply { setPadding(dp(14), 0, dp(14), dp(6)) }
        root.addView(statusView, lp())
        feedEmpty = label("Пока пусто. Нажмите «Обновить».\nПеред первым запуском загрузите модели: Настройки → Модели.", 14f, C.DIM)
            .apply { gravity = Gravity.CENTER; setPadding(dp(24), dp(40), dp(24), dp(40)) }
        root.addView(feedEmpty, lp())
        val rv = RecyclerView(this)
        rv.layoutManager = LinearLayoutManager(this)
        feedAdapter = FeedAdapter()
        rv.adapter = feedAdapter
        root.addView(rv, lp(h = 0, weight = 1f))
        return root
    }

    private fun plain(t: String): String =
        t.replace(Regex("\\[([^\\]]+)\\]\\((https?://[^)\\s]+)\\)"), "$1")
            .replace("**", "").replace("__", "").replace("`", "")
            .replace(Regex("\\s+"), " ").trim()

    /** (title, rest): first line is a title when short and followed by more text. */
    private fun splitTitle(text: String): Pair<String, String> {
        val lines = text.trim().lines()
        if (lines.isEmpty()) return "" to text
        val first = plain(lines[0])
        val rest = lines.drop(1).joinToString("\n").trim()
        return if (first.isNotEmpty() && rest.isNotEmpty() && first.length <= 140) first to rest else "" to text
    }

    /** Telegram markdown -> spans (bold, italic, links) + auto-linked bare URLs. */
    private fun renderMd(text: String): CharSequence {
        val re = Regex("\\[([^\\]]+)\\]\\((https?://[^)\\s]+)\\)|\\*\\*(.+?)\\*\\*|__(.+?)__")
        val sb = android.text.SpannableStringBuilder()
        var pos = 0
        for (m in re.findAll(text)) {
            sb.append(text.substring(pos, m.range.first))
            pos = m.range.last + 1
            val st = sb.length
            when {
                m.groupValues[1].isNotEmpty() -> {
                    sb.append(m.groupValues[1])
                    sb.setSpan(android.text.style.URLSpan(m.groupValues[2]), st, sb.length, 0)
                }
                m.groupValues[3].isNotEmpty() -> {
                    sb.append(m.groupValues[3])
                    sb.setSpan(android.text.style.StyleSpan(android.graphics.Typeface.BOLD), st, sb.length, 0)
                }
                else -> {
                    sb.append(m.groupValues[4])
                    sb.setSpan(android.text.style.StyleSpan(android.graphics.Typeface.ITALIC), st, sb.length, 0)
                }
            }
        }
        sb.append(text.substring(pos))
        android.text.util.Linkify.addLinks(sb, android.text.util.Linkify.WEB_URLS)
        return sb
    }

    private fun bitmap(path: String, maxSide: Int): android.graphics.Bitmap? {
        val key = "$path#$maxSide"
        bmpCache.get(key)?.let { return it }
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, o)
        var sample = 1
        while (o.outWidth / (sample * 2) >= maxSide && o.outHeight / (sample * 2) >= maxSide) sample *= 2
        val b = BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return null
        bmpCache.put(key, b)
        return b
    }

    private fun showFullImage(path: String) {
        val d = android.app.Dialog(this, android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        val iv = ImageView(this)
        iv.setBackgroundColor(0xFF000000.toInt())
        iv.scaleType = ImageView.ScaleType.FIT_CENTER
        bitmap(path, 2048)?.let { iv.setImageBitmap(it) }
        iv.setOnClickListener { d.dismiss() }
        d.setContentView(iv)
        d.show()
    }

    private inner class FeedAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
        var rows: List<Row> = emptyList()
        var order: List<String> = emptyList()
        override fun getItemCount() = rows.size
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
            val v = vbox(0)
            v.layoutParams = RecyclerView.LayoutParams(-1, -2)
            return object : RecyclerView.ViewHolder(v) {}
        }

        override fun onBindViewHolder(h: RecyclerView.ViewHolder, pos: Int) {
            val box = h.itemView as LinearLayout
            box.removeAllViews()
            when (val r = rows[pos]) {
                is Row.Head -> {
                    val col = topicColor(r.topic, order)
                    val t = hbox().apply {
                        background = round(col, 12)
                        setPadding(dp(14), dp(12), dp(14), dp(12))
                        setOnClickListener {
                            if (r.topic.startsWith("Прочее")) {
                                if (!openedOther.add(r.topic)) openedOther.remove(r.topic)
                            } else if (!collapsedSet.add(r.topic)) collapsedSet.remove(r.topic)
                            renderFeed()
                        }
                    }
                    t.addView(label((if (r.closed) "▸  " else "▾  ") + r.topic, 16f, 0xFFFFFFFF.toInt(), true), lp(w = 0, weight = 1f))
                    t.addView(label("${r.n}", 14f, 0xFFFFFFFF.toInt(), true))
                    box.addView(t, lp(l = dp(10), t = dp(10), r = dp(10)))
                }
                is Row.More -> {
                    box.addView(button("Показать ещё (${r.left})", false) {
                        groupLimit[r.topic] = (groupLimit[r.topic] ?: PAGE) + PAGE
                        renderFeed()
                    }, lp(l = dp(10), t = dp(8), r = dp(10)))
                }
                is Row.Item -> bindItem(box, r.p)
            }
        }

        private fun bindItem(box: LinearLayout, p: FeedPost) {
            val ctx = this@MainActivity
            val isOpen = p.link in expanded
            val (title, rest) = splitTitle(p.text)
            val hasImg = p.image.isNotEmpty() && File(p.image).exists()
            val long = hasImg || p.text.length > 230 || p.text.count { it == '\n' } > 3
            val card = vbox(12).apply { background = round(C.CARD, 14) }
            if (long) card.setOnClickListener {
                if (!expanded.add(p.link)) expanded.remove(p.link)
                notifyItemChanged(box.tag as? Int ?: 0)
                renderFeed()
            }

            // header: channel (always) . time
            val meta = hbox()
            val chan = p.channel.ifBlank { "Канал" }
            meta.addView(label(chan, 14f, C.ACC, true), lp(w = 0, weight = 1f))
            meta.addView(label(SimpleDateFormat("d MMM, HH:mm", Locale("ru")).format(Date(p.date * 1000)), 12f, C.DIM))

            val textCol = vbox()
            textCol.addView(meta, lp())
            if (title.isNotEmpty()) textCol.addView(label(title, 16f, C.TEXT, true), lp(t = dp(3)))

            if (!isOpen) {
                val body = plain(if (title.isNotEmpty()) rest else p.text).ifEmpty { "(без текста, только картинка)" }
                val tv = label(body, 14f, C.TEXT)
                tv.maxLines = if (title.isNotEmpty()) 3 else 4
                tv.ellipsize = android.text.TextUtils.TruncateAt.END
                textCol.addView(tv, lp(t = dp(3)))
                if (hasImg) {
                    val row = hbox().apply { gravity = Gravity.TOP }
                    val iv = ImageView(ctx)
                    iv.scaleType = ImageView.ScaleType.CENTER_CROP
                    bitmap(p.image, 360)?.let { iv.setImageBitmap(it) }
                    iv.background = round(C.CARD2, 10)
                    iv.clipToOutline = true
                    row.addView(iv, lp(w = dp(88), h = dp(88), r = dp(10)))
                    row.addView(textCol, lp(w = 0, weight = 1f))
                    card.addView(row, lp())
                } else card.addView(textCol, lp())
            } else {
                card.addView(textCol, lp())
                if (hasImg) {
                    val iv = ImageView(ctx)
                    iv.adjustViewBounds = true
                    iv.scaleType = ImageView.ScaleType.FIT_CENTER
                    bitmap(p.image, 1080)?.let { iv.setImageBitmap(it) }
                    iv.setOnClickListener { showFullImage(p.image) }
                    card.addView(iv, lp(t = dp(8)))
                }
                val tv = label("", 15f, C.TEXT)
                tv.text = renderMd(if (title.isNotEmpty()) rest else p.text)
                tv.setLineSpacing(0f, 1.15f)
                tv.movementMethod = android.text.method.LinkMovementMethod.getInstance()
                tv.setLinkTextColor(C.ACC)
                tv.setTextIsSelectable(true)
                card.addView(tv, lp(t = dp(8)))
            }

            val foot = hbox()
            foot.addView(chip(p.topic, topicColor(p.topic, order)))
            val eng = when (p.engine) { "llm" -> "ИИ"; "emb" -> "смысл"; "kw" -> "слова"; "вручную" -> "вручную"; else -> "" }
            if (eng.isNotEmpty()) foot.addView(chip("$eng ${"%.2f".format(p.score)}", C.CARD2), lp(l = dp(6)))
            foot.addView(View(ctx), lp(w = 0, h = 1, weight = 1f))
            if (long) foot.addView(button(if (isOpen) "Свернуть" else "Подробнее", false) {
                if (!expanded.add(p.link)) expanded.remove(p.link)
                renderFeed()
            })
            var topicBtn: Button? = null
            topicBtn = button("Тема ▾", false) {
                val pm = android.widget.PopupMenu(ctx, topicBtn!!)
                store.topics.map { t -> t.name.trim() }.filter { n -> n.isNotEmpty() }.forEach { n -> pm.menu.add(n) }
                pm.menu.add("Прочее (не подошло)")
                pm.setOnMenuItemClickListener { item ->
                    movePost(p, item.title.toString())
                    true
                }
                pm.show()
            }
            foot.addView(topicBtn, lp(l = dp(6)))
            foot.addView(button("Открыть", false) {
                try { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(p.link))) } catch (_: Exception) {}
            }, lp(l = dp(6)))
            card.addView(foot, lp(t = dp(8)))
            box.addView(card, lp(l = dp(10), t = dp(8), r = dp(10)))
        }
    }

    /** User says: this post belongs to <topic> (or to none). Remembered as an example. */
    private fun movePost(p: FeedPost, topic: String) {
        val saved = if (topic.startsWith("Прочее")) "" else topic
        store.addFeedback(p.text, saved)
        feed = feed.map { if (it.link == p.link) it.copy(topic = topic, engine = "вручную", score = 1.0) else it }
        store.saveFeed(feed)
        renderFeed()
        Toast.makeText(this, "Запомнил. Учту при следующей проверке.", Toast.LENGTH_SHORT).show()
    }

    private fun startScan() {
        if (scanning) { Toast.makeText(this, "Уже идёт проверка", Toast.LENGTH_SHORT).show(); return }
        scanning = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        log("Старт проверки")
        val cfgJson = store.scanSettings()
        val useLlm = store.useLlm
        bg.execute {
            try {
                val msg = helper.load(useLlm)
                if (msg.isNotEmpty()) log(msg)
                if (Downloader.ready(this, Specs.SPM)) core.callAttr("load_tokenizer", Downloader.file(this, Specs.SPM).absolutePath)
                if (!helper.hasEmbed()) log("Модель эмбеддингов не загружена — работаю только по ключевым словам (Настройки → Модели)")
                val res = core.callAttr("scan", cfgJson, helper).toString()
                store.feedJson = res
                applyTopicSuggestions(core.callAttr("suggestions").toString())
                val f = store.feed()
                ui.post { feed = f; renderFeed() }
            } catch (t: Throwable) {
                log("ОШИБКА: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                ui.post { scanning = false; window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            }
        }
    }

    /** Auto-topics found by the scan: new topics are added, similar existing ones get the group's words. */
    private fun applyTopicSuggestions(json: String) {
        try {
            val a = org.json.JSONArray(json)
            val list = store.topics
            for (i in 0 until a.length()) {
                val o = a.getJSONObject(i)
                val kws = (0 until o.getJSONArray("keywords").length()).map { o.getJSONArray("keywords").getString(it) }
                val into = o.optString("merge_into")
                if (into.isNotEmpty()) {
                    val t = list.firstOrNull { it.name.trim().equals(into, true) } ?: continue
                    val have = t.keywords.split(",").map { it.trim().lowercase() }.toSet()
                    val extra = kws.filter { it.lowercase() !in have }
                    if (extra.isEmpty()) continue
                    t.keywords = (t.keywords.trim().trimEnd(',') + ", " + extra.joinToString(", ")).trim(' ', ',')
                    log("Тема «${t.name}» дополнена словами: ${extra.joinToString(", ")}")
                } else {
                    val name = o.getString("name")
                    if (list.any { it.name.trim().equals(name, true) }) continue
                    list.add(Topic(name, kws.joinToString(", ")))
                    log("Новая тема «$name» (постов: ${o.getInt("count")}): ${kws.joinToString(", ")}")
                }
            }
            if (a.length() > 0) store.topics = list
        } catch (t: Throwable) {
            log("Автотемы: ${t.javaClass.simpleName}: ${t.message}")
        }
    }

    // ------------------------------------------------------------ channels
    private fun buildChannels(): View {
        val root = vbox()
        val top = vbox(12)
        top.addView(label("Каналы", 20f, C.TEXT, true))
        val search = edit("Поиск по названию")
        search.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) { chanFilter = s.toString().trim().lowercase(); refreshChannelList() }
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })
        top.addView(search, lp(t = dp(8)))
        val row = hbox()
        row.addView(button("Выбрать все") { setAll(true) }, lp(r = dp(6)))
        row.addView(button("Снять все") { setAll(false) }, lp(r = dp(6)))
        row.addView(button("Загрузить список", false) { loadChannels() })
        top.addView(row, lp(t = dp(8)))
        chanCount = label("", 12f, C.DIM)
        top.addView(chanCount, lp(t = dp(6)))
        root.addView(top, lp())
        val rv = RecyclerView(this)
        rv.layoutManager = LinearLayoutManager(this)
        chanAdapter = ChanAdapter()
        rv.adapter = chanAdapter
        root.addView(rv, lp(h = 0, weight = 1f))
        return root
    }

    private fun visibleChannels() =
        channels.filter { chanFilter.isEmpty() || it.title.lowercase().contains(chanFilter) }

    private fun refreshChannelList() {
        chanAdapter.items = visibleChannels()
        chanAdapter.notifyDataSetChanged()
        val on = channels.count { it.id !in blacklist }
        chanCount.text = "Отслеживается $on из ${channels.size}"
    }

    private fun setAll(on: Boolean) {
        for (c in visibleChannels()) if (on) blacklist.remove(c.id) else blacklist.add(c.id)
        store.blacklist = blacklist
        refreshChannelList()
    }

    private fun loadChannels() {
        log("Загружаю список каналов…")
        bg.execute {
            try {
                val js = core.callAttr("list_channels").toString()
                store.channelsJson = js
                val l = store.channels()
                ui.post { channels = l; refreshChannelList(); log("Каналов: ${l.size} (в архиве ${l.count { it.archived }})") }
            } catch (t: Throwable) { log("Ошибка списка каналов: ${t.message}") }
        }
    }

    private inner class ChanAdapter : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
        var items: List<Chan> = emptyList()
        override fun getItemCount() = items.size
        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
            val cb = CheckBox(this@MainActivity).apply {
                setTextColor(C.TEXT); textSize = 15f
                setPadding(dp(12), dp(10), dp(12), dp(10))
                layoutParams = RecyclerView.LayoutParams(-1, -2)
            }
            return object : RecyclerView.ViewHolder(cb) {}
        }
        override fun onBindViewHolder(h: RecyclerView.ViewHolder, pos: Int) {
            val c = items[pos]
            val cb = h.itemView as CheckBox
            cb.setOnCheckedChangeListener(null)
            cb.text = c.title + (if (c.archived) "  · архив" else "") + (if (c.unread > 0) "  · ${c.unread}" else "")
            cb.isChecked = c.id !in blacklist
            cb.setOnCheckedChangeListener { _, on ->
                if (on) blacklist.remove(c.id) else blacklist.add(c.id)
                store.blacklist = blacklist
                chanCount.text = "Отслеживается ${channels.count { it.id !in blacklist }} из ${channels.size}"
            }
        }
    }

    // ------------------------------------------------------------ settings
    private fun buildSettings(): View {
        val sv = ScrollView(this)
        val box = vbox(14)
        sv.addView(box)

        box.addView(label("Настройки", 20f, C.TEXT, true))

        // models
        box.addView(label("Модели на телефоне", 16f, C.ACC, true), lp(t = dp(16)))
        box.addView(label("Скачиваются один раз по Wi-Fi и работают без интернета.", 12f, C.DIM), lp(b = dp(6)))
        for (spec in listOf(Specs.EMBED, Specs.SPM, Specs.LLM)) {
            val row = vbox(10).apply { background = round(C.CARD, 12) }
            val st = label("", 12f, C.DIM)
            fun refresh() { st.text = if (Downloader.ready(this, spec)) "✓ загружено" else "не загружено (~${spec.approxMb} МБ)" }
            refresh()
            row.addView(label(spec.title, 14f, C.TEXT, true))
            row.addView(st, lp(t = dp(2)))
            row.addView(button("Скачать") {
                if (Downloader.ready(this, spec)) { refresh(); return@button }
                st.text = "Скачивание…"
                bgDl.execute {
                    val err = Downloader.download(this, spec) { d, t ->
                        ui.post { st.text = "Скачано ${d / 1048576} из ${if (t > 0) (t / 1048576).toString() else "?"} МБ" }
                    }
                    ui.post { if (err != null) st.text = "Ошибка: $err. Нажмите «Скачать» ещё раз — докачается." else refresh() }
                    log(if (err != null) "Загрузка ${spec.title}: $err" else "Загружено: ${spec.title}")
                }
            }, lp(t = dp(6)))
            box.addView(row, lp(b = dp(8)))
        }
        val llmSw = Switch(this).apply {
            text = "ИИ-проверка спорных постов"; setTextColor(C.TEXT)
            isChecked = store.useLlm
            setOnCheckedChangeListener { _, v -> store.useLlm = v }
        }
        box.addView(llmSw, lp(t = dp(4)))

        // topics
        box.addView(label("Темы", 16f, C.ACC, true), lp(t = dp(18)))
        box.addView(label("Название и слова через запятую. Чем конкретнее слова, тем точнее.", 12f, C.DIM), lp(b = dp(6)))
        val topicsBox = vbox()
        box.addView(topicsBox, lp())
        fun addRow(t: Topic) {
            val r = vbox(10).apply { background = round(C.CARD, 12) }
            val n = edit("Название темы", t.name)
            val k = edit("Слова через запятую", t.keywords, true)
            r.addView(n, lp()); r.addView(k, lp(t = dp(6)))
            r.addView(button("Удалить", false) { topicsBox.removeView(r) }, lp(t = dp(6)))
            topicsBox.addView(r, lp(b = dp(8)))
        }
        store.topics.forEach { addRow(it) }
        box.addView(button("+ Добавить тему", false) { addRow(Topic("", "")) }, lp())

        // params
        box.addView(label("Параметры", 16f, C.ACC, true), lp(t = dp(18)))
        val thr = edit("Порог схожести (0.30–0.60)", store.threshold.toString())
        val lim = edit("Постов с канала за проход", store.perChannel.toString())
        val bud = edit("Лимит ИИ-проверок за проход", store.llmBudget.toString())
        box.addView(label("Порог: выше — строже", 12f, C.DIM)); box.addView(thr, lp(b = dp(6)))
        box.addView(label("Постов с канала", 12f, C.DIM)); box.addView(lim, lp(b = dp(6)))
        box.addView(label("ИИ-проверок (чем больше, тем дольше)", 12f, C.DIM)); box.addView(bud, lp(b = dp(6)))
        val bc = Switch(this).apply { text = "Только каналы (без групп)"; setTextColor(C.TEXT); isChecked = store.onlyBroadcast }
        val im = Switch(this).apply { text = "Показывать картинки"; setTextColor(C.TEXT); isChecked = store.showImages }
        box.addView(bc, lp(t = dp(4))); box.addView(im, lp(t = dp(4)))

        box.addView(button("Сохранить") {
            val list = mutableListOf<Topic>()
            for (i in 0 until topicsBox.childCount) {
                val r = topicsBox.getChildAt(i) as LinearLayout
                val name = (r.getChildAt(0) as EditText).text.toString().trim()
                val kw = (r.getChildAt(1) as EditText).text.toString().trim()
                if (name.isNotEmpty()) list.add(Topic(name, kw))
            }
            store.topics = list
            thr.text.toString().replace(',', '.').toFloatOrNull()?.let { store.threshold = it.coerceIn(0.2f, 0.8f) }
            lim.text.toString().toIntOrNull()?.let { store.perChannel = it.coerceIn(5, 200) }
            bud.text.toString().toIntOrNull()?.let { store.llmBudget = it.coerceIn(0, 200) }
            store.onlyBroadcast = bc.isChecked
            store.showImages = im.isChecked
            Toast.makeText(this, "Сохранено", Toast.LENGTH_SHORT).show()
            renderFeed()
        }, lp(t = dp(16)))

        box.addView(button("Выйти из аккаунта", false) {
            bg.execute {
                try { core.callAttr("logout") } catch (_: Throwable) {}
                ui.post { showLogin() }
            }
        }, lp(t = dp(10), b = dp(24)))
        return sv
    }

    // ------------------------------------------------------------ log
    private fun buildLog(): View {
        val sv = ScrollView(this)
        val tv = label("", 12f, C.DIM).apply { setPadding(dp(12), dp(12), dp(12), dp(12)); setTextIsSelectable(true) }
        logView = tv
        sv.addView(tv)
        return sv
    }
}
