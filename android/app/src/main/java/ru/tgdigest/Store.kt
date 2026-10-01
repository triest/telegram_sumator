package ru.tgdigest

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class Topic(var name: String, var keywords: String)
data class Chan(val id: Long, val title: String, val username: String, val broadcast: Boolean,
                val unread: Int, val archived: Boolean)
data class FeedPost(val channel: String, val date: Long, val text: String, val topic: String,
                    val score: Double, val engine: String, val image: String, val link: String)

class Store(ctx: Context) {
    private val p = ctx.getSharedPreferences("cfg", Context.MODE_PRIVATE)

    var topics: MutableList<Topic>
        get() {
            val s = p.getString("topics", null)
            if (s == null) return mutableListOf(
                Topic("ИИ и нейросети", "LLM, GPT, Claude, нейросеть, искусственный интеллект, модель, ChatGPT, OpenAI"),
                Topic("Программирование", "Python, код, программирование, разработка, GitHub, библиотека, фреймворк"),
                Topic("Авиация", "самолёт, авиация, авиакомпания, аэропорт, пилот, рейс"),
                Topic("СВО", "СВО, ВС РФ, фронт, наступление, Минобороны, ВСУ"),
                Topic("Походы", "поход, туризм, горы, маршрут, треккинг, палатка, байдарка"),
            )
            val a = JSONArray(s)
            return MutableList(a.length()) { Topic(a.getJSONObject(it).getString("name"), a.getJSONObject(it).getString("kw")) }
        }
        set(v) {
            val a = JSONArray()
            v.forEach { a.put(JSONObject().put("name", it.name).put("kw", it.keywords)) }
            p.edit().putString("topics", a.toString()).apply()
        }

    var blacklist: MutableSet<Long>
        get() = p.getString("black", "")!!.split(",").mapNotNull { it.toLongOrNull() }.toMutableSet()
        set(v) { p.edit().putString("black", v.joinToString(",")).apply() }

    var threshold: Float
        get() = p.getFloat("thr", 0.42f)
        set(v) { p.edit().putFloat("thr", v).apply() }
    var perChannel: Int
        get() = p.getInt("lim", 40)
        set(v) { p.edit().putInt("lim", v).apply() }
    var onlyBroadcast: Boolean
        get() = p.getBoolean("bc", true)
        set(v) { p.edit().putBoolean("bc", v).apply() }
    var showImages: Boolean
        get() = p.getBoolean("img", true)
        set(v) { p.edit().putBoolean("img", v).apply() }
    var useLlm: Boolean
        get() = p.getBoolean("llm", true)
        set(v) { p.edit().putBoolean("llm", v).apply() }
    var llmBudget: Int
        get() = p.getInt("budget", 30)
        set(v) { p.edit().putInt("budget", v).apply() }

    var channelsJson: String
        get() = p.getString("channels", "[]")!!
        set(v) { p.edit().putString("channels", v).apply() }
    var feedJson: String
        get() = p.getString("feed", "[]")!!
        set(v) { p.edit().putString("feed", v).apply() }

    fun channels(): List<Chan> {
        val a = JSONArray(channelsJson)
        return List(a.length()) {
            val o = a.getJSONObject(it)
            Chan(o.getLong("id"), o.getString("title"), o.optString("username"), o.optBoolean("broadcast", true),
                o.optInt("unread"), o.optBoolean("archived"))
        }
    }

    fun feed(): List<FeedPost> {
        val a = JSONArray(feedJson)
        return List(a.length()) {
            val o = a.getJSONObject(it)
            FeedPost(o.getString("channel"), o.getLong("date"), o.getString("text"), o.optString("topic"),
                o.optDouble("score"), o.optString("engine"), o.optString("image"), o.getString("link"))
        }
    }

    fun scanSettings(): String {
        val o = JSONObject()
        val ta = JSONArray()
        topics.filter { it.name.isNotBlank() }.forEach {
            ta.put(JSONObject().put("name", it.name.trim())
                .put("keywords", JSONArray(it.keywords.split(",").map { k -> k.trim() }.filter { k -> k.isNotEmpty() })))
        }
        o.put("topics", ta)
        o.put("blacklist", JSONArray(blacklist.toList()))
        o.put("threshold", threshold.toDouble())
        o.put("per_channel_limit", perChannel)
        o.put("only_broadcast", onlyBroadcast)
        o.put("show_images", showImages)
        o.put("use_llm", useLlm)
        o.put("llm_budget", llmBudget)
        return o.toString()
    }
}
