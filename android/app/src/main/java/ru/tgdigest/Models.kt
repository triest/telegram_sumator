package ru.tgdigest

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.LongBuffer

class ModelSpec(val key: String, val title: String, val file: String, val url: String, val approxMb: Int)

object Specs {
    val EMBED = ModelSpec(
        "embed", "Эмбеддинги (MiniLM, многоязычные)", "minilm_q.onnx",
        "https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model_quantized.onnx", 118
    )
    val SPM = ModelSpec(
        "spm", "Токенайзер", "sentencepiece.bpe.model",
        "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/sentencepiece.bpe.model", 5
    )
    val LLM = ModelSpec(
        "llm", "Локальный ИИ (Qwen2.5 1.5B)", "qwen25_15b_q8.task",
        "https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv1280.task", 1600
    )
}

object Downloader {
    /** Downloads with resume. Returns null on success or an error text. */
    fun download(ctx: Context, spec: ModelSpec, onProgress: (Long, Long) -> Unit): String? {
        val dir = File(ctx.filesDir, "models").also { it.mkdirs() }
        val target = File(dir, spec.file)
        if (target.exists() && target.length() > 1000) return null
        val part = File(dir, spec.file + ".part")
        try {
            var have = if (part.exists()) part.length() else 0L
            val c = URL(spec.url).openConnection() as HttpURLConnection
            c.connectTimeout = 30000
            c.readTimeout = 60000
            c.instanceFollowRedirects = true
            if (have > 0) c.setRequestProperty("Range", "bytes=$have-")
            c.connect()
            val code = c.responseCode
            if (code != 200 && code != 206) return "HTTP $code"
            if (code == 200) have = 0L
            val total = have + c.contentLengthLong
            FileOutputStream(part, code == 206).use { out ->
                c.inputStream.use { inp ->
                    val buf = ByteArray(256 * 1024)
                    var done = have
                    var last = 0L
                    while (true) {
                        val n = inp.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        val now = System.currentTimeMillis()
                        if (now - last > 400) { last = now; onProgress(done, total) }
                    }
                    onProgress(done, total)
                }
            }
            if (!part.renameTo(target)) return "не удалось сохранить файл"
            return null
        } catch (e: Exception) {
            return e.javaClass.simpleName + ": " + (e.message ?: "")
        }
    }

    fun file(ctx: Context, spec: ModelSpec) = File(File(ctx.filesDir, "models"), spec.file)
    fun ready(ctx: Context, spec: ModelSpec) = file(ctx, spec).let { it.exists() && it.length() > 1000 }
}

/** Object handed to Python. Public methods are called from the Python thread. */
class Helper(private val ctx: Context, private val log: (String) -> Unit) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var llm: LlmInference? = null

    @Synchronized
    fun load(useLlm: Boolean): String {
        val sb = StringBuilder()
        if (session == null && Downloader.ready(ctx, Specs.EMBED) && Downloader.ready(ctx, Specs.SPM)) {
            try {
                val e = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions()
                opts.setIntraOpNumThreads(4)
                session = e.createSession(Downloader.file(ctx, Specs.EMBED).absolutePath, opts)
                env = e
                sb.append("эмбеддинги загружены; ")
            } catch (t: Throwable) {
                sb.append("эмбеддинги: ${t.message}; ")
            }
        }
        if (useLlm && llm == null && Downloader.ready(ctx, Specs.LLM)) {
            try {
                val o = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(Downloader.file(ctx, Specs.LLM).absolutePath)
                    .setMaxTokens(1100)
                    .build()
                llm = LlmInference.createFromOptions(ctx, o)
                sb.append("ИИ загружен; ")
            } catch (t: Throwable) {
                sb.append("ИИ: ${t.message}; ")
            }
        }
        return sb.toString()
    }

    @Synchronized
    fun unload() {
        try { session?.close() } catch (_: Throwable) {}
        try { llm?.close() } catch (_: Throwable) {}
        session = null; llm = null
    }

    fun hasEmbed(): Boolean = session != null
    fun hasLlm(): Boolean = llm != null
    fun progress(msg: String) = log(msg)

    /** json: [[id,...],...] -> json flat float array (n*dim, L2-normalised). */
    fun embed(json: String): String {
        val s = session ?: throw IllegalStateException("embedder not loaded")
        val e = env!!
        val arr = JSONArray(json)
        val n = arr.length()
        var maxLen = 1
        val ids = Array(n) { i ->
            val a = arr.getJSONArray(i)
            LongArray(a.length()) { a.getLong(it) }.also { if (it.size > maxLen) maxLen = it.size }
        }
        val inputIds = LongArray(n * maxLen) { 1L } // pad id = 1
        val mask = LongArray(n * maxLen)
        for (i in 0 until n) for (j in ids[i].indices) {
            inputIds[i * maxLen + j] = ids[i][j]; mask[i * maxLen + j] = 1L
        }
        val shape = longArrayOf(n.toLong(), maxLen.toLong())
        val tensors = HashMap<String, OnnxTensor>()
        try {
            val names = s.inputNames
            if ("input_ids" in names) tensors["input_ids"] = OnnxTensor.createTensor(e, LongBuffer.wrap(inputIds), shape)
            if ("attention_mask" in names) tensors["attention_mask"] = OnnxTensor.createTensor(e, LongBuffer.wrap(mask), shape)
            if ("token_type_ids" in names) tensors["token_type_ids"] = OnnxTensor.createTensor(e, LongBuffer.wrap(LongArray(n * maxLen)), shape)
            s.run(tensors).use { res ->
                @Suppress("UNCHECKED_CAST")
                val out = res[0].value as Array<Array<FloatArray>>
                val dim = out[0][0].size
                val flat = StringBuilder("[")
                for (i in 0 until n) {
                    val v = FloatArray(dim)
                    var cnt = 0f
                    for (j in 0 until maxLen) if (mask[i * maxLen + j] == 1L) {
                        cnt += 1f
                        val row = out[i][j]
                        for (k in 0 until dim) v[k] += row[k]
                    }
                    var norm = 0f
                    for (k in 0 until dim) { v[k] /= maxOf(cnt, 1f); norm += v[k] * v[k] }
                    norm = Math.sqrt(norm.toDouble()).toFloat().coerceAtLeast(1e-9f)
                    for (k in 0 until dim) {
                        if (!(i == 0 && k == 0)) flat.append(',')
                        flat.append(v[k] / norm)
                    }
                }
                flat.append(']')
                return flat.toString()
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    fun classify(prompt: String): String {
        val m = llm ?: return ""
        val full = "<|im_start|>system\nТы точный классификатор.<|im_end|>\n" +
            "<|im_start|>user\n$prompt<|im_end|>\n<|im_start|>assistant\n"
        val so = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(1).setTemperature(0.1f).build()
        val sess = LlmInferenceSession.createFromOptions(m, so)
        try {
            sess.addQueryChunk(full)
            return sess.generateResponse()
        } finally {
            sess.close()
        }
    }
}
