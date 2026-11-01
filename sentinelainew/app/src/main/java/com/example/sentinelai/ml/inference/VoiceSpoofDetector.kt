package com.example.sentinelai.ml.inference

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

class VoiceSpoofDetector(context: Context) {
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    var modelSizeBytes: Long = 0
        private set

    init {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            val modelFile = java.io.File(context.cacheDir, "rawnet2.onnx")
            val dataFile = java.io.File(context.cacheDir, "rawnet2.onnx.data")
            
            if (!modelFile.exists() || !dataFile.exists()) {
                Log.d("SentinelAI", "Copying ONNX models to cache...")
                context.assets.open("models/rawnet2.onnx").use { input ->
                    java.io.FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                context.assets.open("models/rawnet2.onnx.data").use { input ->
                    java.io.FileOutputStream(dataFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            
            modelSizeBytes = modelFile.length() + dataFile.length()
            
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
            }
            
            ortSession = ortEnv?.createSession(modelFile.absolutePath, options)
            Log.d("SentinelAI", "ONNX Model loaded successfully! Size: $modelSizeBytes bytes")
        } catch (e: Exception) {
            Log.e("SentinelAI", "Failed to load ONNX model", e)
        }
    }

    data class InferenceResult(
        val isSyntheticProbability: Float,
        val latencyMs: Long
    )

    fun analyzeAudioBuffer(audioData: FloatArray): InferenceResult? {
        val session = ortSession ?: run {
            Log.e("SentinelAI", "ONNX session is null!")
            return null
        }
        val env = ortEnv ?: return null

        if (audioData.size != 16000) {
            Log.e("SentinelAI", "Invalid audio buffer size: ${audioData.size}")
            return null
        }

        try {
            // RawNet2 expects (1, 64600) input
            val targetSize = 64600
            val paddedData = FloatArray(targetSize)
            // Tile the 16000 samples 4 times (16000 * 4 = 64000)
            for (i in 0 until 4) {
                System.arraycopy(audioData, 0, paddedData, i * 16000, 16000)
            }
            // The remaining 600 will be 0.0f by default (padding)

            val floatBuffer = FloatBuffer.wrap(paddedData)
            val shape = longArrayOf(1, targetSize.toLong())
            val tensor = OnnxTensor.createTensor(env, floatBuffer, shape)

            val startTime = System.nanoTime()
            val result = session.run(Collections.singletonMap("input", tensor))
            val endTime = System.nanoTime()

            val outputTensor = result.get(0).value as Array<FloatArray>
            val logits = outputTensor[0] // [bonafide_logit, spoof_logit]
            
            // Softmax for probability of class 1 (spoof)
            val exp0 = Math.exp(logits[0].toDouble())
            val exp1 = Math.exp(logits[1].toDouble())
            val probability = (exp1 / (exp0 + exp1)).toFloat()

            val latency = maxOf(1L, (endTime - startTime) / 1_000_000)
            Log.d("SentinelAI", "ONNX Inference OK -> Prob: $probability, Latency: ${latency}ms")

            tensor.close()
            result.close()

            return InferenceResult(
                isSyntheticProbability = probability,
                latencyMs = latency
            )
        } catch (e: Exception) {
            Log.e("SentinelAI", "Error running ONNX inference", e)
            return null
        }
    }

    fun close() {
        ortSession?.close()
        ortEnv?.close()
        ortSession = null
        ortEnv = null
    }
}
