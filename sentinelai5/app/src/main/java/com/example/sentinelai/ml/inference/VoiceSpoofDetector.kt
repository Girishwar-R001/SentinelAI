package com.example.sentinelai.ml.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VoiceSpoofDetector(context: Context) {
    private var interpreter: Interpreter? = null
    var modelSizeBytes: Long = 0
        private set

    init {
        try {
            val modelName = "models/rawnetlite_float16.tflite"
            val fileDescriptor = context.assets.openFd(modelName)
            val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            
            modelSizeBytes = declaredLength
            
            val options = Interpreter.Options().apply {
                numThreads = 2
            }
            interpreter = Interpreter(mappedByteBuffer, options)
            Log.d("SentinelAI", "TFLite Model loaded successfully! Size: $modelSizeBytes bytes")
        } catch (e: Exception) {
            Log.e("SentinelAI", "Failed to load TFLite model", e)
        }
    }

    data class InferenceResult(
        val isSyntheticProbability: Float,
        val latencyMs: Long
    )

    fun analyzeAudioBuffer(audioData: FloatArray): InferenceResult? {
        val tflite = interpreter ?: run {
            Log.e("SentinelAI", "TFLite interpreter is null!")
            return null
        }

        if (audioData.size != 16000) {
            Log.e("SentinelAI", "Invalid audio buffer size: ${audioData.size}")
            return null
        }

        try {
            // rawnetlite_float16.tflite expects (1, 48000, 1) input
            val targetSize = 48000
            val inputBuffer = ByteBuffer.allocateDirect(targetSize * 4).apply {
                order(ByteOrder.nativeOrder())
                // Tile the 16000 audio data array 3 times to reach 48000
                for (i in 0 until 3) {
                    for (value in audioData) {
                        putFloat(value)
                    }
                }
            }

            // Output is (1, 1) float representing synthetic probability
            val outputBuffer = ByteBuffer.allocateDirect(1 * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            val startTime = System.nanoTime()
            tflite.run(inputBuffer, outputBuffer)
            val endTime = System.nanoTime()

            outputBuffer.rewind()
            val probability = outputBuffer.float

            val latency = maxOf(1L, (endTime - startTime) / 1_000_000)
            Log.d("SentinelAI", "Inference OK -> Prob: $probability, Latency: ${latency}ms")

            return InferenceResult(
                isSyntheticProbability = probability,
                latencyMs = latency
            )
        } catch (e: Exception) {
            Log.e("SentinelAI", "Error running TFLite inference", e)
            return null
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
