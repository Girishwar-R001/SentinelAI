package com.example.sentinelai.data.feedback

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

enum class FeedbackType {
    CORRECT_DETECTION, // User confirmed it's a scam/fake voice
    FALSE_ALARM // User flagged it as a false alarm (was actually a real human)
}

object FeedbackManager {
    private const val TAG = "FeedbackManager"
    private const val RMS_MIN_THRESHOLD = 0.05f
    private const val RMS_MAX_THRESHOLD = 0.9f

    /**
     * Validates the feedback, simulates extracting gradients, and queues a weight update.
     */
    fun processFeedbackForContinuousLearning(
        context: Context,
        audioBuffer: FloatArray,
        feedbackType: FeedbackType,
        modelConfidence: Float
    ): Boolean {
        // 1. Energy Validation (VAD check)
        val rms = calculateRms(audioBuffer)
        if (rms < RMS_MIN_THRESHOLD || rms > RMS_MAX_THRESHOLD) {
            Log.w(TAG, "Feedback rejected: Audio energy ($rms) out of bounds. Likely silence or static.")
            return false
        }

        // 2. Simulate Local Backpropagation
        Log.i(TAG, "Running continuous on-device training...")
        Log.i(TAG, "Extracted gradients...")
        Log.i(TAG, "Raw audio discarded to preserve Zero-Storage Privacy.")

        // 3. Generate Mock Weights Diff
        try {
            val baseDir = File(context.filesDir, "federated_updates")
            if (!baseDir.exists()) {
                baseDir.mkdirs()
            }

            val timestamp = System.currentTimeMillis()
            val file = File(baseDir, "weight_update_${timestamp}.json")

            // Simulate weights (for demo purposes)
            val oldWeights = FloatArray(5) { (Math.random() * 0.1).toFloat() }
            val newWeights = FloatArray(5) { i -> oldWeights[i] + ((Math.random() - 0.5) * 0.05).toFloat() }

            val jsonContent = """
                {
                    "timestamp": $timestamp,
                    "type": "${feedbackType.name}",
                    "oldWeights": [${oldWeights.joinToString()}],
                    "newWeights": [${newWeights.joinToString()}]
                }
            """.trimIndent()

            FileOutputStream(file).use { out ->
                out.write(jsonContent.toByteArray())
            }

            Log.i(TAG, "Saved weight update to ${file.absolutePath}")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save weight update: ${e.message}")
            return false
        }
    }

    /**
     * Retrieves the count of queued weight updates.
     */
    fun getQueuedUpdateCount(context: Context): Int {
        val baseDir = File(context.filesDir, "federated_updates")
        return baseDir.listFiles()?.size ?: 0
    }
    
    /**
     * Clears all queued weight updates (simulate upload).
     */
    fun clearQueuedUpdates(context: Context) {
        val baseDir = File(context.filesDir, "federated_updates")
        baseDir.listFiles()?.forEach { it.delete() }
    }
    
    /**
     * Retrieves the latest weight diff for UI viewing.
     */
    fun getLatestWeightDiff(context: Context): String? {
        val baseDir = File(context.filesDir, "federated_updates")
        val files = baseDir.listFiles() ?: return null
        val latestFile = files.maxByOrNull { it.lastModified() }
        return latestFile?.readText()
    }

    private fun calculateRms(buffer: FloatArray): Float {
        var sum = 0f
        for (sample in buffer) {
            sum += sample * sample
        }
        return sqrt(sum / buffer.size)
    }
}
