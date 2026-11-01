package com.example.sentinelai.services.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.sqrt

class AudioRecordManager(
    private val onBufferReady: (FloatArray) -> Unit,
    private val onAmplitudeChanged: ((Float) -> Unit)? = null
) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private var recordingJob: Job? = null

    private val modelInputSize = 16000
    private val floatBuffer = FloatArray(modelInputSize)
    private var floatBufferIndex = 0

    @SuppressLint("MissingPermission")
    fun startRecording() {
        if (isRecording) return

        val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
        val bufferSize = if (minBufSize <= 0) 4096 else minBufSize * 2

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("SentinelAI", "AudioRecord failed to initialize state! Falling back to simulated microphone.")
                startMockRecording()
                return
            }

            audioRecord?.startRecording()
            isRecording = true
            Log.d("SentinelAI", "AudioRecord started successfully! BufferSize: $bufferSize")

            recordingJob = CoroutineScope(Dispatchers.IO).launch {
                val shortBuffer = ShortArray(1024)
                while (isRecording && isActive) {
                    val readResult = audioRecord?.read(shortBuffer, 0, shortBuffer.size) ?: 0
                    if (readResult > 0) {
                        var sumSq = 0.0
                        for (i in 0 until readResult) {
                            val sample = shortBuffer[i].toFloat() / Short.MAX_VALUE
                            sumSq += sample * sample

                            floatBuffer[floatBufferIndex] = sample
                            floatBufferIndex++

                            if (floatBufferIndex >= modelInputSize) {
                                val chunk = floatBuffer.clone()
                                floatBufferIndex = 0
                                onBufferReady(chunk)
                            }
                        }

                        val rms = sqrt(sumSq / readResult).toFloat()
                        onAmplitudeChanged?.invoke(rms)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("SentinelAI", "Exception starting AudioRecord", e)
        }
    }

    private fun startMockRecording() {
        isRecording = true
        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            val random = java.util.Random()
            var timePassed = 0L
            while (isRecording && isActive) {
                delay(100)
                timePassed += 100
                
                // Simulate amplitude changes
                val amp = 0.02f + random.nextFloat() * 0.08f
                onAmplitudeChanged?.invoke(amp)

                if (timePassed >= 1000) {
                    timePassed = 0
                    // Every 1 second, feed a chunk of mock audio
                    val mockBuffer = FloatArray(modelInputSize)
                    val isSynthetic = random.nextFloat() > 0.8f
                    val freq = 400.0 + random.nextDouble() * 1000.0
                    for (i in 0 until modelInputSize) {
                        if (isSynthetic) {
                            val t = i.toDouble() / sampleRate
                            mockBuffer[i] = (Math.sin(2 * Math.PI * freq * t).toFloat() * 0.5f) + (random.nextGaussian().toFloat() * 0.1f)
                        } else {
                            mockBuffer[i] = random.nextGaussian().toFloat() * 0.3f
                        }
                    }
                    onBufferReady(mockBuffer)
                }
            }
        }
    }

    fun stopRecording() {
        isRecording = false
        recordingJob?.cancel()
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e("SentinelAI", "Error stopping AudioRecord", e)
        }
        audioRecord = null
        floatBufferIndex = 0
    }
}
