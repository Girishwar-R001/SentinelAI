package com.example.sentinelai

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast

import android.content.Intent
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.sentinelai.data.feedback.FeedbackManager
import com.example.sentinelai.data.feedback.FeedbackType
import com.example.sentinelai.detection.conversation.ConversationAnalyzer
import com.example.sentinelai.detection.risk.RiskEngine
import com.example.sentinelai.detection.risk.RiskResult
import com.example.sentinelai.ml.inference.VoiceSpoofDetector
import com.example.sentinelai.services.audio.AudioRecordManager
import com.example.sentinelai.ui.screens.HomeScreen
import com.example.sentinelai.ui.theme.SentinelAITheme

class MainActivity : ComponentActivity() {

    private lateinit var voiceSpoofDetector: VoiceSpoofDetector
    private lateinit var conversationAnalyzer: ConversationAnalyzer
    private lateinit var riskEngine: RiskEngine
    private var audioRecordManager: AudioRecordManager? = null

    private var speechRecognizer: SpeechRecognizer? = null
    private var continuousTranscript = ""

    // Compose States
    private val riskResultState = mutableStateOf<RiskResult?>(null)
    private val isRecordingState = mutableStateOf(false)
    private val latencyMsState = mutableStateOf(0L)
    private val modelSizeMbState = mutableStateOf(0f)
    private val amplitudeState = mutableStateOf(0f)
    private val isDemoModeState = mutableStateOf(false) // Separate Live Mic and Simulated Demo Mode

    // Risk Smoothing State
    private val voiceProbabilityHistory = mutableListOf<Float>()

    // Self-Learning Cache
    private var lastAudioBuffer: FloatArray? = null
    private var lastVoiceProbability = 0.0f

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startDemo()
            } else {
                Toast.makeText(this, "Microphone permission required for Edge AI Demo", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        voiceSpoofDetector = VoiceSpoofDetector(this)
        conversationAnalyzer = ConversationAnalyzer()
        riskEngine = RiskEngine()
        modelSizeMbState.value = voiceSpoofDetector.modelSizeBytes / (1024f * 1024f)

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                Log.e("SentinelAI", "SpeechRecognizer Error: $error")
                if (error == SpeechRecognizer.ERROR_AUDIO || error == SpeechRecognizer.ERROR_CLIENT || error == SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) {
                    Toast.makeText(this@MainActivity, "STT Error ($error). Check Google App Mic/Internet permissions.", Toast.LENGTH_SHORT).show()
                }
                if (isRecordingState.value && !isDemoModeState.value) {
                    android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                        try { startListening() } catch(e: Exception) {}
                    }, 1500)
                }
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    continuousTranscript += " " + matches[0]
                    analyzeLiveTranscript()
                }
                if (isRecordingState.value && !isDemoModeState.value) {
                    startListening()
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {
                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    analyzeLiveTranscript(matches[0])
                }
            }
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        audioRecordManager = AudioRecordManager(
            onBufferReady = { audioBuffer ->
                processAudioBuffer(audioBuffer)
            },
            onAmplitudeChanged = { amp ->
                runOnUiThread {
                    amplitudeState.value = amp
                }
            }
        )

        setContent {
            SentinelAITheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val isRecording by isRecordingState
                    val riskResult by riskResultState
                    val latencyMs by latencyMsState
                    val modelSizeMb by modelSizeMbState
                    val amplitude by amplitudeState
                    val isDemoMode by isDemoModeState
                    val context = androidx.compose.ui.platform.LocalContext.current

                    var queuedUpdateCount by remember { mutableStateOf(FeedbackManager.getQueuedUpdateCount(this@MainActivity)) }

                    HomeScreen(
                        isRecording = isRecording,
                        onToggleRecording = {
                            if (isRecording) {
                                stopDemo()
                            } else {
                                checkPermissionAndStart()
                            }
                        },
                        riskResult = riskResult,
                        latencyMs = latencyMs,
                        modelSizeMb = modelSizeMb,
                        amplitude = amplitude,
                        isDemoMode = isDemoMode,
                        onDemoModeToggle = { enabled ->
                            isDemoModeState.value = enabled
                        },
                        onProvideFeedback = { feedbackType ->
                            lastAudioBuffer?.let { buffer ->
                                val success = FeedbackManager.processFeedbackForContinuousLearning(
                                    context = this@MainActivity,
                                    audioBuffer = buffer,
                                    feedbackType = feedbackType,
                                    modelConfidence = lastVoiceProbability
                                )
                                if (success) {
                                    queuedUpdateCount = FeedbackManager.getQueuedUpdateCount(this@MainActivity)
                                    Toast.makeText(context, "Feedback processed for continuous learning", Toast.LENGTH_SHORT).show()
                                }
                            }
                        },
                        queuedUpdateCount = queuedUpdateCount,
                        onUploadUpdates = {
                            FeedbackManager.clearQueuedUpdates(this@MainActivity)
                            queuedUpdateCount = FeedbackManager.getQueuedUpdateCount(this@MainActivity)
                            Toast.makeText(context, "Queued updates synced to cloud", Toast.LENGTH_SHORT).show()
                        }
                    )
                }
            }
        }
    }

    private fun checkPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startDemo()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }


    private fun startListening() {
        runOnUiThread {
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            }
            speechRecognizer?.startListening(intent)
        }
    }

    private fun analyzeLiveTranscript(partial: String = "") {
        val combined = "$continuousTranscript $partial"
        val analysis = conversationAnalyzer.analyzeTranscript(combined)
        lastConversationScore = analysis.scamScore
        lastConversationTags = analysis.detectedTags
    }

    private fun startDemo() {
        isRecordingState.value = true
        audioRecordManager?.startRecording()
        if (isDemoModeState.value) {
            simulateConversation()
        } else {
            continuousTranscript = ""
            startListening()
        }
    }

    private fun stopDemo() {
        isRecordingState.value = false
        audioRecordManager?.stopRecording()
        runOnUiThread { speechRecognizer?.stopListening() }
        riskResultState.value = null
        latencyMsState.value = 0L
        amplitudeState.value = 0f
        lastConversationScore = 0.0f
        lastConversationTags = emptyList()
        voiceProbabilityHistory.clear() // Clear the temporal smoothing history
    }

    private fun processAudioBuffer(audioBuffer: FloatArray) {
        // Voice Activity Detection (VAD) via RMS energy check
        var sumSq = 0.0
        for (value in audioBuffer) {
            sumSq += value * value
        }
        val rms = kotlin.math.sqrt(sumSq / audioBuffer.size).toFloat()

        // If energy is extremely low (silence or light noise), bypass TFLite inference to prevent false positives
        val result = if (rms < 0.01f) {
            Log.d("SentinelAI", "VAD: Silence/background noise detected (RMS = $rms < 0.01). Bypassing inference.")
            VoiceSpoofDetector.InferenceResult(isSyntheticProbability = 0.0f, latencyMs = 1L)
        } else {
            // Normalize audio buffer by max amplitude to match training data distribution
            var maxAmp = 0.0f
            for (value in audioBuffer) {
                val absVal = kotlin.math.abs(value)
                if (absVal > maxAmp) maxAmp = absVal
            }
            val normalizedBuffer = if (maxAmp > 0.05f) {
                FloatArray(audioBuffer.size) { i -> audioBuffer[i] / maxAmp }
            } else {
                audioBuffer
            }
            // Cache the un-normalized buffer for training export
            lastAudioBuffer = audioBuffer.clone()
            voiceSpoofDetector.analyzeAudioBuffer(normalizedBuffer)
        }

        runOnUiThread {
            if (result != null) {
                latencyMsState.value = result.latencyMs

                // Temporal smoothing (rolling window average of last 5 frames)
                voiceProbabilityHistory.add(result.isSyntheticProbability)
                if (voiceProbabilityHistory.size > 5) {
                    voiceProbabilityHistory.removeAt(0)
                }
                val smoothedProbability = voiceProbabilityHistory.average().toFloat()
                lastVoiceProbability = smoothedProbability

                val risk = riskEngine.calculateRisk(
                    voiceSyntheticProbability = smoothedProbability,
                    conversationScamScore = lastConversationScore,
                    conversationTags = lastConversationTags
                )

                riskResultState.value = risk
            } else {
                Log.w("SentinelAI", "Inference result was null")
            }
        }
    }

    private var lastConversationScore = 0.0f
    private var lastConversationTags = emptyList<String>()

    private fun simulateConversation() {
        Thread {
            val steps = listOf(
                "Hello, this is security department from your bank.",
                "We noticed some suspicious activity on your account. You need to act immediately.",
                "To secure your account, please send a wire transfer or gift card payment.",
                "Lastly, we sent you a text message. Please read the six digit verification code."
            )
            var currentTranscript = ""
            for (step in steps) {
                Thread.sleep(4000)
                if (!isRecordingState.value) break
                currentTranscript += " $step"
                val analysis = conversationAnalyzer.analyzeTranscript(currentTranscript)
                lastConversationScore = analysis.scamScore
                lastConversationTags = analysis.detectedTags
                Log.d("SentinelAI", "Simulated conversation step: $step | Intent matched: ${analysis.detectedTags}")
            }
        }.start()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopDemo()
        voiceSpoofDetector.close()
        speechRecognizer?.destroy()
    }
}
