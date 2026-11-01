package com.example.sentinelai.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sentinelai.detection.risk.RiskLevel
import com.example.sentinelai.detection.risk.RiskResult
import com.example.sentinelai.data.feedback.FeedbackType
import com.example.sentinelai.ui.components.EdgeAiStatusPanel

@Composable
fun HomeScreen(
    isRecording: Boolean,
    onToggleRecording: () -> Unit,
    riskResult: RiskResult?,
    latencyMs: Long,
    modelSizeMb: Float,
    amplitude: Float = 0f,
    isDemoMode: Boolean = false,
    onDemoModeToggle: (Boolean) -> Unit = {},
    onProvideFeedback: (FeedbackType) -> Unit = {},
    queuedUpdateCount: Int = 0,
    onUploadUpdates: () -> Unit = {}
) {
    val scrollState = rememberScrollState()
    val context = androidx.compose.ui.platform.LocalContext.current
    val showWeightsDialog = remember { mutableStateOf(false) }

    if (showWeightsDialog.value) {
        val diffJson = com.example.sentinelai.data.feedback.FeedbackManager.getLatestWeightDiff(context) ?: "No updates found."
        AlertDialog(
            onDismissRequest = { showWeightsDialog.value = false },
            title = { Text("Latest Weight Diff", color = Color.White) },
            text = { Text(diffJson, color = Color.LightGray, fontSize = 12.sp) },
            confirmButton = {
                TextButton(onClick = { showWeightsDialog.value = false }) {
                    Text("Close")
                }
            },
            containerColor = Color(0xFF191919)
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF121212))
            .padding(16.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "SentinelAI",
            color = Color.White,
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Privacy-First Edge Security Assistant",
            color = Color.Gray,
            fontSize = 14.sp
        )

        Spacer(modifier = Modifier.height(24.dp))

        EdgeAiStatusPanel(
            latencyMs = latencyMs,
            modelSizeMb = modelSizeMb,
            isLocal = true
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Live Audio Level Visualizer
        if (isRecording) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Mic Input:",
                        color = Color.Gray,
                        fontSize = 14.sp,
                        modifier = Modifier.width(80.dp)
                    )
                    LinearProgressIndicator(
                        progress = { (amplitude * 5f).coerceIn(0.05f, 1f) },
                        modifier = Modifier
                            .weight(1f)
                            .height(12.dp),
                        color = if (amplitude > 0.1f) Color.Green else Color.DarkGray,
                        trackColor = Color(0xFF2A2A2A)
                    )
                }
            }
            Spacer(modifier = Modifier.height(20.dp))
        }

        // Risk Status Card
        val (cardColor, textColor) = when (riskResult?.level ?: RiskLevel.SAFE) {
            RiskLevel.SAFE -> Color(0xFF1E3A20) to Color.Green
            RiskLevel.LOW -> Color(0xFF333A1E) to Color.Yellow
            RiskLevel.MEDIUM -> Color(0xFF4A3A10) to Color(0xFFFFA500)
            RiskLevel.HIGH -> Color(0xFF4A1E1E) to Color.Red
            RiskLevel.CRITICAL -> Color(0xFF660000) to Color.White
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = cardColor),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "CURRENT SCAM RISK",
                    color = Color.LightGray,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = riskResult?.level?.name ?: "INACTIVE",
                    color = textColor,
                    fontSize = 34.sp,
                    fontWeight = FontWeight.Black
                )

                if (riskResult != null) {
                    Text(
                        text = "Overall risk: ${riskResult.score}%",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(top = 6.dp)
                    )
                    Text(
                        text = "Voice authenticity: ${riskResult.voiceAuthenticity}%",
                        color = Color.LightGray,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                    Text(
                        text = "Behavioural risk: ${riskResult.behaviouralRisk}%",
                        color = Color.LightGray,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(top = 2.dp)
                    )

                    if (riskResult.reasons.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Column(horizontalAlignment = Alignment.Start, modifier = Modifier.fillMaxWidth()) {
                            Text("Detected Signals:", color = Color.LightGray, fontSize = 13.sp)
                            riskResult.reasons.forEach { reason ->
                                Text("• $reason", color = Color.White, fontSize = 14.sp, modifier = Modifier.padding(top = 4.dp))
                            }
                        }
                    }

                    if (riskResult.level == RiskLevel.HIGH || riskResult.level == RiskLevel.CRITICAL) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Continuous On-Device Feedback", color = Color.Gray, fontSize = 12.sp)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 8.dp)) {
                            Button(onClick = { onProvideFeedback(FeedbackType.CORRECT_DETECTION) }, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E88E5))) {
                                Text("👍 Correct")
                            }
                            Button(onClick = { onProvideFeedback(FeedbackType.FALSE_ALARM) }, colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFE53935))) {
                                Text("👎 False Alarm")
                            }
                        }
                    }

                    if (riskResult.level == RiskLevel.CRITICAL) {
                        Spacer(modifier = Modifier.height(20.dp))
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Color(0xFF330000)),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text("⚠️ Critical risk detected. The caller may be impersonating someone you know.", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                                Spacer(modifier = Modifier.height(8.dp))
                                Text("Consider asking a personal verification question.", color = Color.Yellow, fontSize = 13.sp)
                                Spacer(modifier = Modifier.height(12.dp))
                                val customQuestion = remember { mutableStateOf("") }
                                OutlinedTextField(
                                    value = customQuestion.value,
                                    onValueChange = { customQuestion.value = it },
                                    label = { Text("Enter a personal question", color = Color.Gray, fontSize = 12.sp) },
                                    modifier = Modifier.fillMaxWidth(),
                                    textStyle = androidx.compose.ui.text.TextStyle(color = Color.White, fontSize = 14.sp),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = Color.Yellow,
                                        unfocusedBorderColor = Color.Gray
                                    )
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text("Note: DO NOT automatically treat a correct answer as proof that the caller is genuine. It is only an additional verification signal.", color = Color.Gray, fontSize = 11.sp)
                            }
                        }
                    }
                } else {
                    Text("Tap Start Demo to activate live monitoring", color = Color.Gray, fontSize = 14.sp, modifier = Modifier.padding(top = 6.dp))
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        
        // Federated Edge AI Sandbox Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color(0xFF191919)),
            shape = RoundedCornerShape(12.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Continuous On-Device Learning", color = Color(0xFF1976D2), fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Text("Queued Weight Updates: $queuedUpdateCount", color = Color.White, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp))
                Text("Raw audio is discarded. Local weight updates will be synced when online.", color = Color.Gray, fontSize = 12.sp, modifier = Modifier.padding(top = 4.dp))
                
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 12.dp)) {
                    Button(
                        onClick = onUploadUpdates,
                        enabled = queuedUpdateCount > 0,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E2E2E)),
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Sync Updates", color = Color.White)
                    }
                    Button(
                        onClick = { showWeightsDialog.value = true },
                        enabled = queuedUpdateCount > 0,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E2E2E)),
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("View Diff", color = Color.White)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        if (!isRecording) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Simulate Scam Script",
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Escalates text triggers in background",
                            color = Color.Gray,
                            fontSize = 12.sp
                        )
                    }
                    Switch(
                        checked = isDemoMode,
                        onCheckedChange = onDemoModeToggle,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF1976D2),
                            checkedTrackColor = Color(0xFF1976D2).copy(alpha = 0.5f)
                        )
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = onToggleRecording,
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) Color(0xFFD32F2F) else Color(0xFF1976D2)
            ),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text(
                text = if (isRecording) "STOP MONITORING" else if (isDemoMode) "START DEMO (MIC)" else "START LIVE MIC",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Analysis runs 100% on-device via TensorFlow Lite.",
            color = Color.Gray,
            fontSize = 12.sp,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(24.dp))
    }
}
