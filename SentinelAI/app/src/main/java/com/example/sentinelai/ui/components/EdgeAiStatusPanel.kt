package com.example.sentinelai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun EdgeAiStatusPanel(
    modelName: String = "Voice Spoof Detector",
    isLocal: Boolean = true,
    latencyMs: Long,
    modelSizeMb: Float
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF1E1E1E), shape = RoundedCornerShape(12.dp))
            .border(1.dp, Color(0xFF333333), shape = RoundedCornerShape(12.dp))
            .padding(16.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(10.dp)
                    .background(Color.Green, shape = RoundedCornerShape(5.dp))
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "EDGE AI ACTIVE",
                color = Color.Green,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
        }
        
        Spacer(modifier = Modifier.height(12.dp))
        
        StatusRow("Model", modelName)
        StatusRow("Inference", if (isLocal) "ON DEVICE" else "CLOUD")
        StatusRow("Network Required", "NO")
        StatusRow("Inference Latency", "${latencyMs} ms")
        StatusRow("Model Size", "${"%.2f".format(modelSizeMb)} MB")
    }
}

@Composable
private fun StatusRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, color = Color.Gray, fontSize = 14.sp)
        Text(text = value, color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Medium)
    }
}
