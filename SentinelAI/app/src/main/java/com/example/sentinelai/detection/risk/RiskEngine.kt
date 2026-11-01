package com.example.sentinelai.detection.risk

enum class RiskLevel {
    SAFE, LOW, MEDIUM, HIGH, CRITICAL
}

data class RiskResult(
    val level: RiskLevel,
    val score: Int, // 0 to 100
    val reasons: List<String>
)

class RiskEngine {

    fun calculateRisk(
        voiceSyntheticProbability: Float, // 0.0 to 1.0
        conversationScamScore: Float, // 0.0 to 1.0
        conversationTags: List<String>
    ): RiskResult {
        // Use the highest risk signal, but dampen voice probability if there's no suspicious conversation
        val combinedRaw = if (voiceSyntheticProbability > 0.85f || conversationScamScore > 0.6f) {
            maxOf(voiceSyntheticProbability, conversationScamScore)
        } else {
            maxOf(voiceSyntheticProbability * 0.7f, conversationScamScore)
        }
        val finalScore = (combinedRaw * 100).toInt().coerceIn(0, 100)

        val reasons = mutableListOf<String>()
        
        if (voiceSyntheticProbability > 0.75f) {
            reasons.add("Possible AI-generated voice detected (${(voiceSyntheticProbability * 100).toInt()}%)")
        }
        
        if (conversationScamScore > 0.5f) {
            reasons.add("Suspicious conversation patterns detected")
        }
        
        reasons.addAll(conversationTags)

        val level = when {
            finalScore >= 80 -> RiskLevel.CRITICAL
            finalScore >= 60 -> RiskLevel.HIGH
            finalScore >= 40 -> RiskLevel.MEDIUM
            finalScore >= 20 -> RiskLevel.LOW
            else -> RiskLevel.SAFE
        }

        return RiskResult(
            level = level,
            score = finalScore,
            reasons = reasons.distinct()
        )
    }
}
