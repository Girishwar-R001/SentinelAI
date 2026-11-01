package com.example.sentinelai.detection.conversation

class ConversationAnalyzer {

    // Simple rule-based intent mapping for MVP
    private val triggers = mapOf(
        "financial_request" to listOf(
            "send money", "wire transfer", "gift card", "bitcoin", "crypto",
            "bank account", "credit card", "payment", "social security"
        ),
        "artificial_urgency" to listOf(
            "immediately", "right now", "arrest", "warrant", "before it's too late",
            "act fast", "urgent", "police"
        ),
        "impersonation" to listOf(
            "this is your bank", "irs", "fbi", "police department", "grandson",
            "granddaughter", "in jail", "hospital"
        ),
        "otp_request" to listOf(
            "one time passcode", "verification code", "read the code", "otp",
            "six digit code", "text message code"
        )
    )

    data class AnalysisResult(
        val scamScore: Float, // 0.0 to 1.0
        val detectedTags: List<String>
    )

    fun analyzeTranscript(transcript: String): AnalysisResult {
        val lowerText = transcript.lowercase()
        val tags = mutableListOf<String>()
        var matchedTriggers = 0

        for ((intent, keywords) in triggers) {
            for (keyword in keywords) {
                if (lowerText.contains(keyword)) {
                    tags.add(intent)
                    matchedTriggers++
                    break // count intent only once
                }
            }
        }

        // Calculate a score based on matched intents
        val score = when {
            matchedTriggers >= 3 -> 0.95f
            matchedTriggers == 2 -> 0.70f
            matchedTriggers == 1 -> 0.40f
            else -> 0.10f // baseline
        }

        return AnalysisResult(
            scamScore = score,
            detectedTags = tags.distinct()
        )
    }
}
