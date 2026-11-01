# 🛡️ SentinelAI – AI-Powered Voice Scam Detection

SentinelAI is a privacy-focused AI application designed to detect AI-generated voice clones and prevent voice-based scams. It analyzes audio to identify whether a voice is genuine or artificially generated, helping users recognize potential scam calls before they become victims.

## 🚨 Problem

With the rise of AI voice cloning technology, scammers can now replicate someone's voice using just a small audio sample. These cloned voices are increasingly being used in phone scams, impersonation attacks, and fraudulent emergency calls.

Traditional scam detection methods struggle to identify these attacks because AI-generated voices can sound almost identical to real human voices.

## 💡 Our Solution

SentinelAI uses machine learning and deep learning models to analyze voice recordings and identify signs of AI-generated speech.

The application focuses on privacy by performing audio analysis locally rather than depending entirely on cloud-based processing.

It also includes an Android call-audio blocking demonstration to showcase how suspicious calls could be identified and prevented from reaching users.

## ✨ Key Features

- **AI Voice Detection:** Identifies whether an audio recording is human-generated or AI-generated.
- **Voice Clone Detection:** Detects potential AI voice cloning attacks used in impersonation scams.
- **Privacy-Focused Processing:** Supports local audio analysis to minimize the need to send sensitive voice recordings to external servers.
- **Real-Time Detection Concept:** Demonstrates how suspicious voice calls can be analyzed to identify potential threats.
- **Android Call Blocking Demo:** Demonstrates how suspicious calls can be blocked before users interact with them.
- **Audio Preprocessing:** Processes audio into a standardized 16 kHz format for model analysis.
- **Visual Results:** Presents detection results to help users understand potential voice-based threats.

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| TensorFlow | 1D CNN model |
| PyTorch | RawNet2 deep learning model |
| Librosa | Audio processing and feature extraction |
| ASVspoof | Audio spoofing detection dataset |
| gTTS | Generating synthetic speech for testing |

## ⚙️ How It Works

1. The audio is preprocessed and converted to a standardized 16 kHz format.
2. The processed audio is passed through deep learning models.
3. The models analyze voice patterns and identify potential synthetic speech.
4. SentinelAI generates a classification result indicating whether the audio is likely genuine or AI-generated.

## 🌍 Real-World Impact

SentinelAI aims to make voice-based communication safer by helping people identify AI voice impersonation attacks.

Potential applications include:
- Preventing family emergency impersonation scams.
- Identifying fraudulent phone calls.
- Protecting individuals from AI voice cloning attacks.
- Improving awareness of synthetic audio threats.
- Supporting privacy-conscious voice security solutions.

## 🔮 Future Scope

- Improved detection accuracy against advanced voice cloning models.
- Integration with live phone calls.
- Enhanced Android integration for automatic suspicious-call detection.
- Support for multiple languages and accents.
- Continuous model improvements against emerging AI-generated voices.

## 👨‍💻 Project

SentinelAI is developed as an AI-powered security solution exploring the use of edge AI and deep learning to address the growing threat of voice cloning scams.

**Built with a focus on AI security, privacy, and real-world protection.**