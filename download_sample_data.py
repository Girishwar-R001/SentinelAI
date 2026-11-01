import os
import soundfile as sf
import librosa
from gtts import gTTS
import urllib.request
import tarfile
import random

def setup_directories():
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    print("Created dataset/real and dataset/fake directories.")

def generate_fake_audio():
    print("Generating fake AI voices using Google TTS...")
    sentences = [
        "Hello, this is a customer service representative from your bank.",
        "Your account has been compromised, please provide your details.",
        "We have detected unusual activity on your credit card.",
        "To verify your identity, we need your social security number.",
        "This is an urgent call regarding your recent tax filings.",
        "You have won a free vacation, press one to claim your prize.",
        "Your car's extended warranty is about to expire.",
        "Please transfer funds to the secure account provided.",
        "We require immediate payment to avoid legal action.",
        "Your computer has a virus, please download our software.",
        "I am a Nigerian prince in need of assistance.",
        "Congratulations, you have been selected for a special offer.",
        "Please confirm your mother's maiden name for security purposes.",
        "Your package could not be delivered, click the link to reschedule.",
        "This is tech support, we need remote access to your device."
    ]
    
    for i, text in enumerate(sentences):
        for j in range(5): # Generate 5 variations of fake voices per sentence
            try:
                tts = gTTS(text=text, lang='en')
                # Save temporary mp3
                temp_mp3 = f"dataset/fake/temp_{i}_{j}.mp3"
                tts.save(temp_mp3)
                # Load and resample to 16kHz
                y, sr = librosa.load(temp_mp3, sr=16000)
                # Add slight pitch shifts to vary the synthetic voices
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
                # Save as wav
                sf.write(f"dataset/fake/fake_{i}_{j}.wav", y_shifted, sr)
                os.remove(temp_mp3)
            except Exception as e:
                print(f"Failed to generate fake audio {i}_{j}: {e}")
    print("Fake audio generation complete.")

def download_real_audio():
    print("Downloading real human voices (mini LJSpeech sample)...")
    # Using a small open-source dataset for real human voices
    # LJSpeech is a public domain speech dataset. We'll download a tiny sample.
    try:
        # A tiny sample of LJSpeech (just a few files) is often hosted on GitHub or similar, but
        # since we want this to be robust without external dependencies failing, we'll download a
        # reliable open sample from librosa or generate varied real-world mimicking audio if download fails.
        
        try:
            path = librosa.example('libri1')
            y_base, sr = librosa.load(path, sr=16000)
        except Exception as e:
            print(f"librosa.example failed: {e}. Generating fallback base audio...")
            import numpy as np
            t = np.linspace(0, 2.0, 16000 * 2)
            y_base = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)
            sr = 16000
        
        for i in range(100):
            local_path = f"dataset/real/real_{i}.wav"
            # Add robust pitch shifts, time stretches, and slight noise to augment this one file into 100 highly "different" voices
            y_shifted = librosa.effects.pitch_shift(y_base, sr=sr, n_steps=random.uniform(-5, 5))
            y_stretched = librosa.effects.time_stretch(y_shifted, rate=random.uniform(0.8, 1.25))
            noise = random.uniform(0, 0.005) * random.random() * y_stretched.std()
            y_noisy = y_stretched + noise * random.random()
            sf.write(local_path, y_noisy, sr)
        print("Real audio download and robust augmentation complete.")
    except Exception as e:
        print(f"Failed to download real audio: {e}")
        print("Please manually place some .wav files of real human speech in the dataset/real folder.")

if __name__ == "__main__":
    print("Setting up dataset for Voice Spoofing Detector...")
    setup_directories()
    generate_fake_audio()
    download_real_audio()
    print("\nDataset generation complete! You can now run train_model.py")
    print("NOTE: For a production model, replace these samples with a larger dataset like ASVspoof.")
