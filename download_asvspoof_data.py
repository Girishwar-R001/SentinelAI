import os
import numpy as np
import soundfile as sf

def generate_mock_asvspoof():
    base_dir = "dataset/asvspoof"
    wav_dir = os.path.join(base_dir, "ASVspoof2019_LA_train/wav")
    protocols_dir = os.path.join(base_dir, "ASVspoof2019_LA_protocols")
    
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(protocols_dir, exist_ok=True)
    
    import librosa
    from gtts import gTTS
    import random
    
    print("Setting up mock ASVspoof directory structure...")
    
    num_samples = 40
    protocol_entries = []
    sr = 16000
    
    # Pre-generate a real voice sample base using librosa
    try:
        path = librosa.example('libri1')
        real_base, _ = librosa.load(path, sr=sr)
    except:
        # Fallback if librosa example fails
        t = np.linspace(0, 1.0, sr)
        real_base = 0.5 * np.sin(2 * np.pi * 300 * t)
        
    sentences = [
        "Your account is compromised.", "Please transfer funds.",
        "Your car warranty expired.", "I am a prince.", "Download our software."
    ]
    
    for i in range(num_samples):
        file_id = f"LA_T_{1000000 + i}"
        speaker_id = f"LA_{1000 + (i % 5)}"
        is_spoof = (i % 2 == 1)
        
        file_path = os.path.join(wav_dir, f"{file_id}.wav")
        
        if is_spoof:
            attack_id = f"A0{1 + (i % 6)}"
            label = "spoof"
            # Fake voice using TTS
            text = random.choice(sentences)
            temp_mp3 = f"dataset/asvspoof/temp_{i}.mp3"
            try:
                tts = gTTS(text=text, lang='en')
                tts.save(temp_mp3)
                y, _ = librosa.load(temp_mp3, sr=sr)
                os.remove(temp_mp3)
            except:
                y = np.random.normal(0, 0.05, sr) # Fallback noise
        else:
            attack_id = "-"
            label = "bonafide"
            # Real voice using augmented librosa sample
            y = librosa.effects.pitch_shift(real_base, sr=sr, n_steps=random.uniform(-3, 3))
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
            
        # Pad or trim to exactly 1 second (16000 samples)
        if len(y) > sr:
            y = y[:sr]
        elif len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
            
        sf.write(file_path, y, sr)
        
        # Protocol entry: Speaker_ID File_ID - Attack_ID Label
        entry = f"{speaker_id} {file_id} - {attack_id} {label}"
        protocol_entries.append(entry)
        
    # Write protocol file
    protocol_path = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    with open(protocol_path, "w") as f:
        f.write("\n".join(protocol_entries))
        
    print(f"Generated {num_samples} mock speech files in {wav_dir}")
    print(f"Written mock ASVspoof protocol to {protocol_path}")

if __name__ == "__main__":
    generate_mock_asvspoof()
