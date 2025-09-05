import numpy as np
import soundfile as sf
import os

SR = 16000
DURATION_S = 2
NUM_SAMPLES = 5000

def create_noisy_sine_wave(freq=440, sr=SR, duration_s=DURATION_S, noise_amp=0.1):
    """Generates a clean sine wave and adds white noise."""
    t = np.linspace(0., duration_s, int(sr * duration_s))
    clean_wave = 0.5 * np.sin(2. * np.pi * freq * t)
    noise = noise_amp * np.random.randn(len(t))
    noisy_wave = clean_wave + noise
    
    noisy_wave = np.clip(noisy_wave, -1.0, 1.0).astype(np.float32)
    clean_wave = np.clip(clean_wave, -1.0, 1.0).astype(np.float32)
    
    return clean_wave, noisy_wave

def generate_dataset(output_dir="denoiser_dataset"):
    """Generates a dataset of clean and noisy audio pairs."""
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, 'clean'))
    ensure_dir(os.path.join(output_dir, 'noisy'))
    
    for i in range(NUM_SAMPLES):
        freq = np.random.randint(200, 1000)
        clean, noisy = create_noisy_sine_wave(freq=freq)
        
        sf.write(os.path.join(output_dir, 'clean', f'{i:05d}.wav'), clean, SR)
        sf.write(os.path.join(output_dir, 'noisy', f'{i:05d}.wav'), noisy, SR)
        
    print(f"Generated {NUM_SAMPLES} pairs of clean and noisy audio in '{output_dir}'.")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

if _name_ == '_main_':
    generate_dataset()
