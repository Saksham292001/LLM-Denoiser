import torch
import soundfile as sf
import os
import numpy as np
from model import DeeperDenoiser
from utils import get_device

SR = 16000
DURATION_S = 2

def denoise_audio(input_path, output_path, checkpoint_path):
    device = get_device()
    model = DeeperDenoiser().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    noisy_audio, _ = sf.read(input_path)
    noisy_tensor = torch.from_numpy(noisy_audio).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
        
    denoised_audio = denoised_tensor.squeeze().cpu().numpy()
    sf.write(output_path, denoised_audio, SR)
    
    print(f"Denoised audio saved to {output_path}")

if _name_ == '_main_':
    t = np.linspace(0., DURATION_S, int(SR * DURATION_S))
    clean_wave = 0.5 * np.sin(2. * np.pi * 600 * t)
    noise = 0.1 * np.random.randn(len(t))
    noisy_wave = clean_wave + noise
    sf.write('test_noisy_sample.wav', noisy_wave, SR)
    
    denoise_audio(
        input_path='test_noisy_sample.wav',
        output_path='denoised_output.wav',
        checkpoint_path='denoiser_checkpoints/denoiser_epoch_30.pth'
    )
