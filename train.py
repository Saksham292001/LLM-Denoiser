import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from data import SR, DURATION_S, NUM_SAMPLES
from model import DeeperDenoiser
from utils import get_device, ensure_dir
import torchaudio

class DenoiseDataset(Dataset):
    def _init_(self, data_dir):
        self.data_dir = data_dir
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')
        self.file_list = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]

    def _len_(self):
        return len(self.file_list)

    def _getitem_(self, idx):
        file_name = self.file_list[idx]
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)
        
        clean_audio, _ = sf.read(clean_path)
        noisy_audio, _ = sf.read(noisy_path)
        
        return torch.from_numpy(clean_audio).unsqueeze(0).float(), torch.from_numpy(noisy_audio).unsqueeze(0).float()

def train_model(epochs=30, batch_size=32, lr=1e-4, data_dir="denoiser_dataset", save_dir="denoiser_checkpoints"):
    device = get_device()
    print(f"Using device: {device}")
    
    if not os.path.exists(data_dir):
        print("Dataset not found. Please run data.py to generate it first.")
        return
        
    dataset = DenoiseDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeeperDenoiser().to(device)
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256).to(device)
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ensure_dir(save_dir)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for clean, noisy in pbar:
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            output = model(noisy)
            
            # Use Spectrogram-based Loss
            clean_spec = spectrogram_transform(clean)
            output_spec = spectrogram_transform(output)
            loss = criterion_mse(output_spec, clean_spec)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(save_dir, f'denoiser_epoch_{epoch+1}.pth'))
        print(f"Checkpoint saved to {os.path.join(save_dir, f'denoiser_epoch_{epoch+1}.pth')}")

if _name_ == '_main_':
    train_model()
