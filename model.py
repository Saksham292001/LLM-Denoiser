import torch
import torch.nn as nn

class DeeperDenoiser(nn.Module):
    """A deeper CNN-based denoiser model."""
    def _init_(self):
        super(DeeperDenoiser, self)._init_()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=32, stride=1, padding=16),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=32, stride=1, padding=16),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=32, stride=1, padding=16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=32, stride=1, padding=16),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=32, stride=1, padding=16),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=32, stride=1, padding=16),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
