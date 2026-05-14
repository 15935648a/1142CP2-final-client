import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .game import BOARD_SIZE, ACTION_SIZE

IN_CHANNELS = 6


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


class PolicyValueNet(nn.Module):
    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs: np.ndarray):
        """Single obs (6,15,15) → (policy[450], value float)."""
        self.eval()
        device = next(self.parameters()).device
        t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, v = self(t)
        policy = F.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()
        return policy, float(v.float().squeeze())

    @torch.no_grad()
    def predict_batch(self, obs_batch: np.ndarray):
        """Batch (N,6,15,15) → policies (N,450), values (N,). GPU-friendly."""
        self.eval()
        device = next(self.parameters()).device
        t = torch.from_numpy(obs_batch).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, v = self(t)
        policies = F.softmax(logits.float(), dim=-1).cpu().numpy()
        values   = v.float().squeeze(-1).cpu().numpy()
        return policies, values


def build_net(num_res_blocks=4, channels=64, device=None) -> PolicyValueNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PolicyValueNet(num_res_blocks=num_res_blocks, channels=channels)
    return net.to(device)
