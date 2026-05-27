"""Train the PolicyValueNet on self-play examples (GPU + AMP)."""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from .network import PolicyValueNet
from .config import Config


class ReplayBuffer:
    def __init__(self, max_size):
        self.buf = deque(maxlen=max_size)

    def extend(self, examples):
        self.buf.extend(examples)

    def sample(self, batch_size, device, pool=None):
        if pool is None:
            pool = list(self.buf)
        batch = random.sample(pool, min(batch_size, len(pool)))
        # Support both legacy 3-tuple and new 4-tuple (with is_full flag)
        if len(batch[0]) == 4:
            obs, pis, zs, is_full = zip(*batch)
            return (
                torch.from_numpy(np.stack(obs)).to(device),
                torch.from_numpy(np.stack(pis)).to(device),
                torch.from_numpy(np.array(zs, dtype=np.float32)).to(device),
                torch.tensor(is_full, dtype=torch.bool).to(device),
            )
        obs, pis, zs = zip(*batch)
        n = len(batch)
        return (
            torch.from_numpy(np.stack(obs)).to(device),
            torch.from_numpy(np.stack(pis)).to(device),
            torch.from_numpy(np.array(zs, dtype=np.float32)).to(device),
            torch.ones(n, dtype=torch.bool).to(device),
        )

    def __len__(self):
        return len(self.buf)


class Trainer:
    def __init__(self, network: PolicyValueNet, config: Config):
        self.net     = network
        self.cfg     = config
        self.device  = torch.device(config.DEVICE)
        self.replay  = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        self.scaler  = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(
            network.parameters(), lr=config.LR, weight_decay=config.L2_REG
        )
        self._iter = 0   # training iteration counter for LR decay

    # ------------------------------------------------------------------
    def add_examples(self, examples):
        self.replay.extend(examples)

    def train_epoch(self):
        if len(self.replay) < self.cfg.BATCH_SIZE:
            return None

        self.net.train()
        cap = getattr(self.cfg, "BATCHES_PER_EPOCH", None)
        n_batches = max(1, cap if cap else len(self.replay) // self.cfg.BATCH_SIZE)
        total_loss = total_v = total_p = total_te = 0.0
        pool = list(self.replay.buf)

        for _ in range(n_batches):
            obs, pis, zs, is_full = self.replay.sample(self.cfg.BATCH_SIZE, self.device, pool=pool)

            with torch.autocast(device_type=self.device.type,
                                enabled=self.device.type == "cuda"):
                logits, v = self.net(obs)
                v_loss    = F.mse_loss(v.squeeze(-1), zs)
                log_p     = F.log_softmax(logits, dim=-1)
                # Policy loss only on full-search positions (playout cap)
                if is_full.any():
                    p_loss = -(pis[is_full] * log_p[is_full]).sum(dim=-1).mean()
                else:
                    p_loss = torch.tensor(0.0, device=self.device)
                loss = self.cfg.VALUE_LOSS_WEIGHT * v_loss + p_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            full_pis   = pis[is_full] if is_full.any() else pis
            target_ent = -(full_pis * torch.log(full_pis + 1e-8)).sum(dim=-1).mean()
            total_loss += loss.item()
            total_v    += v_loss.item()
            total_p    += p_loss.item()
            total_te   += target_ent.item()

        return {
            "total":         total_loss / n_batches,
            "value":         total_v    / n_batches,
            "policy":        total_p    / n_batches,
            "target_entropy": total_te  / n_batches,
        }

    def train(self, num_epochs=None):
        import time
        n      = num_epochs or self.cfg.NUM_EPOCHS
        losses = []
        t0 = time.time()
        for epoch in range(n):
            result = self.train_epoch()
            if result:
                losses.append(result)
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{n}  "
                      f"loss={result['total']:.4f}  "
                      f"v={result['value']:.4f}  "
                      f"p={result['policy']:.4f}  "
                      f"H(target)={result['target_entropy']:.4f}  "
                      f"t={elapsed:.0f}s")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return losses

    def step_lr_decay(self):
        """Call once per training iteration. Decays LR on schedule."""
        self._iter += 1
        if self._iter % self.cfg.LR_DECAY_ITERS == 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= self.cfg.LR_DECAY
            print(f"  LR decayed → {self.optimizer.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter": self._iter,
        }, path)
        print(f"Saved → {path}")

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.net.load_state_dict(ckpt["model"])
            if load_optimizer and "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self._iter = ckpt.get("iter", 0)
        else:
            self.net.load_state_dict(ckpt)   # legacy format
        print(f"Loaded ← {path}  (iter={self._iter})")

    def gpu_memory_str(self):
        if self.device.type != "cuda":
            return ""
        alloc = torch.cuda.memory_allocated(self.device) / 1e9
        reserv = torch.cuda.memory_reserved(self.device) / 1e9
        return f"GPU {alloc:.2f}/{reserv:.2f} GB"
