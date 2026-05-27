"""
ICM (Intrinsic Curiosity Module)
适配 simple_search.py 的观测结构：
  obs = [pos(2), vel(2), other_agents(n*2), coverage(1), map_flat(3*64*64)]
  vec_dim = 4 + (n_searchers-1)*2 + 1   ← 非地图部分
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICMEncoder(nn.Module):
    """地图用CNN，向量用MLP，最后融合"""
    def __init__(self, vec_dim: int, map_channels: int, map_size: int, feature_dim: int = 128):
        super().__init__()
        self.vec_dim = vec_dim
        self.map_channels = map_channels
        self.map_size = map_size

        # CNN 处理地图部分：64→32→16→8
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = 32 * (map_size // 8) * (map_size // 8)  # 32*8*8=2048

        # MLP 处理向量部分
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) → (B, feature_dim)"""
        B = obs.shape[0]
        map_flat_dim = self.map_channels * self.map_size * self.map_size

        vec_part = obs[:, :self.vec_dim]
        map_part = obs[:, self.vec_dim: self.vec_dim + map_flat_dim]

        map_3d   = map_part.view(B, self.map_channels, self.map_size, self.map_size)
        cnn_feat = self.cnn(map_3d)
        vec_feat = self.vec_mlp(vec_part)

        return self.fusion(torch.cat([cnn_feat, vec_feat], dim=-1))


class ICMForwardModel(nn.Module):
    """给定 φ(s_t) + a_t，预测 φ(s_{t+1})"""
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, phi_s, action):
        return self.net(torch.cat([phi_s, action], dim=-1))


class ICMInverseModel(nn.Module):
    """给定 φ(s_t) + φ(s_{t+1})，预测动作 a_t"""
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, phi_s, phi_s_next):
        return self.net(torch.cat([phi_s, phi_s_next], dim=-1))


class ICMModule(nn.Module):
    def __init__(
        self,
        vec_dim: int,
        map_channels: int,
        map_size: int,
        action_dim: int,
        feature_dim: int = 128,
        eta: float  = 0.01,
        beta: float = 0.2,
        lr: float   = 3e-4,
        device: str = 'cpu'
    ):
        super().__init__()
        self.eta    = eta
        self.beta   = beta
        self.device = torch.device(device)

        self.encoder       = ICMEncoder(vec_dim, map_channels, map_size, feature_dim)
        self.forward_model = ICMForwardModel(feature_dim, action_dim)
        self.inverse_model = ICMInverseModel(feature_dim, action_dim)
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # ── 推理接口：计算内在奖励，不更新权重 ──────────────────────────────
    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        obs:      np.ndarray,   # (n_agents, obs_dim)
        actions:  np.ndarray,   # (n_agents, action_dim)
        obs_next: np.ndarray    # (n_agents, obs_dim)
    ) -> np.ndarray:            # (n_agents,)
        obs_t  = torch.FloatTensor(obs).to(self.device)
        act_t  = torch.FloatTensor(actions).to(self.device)
        obs_n  = torch.FloatTensor(obs_next).to(self.device)

        phi_s      = self.encoder(obs_t)
        phi_s_next = self.encoder(obs_n)
        phi_pred   = self.forward_model(phi_s, act_t)

        errors = F.mse_loss(phi_pred, phi_s_next, reduction='none')  # (B, feature_dim)
        return (self.eta * errors.mean(dim=-1)).cpu().numpy()         # (B,)

    # ── 训练接口：用 Replay Buffer 的 batch 更新网络 ─────────────────────
    def update(
        self,
        obs:      torch.Tensor,   # (B, obs_dim)
        actions:  torch.Tensor,   # (B, action_dim)
        obs_next: torch.Tensor    # (B, obs_dim)
    ) -> dict:
        phi_s      = self.encoder(obs)
        phi_s_next = self.encoder(obs_next).detach()

        forward_loss = F.mse_loss(self.forward_model(phi_s, actions), phi_s_next)
        inverse_loss = F.mse_loss(self.inverse_model(phi_s, phi_s_next), actions)
        loss = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'icm_loss':     loss.item(),
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item()
        }

    # ── 适配 XuanCe batch 格式的便捷接口 ────────────────────────────────
    def update_from_batch(self, obs_batch, action_batch, obs_next_batch):
        # 输入已经是 (B*n_agents, dim) 的 2D array，直接转 tensor
        if obs_batch.ndim == 3:
            B, N, D = obs_batch.shape
            obs_batch = obs_batch.reshape(B * N, D)
            action_batch = action_batch.reshape(B * N, -1)
            obs_next_batch = obs_next_batch.reshape(B * N, -1)
        # ndim==2 时不做任何 reshape，直接往下走

        return self.update(
            torch.FloatTensor(obs_batch).to(self.device),
            torch.FloatTensor(action_batch).to(self.device),
            torch.FloatTensor(obs_next_batch).to(self.device)
        )

