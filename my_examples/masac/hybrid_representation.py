import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box


class CNNEncoder(nn.Module):
    """
    处理多通道地图的 CNN 编码器。
    输入: (batch, C, H, W) = (batch, 3, 64, 64)
    输出: (batch, cnn_output_dim)

    改动：
      1. 加入 CoordConv：自动在输入地图上附加 X/Y 坐标通道
      2. 换用小步长卷积，保留更多空间位置信息
    """

    def __init__(self, in_channels: int, grid_size: int, cnn_output_dim: int = 256):
        super().__init__()

        self.grid_size = grid_size

        # ── 改动 1：CoordConv，输入通道数 +2（x坐标通道 + y坐标通道）──
        actual_in_channels = in_channels + 2

        self.cnn = nn.Sequential(
            # ── 改动 2：stride 从 4 改为 2，保留更多空间细节 ──
            # (batch, C+2, 64, 64) → (batch, 32, 31, 31)
            nn.Conv2d(actual_in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 32, 31, 31) → (batch, 64, 15, 15)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 64, 15, 15) → (batch, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 128, 7, 7) → (batch, 64, 3, 3)
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 自动推导展平维度（加了坐标通道后尺寸变了，用 dummy 自动算）
        dummy = torch.zeros(1, actual_in_channels, grid_size, grid_size)
        cnn_flat_dim = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, cnn_output_dim),
            nn.ReLU()
        )

    def _add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        给输入地图附加两个坐标通道。
        x: (batch, C, H, W)
        返回: (batch, C+2, H, W)

        x_coord 通道：每列的值从 -1 到 1（代表水平位置）
        y_coord 通道：每行的值从 -1 到 1（代表垂直位置）
        """
        batch, C, H, W = x.shape

        # 生成 x 坐标：shape (1, 1, 1, W)，广播到 (batch, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=x.device)
        x_coords = x_coords.view(1, 1, 1, W).expand(batch, 1, H, W)

        # 生成 y 坐标：shape (1, 1, H, 1)，广播到 (batch, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device)
        y_coords = y_coords.view(1, 1, H, 1).expand(batch, 1, H, W)

        # 拼接到原始通道后面
        return torch.cat([x, x_coords, y_coords], dim=1)  # (batch, C+2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先加坐标通道，再过 CNN
        x = self._add_coord_channels(x)  # (batch, C+2, H, W)
        return self.fc(self.cnn(x))  # (batch, cnn_output_dim)


class HybridRepresentation(nn.Module):
    def __init__(self, input_space: Box, config=None, **kwargs):
        super().__init__()

        # 从 config 读取，带默认值
        vec_dim        = getattr(config, 'vec_dim', 9)
        map_channels   = getattr(config, 'map_channels', 3)
        map_h          = getattr(config, 'map_h', 64)
        map_w          = getattr(config, 'map_w', 64)
        cnn_output_dim = getattr(config, 'cnn_output_dim', 256)

        self.vec_dim      = vec_dim
        self.map_channels = map_channels
        self.map_h        = map_h
        self.map_w        = map_w
        self.map_flat_dim = map_channels * map_h * map_w

        total_dim = input_space.shape[0]
        assert total_dim == vec_dim + self.map_flat_dim, (
            f"维度不匹配: input_space={total_dim}, "
            f"vec_dim({vec_dim}) + map_flat_dim({self.map_flat_dim}) = {vec_dim + self.map_flat_dim}"
        )

        # CNN 编码器
        self.cnn_encoder = CNNEncoder(map_channels, map_h, cnn_output_dim)

        # 输出维度：向量直接透传 + CNN 压缩结果
        self.output_dim = vec_dim + cnn_output_dim

        # XuanCe 的 representation 需要暴露这个属性
        self.output_shapes = {'state': (self.output_dim,)}

    def forward(self, x: torch.Tensor):
        """
        x: (batch, obs_dim) 展平的观测
        """
        # ── 保证输入是 tensor ──────────────────────────────────────
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        # 1. 切分向量部分和地图部分
        vec_part = x[:, :self.vec_dim]                          # (batch, vec_dim)
        map_part = x[:, self.vec_dim:]                          # (batch, map_flat_dim)

        # 2. 地图 reshape 成 (batch, C, H, W)
        map_tensor = map_part.reshape(-1, self.map_channels, self.map_h, self.map_w)

        # 3. CNN 编码地图
        map_encoded = self.cnn_encoder(map_tensor)              # (batch, cnn_output_dim)

        # 4. 拼接
        out = torch.cat([vec_part, map_encoded], dim=-1)        # (batch, vec_dim + cnn_output_dim)

        # XuanCe representation 的标准输出格式
        return {'state': out}

    @property
    def input_shapes(self):
        return {'obs': (self.vec_dim + self.map_flat_dim,)}

class HybridCriticRepresentation(nn.Module):
    """
    Critic 专用混合表示层。

    Critic 的输入是所有 agent 的 [obs + action] 拼接成的大向量：
      [obs_0 | obs_1 | ... | obs_n | action_0 | action_1 | ... | action_n]

    处理方式：
      - 每个 obs 中的地图部分 → 共享 CNN → 压缩特征
      - 每个 obs 中的向量部分 → 直接保留
      - action 部分 → 直接保留
      - 全部拼接后输出
    """

    def __init__(self, input_space, config=None, action_dim=2, **kwargs):
        super().__init__()

        vec_dim = getattr(config, 'vec_dim', 9)
        map_channels = getattr(config, 'map_channels', 3)
        map_h = getattr(config, 'map_h', 64)
        map_w = getattr(config, 'map_w', 64)
        cnn_output_dim = getattr(config, 'cnn_output_dim', 256)
        n_agents = getattr(config, 'num_searchers', 3)

        print(f"[DEBUG Critic] vec_dim={vec_dim}, obs_dim={vec_dim + map_channels * map_h * map_w}")

        self.n_agents       = n_agents
        self.vec_dim        = vec_dim
        self.map_channels   = map_channels
        self.map_h          = map_h
        self.map_w          = map_w
        self.map_flat_dim   = map_channels * map_h * map_w
        self.obs_dim        = vec_dim + self.map_flat_dim
        self.action_dim     = action_dim
        self.cnn_output_dim = cnn_output_dim

        self.shared_cnn = CNNEncoder(map_channels, map_h, cnn_output_dim)

        self.output_dim = n_agents * (vec_dim + cnn_output_dim) + n_agents * action_dim
        self.output_shapes = {'state': (self.output_dim,)}

    def forward(self, x: torch.Tensor):
        """
        x: (batch, n_agents * obs_dim + n_agents * action_dim)
        """
        # print(f"[DEBUG Critic] x.shape = {x.shape}")
        # print(f"[DEBUG Critic] 期望 = (batch, {self.n_agents * self.obs_dim + self.n_agents * self.action_dim})")
        batch = x.shape[0]

        # ── Step 1：切分 obs 部分 和 action 部分 ──────────────────
        obs_total_dim = self.n_agents * self.obs_dim
        obs_all = x[:, :obs_total_dim]          # (batch, n_agents * obs_dim)
        action_all = x[:, obs_total_dim:]       # (batch, n_agents * action_dim)

        # ── Step 2：逐个 agent 处理 obs ───────────────────────────
        vec_parts = []
        map_encoded_parts = []

        for i in range(self.n_agents):
            start = i * self.obs_dim
            end = start + self.obs_dim
            obs_i = obs_all[:, start:end]                           # (batch, obs_dim)

            # 切分向量和地图
            vec_i = obs_i[:, :self.vec_dim]                         # (batch, vec_dim)
            map_i = obs_i[:, self.vec_dim:]                         # (batch, map_flat_dim)

            # 地图 reshape → CNN
            map_tensor = map_i.reshape(batch, self.map_channels, self.map_h, self.map_w)
            map_feat = self.shared_cnn(map_tensor)                  # (batch, cnn_output_dim)

            vec_parts.append(vec_i)
            map_encoded_parts.append(map_feat)

        # ── Step 3：拼接所有特征 ──────────────────────────────────
        # [vec_0, cnn_0, vec_1, cnn_1, ..., action_all]
        all_parts = []
        for i in range(self.n_agents):
            all_parts.append(vec_parts[i])
            all_parts.append(map_encoded_parts[i])
        all_parts.append(action_all)

        out = torch.cat(all_parts, dim=-1)      # (batch, output_dim)

        return {'state': out}

    @property
    def input_shapes(self):
        return self.output_shapes
