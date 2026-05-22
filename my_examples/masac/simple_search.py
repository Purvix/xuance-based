from types import SimpleNamespace
import pygame
import numpy as np
from gymnasium import spaces
from xuance.environment import RawMultiAgentEnv
import matplotlib
matplotlib.use('TkAgg')   # 如果是无头服务器改成 'Agg'
import matplotlib.pyplot as plt

class SearchEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(SearchEnv, self).__init__()

        # --- 配置参数 ---
        self.env_id = getattr(env_config, 'env_id', 'Search_v1')
        self.grid_size = getattr(env_config, 'grid_size', 64)  # 64x64 网格
        self.n_searchers = getattr(env_config, 'num_searchers', 3)  # RL控制的智能体
        self.n_targets = getattr(env_config, 'num_targets', 2)  # 规则控制的智能体
        self.n_obstacles = getattr(env_config, 'num_obstacles', 5)
        self.max_obstacle_ratio = getattr(env_config, 'max_obstacle_ratio', 0.20)
        self.max_episode_steps = getattr(env_config, 'max_episode_steps', 200)
        self.vec_dim = getattr(env_config, 'vec_dim', 8)

        # 渲染相关变量
        self.screen = None
        self.clock = None
        self.render_scale = 15  # 缩放比例: 1个网格 = 10个像素 (64x64 -> 640x640窗口)
        self.render_fps = 10    # 渲染帧率

        # 探测与衰减参数
        self.detect_radius = getattr(env_config, 'detect_radius', 5.0)  # 探测半径
        self.decay_rate = getattr(env_config, 'decay_rate', 0.98)  # 记忆衰减率
        self.min_decay_val = getattr(env_config, 'min_decay_val', 0.1)  # 衰减下限
        self.collision_dist = 3.0  # 判定抓捕/碰撞距离

        #奖励相关
        self.step_penalty = getattr(env_config, 'step_penalty', -0.01)  # 每步扣 0.01 分
        self.agent_collision_dist = getattr(env_config, 'agent_collision_dist', 2.0)  # 碰撞判定距离
        self.agent_collision_penalty = getattr(env_config, 'agent_collision_penalty', -0.5)  # 碰撞扣 0.2 分
        self.coverage_reward_scale = getattr(env_config, 'coverage_reward_scale', 10.0)
        self.coverage_threshold = getattr(env_config, 'coverage_threshold', 0.90)
        self.coverage_done_reward = getattr(env_config, 'coverage_done_reward', 20.0)
        self._prev_coverage = 0.0  # 用于计算每步覆盖率增量

        # 智能体相关
        self.searcher_ids = [f"searcher_{i}" for i in range(self.n_searchers)]
        self.target_ids = [f"target_{i}" for i in range(self.n_targets)]
        self.agents = self.searcher_ids
        self.num_agents = len(self.agents)

        self.max_speed = getattr(env_config, 'max_speed', 2.0)  # 最大速度
        self.accel_scale = getattr(env_config, 'accel_scale', 1.0)  # 加速度缩放因子（动作转化为力的倍数）
        self.damping = getattr(env_config, 'damping', 0.25)  # 阻尼/摩擦力系数 (0~1)，越大刹车越快
        # --- 地图参数 ---
        self.map_channels = getattr(env_config, 'map_channels', 3)
        self.map_h = getattr(env_config, 'map_h', self.grid_size)
        self.map_w = getattr(env_config, 'map_w', self.grid_size)
        self.max_other_agents = self.n_searchers - 1


        # --- 内部状态初始化 ---
        self._current_step = 0
        self.searcher_pos = np.zeros((self.n_searchers, 2))
        self.target_pos = np.zeros((self.n_targets, 2))
        self.target_alive = np.ones(self.n_targets, dtype=bool)  # ← 新增，True=存活
        self.searcher_vel = np.zeros((self.n_searchers, 2))

        # 地图状态
        # global_grid_map: 真实环境地图 (0:空地, 1:障碍物)
        self.global_obstacle_map = np.zeros((self.grid_size, self.grid_size))

        # 多通道地图: [N, C, H, W]
        # ch0: 探索记忆  ch1: 障碍物  ch2: 目标位置（实时+衰减）
        self.agent_maps = np.zeros((self.n_searchers, self.map_channels, self.grid_size, self.grid_size))

        # 保留 agent_memory_maps 作为 ch0 的别名，供 state() 和 render() 兼容使用
        # （指向同一块内存，修改 agent_maps[:,0] 即修改 agent_memory_maps）
        self.agent_memory_maps = self.agent_maps[:, 0, :, :]

        # 预计算网格坐标，用于加速距离计算
        x = np.linspace(0, self.grid_size - 1, self.grid_size)
        y = np.linspace(0, self.grid_size - 1, self.grid_size)
        self.grid_x, self.grid_y = np.meshgrid(x, y, indexing='ij')  # [64, 64]

        # --- 调试可视化开关 ---
        self.debug_obs_vis = getattr(env_config, 'debug_obs_vis', False)  # 是否开启观测可视化
        self.debug_obs_interval = getattr(env_config, 'debug_obs_interval', 20)  # 每隔多少步刷新
        self._fig = None  # matplotlib figure 句柄

        # --- 空间定义 ---
        # 动作空间: 连续控制 [vx, vy], 范围 [-1, 1]
        self.action_space = {agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                             for agent in self.agents}
        # --- 观测空间维度计算 ---
        # 1. 自身位置归一化: 2 维 (x, y)
        # 2. 自身速度归一化: 2 维 (vx, vy)
        base_obs_dim = 4
        # 3. 队友相对位置（全局通信，始终可知，Padding 到固定长度）
        self.other_agent_dim = (self.n_searchers - 1) * 2
        # 4. 多通道地图 flatten
        self.map_flat_dim = self.map_channels * self.map_h * self.map_w
        # 总维度
        obs_dim = base_obs_dim + self.other_agent_dim + self.map_flat_dim + 1
        self.observation_space = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }

        # # 计算状态维度
        # state_dim = self.grid_size * self.grid_size + 2 * self.n_searchers + 2 * self.n_targets  # 4096 + 6 + 4 = 4106
        # self.state_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        # 1. 状态空间维度: (n_searchers * 4) + 多通道地图展平维度 (map_channels * grid_size * grid_size)
        state_dim = (self.n_searchers * 4) + (self.map_channels * self.grid_size * self.grid_size)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        # 连续动作空间通常不需要mask，除非特定算法需要
        return None

    def agent_mask(self):
        return {agent: True for agent in self.agents}

    def state(self):
        # 1. 所有智能体位置归一化到 [0, 1]
        pos_flat = (self.searcher_pos / self.grid_size).flatten()

        # 2. 所有智能体速度归一化到 [-1, 1]
        vel_flat = (self.searcher_vel / self.max_speed).flatten()

        # 3. 合并多个智能体的多通道地图为一个多通道地图
        # self.agent_maps 的 shape 为 (n_searchers, map_channels, grid_size, grid_size)
        # 使用 np.max 仅在智能体维度 (axis=0) 上取最大值
        # 结果 merged_map 的 shape 为 (map_channels, grid_size, grid_size)
        merged_map = np.max(self.agent_maps, axis=0)

        # 将多通道地图展平为一维向量
        merged_map_flat = merged_map.flatten()

        # 4. 拼接成一维向量
        state_vector = np.concatenate([
            pos_flat,
            vel_flat,
            merged_map_flat
        ]).astype(np.float32)

        return state_vector

    def _global_coverage(self):
        """返回 0~1 之间的全局覆盖率（向量化版本）"""
        # np.max 在 axis=0 上合并所有 agent 的探索记忆
        global_map = np.max(self.agent_maps[:, 0, :, :], axis=0)  # (H, W)
        explored = np.count_nonzero(global_map)  # 比 np.sum(>0) 更快
        total = self.grid_size * self.grid_size
        return float(explored) / float(total)

    def _generate_obstacles(self):
        """
        生成混合类型障碍物地图，包含：
          - 矩形障碍物（房间/墙壁）
          - L形障碍物（拐角）
          - 走廊（细长条）
        保证边界留有安全通道，不封死地图。
        """
        obs_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        G = self.grid_size

        # 安全边距：边界附近不放障碍物，保证智能体出生区域畅通
        margin = int(self.detect_radius) + 2

        # 目标覆盖率：障碍物占地图总面积的比例上限
        max_cells = int(G * G * self.max_obstacle_ratio)

        placed_cells = 0

        for attempt in range(self.n_obstacles):
            if placed_cells >= max_cells:
                break

            # 随机选择障碍物类型
            obs_type = np.random.choice(['rect', 'rect', 'L', 'corridor'], p=[0.4, 0.2, 0.2, 0.2])

            # 随机选择中心点（避开边界安全区）
            cx = np.random.randint(margin, G - margin)
            cy = np.random.randint(margin, G - margin)

            # 生成候选障碍物的格子列表
            cells = []

            if obs_type == 'rect':
                # 矩形：宽高随机，3~8 格
                w = np.random.randint(3, 9)
                h = np.random.randint(3, 9)
                x0 = max(margin, cx - w // 2)
                x1 = min(G - margin, cx + w // 2)
                y0 = max(margin, cy - h // 2)
                y1 = min(G - margin, cy + h // 2)
                for x in range(x0, x1):
                    for y in range(y0, y1):
                        cells.append((x, y))

            elif obs_type == 'L':
                # L 形：两段矩形拼接
                arm_len = np.random.randint(4, 10)
                arm_w = np.random.randint(1, 3)
                # 横臂
                x0 = max(margin, cx - arm_len // 2)
                x1 = min(G - margin, cx + arm_len // 2)
                for x in range(x0, x1):
                    for y in range(cy, min(G - margin, cy + arm_w)):
                        cells.append((x, y))
                # 竖臂
                y0 = max(margin, cy - arm_len // 2)
                y1 = min(G - margin, cy + arm_len // 2)
                for y in range(y0, y1):
                    for x in range(cx, min(G - margin, cx + arm_w)):
                        cells.append((x, y))

            elif obs_type == 'corridor':
                # 走廊：细长条，随机水平或垂直
                length = np.random.randint(8, 20)
                width = np.random.randint(1, 3)
                horizontal = np.random.rand() > 0.5
                if horizontal:
                    x0 = max(margin, cx - length // 2)
                    x1 = min(G - margin, cx + length // 2)
                    for x in range(x0, x1):
                        for y in range(cy, min(G - margin, cy + width)):
                            cells.append((x, y))
                else:
                    y0 = max(margin, cy - length // 2)
                    y1 = min(G - margin, cy + length // 2)
                    for y in range(y0, y1):
                        for x in range(cx, min(G - margin, cx + width)):
                            cells.append((x, y))

            # 去重
            cells = list(set(cells))

            if len(cells) == 0:
                continue

            # 写入地图
            new_cells = 0
            for x, y in cells:
                if obs_map[x, y] == 0:  # 只统计真正新增的格子
                    obs_map[x, y] = 1.0
                    new_cells += 1
            placed_cells += new_cells

        return obs_map

    def reset(self):
        self._current_step = 0

        # --- 1. 先生成障碍物 ---
        self.global_obstacle_map = self._generate_obstacles()

        # --- 2. 定义安全边距（必须在目标位置初始化之前）---
        safe_margin = self.detect_radius + 1.0
        min_pos = safe_margin
        max_pos = self.grid_size - safe_margin

        if max_pos <= min_pos:
            min_pos = self.grid_size / 2.0 - 1.0
            max_pos = self.grid_size / 2.0 + 1.0

        # --- 3. 初始化搜索者位置与速度 ---
        self.searcher_pos = np.zeros((self.n_searchers, 2))
        self.searcher_vel = np.zeros((self.n_searchers, 2))

        # --- 4. 初始化目标位置 + 重置存活状态（放在 safe_margin 定义之后）---
        self.target_pos = np.random.uniform(
            low=min_pos,  # ← 用 min_pos / max_pos，与搜索者保持一致
            high=max_pos,
            size=(self.n_targets, 2)
        ).astype(np.float32)
        self.target_alive = np.ones(self.n_targets, dtype=bool)

        # --- 5. 搜索者避障随机生成位置 ---
        for i in range(self.n_searchers):
            max_retries = 100  # 最大尝试次数，防止死循环
            for _ in range(max_retries):
                x = np.random.uniform(min_pos, max_pos)
                y = np.random.uniform(min_pos, max_pos)

                # 检查该位置周围是否有障碍物（保持至少 1.5 格的安全出生距离）
                safe_radius = 1.5
                x_min_check = max(0, int(x - safe_radius))
                x_max_check = min(self.grid_size, int(x + safe_radius) + 1)
                y_min_check = max(0, int(y - safe_radius))
                y_max_check = min(self.grid_size, int(y + safe_radius) + 1)

                # ---检查与其他已生成智能体的距离 ---
                conflict = False
                for j in range(i):  # 只和已经成功生成的智能体比较
                    dist = np.sqrt((x - self.searcher_pos[j, 0]) ** 2 + (y - self.searcher_pos[j, 1]) ** 2)
                    if dist < self.agent_collision_dist + 1.0:  # 保持比碰撞距离稍微大一点的安全距离
                        conflict = True
                        break
                # 如果这个矩形区域内没有障碍物 且 没有和其他智能体冲突
                if np.sum(self.global_obstacle_map[
                              x_min_check:x_max_check, y_min_check:y_max_check]) == 0 and not conflict:
                    self.searcher_pos[i, 0] = x
                    self.searcher_pos[i, 1] = y
                    break
            else:
                # 如果运气极差，100次都没找到空地（比如障碍物太密集），就强行生成一个位置
                self.searcher_pos[i, 0] = np.random.uniform(min_pos, max_pos)
                self.searcher_pos[i, 1] = np.random.uniform(min_pos, max_pos)

        # 3. 初始化/重置多通道地图
        self.agent_maps = np.zeros((self.n_searchers, self.map_channels, self.grid_size, self.grid_size))
        # 同步别名（reset 时需要重新绑定，因为 agent_maps 是新数组）
        self.agent_memory_maps = self.agent_maps[:, 0, :, :]

        # 4. 初始探测更新
        self._update_maps_and_memory()

        observation = self._get_observations()
        info = {}
        self._prev_coverage = self._global_coverage()  # reset 后重新记录基准
        return observation, info

    def step(self, action_dict):
        self._current_step += 1
        prev_pos = self.searcher_pos.copy()  # ← 在所有移动逻辑之前保存
        coverage_before = self._prev_coverage
        rewards = {agent: 0.0 for agent in self.agents}
        # 用于存储每个 agent 的奖励组成，方便调试
        reward_info = {agent: {'step': 0.0, 'explore': 0.0, 'target': 0.0, 'collision': 0.0} for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}  # 注意：XuanCe 通常需要 truncated

        # --- 1. 智能体移动与越界判定 (二阶粒子运动学) ---
        for i, agent_key in enumerate(self.agents):
            action = action_dict[agent_key]

            # 此时神经网络输出的 action 代表【力/加速度】
            ax, ay = action[0], action[1]
            force = np.array([ax, ay]) * self.accel_scale

            # 1. 更新速度: v_{t+1} = v_t + a * dt (假设 dt=1)
            self.searcher_vel[i] += force
            # 2. 施加物理阻尼/摩擦力: 模拟空气阻力或地面摩擦
            self.searcher_vel[i] *= (1.0 - self.damping)
            # 3. 限制最大速度 (防止速度无限叠加)
            speed = np.linalg.norm(self.searcher_vel[i])
            if speed > self.max_speed:
                self.searcher_vel[i] = (self.searcher_vel[i] / speed) * self.max_speed
            # 4. 更新位置: p_{t+1} = p_t + v_{t+1} * dt
            next_pos = self.searcher_pos[i] + self.searcher_vel[i]

            # 5. 边界处理
            clipped_pos = np.clip(next_pos, 0, self.grid_size - 0.01)
            if clipped_pos[0] != next_pos[0]:
                self.searcher_vel[i][0] = 0.0
            if clipped_pos[1] != next_pos[1]:
                self.searcher_vel[i][1] = 0.0

            # 6. 障碍物碰撞检测（先检查，再决定是否更新位置）
            nx = int(np.clip(clipped_pos[0], 0, self.grid_size - 1))
            ny = int(np.clip(clipped_pos[1], 0, self.grid_size - 1))

            if self.global_obstacle_map[nx, ny] > 0:
                # 撞到障碍物：速度清零，位置保持不动（不更新）
                self.searcher_vel[i] = np.zeros(2)
            else:
                # 正常移动
                self.searcher_pos[i] = clipped_pos

        # # 目标移动 (规则控制：随机游走)
        # for i in range(self.n_targets):
        #     if not self.target_alive[i]:  # ← 新增：跳过已被捕获的目标
        #         continue
        #     noise = np.random.randn(2) * 1.0
        #     self.target_pos[i] += noise
        #     self.target_pos[i] = np.clip(self.target_pos[i], 0, self.grid_size - 0.01)

        # --- 2. 环境状态更新 (探测与衰减) ---
        # 计算探测增量用于奖励
        # new_explored_counts = self._update_maps_and_memory()
        new_explored_rewards = self._update_maps_and_memory()

        # --- 3. 奖励计算 ---
        total_reward = 0.0
        for i, agent_key in enumerate(self.agents):
            if terminated[agent_key]:
                total_reward += rewards[agent_key]
                # 记录 info 方便调试
                reward_info[agent_key] = {
                    'total': rewards[agent_key], 'step': 0, 'explore': 0,
                    'target': 0, 'collision': 0, 'boundary': rewards[agent_key]
                }
                continue  # 跳过下面的计算
            # A. 基础步数惩罚 (Time Penalty)
            # 迫使智能体尽快找到目标，不要在原地打转
            r_step = -0.005

            # B. 探测新区域奖励
            r_explore = new_explored_rewards[i] * 0.05

            # C. 势能奖励 + 捕获奖励（仅对探测范围内的目标生效）
            r_target = 0.0
            agent_pos = self.searcher_pos[i]

            # 找出当前智能体能"看见"的存活目标（在探测范围内）
            visible_targets = []
            for j in range(self.n_targets):
                if not self.target_alive[j]:
                    continue
                dist = np.linalg.norm(agent_pos - self.target_pos[j])
                if dist <= self.detect_radius:
                    visible_targets.append(j)

            # 势能奖励：只对可见目标计算靠近/远离
            if visible_targets:
                dist_now = min(np.linalg.norm(agent_pos - self.target_pos[j]) for j in visible_targets)
                dist_prev = min(np.linalg.norm(prev_pos[i] - self.target_pos[j]) for j in visible_targets)
                r_target += (dist_prev - dist_now) * 0.01

            # 捕获奖励：只对可见目标判断是否进入捕获距离
            for j in visible_targets:
                if not self.target_alive[j]:  # 防止同一步被其他智能体先捕获
                    continue
                dist = np.linalg.norm(agent_pos - self.target_pos[j])
                if dist < self.collision_dist:
                    r_target += 20.0
                    self.target_alive[j] = False

            # D. 智能体间的碰撞惩罚 (Agent Collision Penalty)
            # 遍历其他所有搜索者
            r_collision = 0.0
            for j in range(self.n_searchers):
                if i != j:  # 不和自己计算距离
                    dist = np.sqrt((self.searcher_pos[i, 0] - self.searcher_pos[j, 0]) ** 2 + (
                                self.searcher_pos[i, 1] - self.searcher_pos[j, 1]) ** 2)
                    if dist < self.agent_collision_dist:
                        r_collision += self.agent_collision_penalty
                        if dist < self.agent_collision_dist / 2.0:
                            r_collision += self.agent_collision_penalty * 2.0


            # E. 边界持续惩罚 (Continuous Boundary Penalty)
            r_boundary = 0.0
            px, py = self.searcher_pos[i]

            # 计算到四个边界的距离
            dist_left = px
            dist_right = self.grid_size - px
            dist_top = py
            dist_bottom = self.grid_size - py

            # 遍历四个边界的距离，独立计算惩罚并叠加
            for dist in [dist_left, dist_right, dist_top, dist_bottom]:
                if dist < self.detect_radius:
                    # 平方递增
                    penalty_factor = (1.0 - (dist / self.detect_radius)) * 2
                    r_boundary += -0.5 * penalty_factor

            # --- F. 障碍物持续避障惩罚（基于感知记忆，非全知）---
            r_obstacle = 0.0
            px, py = self.searcher_pos[i]

            known_obstacle_map = self.agent_maps[i, 1, :, :]  # 只用已知障碍物

            dist_map = np.sqrt(
                (self.grid_x - px) ** 2 +
                (self.grid_y - py) ** 2
            )

            # ── 预警半径改为 detect_radius，障碍物一进入视野就开始软惩罚 ──
            warn_radius = self.detect_radius  # 5.0：远距离软警告
            danger_radius = 2.0  # 2.0：近距离强惩罚
            lethal_radius = 1.0  # 1.0：致命区

            obs_known = (known_obstacle_map > 0)
            obs_in_warn = (dist_map < warn_radius) & obs_known

            if np.any(obs_in_warn):
                min_dist = float(np.min(dist_map[obs_in_warn]))

                # ── 1. 远距离软警告（warn_radius ~ danger_radius）──
                # 线性惩罚，量级很小，只是给智能体一个"有障碍物"的梯度信号
                if min_dist < warn_radius:
                    soft_factor = (1.0 - min_dist / warn_radius)  # 0~1
                    r_obstacle -= 0.3 * soft_factor  # 最大 -0.3

                # ── 2. 近距离强惩罚（< danger_radius）──
                # 平方递增，让梯度在靠近时急剧变大
                if min_dist < danger_radius:
                    hard_factor = (1.0 - min_dist / danger_radius) ** 2
                    r_obstacle -= 3.0 * hard_factor  # 最大 -3.0

                # ── 3. 致命区重罚（< lethal_radius）──
                if min_dist < lethal_radius:
                    r_obstacle -= 5.0 * (1.0 - min_dist)  # 最大 -5.0

                # ── 4. 速度方向惩罚：朝已知障碍物冲时额外惩罚 ──
                vel = self.searcher_vel[i]
                speed = np.linalg.norm(vel)

                if speed > 0.1 and min_dist < danger_radius:
                    nearest_idx = np.unravel_index(
                        np.argmin(np.where(obs_in_warn, dist_map, np.inf)),
                        dist_map.shape
                    )
                    obs_dir = np.array([nearest_idx[0] + 0.5 - px, nearest_idx[1] + 0.5 - py])
                    obs_dist_norm = np.linalg.norm(obs_dir)

                    if obs_dist_norm > 0:
                        obs_dir_unit = obs_dir / obs_dist_norm
                        vel_unit = vel / speed
                        approach_rate = float(np.dot(vel_unit, obs_dir_unit))
                        if approach_rate > 0:
                            # 速度越快、越正对障碍物，惩罚越大
                            r_obstacle -= approach_rate * (speed / self.max_speed) * 2.0

            rewards[agent_key] = r_step + r_explore + r_target + r_collision + r_boundary + r_obstacle
            # total_reward += rewards[agent_key]

            # 存入 info 供测试打印
            reward_info[agent_key] = {
                'total': rewards[agent_key],
                'step': r_step,
                'explore': r_explore,
                'target': r_target,
                'collision': r_collision,
                'boundary': r_boundary
            }
        # --- G. 全局覆盖率增量奖励（循环外，只算一次）---
        coverage_after = self._global_coverage()
        coverage_delta = coverage_after - coverage_before
        r_coverage = coverage_delta * self.coverage_reward_scale if coverage_delta > 0 else 0.0
        self._prev_coverage = coverage_after

        total_reward = 0.0
        for ak in self.agents:  # 用新变量名 ak，不污染外层
            rewards[ak] += r_coverage
            reward_info[ak]['coverage'] = r_coverage
            reward_info[ak]['total'] = rewards[ak]  # 同步更新 total
            total_reward += rewards[ak]

        info = {"reward_details": reward_info, "total_step_reward": total_reward,
                "episode_score": rewards}

        # --- 4. 结束条件 ---
        truncated = False
        if self._current_step >= self.max_episode_steps:
            truncated = True

        # --- 覆盖率达标终止 ---
        if coverage_after >= self.coverage_threshold:
            for ak in self.agents:
                terminated[ak] = True
                rewards[ak] += self.coverage_done_reward

        # --- 目标全部捕获终止 ---
        # if not np.any(self.target_alive):
        #     time_bonus = (self.max_episode_steps - self._current_step) * 0.1
        #     for ak in self.agents:
        #         terminated[ak] = True
        #         rewards[ak] += time_bonus

        observation = self._get_observations()

        # episode 结束时打印覆盖率
        # if any(terminated.values()) or truncated:
        #     coverage = self._episode_coverage()
            # print(f"[Episode] 可通行区域覆盖率: {coverage:.2%}")

        return observation, rewards, terminated, truncated, info

    def _episode_coverage(self):
        """
        计算所有 agent 联合覆盖的可通行格子比例（排除障碍物）。
        ch0 是探索记忆通道。
        """
        # 合并所有 agent 的 ch0 探索记忆（任意一个探索过即算）
        combined = np.zeros((self.map_h, self.map_w), dtype=bool)
        for i in range(self.n_searchers):
            combined |= (self.agent_maps[i, 0, :, :] > 0)  # ch0 通道

        # 可通行格子总数（排除障碍物）
        passable_mask = self.global_obstacle_map == 0  # ← 字段名修正
        passable_total = np.sum(passable_mask)

        if passable_total == 0:
            return 0.0

        # 已探索 且 可通行 的格子
        explored_passable = np.sum(combined & passable_mask)

        return float(explored_passable) / float(passable_total)

    # def _update_maps_and_memory(self):
    #     """
    #     更新三通道地图：
    #       ch0: 探索记忆（空地，带衰减）
    #       ch1: 障碍物（永久标记）
    #       ch2: 目标位置（实时更新 + 带衰减记忆）
    #     返回每个智能体的信息增益，用于探索奖励。
    #     """
    #
    #     # ── Step A：衰减 ──────────────────────────────────────────────
    #     if self._current_step > 1:
    #         # ch0 探索记忆衰减（只衰减空地，障碍物格子不参与）
    #         ch0 = self.agent_maps[:, 0, :, :]
    #         explored_mask = ch0 > 0
    #         ch0[explored_mask] *= self.decay_rate
    #         ch0[explored_mask] = np.maximum(
    #             ch0[explored_mask],
    #             self.min_decay_val
    #         )
    #
    #         # # ch2 目标位置记忆衰减（目标移走后痕迹逐渐消失）
    #         # ch2 = self.agent_maps[:, 2, :, :]
    #         # target_mem_mask = ch2 > 0
    #         # ch2[target_mem_mask] *= self.decay_rate
    #         # ch2[target_mem_mask] = np.maximum(
    #         #     ch2[target_mem_mask],
    #         #     self.min_decay_val
    #         # )
    #
    #     new_explored_rewards = np.zeros(self.n_searchers)
    #
    #     # ── Step B：更新 ch0 探索记忆 和 ch1 障碍物 ──────────────────
    #     for i in range(self.n_searchers):
    #         pos = self.searcher_pos[i]
    #         r = int(self.detect_radius)
    #         cx, cy = int(pos[0]), int(pos[1])
    #
    #         x_min = max(0, cx - r)
    #         x_max = min(self.grid_size, cx + r + 1)
    #         y_min = max(0, cy - r)
    #         y_max = min(self.grid_size, cy + r + 1)
    #
    #         for x in range(x_min, x_max):
    #             for y in range(y_min, y_max):
    #                 dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
    #                 if dist <= self.detect_radius:
    #                     if self.global_obstacle_map[x, y] > 0:
    #                         # ch1：障碍物永久标记，ch0 清零（障碍物格子不算探索）
    #                         self.agent_maps[i, 1, x, y] = 1.0
    #                         self.agent_maps[i, 0, x, y] = 0.0
    #                     else:
    #                         # ch0：计算信息增益，刷新新鲜度
    #                         old_val = self.agent_maps[i, 0, x, y]
    #                         new_explored_rewards[i] += (1.0 - max(0.0, old_val))
    #                         self.agent_maps[i, 0, x, y] = 1.0
    #     # 每步清空 ch2，只标记当前存活的目标
    #     self.agent_maps[:, 2, :, :] = 0.0  # ← 新增：先清零
    #     # ── Step C：更新 ch2 目标位置（目标在探测范围内时，标记其四邻格为1）────
    #     for i in range(self.n_searchers):
    #         for j, t_pos in enumerate(self.target_pos):
    #             if not self.target_alive[j]:
    #                 continue
    #
    #             # 判断目标是否在该智能体的探测范围内
    #             dist_to_target = np.sqrt(
    #                 (t_pos[0] - self.searcher_pos[i, 0]) ** 2 +
    #                 (t_pos[1] - self.searcher_pos[i, 1]) ** 2
    #             )
    #             if dist_to_target > self.detect_radius:
    #                 continue  # 目标不在探测范围内，跳过
    #
    #             # 目标在探测范围内 → 标记目标周围四个相邻格子
    #             tx = int(np.clip(t_pos[0], 0, self.grid_size - 1))
    #             ty = int(np.clip(t_pos[1], 0, self.grid_size - 1))
    #
    #             for nr, nc in [(tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)]:
    #                 if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
    #                     self.agent_maps[i, 2, nr, nc] = 1.0
    #
    #     return new_explored_rewards
    def _update_maps_and_memory(self):
        """
        更新三通道地图（全向量化版本，无 Python 循环遍历格子）：
          ch0: 探索记忆（空地，带衰减）
          ch1: 障碍物（永久标记）
          ch2: 目标位置（实时更新）
        返回每个智能体的信息增益，用于探索奖励。
        """

        # ── Step A：衰减（每5步执行一次，等效连续衰减）──────────────────────
        if self._current_step > 1 and self._current_step % 5 == 0:
            ch0 = self.agent_maps[:, 0, :, :]  # shape: (n_searchers, H, W)
            explored_mask = ch0 > 0
            effective_rate = self.decay_rate ** 5  # 等效5步衰减
            ch0[explored_mask] *= effective_rate
            np.maximum(ch0, 0.0, out=ch0)  # 清除负值（数值安全）
            # 低于 min_decay_val 的已探索格子保持下限
            below_min = explored_mask & (ch0 < self.min_decay_val) & (ch0 > 0)
            ch0[below_min] = self.min_decay_val

        new_explored_rewards = np.zeros(self.n_searchers)

        # ── Step B：向量化更新 ch0 探索记忆 和 ch1 障碍物 ────────────────────
        # self.grid_x, self.grid_y 已在 __init__ 中预计算，shape: (H, W)
        for i in range(self.n_searchers):
            pos = self.searcher_pos[i]  # (2,)

            # 一次性计算该智能体到所有格子的距离，shape: (H, W)
            dist_map = np.sqrt(
                (self.grid_x - pos[0]) ** 2 +
                (self.grid_y - pos[1]) ** 2
            )

            in_range = dist_map <= self.detect_radius  # bool mask, shape: (H, W)

            # 分离障碍物格子 和 空地格子
            obstacle_mask = in_range & (self.global_obstacle_map > 0)
            free_mask = in_range & (self.global_obstacle_map == 0)

            # ch1：标记障碍物，同时清除 ch0 中障碍物格子的探索记忆
            self.agent_maps[i, 1][obstacle_mask] = 1.0
            self.agent_maps[i, 0][obstacle_mask] = 1.0

            # ch0：计算信息增益（新探索的格子贡献更多奖励）
            old_vals = self.agent_maps[i, 0][free_mask]  # 已有记忆值
            new_explored_rewards[i] = float(np.sum(1.0 - np.maximum(0.0, old_vals)))
            self.agent_maps[i, 0][free_mask] = 1.0  # 刷新为新鲜

        # ── Step C：更新 ch2 目标位置（先清零，再标记可见目标）────────────────
        self.agent_maps[:, 2, :, :] = 0.0

        for i in range(self.n_searchers):
            for j in range(self.n_targets):
                if not self.target_alive[j]:
                    continue

                t_pos = self.target_pos[j]
                dist_to_target = np.linalg.norm(self.searcher_pos[i] - t_pos)
                if dist_to_target > self.detect_radius:
                    continue

                # 目标在探测范围内 → 标记目标格子及四邻格
                tx = int(np.clip(t_pos[0], 0, self.grid_size - 1))
                ty = int(np.clip(t_pos[1], 0, self.grid_size - 1))

                for nr, nc in [(tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1), (tx, ty)]:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.agent_maps[i, 2, nr, nc] = 1.0

        return new_explored_rewards

    def _get_observations(self):
        obs_dict = {}
        # 地图对角线长度，用于归一化任意相对距离到 [-1, 1]
        diag = self.grid_size * np.sqrt(2)
        # --- 全局覆盖率 ---
        global_coverage = np.array([self._global_coverage()], dtype=np.float32)

        for i, agent_key in enumerate(self.agents):
            # --- 1. 自身位置归一化 [2] ---
            pos_norm = self.searcher_pos[i] / self.grid_size

            # --- 2. 自身速度归一化 [2] ---
            vel_norm = self.searcher_vel[i] / self.max_speed

            # --- 3. 队友相对位置（全局通信，始终可知，Padding 到固定长度）---
            other_agent_obs = []
            slot = 0
            for j in range(self.n_searchers):
                if i == j:
                    continue
                if slot >= self.max_other_agents:
                    break  # 超过上限截断（正常不会发生）

                dx = self.searcher_pos[j, 0] - self.searcher_pos[i, 0]
                dy = self.searcher_pos[j, 1] - self.searcher_pos[i, 1]

                # 归一化到 [-1, 1]，除以地图对角线保证任意位置都在范围内
                dx_norm = dx / diag
                dy_norm = dy / diag

                other_agent_obs.extend([dx_norm, dy_norm])
                slot += 1

            # 剩余槽位用 [0, 0] 填充（Padding）
            remaining = self.max_other_agents - slot
            other_agent_obs.extend([0.0, 0.0] * remaining)

            # --- 4. 多通道地图 flatten [map_channels * map_h * map_w] ---
            map_flat = self.agent_maps[i].flatten()  # shape: (3, 64, 64) → (12288,)



            # --- 6. 拼接所有观测 ---
            obs = np.concatenate([
                pos_norm,  # [2]
                vel_norm,  # [2]
                other_agent_obs,  # [max_other_agents * 2]，固定维度
                global_coverage,  # 1   ← 必须在 map_flat 之前
                map_flat  # map_channels * H * W
            ]).astype(np.float32)

            obs_dict[agent_key] = obs

        return obs_dict

    def render(self, mode="human"):
        """
        PyGame 渲染函数
        mode: "human" (弹出窗口显示) 或 "rgb_array" (返回图像数组用于保存视频)
        """
        # print("render called")  # 添加这一行

        # 1. 计算窗口尺寸
        width = self.grid_size * self.render_scale
        height = self.grid_size * self.render_scale

        # 2. 初始化 PyGame (懒加载，只有第一次调用render时才初始化)
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Multi-Agent Search Environment")
            else:
                # 如果是 rgb_array 模式，创建一个不可见的 surface
                self.screen = pygame.Surface((width, height))


            self.clock = pygame.time.Clock()

        # 3. 绘制背景 (白色)
        self.screen.fill((255, 255, 255))

        # --- 图层 A: 绘制记忆热力图 (柔和绿色) ---
        # 我们取所有智能体记忆的最大值，展示"全局已知信息"
        # global_memory shape: [64, 64], 值域 0.0 ~ 1.0
        global_memory = np.max(self.agent_memory_maps, axis=0)

        # 找到所有有记忆的地方 (>0)
        mem_indices = np.where(global_memory > 0.05)
        for x, y in zip(mem_indices[0], mem_indices[1]):
            val = global_memory[x, y]

            # --- 【修改点】颜色计算: 柔和过渡 ---
            # 新鲜 (val=1.0): 柔和的薄荷绿 (90, 190, 120)
            # 陈旧 (val=0.0): 极淡的灰绿色 (245, 250, 245)，自然融入白色背景
            r = int(245 - 155 * val)  # 245 -> 90
            g = int(250 - 60 * val)  # 250 -> 190
            b = int(245 - 125 * val)  # 245 -> 120

            # 确保 RGB 值在合法范围内 (0-255)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            color = (r, g, b)

            rect = pygame.Rect(
                x * self.render_scale,
                y * self.render_scale,
                self.render_scale,
                self.render_scale
            )
            pygame.draw.rect(self.screen, color, rect)

        # --- 图层 B: 绘制障碍物 (黑色) ---
        obs_indices = np.where(self.global_obstacle_map == 1)
        for x, y in zip(obs_indices[0], obs_indices[1]):
            rect = pygame.Rect(
                x * self.render_scale,
                y * self.render_scale,
                self.render_scale,
                self.render_scale
            )
            pygame.draw.rect(self.screen, (40, 40, 40), rect)

        # --- 图层 C: 绘制探测范围 (半透明圆) ---
        # PyGame 原生不支持直接画带 Alpha 的圆，需要画在临时 Surface 上再 blit
        surf_alpha = pygame.Surface((width, height), pygame.SRCALPHA)

        for pos in self.searcher_pos:
            center = (int(pos[0] * self.render_scale), int(pos[1] * self.render_scale))
            radius = int(self.detect_radius * self.render_scale)
            # 蓝色，透明度 50/255
            pygame.draw.circle(surf_alpha, (0, 0, 255, 30), center, radius)

        self.screen.blit(surf_alpha, (0, 0))

        # --- 图层 D: 绘制实体 ---

        # 1. 目标 (红色圆点)
        for j, pos in enumerate(self.target_pos):
            if not self.target_alive[j]:  # ← 新增：跳过已死亡目标
                continue
            center = (int(pos[0] * self.render_scale), int(pos[1] * self.render_scale))
            radius = int(self.render_scale * 0.8)
            pygame.draw.circle(self.screen, (255, 50, 50), center, radius)

        # 2. 搜索者 (蓝色实心圆 + 编号)
        for i, pos in enumerate(self.searcher_pos):
            center = (int(pos[0] * self.render_scale), int(pos[1] * self.render_scale))
            radius = int(self.render_scale * 0.8)

            # 实心圆
            pygame.draw.circle(self.screen, (50, 50, 255), center, radius)
            # 白色边框，方便在深色背景下看清
            pygame.draw.circle(self.screen, (255, 255, 255), center, radius, 1)

            # 编号文字
            if not hasattr(self, '_font') or self._font is None:
                pygame.font.init()
                self._font = pygame.font.SysFont("Arial", max(8, self.render_scale - 2), bold=True)

            label = self._font.render(str(i), True, (255, 255, 255))
            # 文字居中偏移
            label_rect = label.get_rect(center=center)
            self.screen.blit(label, label_rect)

            # 边框 (区分不同智能体)
            # pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 1)

        # ─── 调试：绘制每个智能体的三通道观测地图 ───────────────────────────────
        if self.debug_obs_vis and (self._current_step % self.debug_obs_interval == 0):
            self._render_obs_channels()

        # --- 结束绘制 ---

        if mode == "human":
            pygame.event.pump()  # 处理窗口事件，防止卡死
            pygame.display.flip()
            self.clock.tick(self.render_fps)  # 控制帧率
            return None

        elif mode == "rgb_array":
            # 将 PyGame Surface 转为 Numpy Array (H, W, 3)
            # PyGame 是 (W, H, 3)，需要转置
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _render_obs_channels(self):
        """
        将每个搜索者的三通道观测地图画在一张 matplotlib 图上。
        布局：行 = 智能体，列 = 通道（ch0探索记忆 | ch1障碍物 | ch2目标位置）
        每个子图叠加：自身位置（蓝星）、队友位置（绿圆）、存活目标（红三角）
        """
        n_agents = self.n_searchers
        n_channels = self.map_channels  # 3

        col_titles = ["ch0: Exploration Memory", "ch1: Obstacles", "ch2: Target Positions"]
        channel_cmaps = ['Greens', 'Greys', 'Reds']
        channel_vmax = [1.0, 1.0, 1.0]

        # ── 第一次调用：创建 figure ──────────────────────────────────────────
        if self._fig is None:
            fig_w = n_channels * 3.8
            fig_h = n_agents * 3.8 + 0.6  # 额外留给 suptitle
            self._fig, self._axes = plt.subplots(
                n_agents, n_channels,
                figsize=(fig_w, fig_h),
                squeeze=False,
                gridspec_kw={'hspace': 0.35, 'wspace': 0.15}
            )
            plt.ion()

        # ── 每帧刷新 ─────────────────────────────────────────────────────────
        for i in range(n_agents):
            for c in range(n_channels):
                ax = self._axes[i][c]
                ax.cla()

                # 地图数据（转置使 x=列, y=行，与 pygame 坐标系一致）
                channel_map = self.agent_maps[i, c, :, :].T  # (W, H) → imshow (H, W)

                ax.imshow(
                    channel_map,
                    cmap=channel_cmaps[c],
                    vmin=0.0,
                    vmax=channel_vmax[c],
                    origin='upper',
                    interpolation='nearest',
                    aspect='equal'
                )

                # ── 叠加：队友位置（绿色圆圈，空心）──
                for j in range(n_agents):
                    if j == i:
                        continue
                    px, py = self.searcher_pos[j]
                    ax.plot(px, py,
                            marker='o', markersize=9,
                            markerfacecolor='none',
                            markeredgecolor='limegreen',
                            markeredgewidth=1.8,
                            zorder=4)
                    ax.text(px + 1.0, py - 1.0, str(j),
                            color='limegreen', fontsize=7,
                            fontweight='bold', zorder=5)

                # ── 叠加：自身位置（蓝色星形 + 编号）──
                sx, sy = self.searcher_pos[i]
                ax.plot(sx, sy,
                        marker='*', markersize=13,
                        color='royalblue',
                        markeredgecolor='white',
                        markeredgewidth=0.8,
                        zorder=5)
                ax.text(sx + 1.0, sy - 1.5, f'S{i}',
                        color='royalblue', fontsize=7,
                        fontweight='bold', zorder=6)

                # ── 叠加：存活目标（红色三角 + 编号）──
                for j, t_pos in enumerate(self.target_pos):
                    if self.target_alive[j]:
                        tx, ty = t_pos
                        ax.plot(tx, ty,
                                marker='^', markersize=8,
                                color='tomato',
                                markeredgecolor='darkred',
                                markeredgewidth=0.8,
                                zorder=4)
                        ax.text(tx + 1.0, ty - 1.0, f'T{j}',
                                color='darkred', fontsize=7,
                                fontweight='bold', zorder=5)

                # ── 标题和轴标签 ──
                if i == 0:
                    ax.set_title(col_titles[c], fontsize=9, fontweight='bold', pad=4)

                if c == 0:
                    ax.set_ylabel(f'Searcher {i}', fontsize=9,
                                  fontweight='bold', labelpad=4)

                # 轴范围与刻度
                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(self.grid_size, 0)  # 翻转 y 轴，与 pygame 一致
                ax.set_xticks([])
                ax.set_yticks([])

                # 给每个子图加细边框，视觉上区分智能体行
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor('#aaaaaa')

        # ── 全局标题（含步数和覆盖率）────────────────────────────────────────
        coverage = self._global_coverage()
        self._fig.suptitle(
            f"Agent Observation Channels  |  Step: {self._current_step}"
            f"  |  Coverage: {coverage:.1%}",
            fontsize=11, fontweight='bold', y=0.995
        )

        # ── 图例说明（画在 figure 底部）─────────────────────────────────────
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='royalblue', markersize=10, label='Self (★)'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='none', markeredgecolor='limegreen',
                       markersize=8, label='Teammate (○)'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor='tomato', markersize=8, label='Target (▲)'),
        ]
        self._fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=8,
            framealpha=0.7,
            bbox_to_anchor=(0.5, 0.0)
        )

        self._fig.tight_layout(rect=[0, 0.04, 1, 0.99])
        plt.pause(0.001)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return

