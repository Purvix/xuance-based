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
        self.render_scale = 10  # 缩放比例: 1个网格 = 10个像素 (64x64 -> 640x640窗口)
        self.render_fps = 10    # 渲染帧率

        # 探测与衰减参数
        self.detect_radius = getattr(env_config, 'detect_radius', 5.0)  # 探测半径
        self.decay_rate = getattr(env_config, 'decay_rate', 0.98)  # 记忆衰减率
        self.min_decay_val = getattr(env_config, 'min_decay_val', 0.1)  # 衰减下限
        self.collision_dist = 3.0  # 判定抓捕/碰撞距离

        #奖励相关
        self.step_penalty = getattr(env_config, 'step_penalty', -0.01)  # 每步扣 0.01 分
        self.agent_collision_dist = getattr(env_config, 'agent_collision_dist', 2.0)  # 碰撞判定距离
        self.agent_collision_penalty = getattr(env_config, 'agent_collision_penalty', -0.2)  # 碰撞扣 0.2 分
        self.coverage_reward_scale = getattr(env_config, 'coverage_reward_scale', 10.0)
        self.coverage_threshold = getattr(env_config, 'coverage_threshold', 0.85)
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

        # 1. 状态空间维度: (n_searchers * 2) + 1
        state_dim = (self.n_searchers * 4) + 1
        self.state_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)

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

        # 3. 计算全图探索率 (0.0 到 1.0)
        total_cells = self.grid_size * self.grid_size
        global_explored_map = np.max(self.agent_memory_maps, axis=0)
        explored_cells = np.sum(global_explored_map > 0)
        exploration_rate = explored_cells / total_cells

        # 4. 拼接成一维向量
        state_vector = np.concatenate([
            pos_flat,
            vel_flat,
            [exploration_rate]
        ]).astype(np.float32)

        return state_vector

    def _global_coverage(self):
        """返回 0~1 之间的全局覆盖率（所有智能体联合探索的格子比例）"""
        global_map = np.max(self.agent_memory_maps, axis=0)  # (H, W)
        explored = np.sum(global_map > 0)
        total = self.grid_size * self.grid_size         #没有考虑障碍物
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

            # 检查：不能封死地图（简单检查：障碍物不能紧贴边界 margin 区域）
            valid = all(
                margin <= x < G - margin and margin <= y < G - margin
                for x, y in cells
            )
            if not valid or len(cells) == 0:
                continue

            # 写入地图
            for x, y in cells:
                obs_map[x, y] = 1.0
            placed_cells += len(cells)

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
        coverage_before = self._global_coverage()
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
            # 5. 边界处理 (Clip 限制)
            clipped_pos = np.clip(next_pos, 0, self.grid_size - 0.01)
            # 如果撞到墙（被 clip 了），垂直于墙面方向的速度清零，
            if clipped_pos[0] != next_pos[0]:
                self.searcher_vel[i][0] = 0.0  # X轴撞墙，X轴速度清零
            if clipped_pos[1] != next_pos[1]:
                self.searcher_vel[i][1] = 0.0  # Y轴撞墙，Y轴速度清零

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

            # C. 势能奖励 + 捕获奖励
            r_target = 0.0
            agent_pos = self.searcher_pos[i]

            def min_dist_to_alive_targets(pos):
                min_d = float('inf')
                for j in range(self.n_targets):
                    if not self.target_alive[j]:
                        continue
                    d = np.linalg.norm(pos - self.target_pos[j])
                    if d < min_d:
                        min_d = d
                # ← 关键修复：inf 说明没有存活目标，返回 0 避免 inf 参与运算
                return min_d if min_d != float('inf') else 0.0

            # 先算好两个距离（此时 target_alive 还未被本步修改）
            dist_now = min_dist_to_alive_targets(agent_pos)
            dist_prev = min_dist_to_alive_targets(prev_pos[i])

            # 势能奖励
            if np.any(self.target_alive):
                r_potential = (dist_prev - dist_now) * 0.01
                r_target += r_potential

            # 捕获奖励（在势能计算之后再修改 target_alive）
            for j in range(self.n_targets):
                if not self.target_alive[j]:
                    continue
                dist = np.linalg.norm(agent_pos - self.target_pos[j])
                if dist < self.collision_dist:
                    r_target += 20.0              # 捕获一次性奖励
                    self.target_alive[j] = False  # 立刻标记死亡，下一步不再计算

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

            # --- 【修改】F. 障碍物持续避障惩罚 (Continuous Obstacle Avoidance Penalty) ---
            r_obstacle = 0.0
            px, py = self.searcher_pos[i]

            # 1. 在智能体的探测范围内寻找最近的障碍物
            r = int(self.detect_radius)
            cx, cy = int(px), int(py)

            x_min, x_max = max(0, cx - r), min(self.grid_size, cx + r + 1)
            y_min, y_max = max(0, cy - r), min(self.grid_size, cy + r + 1)

            min_dist_to_obs = float('inf')

            # 遍历周围的格子，找到距离最近的障碍物
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    if self.global_obstacle_map[x, y] > 0:
                        # 计算智能体到障碍物格子中心点的距离 (加0.5是为了算格子中心)
                        dist = np.sqrt((x + 0.5 - px) ** 2 + (y + 0.5 - py) ** 2)
                        if dist < min_dist_to_obs:
                            min_dist_to_obs = dist

            # 2. 根据最近障碍物的距离计算惩罚
            if min_dist_to_obs < self.detect_radius:
                # (1) 靠近惩罚：距离越近，惩罚变大得越快（平方递增）
                # 这样智能体在边缘时只会受到微小警告，但越靠近扣分越狠，逼迫它转向
                penalty_factor = (1.0 - (min_dist_to_obs / self.detect_radius)) ** 2
                r_obstacle -= 0.5 * penalty_factor

                # (2) 致命碰撞惩罚：如果距离小于 1.0（认为已经贴上或踩进去了）
                if min_dist_to_obs < 1.0:
                    r_obstacle -= 2.0  # 给予重罚！

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
        if not np.any(self.target_alive):
            time_bonus = (self.max_episode_steps - self._current_step) * 0.1
            for ak in self.agents:
                terminated[ak] = True
                rewards[ak] += time_bonus

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

    def _update_maps_and_memory(self):
        """
        更新三通道地图：
          ch0: 探索记忆（空地，带衰减）
          ch1: 障碍物（永久标记）
          ch2: 目标位置（实时更新 + 带衰减记忆）
        返回每个智能体的信息增益，用于探索奖励。
        """

        # ── Step A：衰减 ──────────────────────────────────────────────
        if self._current_step > 1:
            # ch0 探索记忆衰减（只衰减空地，障碍物格子不参与）
            ch0 = self.agent_maps[:, 0, :, :]
            explored_mask = ch0 > 0
            ch0[explored_mask] *= self.decay_rate
            ch0[explored_mask] = np.maximum(
                ch0[explored_mask],
                self.min_decay_val
            )

            # # ch2 目标位置记忆衰减（目标移走后痕迹逐渐消失）
            # ch2 = self.agent_maps[:, 2, :, :]
            # target_mem_mask = ch2 > 0
            # ch2[target_mem_mask] *= self.decay_rate
            # ch2[target_mem_mask] = np.maximum(
            #     ch2[target_mem_mask],
            #     self.min_decay_val
            # )

        new_explored_rewards = np.zeros(self.n_searchers)

        # ── Step B：更新 ch0 探索记忆 和 ch1 障碍物 ──────────────────
        for i in range(self.n_searchers):
            pos = self.searcher_pos[i]
            r = int(self.detect_radius)
            cx, cy = int(pos[0]), int(pos[1])

            x_min = max(0, cx - r)
            x_max = min(self.grid_size, cx + r + 1)
            y_min = max(0, cy - r)
            y_max = min(self.grid_size, cy + r + 1)

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                    if dist <= self.detect_radius:
                        if self.global_obstacle_map[x, y] > 0:
                            # ch1：障碍物永久标记，ch0 清零（障碍物格子不算探索）
                            self.agent_maps[i, 1, x, y] = 1.0
                            self.agent_maps[i, 0, x, y] = 0.0
                        else:
                            # ch0：计算信息增益，刷新新鲜度
                            old_val = self.agent_maps[i, 0, x, y]
                            new_explored_rewards[i] += (1.0 - max(0.0, old_val))
                            self.agent_maps[i, 0, x, y] = 1.0
        # 每步清空 ch2，只标记当前存活的目标
        self.agent_maps[:, 2, :, :] = 0.0  # ← 新增：先清零
        # ── Step C：更新 ch2 目标位置 ────
        for i in range(self.n_searchers):

            # 注意：不清零会保留衰减记忆，这里选择保留衰减痕迹，只不再刷新
            for j, t_pos in enumerate(self.target_pos):
                if not self.target_alive[j]:  # ← 新增：跳过已死亡目标
                    continue
                tx = int(np.clip(t_pos[0], 0, self.grid_size - 1))
                ty = int(np.clip(t_pos[1], 0, self.grid_size - 1))
                self.agent_maps[i, 2, tx, ty] = 1.0

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
            radius = int(self.render_scale * 0.4)
            pygame.draw.circle(self.screen, (255, 50, 50), center, radius)

        # 2. 搜索者 (蓝色实心圆 + 黑色边框)
        for i, pos in enumerate(self.searcher_pos):
            center = (int(pos[0] * self.render_scale), int(pos[1] * self.render_scale))
            radius = int(self.render_scale * 0.4)
            # 实心
            pygame.draw.circle(self.screen, (50, 50, 255), center, radius)
            # 边框 (区分不同智能体)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 1)

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
        """
        n_agents = self.n_searchers
        n_channels = self.map_channels  # 3

        # 第一次调用时创建 figure
        if self._fig is None:
            self._fig, self._axes = plt.subplots(
                n_agents, n_channels,
                figsize=(n_channels * 3.5, n_agents * 3.5),
                squeeze=False
            )
            plt.ion()  # 开启交互模式，允许动态刷新
            self._fig.suptitle("Agent Observation Channels", fontsize=13, fontweight='bold')

            # 预先设置每列的标题（只设一次）
            col_titles = ["ch0: Exploration Memory", "ch1: Obstacles", "ch2: Target Memory"]
            for col, title in enumerate(col_titles):
                self._axes[0][col].set_title(title, fontsize=10)

        channel_cmaps = ['Greens', 'Greys', 'Reds']  # 三个通道各用不同色系

        for i in range(n_agents):
            for c in range(n_channels):
                ax = self._axes[i][c]
                ax.cla()  # 清除上一帧

                # 取出该通道的地图，shape: (grid_size, grid_size)
                # 注意：地图存储是 [x, y]，matplotlib imshow 默认 [row=y, col=x]，需要转置
                channel_map = self.agent_maps[i, c, :, :].T  # (H, W)

                im = ax.imshow(
                    channel_map,
                    cmap=channel_cmaps[c],
                    vmin=0.0,
                    vmax=1.0,
                    origin='upper',
                    interpolation='nearest'
                )

                # 在地图上叠加智能体自身位置（蓝色星形）
                px, py = self.searcher_pos[i]
                ax.plot(px, py, marker='*', color='blue', markersize=8, label=f'searcher_{i}')

                # 叠加存活目标位置（红色三角）
                for j, t_pos in enumerate(self.target_pos):
                    if self.target_alive[j]:
                        ax.plot(t_pos[0], t_pos[1], marker='^', color='red', markersize=6)

                # 行标签（只在第一列显示）
                if c == 0:
                    ax.set_ylabel(f'searcher_{i}', fontsize=9, fontweight='bold')

                ax.set_xlim(0, self.grid_size)
                ax.set_ylim(self.grid_size, 0)  # y轴翻转，和pygame坐标系一致
                ax.set_xticks([])
                ax.set_yticks([])

        # 在图的右下角显示当前步数
        self._fig.text(
            0.99, 0.01,
            f"Step: {self._current_step}",
            ha='right', va='bottom', fontsize=9, color='gray'
        )

        self._fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.pause(0.001)  # 非阻塞刷新

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return

