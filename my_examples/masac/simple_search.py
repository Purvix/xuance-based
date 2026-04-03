from types import SimpleNamespace
import pygame
import numpy as np
from gymnasium import spaces
from xuance.environment import RawMultiAgentEnv

class SearchEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(SearchEnv, self).__init__()

        # --- 配置参数 ---
        self.env_id = getattr(env_config, 'env_id', 'Search_v1')
        self.grid_size = getattr(env_config, 'grid_size', 64)  # 64x64 网格
        self.n_searchers = getattr(env_config, 'num_searchers', 3)  # RL控制的智能体
        self.n_targets = getattr(env_config, 'num_targets', 2)  # 规则控制的智能体
        self.n_obstacles = getattr(env_config, 'num_obstacles', 5)
        self.max_episode_steps = getattr(env_config, 'max_episode_steps', 200)

        # 渲染相关变量
        self.screen = None
        self.clock = None
        self.render_scale = 10  # 缩放比例: 1个网格 = 10个像素 (64x64 -> 640x640窗口)
        self.render_fps = 10    # 渲染帧率

        # 探测与衰减参数
        self.detect_radius = getattr(env_config, 'detect_radius', 5.0)  # 探测半径
        self.decay_rate = getattr(env_config, 'decay_rate', 0.98)  # 记忆衰减率
        self.min_decay_val = getattr(env_config, 'min_decay_val', 0.1)  # 衰减下限
        self.collision_dist = 1.0  # 判定抓捕/碰撞距离

        #奖励相关
        self.step_penalty = getattr(env_config, 'step_penalty', -0.01)  # 每步扣 0.01 分
        self.agent_collision_dist = getattr(env_config, 'agent_collision_dist', 2.0)  # 碰撞判定距离
        self.agent_collision_penalty = getattr(env_config, 'agent_collision_penalty', -0.2)  # 碰撞扣 0.2 分

        # 智能体相关
        self.searcher_ids = [f"searcher_{i}" for i in range(self.n_searchers)]
        self.target_ids = [f"target_{i}" for i in range(self.n_targets)]
        self.agents = self.searcher_ids
        self.num_agents = len(self.agents)

        self.max_speed = getattr(env_config, 'max_speed', 2.0)  # 最大速度
        self.accel_scale = getattr(env_config, 'accel_scale', 1.0)  # 加速度缩放因子（动作转化为力的倍数）
        self.damping = getattr(env_config, 'damping', 0.25)  # 阻尼/摩擦力系数 (0~1)，越大刹车越快

        # --- 空间定义 ---
        # 动作空间: 连续控制 [vx, vy], 范围 [-1, 1]
        self.action_space = {agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                             for agent in self.agents}
        # --- 观测空间维度计算 ---
        # 1. 自身位置归一化: 2 维 (x, y)
        # 2. 自身速度归一化: 2 维 (vx, vy)
        base_obs_dim = 4
        # 3. 其他智能体相对信息: (n_searchers - 1) 个智能体，每个占 3 维 (dx, dy, is_visible)
        self.other_agent_dim = (self.n_searchers - 1) * 3
        # 总维度
        obs_dim = base_obs_dim + self.other_agent_dim
        self.observation_space = {agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        # # 计算状态维度
        # state_dim = self.grid_size * self.grid_size + 2 * self.n_searchers + 2 * self.n_targets  # 4096 + 6 + 4 = 4106
        # self.state_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        # 1. 状态空间维度: (n_searchers * 2) + 1
        state_dim = (self.n_searchers * 4) + 1
        self.state_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)



        # --- 内部状态初始化 ---
        self._current_step = 0
        self.searcher_pos = np.zeros((self.n_searchers, 2))
        self.target_pos = np.zeros((self.n_targets, 2))
        self.searcher_vel = np.zeros((self.n_searchers, 2))

        # 地图状态
        # global_grid_map: 真实环境地图 (0:空地, 1:障碍物)
        self.global_obstacle_map = np.zeros((self.grid_size, self.grid_size))

        # individual_maps: 每个智能体的记忆 [N, W, H]
        self.agent_memory_maps = np.zeros((self.n_searchers, self.grid_size, self.grid_size))

        # 预计算网格坐标，用于加速距离计算
        x = np.linspace(0, self.grid_size - 1, self.grid_size)
        y = np.linspace(0, self.grid_size - 1, self.grid_size)
        self.grid_x, self.grid_y = np.meshgrid(x, y, indexing='ij')  # [64, 64]

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

    def reset(self):
        self._current_step = 0

        # --- 1. 先生成障碍物 ---
        # (必须在生成智能体位置之前生成，否则无法判断避障)
        self.global_obstacle_map = np.zeros((self.grid_size, self.grid_size))
        # 示例：随机放置5个障碍块
        for _ in range(self.n_obstacles):
            ox, oy = np.random.randint(0, self.grid_size, 2)
            self.global_obstacle_map[ox:min(ox + 5, 64), oy:min(oy + 5, 64)] = 1.0

        # --- 2. 随机初始化位置与速度 (避开边界和障碍物) ---
        self.searcher_pos = np.zeros((self.n_searchers, 2))
        self.searcher_vel = np.zeros((self.n_searchers, 2))

        # 定义安全边距，确保不靠近边界
        safe_margin = self.detect_radius + 1.0
        min_pos = safe_margin
        max_pos = self.grid_size - safe_margin

        if max_pos <= min_pos:
            min_pos = self.grid_size / 2.0 - 1.0
            max_pos = self.grid_size / 2.0 + 1.0

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

        # 3. 初始化/重置地图记忆
        # 0表示未探测，探测后置为1，随时间衰减
        self.agent_memory_maps = np.zeros((self.n_searchers, self.grid_size, self.grid_size))

        # 4. 初始探测更新
        self._update_maps_and_memory()

        observation = self._get_observations()
        info = {}
        return observation, info

    def step(self, action_dict):
        self._current_step += 1
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

        # 目标移动 (规则控制：随机游走)
        # 后期可改为势场法：避开搜索者
        for i in range(self.n_targets):
            noise = np.random.randn(2) * 1.0  # 随机移动
            self.target_pos[i] += noise
            self.target_pos[i] = np.clip(self.target_pos[i], 0, self.grid_size - 0.01)

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
            r_step = -0.01

            # B. 探测新区域奖励
            r_explore = new_explored_rewards[i] * 0.05

            # C. 搜索与捕获目标奖励
            r_target = 0.0
            agent_pos = self.searcher_pos[i]
            for j, t_pos in enumerate(self.target_pos):
                dist = np.linalg.norm(agent_pos - t_pos)
                if dist < self.detect_radius:
                    r_target += 0.5
                    if dist < self.collision_dist:
                        r_target += 5.0

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
            total_reward += rewards[agent_key]

            # 存入 info 供测试打印
            reward_info[agent_key] = {
                'total': rewards[agent_key],
                'step': r_step,
                'explore': r_explore,
                'target': r_target,
                'collision': r_collision,
                'boundary': r_boundary
            }

        info = {"reward_details": reward_info, "total_step_reward": total_reward}

        # --- 4. 结束条件 ---
        # 两种处理方式：
        # 1. 只要有一个撞墙，全场结束（简单，适合初学者）
        # 2. 撞墙的智能体停在那，其他继续（复杂，需要处理 mask）

        # 采用方式 1：一人失误，全队重来
        if any(terminated.values()):
            for agent in self.agents:
                terminated[agent] = True
        truncated = False
        if self._current_step >= self.max_episode_steps:
            truncated = True

        observation = self._get_observations()

        return observation, rewards, terminated, truncated, info

    def _update_maps_and_memory(self):
        """
        更新每个智能体的记忆地图，处理衰减，并返回“信息增益”作为奖励依据
        """
        # info_gains = np.zeros(self.n_searchers)
        # 1. 全局衰减：所有智能体的记忆地图随时间变淡
        # 只有大于 min_decay_val 的地方才衰减
        # 找到所有已经被探索过的格子 (值 > 0)
        explored_mask = self.agent_memory_maps > 0
        non_obstacle_mask = self.global_obstacle_map == 0
        decay_mask = explored_mask & non_obstacle_mask

        # 防止第一步“白嫖”初始位置的衰减恢复奖励
        if self._current_step > 1:
            # 1. 仅对符合条件的区域进行衰减
            self.agent_memory_maps[decay_mask] *= self.decay_rate

            # 2. 仅对符合条件的区域限制下限
            self.agent_memory_maps[decay_mask] = np.maximum(
                self.agent_memory_maps[decay_mask],
                self.min_decay_val
            )

        new_explored_rewards = np.zeros(self.n_searchers)

        for i in range(self.n_searchers):
            pos = self.searcher_pos[i]

            # 2. 计算当前探测范围内的网格坐标
            # 优化：只计算以智能体为中心的一个正方形区域，减少计算量
            r = int(self.detect_radius)
            cx, cy = int(pos[0]), int(pos[1])

            x_min, x_max = max(0, cx - r), min(self.grid_size, cx + r + 1)
            y_min, y_max = max(0, cy - r), min(self.grid_size, cy + r + 1)

            # 遍历探测半径内的格子
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                    if dist <= self.detect_radius:
                        old_val = self.agent_memory_maps[i, x, y]

                        # 判断该位置是否为障碍物
                        if self.global_obstacle_map[x, y] > 0:
                            # 【修改点】发现障碍物，将其新鲜度永久赋值为 1.0，且【不给探索奖励】
                            self.agent_memory_maps[i, x, y] = 1.0
                        else:
                            # 正常空地逻辑
                            increment = 1.0 - max(0, old_val)
                            new_explored_rewards[i] += increment

                            # 更新地图：踩过的空地格子新鲜度恢复到 1.0
                            self.agent_memory_maps[i, x, y] = 1.0

        return new_explored_rewards

    def _get_observations(self):
        obs_dict = {}
        for i, agent_key in enumerate(self.agents):
            # --- 1. 自身位置归一化 (2维) ---
            my_pos = self.searcher_pos[i]
            # 归一化到 [0, 1]
            pos_norm = my_pos / self.grid_size

            # --- 2. 自身速度归一化 (2维) ---
            my_vel = self.searcher_vel[i]
            # 归一化到 [-1, 1] 左右
            vel_norm = my_vel / self.max_speed

            # --- 3. 其他智能体的相对位置与可见性 (3维/个) ---
            other_agent_obs = []
            for j in range(self.n_searchers):
                if i == j:
                    continue  # 跳过自己

                # 计算相对位置: 目标位置 - 自身位置
                dx = self.searcher_pos[j, 0] - my_pos[0]
                dy = self.searcher_pos[j, 1] - my_pos[1]
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist <= self.detect_radius:
                    # 【可见】：在探测范围内
                    # 除以探测半径，将相对坐标严格归一化到 [-1, 1] 之间
                    dx_norm = dx / self.detect_radius
                    dy_norm = dy / self.detect_radius
                    is_visible = 1.0  # 1.0 表示可见
                else:
                    # 【不可见】：在探测范围外
                    # 坐标给 0，标志位给 0
                    dx_norm = 2.0
                    dy_norm = 2.0
                    is_visible = 0.0  # 0.0 表示不可见

                # 将这 3 个值加入列表
                other_agent_obs.extend([dx_norm, dy_norm, is_visible])

            # --- 4. 拼接所有观测 ---
            obs = np.concatenate([
                pos_norm,
                vel_norm,
                other_agent_obs
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
        for pos in self.target_pos:
            center = (int(pos[0] * self.render_scale), int(pos[1] * self.render_scale))
            # 半径设为缩放的一半，稍微小一点
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

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        return

