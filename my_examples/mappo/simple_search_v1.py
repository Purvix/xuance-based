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
        self.n_obstacles = getattr(env_config, 'num_obstacles', 5)  #障碍物
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

        # 智能体列表
        self.searcher_ids = [f"searcher_{i}" for i in range(self.n_searchers)]
        self.target_ids = [f"target_{i}" for i in range(self.n_targets)]
        self.agents = self.searcher_ids  # XuanCe只关注RL控制的智能体ID
        self.num_agents = len(self.agents)

        # --- 空间定义 ---
        # 动作空间: 连续控制 [vx, vy], 范围 [-1, 1]

        # 离散控制: 5个动作 (0:不动, 1:上, 2:下, 3:左, 4:右)
        self.action_space = {agent: spaces.Discrete(5) for agent in self.agents}

        self.other_agent_dim = (self.n_searchers - 1) * 3
        obs_dim = 2 + 4

        # 更新观测空间定义
        self.observation_space = {agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                                  for agent in self.agents}
        # # 计算状态维度
        # state_dim = self.grid_size * self.grid_size + 2 * self.n_searchers + 2 * self.n_targets  # 4096 + 6 + 4 = 4106
        # self.state_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        # 状态空间: (n_searchers * 2) + 1
        state_dim = (self.n_searchers * 6) + 1
        self.state_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)

        # --- 内部状态初始化 ---
        self._current_step = 0
        self.searcher_pos = np.zeros((self.n_searchers, 2))
        self.target_pos = np.zeros((self.n_targets, 2))

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
        # 1. 获取当前所有智能体的观测
        obs_dict = self._get_observations()

        # 按照智能体顺序提取观测向量，放在一个列表中
        all_obs = []
        for agent_key in self.agents:
            all_obs.append(obs_dict[agent_key])

        # 将所有观测拼接成一个长的一维数组
        all_obs_flat = np.concatenate(all_obs)

        # 2. 计算全图探索率 (1 维)
        total_cells = self.grid_size * self.grid_size
        global_explored_map = np.max(self.agent_memory_maps, axis=0)
        explored_cells = np.sum(global_explored_map > 0)
        exploration_rate = explored_cells / total_cells

        # 3. 拼接观测数组与探索率并返回
        state_vector = np.concatenate([
            all_obs_flat,
            [exploration_rate]
        ]).astype(np.float32)

        return state_vector

        # # 1. 全局记忆地图（所有智能体记忆的最大值），二值化后展平
        # global_memory = np.max(self.agent_memory_maps, axis=0)  # shape: (64,64)
        # global_memory_binary = (global_memory > 0).astype(np.float32)
        # memory_flat = global_memory_binary.flatten()
        #
        # # 2. 所有智能体位置归一化
        # searchers_pos_flat = (self.searcher_pos / self.grid_size).flatten()  # shape: (6,)
        #
        # # 3. 所有目标位置归一化
        # targets_pos_flat = (self.target_pos / self.grid_size).flatten()  # shape: (4,)
        #
        # # 拼接
        # state_vec = np.concatenate([memory_flat, searchers_pos_flat, targets_pos_flat]).astype(np.float32)
        # return state_vec

        # """
        # 返回全局状态图像 (H, W, C)
        # """
        # # 修复：直接使用 self.common_shape 初始化
        # global_map = np.zeros(self.common_shape, dtype=np.float32)
        #
        # # Channel 0: 全局障碍物
        # global_map[:, :, 0] = self.global_obstacle_map
        #
        # # Channel 1: 全局探索情况
        # if self.n_searchers > 0:
        #     global_map[:, :, 1] = np.max(self.agent_memory_maps, axis=0)
        #
        # # Channel 2: 所有智能体位置
        # for pos in self.searcher_pos:
        #     x, y = int(pos[0]), int(pos[1])
        #     if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
        #         global_map[x, y, 2] = 1.0
        #
        # # Channel 3: 所有目标位置
        # for pos in self.target_pos:
        #     x, y = int(pos[0]), int(pos[1])
        #     if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
        #         global_map[x, y, 3] = 1.0

        # return global_map

    def reset(self):
        self._current_step = 0

        # --- 1. 固定初始化位置 ---
        self.searcher_pos = np.zeros((self.n_searchers, 2))

        # 计算 Y 轴的均匀分布间隔
        # 例如 grid_size=64, n_searchers=3 时，Y坐标分别为 16, 32, 48
        y_gap = self.grid_size / (self.n_searchers + 1)

        for i in range(self.n_searchers):
            # X轴固定为探测半径 (距离左边界 detect_radius)
            self.searcher_pos[i, 0] = self.detect_radius
            # Y轴均匀分布
            self.searcher_pos[i, 1] = y_gap * (i + 1)

        # 2. 初始化/重置地图记忆
        # 0表示未探测，探测后置为1，随时间衰减
        self.agent_memory_maps = np.zeros((self.n_searchers, self.grid_size, self.grid_size))
        #
        # 3. 生成障碍物 (简单随机生成
        self.global_obstacle_map = np.zeros((self.grid_size, self.grid_size))
        # 随机放置障碍块
        for _ in range(self.n_obstacles):
            ox, oy = np.random.randint(0, self.grid_size, 2)
            self.global_obstacle_map[ox:min(ox + 5, 64), oy:min(oy + 5, 64)] = 1.0

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

        # --- 1. 智能体移动与越界判定 ---
        for i, agent_key in enumerate(self.agents):
            action = action_dict[agent_key]

            # 将离散的整数动作 (0~4) 映射为二维速度向量
            if action == 0:
                vel = np.array([0.0, 0.0])  # 不动
            elif action == 1:
                vel = np.array([0.0, 1.0])  # 向上
            elif action == 2:
                vel = np.array([0.0, -1.0])  # 向下
            elif action == 3:
                vel = np.array([-1.0, 0.0])  # 向左
            elif action == 4:
                vel = np.array([1.0, 0.0])  # 向右
            else:
                vel = np.array([0.0, 0.0])

            # 设置每次移动的固定步长（相当于原来的 max_speed）
            step_size = 0.5
            vel = vel * step_size

            next_pos = self.searcher_pos[i] + vel

            # --- 【判定撞墙】 ---
            # 检查是否超出 [0, grid_size] 边界
            out_of_bounds = (next_pos[0] < 0 or next_pos[0] >= self.grid_size or
                             next_pos[1] < 0 or next_pos[1] >= self.grid_size)

            if out_of_bounds:
                # 1. 给予撞墙的瞬间惩罚
                rewards[agent_key] = -5.0

                # 2. 弹性反弹计算 (越过边界多少，就弹回来多少)
                # 处理 X 轴反弹
                if next_pos[0] < 0:
                    next_pos[0] = -next_pos[0]  # 比如走到 -1.5，就弹回 1.5
                elif next_pos[0] >= self.grid_size:
                    next_pos[0] = 2 * self.grid_size - next_pos[0] - 0.01

                # 处理 Y 轴反弹
                if next_pos[1] < 0:
                    next_pos[1] = -next_pos[1]
                elif next_pos[1] >= self.grid_size:
                    next_pos[1] = 2 * self.grid_size - next_pos[1] - 0.01

                # 3. 兜底保护：防止速度极大导致反弹后依然越界
                self.searcher_pos[i] = np.clip(next_pos, 0, self.grid_size - 0.01)
            else:
                # 正常移动
                self.searcher_pos[i] = next_pos

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
            r_step = -0.01  # 存活奖励

            # B. 探测新区域奖励 (保持不变)
            r_explore = new_explored_rewards[i] * 0.05

            # C. 搜索与捕获目标奖励 (保持不变)
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
            for k in range(self.n_searchers):
                if i == k: continue  # 跳过自己
                dist_to_other = np.linalg.norm(agent_pos - self.searcher_pos[k])
                if dist_to_other < self.agent_collision_dist:
                    # 如果距离太近，给予惩罚
                    r_collision -= 0.3

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
                    # 距离越近，惩罚越大。单边最大惩罚设为 -0.5 (当距离接近0时)
                    # 比例因子: (1 - dist / detect_radius)
                    penalty_factor = 1.0 - (dist / self.detect_radius)
                    r_boundary += -0.5 * penalty_factor


            # C. 衰减区域重访逻辑 (隐含在A中)
            # 如果一个区域很久没去，值衰减得很低，_update_maps_and_memory 会将其重新置为1
            # 增量 (1.0 - old_value) 越大，奖励越高，鼓励去“旧”地方或“新”地方

            rewards[agent_key] = r_step + r_explore + r_target + r_collision + r_boundary
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

        # 1. 仅对已探索区域进行衰减
        self.agent_memory_maps[explored_mask] *= self.decay_rate

        # 2. 仅对已探索区域限制下限 (保证其不会衰减回 0，0 永远只代表"未探索")
        self.agent_memory_maps[explored_mask] = np.maximum(
            self.agent_memory_maps[explored_mask],
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
                        # --- 【关键奖励逻辑】 ---
                        old_val = self.agent_memory_maps[i, x, y]
                        # 增量奖励 = 1.0 - 旧的新鲜度
                        # 如果 old_val 是 0 (全新)，奖励是 1.0
                        # 如果 old_val 是 0.9 (刚走过)，奖励是 0.1
                        increment = 1.0 - old_val
                        new_explored_rewards[i] += increment

                        # 更新地图：踩过的格子新鲜度恢复到 1.0
                        self.agent_memory_maps[i, x, y] = 1.0

        return new_explored_rewards

        # ###开始
        # # 不再进行衰减，只更新探测区域
        # for i in range(self.n_searchers):
        #     pos = self.searcher_pos[i]
        #     dist_sq = (self.grid_x - pos[0]) ** 2 + (self.grid_y - pos[1]) ** 2
        #     in_range_mask = dist_sq <= (self.detect_radius ** 2)
        #
        #     # 当前未探测（值为0）的格子数量即为增益
        #     current_vals = self.agent_memory_maps[i, in_range_mask]
        #     gain = np.sum(1.0 - current_vals)  # 原来为0的格子变为1，增益为1
        #     info_gains[i] = gain
        # ####结束
        #     # 更新探测区域为 1.0
        #     self.agent_memory_maps[i, in_range_mask] = 1.0
        #
        # return info_gains

    def _get_observations(self):
        obs_dict = {}
        for i, agent_key in enumerate(self.agents):
            # --- 1. 自身位置归一化 (2维) ---
            my_pos = self.searcher_pos[i]
            pos_norm = my_pos / self.grid_size

            # # --- 2. 边界距离归一化 (4维) ---
            # # 逻辑：距离 < 探测半径 ? 0 (危险) : 1 (安全)
            # px, py = my_pos[0], my_pos[1]
            # # 计算到四面墙的距离
            # dist_left = px
            # dist_right = self.grid_size - px
            # dist_top = py
            # dist_bottom = self.grid_size - py
            #
            # # 生成标志位：1.0 表示安全，0.0 表示靠近边界
            # obs_boundary = [
            #     min(1.0, dist_left / self.detect_radius),
            #     min(1.0, dist_right / self.detect_radius),
            #     min(1.0, dist_top / self.detect_radius),
            #     min(1.0, dist_bottom / self.detect_radius)
            # ]


            # # --- 3. 其他智能体信息 ((N-1)*3 维) ---
            # others_info = []
            # for j in range(self.n_searchers):
            #     if i == j: continue
            #
            #     other_pos = self.searcher_pos[j]
            #     rel_pos = (other_pos - my_pos) / self.grid_size  # 相对位置归一化
            #     dist = np.linalg.norm(rel_pos)
            #
            #     others_info.extend([rel_pos[0], rel_pos[1], dist])
            #
            # others_info = np.array(others_info, dtype=np.float32)

            # --- 4. 拼接所有特征 ---
            # 总维度: 2 (pos) + 4 (wall) + (N-1)*3 (others)
            obs = np.concatenate([
                pos_norm,
                # obs_boundary,
                # others_info
            ]).astype(np.float32)

            obs_dict[agent_key] = obs

        return obs_dict

        # 1. 二进制记忆地图展平（将 >0 的视为1，否则0）
            # memory_flat = (self.agent_memory_maps[i] > 0).astype(np.float32).flatten()

            # 2. 自身位置归一化
            # pos_norm = self.searcher_pos[i] / self.grid_size  # 归一化到 [0,1]

            # # 3. 探测目标标志：是否有任何目标在探测半径内
            # my_pos = self.searcher_pos[i]
            # target_detected = 0.0
            # for t_pos in self.target_pos:
            #     if np.linalg.norm(my_pos - t_pos) <= self.detect_radius:
            #         target_detected = 1.0
            #         break

            # # 拼接成一维数组
            # obs = np.concatenate([memory_flat, pos_norm, [target_detected]]).astype(np.float32)
            # obs_dict[agent_key] = obs
        # obs_dict = {}
        # for i, agent_key in enumerate(self.agents):
        #     # 构建 4通道 观测矩阵
        #     obs = np.zeros(self.common_shape, dtype=np.float32)
        #
        #     # Channel 0: 障碍物 (仅当热力图>0即探测过时，才显示障碍物)
        #     known_mask = self.agent_memory_maps[i] > 0  # 使用memory map作为已知区域掩码
        #     obs[:, :, 0] = np.where(known_mask, self.global_obstacle_map, 0)
        #
        #     # Channel 1: 覆盖热力图 (记忆)
        #     obs[:, :, 1] = self.agent_memory_maps[i]
        #
        #     # Channel 2: 自身位置 (在网格上画一个点或高斯)
        #     px, py = int(self.searcher_pos[i][0]), int(self.searcher_pos[i][1])
        #     if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
        #         obs[px, py, 2] = 1.0
        #
        #     # Channel 3: 探测范围内的目标
        #     # 只有在探测半径内的目标才会在观测中显示
        #     my_pos = self.searcher_pos[i]
        #     for t_pos in self.target_pos:
        #         dist = np.linalg.norm(my_pos - t_pos)
        #         if dist <= self.detect_radius:
        #             tx, ty = int(t_pos[0]), int(t_pos[1])
        #             if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
        #                 obs[tx, ty, 3] = 1.0
        #
        #
        #
        #     obs_dict[agent_key] = obs
        # return obs_dict

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

        # --- 图层 A: 绘制记忆热力图 (绿色) ---
        # 我们取所有智能体记忆的最大值，展示"全局已知信息"
        # global_memory shape: [64, 64], 值域 0.0 ~ 1.0
        global_memory = np.max(self.agent_memory_maps, axis=0)

        # 找到所有有记忆的地方 (>0)
        mem_indices = np.where(global_memory > 0.05)
        for x, y in zip(mem_indices[0], mem_indices[1]):
            val = global_memory[x, y]
            # 颜色计算: 越新鲜(1.0)越绿，越旧(0.1)越淡
            # 使用 RGBA，但 PyGame draw.rect 不支持 alpha，所以我们用混合颜色
            # 白(255) -> 绿(0, 255, 0)
            c_val = int(255 * (1 - val))
            color = (c_val, 255, c_val)  # 浅绿到深绿

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
        return

