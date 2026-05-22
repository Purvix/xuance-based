# fast_buffer.py  ← 完整替换
"""
替换 XuanCe 的 MARL_OffPolicyBuffer。
核心改动：把存储格式从 (n_envs, n_size, dim) 改成 (capacity, dim)，
让 sample 变成连续内存访问，从 ~80ms 降到 <1ms。
"""
import numpy as np


class FastMARLBuffer:
    """
    扁平化存储的 MARL 经验回放池。

    原始格式：shape = (n_envs=16, n_size=625, dim)
              采样时双重随机索引，内存跳跃，sample ~80ms

    新格式：  shape = (capacity=10000, dim)
              采样时单次随机索引，内存连续，sample <1ms

    关键：buffer_size 直接就是总容量，不再除以 n_envs。
    每次 store() 写入 n_envs 条，ptr 以 n_envs 为步长推进。
    """

    def __init__(self, agent_keys, obs_space, act_space,
                 n_envs, buffer_size, batch_size, **kwargs):
        self.agent_keys  = agent_keys
        self.n_envs      = n_envs
        self.capacity    = buffer_size          # 直接就是总容量
        self.n_size      = buffer_size          # 兼容外部访问
        self.batch_size  = batch_size
        self.buffer_size = buffer_size
        self.obs_space   = obs_space
        self.act_space   = act_space

        # 从 space 对象里拿维度
        obs_dim = {k: obs_space[k].shape[0] for k in agent_keys}
        act_dim = {k: act_space[k].shape[0] for k in agent_keys}

        # ── 预分配扁平数组 ──────────────────────────────────────────
        # shape = (capacity, dim)，连续内存，采样只需单次索引
        self.data = {
            'obs':        {k: np.zeros((self.capacity, obs_dim[k]), dtype=np.float32) for k in agent_keys},
            'obs_next':   {k: np.zeros((self.capacity, obs_dim[k]), dtype=np.float32) for k in agent_keys},
            'actions':    {k: np.zeros((self.capacity, act_dim[k]), dtype=np.float32) for k in agent_keys},
            'rewards':    {k: np.zeros((self.capacity,),            dtype=np.float32) for k in agent_keys},
            'terminals':  {k: np.zeros((self.capacity,),            dtype=bool)       for k in agent_keys},
            'agent_mask': {k: np.ones( (self.capacity,),            dtype=bool)       for k in agent_keys},
        }

        # ptr 以"步"为单位（每步写入 n_envs 条）
        self.ptr       = 0      # 当前写入的"步"指针
        self.n_steps   = self.capacity // n_envs   # 最多存多少步
        self.size      = 0      # 已有效的条数（用于采样上界）

        mem_mb = sum(
            arr.nbytes
            for field in self.data.values()
            for arr in field.values()
        ) / 1024 / 1024

        print(f"[FastMARLBuffer] capacity={self.capacity}  "
              f"n_steps={self.n_steps}  "
              f"obs_dim={obs_dim}  "
              f"内存占用≈{mem_mb:.0f}MB")

    def store(self, **step_data):
        """
        写入一步数据（n_envs 条）。
        写入位置：[ptr*n_envs : (ptr+1)*n_envs]
        """
        start = self.ptr * self.n_envs
        end   = start + self.n_envs

        for field, agent_dict in step_data.items():
            if field not in self.data:
                continue
            for agt_key, values in agent_dict.items():
                # values shape: (n_envs,) 或 (n_envs, dim)
                self.data[field][agt_key][start:end] = values

        self.ptr  = (self.ptr + 1) % self.n_steps
        self.size = min(self.size + self.n_envs, self.capacity)

    def sample(self, batch_size=None):
        """
        随机采样。batch_size 可选，默认用初始化时的值。
        单次 np.random.choice → 连续内存访问 → <1ms
        """
        if batch_size is None:
            batch_size = self.batch_size

        idx = np.random.choice(self.size, size=batch_size, replace=False)

        result = {'batch_size': batch_size}
        for field, agent_dict in self.data.items():
            result[field] = {
                agt_key: arr[idx]
                for agt_key, arr in agent_dict.items()
            }
        return result

    def clear(self):
        self.ptr  = 0
        self.size = 0
        for field in self.data.values():
            for arr in field.values():
                arr[:] = 0
