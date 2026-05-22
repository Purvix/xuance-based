# my_masac_agents.py
"""
在不修改 XuanCe 源码的前提下，通过继承覆写来加速训练。
"""
import numpy as np
from xuance.torch.agents.multi_agent_rl.masac_agents import MASAC_Agents as _BaseAgents
from fast_buffer import FastMARLBuffer


class MASAC_Agents(_BaseAgents):
    """
    覆写两个方法：
      1. __init__   → 替换为 FastMARLBuffer
      2. store_experience → 用 np.stack 消除 Python 循环
    """

    def __init__(self, config, envs):
        super().__init__(config, envs)

        # ── 替换 buffer ──────────────────────────────────────────────
        self.memory = FastMARLBuffer(
            agent_keys  = self.agent_keys,
            obs_space   = envs.observation_space,
            act_space   = envs.action_space,
            n_envs      = config.parallels,
            buffer_size = config.buffer_size,
            batch_size  = config.batch_size,
        )
        # learner 内部也持有 memory 引用，一并替换
        if hasattr(self, 'learner') and self.learner is not None:
            self.learner.memory = self.memory

        print(f"[MASAC_Agents] FastMARLBuffer 已替换原始 buffer ✓")

    def store_experience(self, obs_dict, avail_actions, actions_dict, obs_next_dict,
                         avail_actions_next, rewards_dict, terminals_dict, info, **kwargs):
        """用 np.stack 批量构建，避免逐条 Python 循环。"""
        experience_data = {
            'obs':        {k: np.stack([d[k] for d in obs_dict])       for k in self.agent_keys},
            'actions':    {k: np.stack([d[k] for d in actions_dict])   for k in self.agent_keys},
            'obs_next':   {k: np.stack([d[k] for d in obs_next_dict])  for k in self.agent_keys},
            'rewards':    {k: np.array([d[k] for d in rewards_dict],   dtype=np.float32)
                           for k in self.agent_keys},
            'terminals':  {k: np.array([d[k] for d in terminals_dict], dtype=bool)
                           for k in self.agent_keys},
            'agent_mask': {k: np.array([d['agent_mask'][k] for d in info], dtype=bool)
                           for k in self.agent_keys},
        }
        if self.use_global_state:
            experience_data['state']      = np.asarray(kwargs['state'],      dtype=np.float32)
            experience_data['state_next'] = np.asarray(kwargs['next_state'], dtype=np.float32)
        if self.use_actions_mask:
            experience_data['avail_actions'] = {
                k: np.stack([d[k] for d in avail_actions]) for k in self.agent_keys}
            experience_data['avail_actions_next'] = {
                k: np.stack([d[k] for d in avail_actions_next]) for k in self.agent_keys}

        self.memory.store(**experience_data)
