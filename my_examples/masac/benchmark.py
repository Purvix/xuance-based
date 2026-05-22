# benchmark.py
import time
import argparse
import numpy as np
import torch

from xuance.common import get_configs
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.utils.operations import set_seed

# ── 关键：从自己的文件导入，不从 xuance 导入 ──
from my_masac_agents import MASAC_Agents
from hybrid_representation import HybridRepresentation
from simple_search import SearchEnv

configs_dict = get_configs(file_dir="masac_search_configs/masac_simple_search.yaml")
configs = argparse.Namespace(**configs_dict)
set_seed(configs.seed)

REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv
REGISTRY_Representation["Hybrid_Representation"] = HybridRepresentation

_tmp = SearchEnv(configs)
configs.vec_dim = _tmp.vec_dim
del _tmp

envs = make_envs(configs)

print("创建 Agent...")
Agent = MASAC_Agents(config=configs, envs=envs)

# 确认 buffer 类型
print(f"buffer 类型: {type(Agent.memory).__name__}")
print(f"buffer 类: {type(Agent.memory)}")

print("\n预热 buffer...")
Agent.train(configs.start_training // configs.parallels + 10)
print("预热完成\n")

# 计时
N = 30
times = []
for i in range(N):
    t0 = time.perf_counter()
    Agent.train(configs.training_frequency)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    times.append(ms)
    print(f"  轮 {i+1:2d}: {ms:.1f}ms  ({configs.training_frequency * configs.parallels / (ms/1000):.0f} steps/s)")

print(f"\n{'='*50}")
print(f"training_frequency = {configs.training_frequency}")
print(f"train() 均值 = {np.mean(times):.1f}ms  中位 = {np.median(times):.1f}ms")
print(f"吞吐量 = {configs.training_frequency * configs.parallels / (np.mean(times)/1000):.0f} env_steps/秒")
print(f"对比基线 576 steps/s，提升 {configs.training_frequency*configs.parallels/(np.mean(times)/1000)/576:.1f}×")
