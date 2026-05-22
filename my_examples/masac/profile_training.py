# profile_training.py  ← 完整替换
import time
import argparse
import numpy as np
import torch
import inspect

from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MASAC_Agents
from hybrid_representation import HybridRepresentation, HybridCriticRepresentation
from simple_search import SearchEnv

# ── 初始化 ──────────────────────────────────────────────────────────
configs_dict = get_configs(file_dir="masac_search_configs/masac_simple_search.yaml")
configs = argparse.Namespace(**configs_dict)
set_seed(configs.seed)
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv
REGISTRY_Representation["Hybrid_Representation"] = HybridRepresentation

_tmp_env = SearchEnv(configs)
configs.vec_dim = _tmp_env.vec_dim
del _tmp_env

envs = make_envs(configs)
Agent = MASAC_Agents(config=configs, envs=envs)

# ── 预热 buffer ──────────────────────────────────────────────────────
print("预热 buffer 中...")
Agent.train(configs.start_training // configs.parallels + 50)
print("预热完成\n")

# ================================================================
# 第一部分：打印 learner.update 和 memory.store 源码
# ================================================================
print("=" * 60)
print("【learner.update() 源码】")
print("=" * 60)
try:
    src = inspect.getsource(Agent.learner.update)
    for i, line in enumerate(src.split('\n')[:100]):
        print(f"{i+1:3d} | {line}")
except Exception as e:
    print(f"报错: {e}")

print("\n" + "=" * 60)
print("【memory.store() 源码】")
print("=" * 60)
try:
    src = inspect.getsource(Agent.memory.store)
    for i, line in enumerate(src.split('\n')[:60]):
        print(f"{i+1:3d} | {line}")
except Exception as e:
    print(f"报错: {e}")

print("\n" + "=" * 60)
print("【memory 内部数据结构】")
print("=" * 60)
try:
    mem = Agent.memory
    # 打印 memory 对象的所有属性
    for attr in dir(mem):
        if attr.startswith('_'):
            continue
        val = getattr(mem, attr)
        if isinstance(val, np.ndarray):
            print(f"  {attr:<30} ndarray  shape={val.shape}  dtype={val.dtype}")
        elif isinstance(val, dict):
            print(f"  {attr:<30} dict  keys={list(val.keys())[:5]}")
            for k, v in list(val.items())[:2]:
                if isinstance(v, np.ndarray):
                    print(f"    [{k}]  shape={v.shape}  dtype={v.dtype}")
        elif isinstance(val, (int, float, bool, str)):
            print(f"  {attr:<30} {type(val).__name__}  = {val}")
except Exception as e:
    print(f"报错: {e}")

# ================================================================
# 第二部分：patch 计时，精确拆解 train() 内部
# ================================================================
print("\n" + "=" * 60)
print("【精确计时：train() 内部各阶段】")
print("=" * 60)

original_update = Agent.learner.update
original_sample = Agent.memory.sample
original_store  = Agent.memory.store

update_times = []
sample_times = []
store_times  = []

def timed_update(*args, **kwargs):
    t0 = time.perf_counter()
    result = original_update(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    update_times.append((t1 - t0) * 1000)
    return result

def timed_sample(*args, **kwargs):
    t0 = time.perf_counter()
    result = original_sample(*args, **kwargs)
    t1 = time.perf_counter()
    sample_times.append((t1 - t0) * 1000)
    return result

def timed_store(*args, **kwargs):
    t0 = time.perf_counter()
    result = original_store(*args, **kwargs)
    t1 = time.perf_counter()
    store_times.append((t1 - t0) * 1000)
    return result

Agent.learner.update  = timed_update
Agent.memory.sample   = timed_sample
Agent.memory.store    = timed_store

# 跑 20 轮
train_total_times = []
for i in range(20):
    t0 = time.perf_counter()
    Agent.train(configs.training_frequency)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    train_total_times.append((t1 - t0) * 1000)
    print(f"  轮 {i+1:2d}: train()={train_total_times[-1]:.1f}ms  "
          f"update={update_times[-1] if update_times else 0:.1f}ms  "
          f"sample={sample_times[-1] if sample_times else 0:.2f}ms  "
          f"store={np.mean(store_times[-configs.training_frequency:]) if store_times else 0:.2f}ms(均)")

# ── 汇总 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【汇总】")
print("=" * 60)

def stats(arr, name):
    if not arr:
        print(f"  {name:<25} 无数据")
        return
    print(f"  {name:<25} 均值={np.mean(arr):.2f}ms  中位={np.median(arr):.2f}ms  最大={np.max(arr):.2f}ms")

stats(train_total_times, "train() 总耗时")
stats(update_times,      "  learner.update()")
stats(sample_times,      "  memory.sample()")
stats(store_times,       "  memory.store() 单次")

total_store = np.sum(store_times)
total_train = np.sum(train_total_times)
total_update = np.sum(update_times)
total_sample = np.sum(sample_times)
other = total_train - total_update - total_sample - total_store

print(f"\n  时间占比分析（{len(train_total_times)}轮合计）:")
print(f"    learner.update  : {total_update/total_train*100:.1f}%  ({total_update:.0f}ms)")
print(f"    memory.sample   : {total_sample/total_train*100:.1f}%  ({total_sample:.0f}ms)")
print(f"    memory.store    : {total_store/total_train*100:.1f}%  ({total_store:.0f}ms)")
print(f"    其他框架开销    : {other/total_train*100:.1f}%  ({other:.0f}ms)")

print(f"\n  等效吞吐: {configs.training_frequency * configs.parallels / (np.mean(train_total_times)/1000):.0f} env_steps/秒")
print(f"  GPU实际工作占比: {total_update/total_train*100:.1f}%")
sample = Agent.memory.sample(configs.batch_size)
print("【memory.data 内部 shape】")
for k, v in Agent.memory.data.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  data['{k}']['{kk}']  shape={vv.shape}  dtype={vv.dtype}")
    elif isinstance(v, np.ndarray):
        print(f"  data['{k}']  shape={v.shape}  dtype={v.dtype}")

print("\n【sample 返回的 shape】")
for k, v in sample.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            if hasattr(vv, 'shape'):
                print(f"  sample['{k}']['{kk}']  shape={vv.shape}  dtype={vv.dtype}")
    elif hasattr(v, '__len__'):
        print(f"  sample['{k}']  = {v}")