import argparse
import numpy as np
import torch
from copy import deepcopy
from operator import itemgetter

from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.representations import REGISTRY_Representation
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MASAC_Agents
from hybrid_representation import HybridRepresentation, HybridCriticRepresentation

from simple_search import SearchEnv
from icm_module import ICMModule
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ← 新增：安全注入函数，放在所有函数定义最前面
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _inject_icm_to_envs(vec_envs, icm_instance):
    """
    安全穿透 XuanCe wrapper 层，找到真正的 SearchEnv 实例并注入 ICM。
    遇到 None 或超过10层时停止，避免无限循环。
    """
    injected = 0
    for env_wrapper in vec_envs.envs:
        inner = env_wrapper
        depth = 0
        # 只要还有 .env 属性、不是 None、还不是 SearchEnv，就继续往里钻
        while (
            hasattr(inner, 'env')
            and inner.env is not None
            and not isinstance(inner, SearchEnv)
            and depth < 10
        ):
            inner = inner.env
            depth += 1

        if isinstance(inner, SearchEnv):
            inner.icm = icm_instance
            inner._icm_obs_buffer = None   # reset() 时会自动初始化
            injected += 1
        else:
            print(f"[ICM WARNING] 未能找到 SearchEnv，实际类型: {type(inner).__name__}")

    print(f"[ICM] 成功注入 {injected}/{len(vec_envs.envs)} 个环境")


def _print_env_hierarchy(vec_envs):
    """调试用：打印第一个环境的 wrapper 层级，确认 SearchEnv 在哪一层"""
    inner = vec_envs.envs[0]
    depth = 0
    print("[ICM DEBUG] 环境 wrapper 层级：")
    while inner is not None and depth < 10:
        print(f"  层级 {depth}: {type(inner).__name__}")
        inner = getattr(inner, 'env', None)
        depth += 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _update_icm(agent, icm, configs):
    if not hasattr(agent, 'memory'):
        return None

    buf = agent.memory
    buf_size = getattr(buf, 'current_size',
               getattr(buf, 'size',
               getattr(buf, 'filled_i',
               getattr(buf, 'data_count', None))))

    if buf_size is None or buf_size < configs.batch_size:
        return None

    _sample = buf.sample()

    _obs      = _sample.get('obs',      None)
    _act      = _sample.get('actions',  None)
    _obs_next = _sample.get('obs_next', None)

    if _obs is None or _act is None or _obs_next is None:
        print("[ICM WARNING] 字段缺失:", list(_sample.keys()))
        return None

    def dict_to_array(d):
        if isinstance(d, dict):
            arrays = []
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    arrays.append(v.cpu().numpy())
                else:
                    arrays.append(np.array(v))
            return np.concatenate(arrays, axis=0)
        elif isinstance(d, torch.Tensor):
            return d.cpu().numpy()
        else:
            return np.array(d)

    obs_arr      = dict_to_array(_obs)
    act_arr      = dict_to_array(_act)
    obs_next_arr = dict_to_array(_obs_next)

    icm_logs = icm.update_from_batch(obs_arr, act_arr, obs_next_arr)
    return icm_logs  # ← 修改：返回结果，不在这里打印

def get_icm_eta(i_epoch, configs):
    """
    前 warmup_epochs 个 epoch：eta = 0（ICM 只训练，不注入奖励）
    warmup 结束后线性升到 eta_max，经过 rampup_epochs 达到最大值
    """
    warmup  = getattr(configs, 'icm_warmup_epochs',   300)
    rampup  = getattr(configs, 'icm_eta_rampup_epochs', 100)
    eta_max = getattr(configs, 'icm_eta_max',          0.01)

    if i_epoch < warmup:
        return 0.0
    elif i_epoch < warmup + rampup:
        progress = (i_epoch - warmup) / rampup   # 0.0 → 1.0
        return eta_max * progress
    else:
        return eta_max


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: MASAC for MPE.")
    parser.add_argument("--env-id", type=str, default="masac_simple_search")
    # parser.add_argument("--test", type=int, default=0)
    # parser.add_argument("--benchmark", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir=f"masac_search_configs/{parser.env_id}.yaml")
    # print(f"DEBUG: 正在尝试加载配置文件: {configs_dict}")  # 打印出来确认一下
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)
    set_seed(configs.seed)
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv
    REGISTRY_Representation["Hybrid_Representation"] = HybridRepresentation
    # 动态同步 vec_dim
    _tmp_env = SearchEnv(configs)
    configs.vec_dim = _tmp_env.vec_dim
    # print(f"[DEBUG] 同步后 configs.vec_dim = {configs.vec_dim}")
    del _tmp_env

    envs = make_envs(configs)
    # print(f"[DEBUG] make_envs 后 configs.vec_dim = {configs.vec_dim}")  # 确认 make_envs 没有覆盖 configs
    # print("Config object content:")
    # print(configs)
    # print("hidden_sizes in config:", getattr(configs, 'hidden_sizes', None))
    # print("fc_hidden_sizes in config:", getattr(configs, 'fc_hidden_sizes', None))
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ← 新增：创建 ICM 并注入训练环境
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _use_icm = getattr(configs, 'use_icm', False)
    icm = None

    if _use_icm:
        _print_env_hierarchy(envs)

        icm = ICMModule(
            vec_dim=configs.vec_dim,
            map_channels=getattr(configs, 'map_channels', 3),
            map_size=getattr(configs, 'grid_size', 64),
            action_dim=2,
            feature_dim=getattr(configs, 'icm_feature_dim', 128),
            eta=0.0,  # ← 修改：初始 eta=0，后期由 get_icm_eta 动态设置
            beta=getattr(configs, 'icm_beta', 0.2),
            lr=getattr(configs, 'icm_lr', 3e-4),
            device=configs.device
        )
        warmup = getattr(configs, 'icm_warmup_epochs', 300)
        rampup = getattr(configs, 'icm_eta_rampup_epochs', 100)
        eta_max = getattr(configs, 'icm_eta_max', 0.01)
        # ↓ 新增：启动时打印完整 ICM 计划
        print(f"[ICM] 已创建 | beta={icm.beta} | device={icm.device}")
        # print(type(warmup), type(rampup), type(eta_max))
        # print(repr(warmup), repr(rampup), repr(eta_max))

        print(
            f"[ICM] 激活计划: epoch 0~{warmup} 预热(eta=0) → "
            f"epoch {warmup}~{warmup + rampup} 线性升至 eta={eta_max} → "
            f"epoch {warmup + rampup}+ 全速运行"
        )

        _inject_icm_to_envs(envs, icm)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Agent = MASAC_Agents(config=configs, envs=envs)
    # print(f"[DEBUG] critic_1_representation 类型: {type(list(Agent.policy.critic_1_representation.values())[0])}")
    # print(
    #     f"[DEBUG] critic_1_representation output_shapes: {list(Agent.policy.critic_1_representation.values())[0].output_shapes}")

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        configs_test = deepcopy(configs)
        configs_test.parallels = configs_test.test_episode
        test_envs = make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agent.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ← 新增：benchmark 模式下同步更新 ICM
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if icm is not None:
                # 1. 动态更新 eta（前期=0，warmup 后线性升起）
                current_eta = get_icm_eta(i_epoch, configs)
                icm.eta = current_eta

                # 2. 无论 eta 是否为 0，ICM 网络都持续训练
                n_icm_updates = eval_interval // getattr(configs, 'training_frequency', 25)
                last_logs = None
                for _ in range(n_icm_updates):
                    last_logs = _update_icm(Agent, icm, configs)

                # 3. 每 10 个 epoch 打印一次，显示 eta 状态
                if last_logs is not None and i_epoch % 10 == 0:
                    warmup = getattr(configs, 'icm_warmup_epochs', 300)
                    status = "预热中" if i_epoch < warmup else "已激活"
                    print(
                        f"[ICM] epoch={i_epoch:4d} | "
                        f"eta={current_eta:.5f}({status}) | "
                        f"loss={last_logs['icm_loss']:.4f} | "
                        f"fwd={last_logs['forward_loss']:.4f} | "
                        f"inv={last_logs['inverse_loss']:.4f}"
                    )
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
                Agent.save_model(model_name="final_train_model.pth")
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            test_envs = make_envs(configs)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ← 新增：test 模式下注入 ICM 到测试环境
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if icm is not None:
                _inject_icm_to_envs(test_envs, icm)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Agent.load_model(path=Agent.model_dir_load)
            # scores = Agent.test(test_episodes=configs.test_episode, test_envs=test_envs, close_envs=True)
            # print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            # print("Finish testing.")
            # ── 热力图初始化（受 yaml 开关控制）──────────────────────────
            visualizer = None
            if getattr(configs, 'heatmap_vis', False):
                from visualize_cnn import CNNHeatmapVisualizer

                _repr_dict = Agent.policy.actor_representation

                # 参数共享时 key 可能是 'share' 或第一个 agent 名，取第一个值即可
                _repr_obj = list(_repr_dict.values())[0]

                # print(f"[DEBUG] representation 类型: {type(_repr_obj)}")
                # print(f"[DEBUG] representation keys: {list(_repr_dict.keys())}")

                visualizer = CNNHeatmapVisualizer(
                    cnn_encoder=_repr_obj.cnn_encoder,  # ← 从对象上访问
                    vec_dim=configs.vec_dim,
                    grid_size=getattr(configs, 'grid_size', 64),
                )
                print(f"[热力图] 已开启，每 {getattr(configs, 'heatmap_interval', 20)} 步刷新一次")
            # print(f"[DEBUG] heatmap_vis={getattr(configs, 'heatmap_vis', '未找到')}, "
            #           f"heatmap_save={getattr(configs, 'heatmap_save', '未找到')}")
            # ─────────────────────────────────────────────────────────────
            all_scores = []

            for i_episode in range(configs.test_episode):
                obs_dict, info = test_envs.reset()
                terminated = [False]
                truncated = [False]
                episode_rewards = {agent: 0.0 for agent in envs.agents}   # 累积奖励
                step_count = 0

                # 运行单个 Episode
                while not (any(terminated) or any(truncated)):
                    # 【关键】显式调用渲染函数
                    test_envs.render(configs.render_mode)

                    # 格式化打印每个智能体的观测
                    # print(f"--- Step {step_count} 观测 ---")
                    # 注意：XuanCe 的 VecEnv 返回的 obs_dict 可能被包裹在列表中，即 obs_dict[0]
                    current_obs = obs_dict[0] if isinstance(obs_dict, list) else obs_dict
                    # for agent_id, obs in current_obs.items():
                    #     # 保留两位小数，方便查看
                    #     formatted_obs = [round(float(x), 2) for x in obs]
                    #     print(f"  {agent_id}: {formatted_obs}")

                    # ── 热力图（受开关和间隔控制）────────────────────────────────
                    if visualizer is not None:
                        heatmap_interval = getattr(configs, 'heatmap_interval', 20)
                        if step_count % heatmap_interval == 0:

                            # 取第一个智能体的观测
                            obs_sample = current_obs['searcher_0']

                            # 从 VecEnv 里取出真实环境实例，获取位置信息
                            env_instance = test_envs.envs[0].env
                            print(f"[DEBUG] env_instance 类型: {type(env_instance)}")
                            print(f"[DEBUG] searcher_pos: {env_instance.searcher_pos}")

                            # 决定是否保存文件
                            save_path = None
                            if getattr(configs, 'heatmap_save', False):
                                save_path = f"heatmap_ep{i_episode + 1:02d}_step{step_count:04d}.png"

                            visualizer.show(
                                obs=obs_sample,
                                searcher_pos=env_instance.searcher_pos,
                                target_pos=env_instance.target_pos,
                                target_alive=env_instance.target_alive,
                                step=step_count,
                                save_path=save_path,
                            )
                    # ─────────────────────────────────────────────────────────────

                    # 获取动作 (使用 Agent 的 action 接口)
                    policy_out = Agent.action(obs_dict=obs_dict, test_mode=True)

                    actions_dict = policy_out['actions']

                    # 环境步进
                    next_obs_dict, rewards, terminated_dict, truncated_list, info_list = test_envs.step(actions_dict)

                    # --- 提取并累加奖励 ---
                    step_rewards = rewards[0]  # 当前 step 的奖励字典

                    # 先累加当前步的奖励到累计奖励中
                    for agent, r in step_rewards.items():
                        episode_rewards[agent] += r

                    # --- 格式化打印：当前步奖励 + 累计奖励 ---
                    # print_str = f"[Step {step_count:03d}] "
                    # for agent in step_rewards.keys():
                    #     print_str += f"{agent}: 步奖励={step_rewards[agent]:+.2f}, 累计={episode_rewards[agent]:+.2f} | "
                    # print(print_str)

                    # # 可选：打印详细分解 (如果你需要看各项惩罚的具体数值，可以取消注释)
                    # if "reward_details" in info_list[0]:
                    #     print(f"  详细 = {info_list[0]['reward_details']}")

                    # 状态更新
                    obs_dict = next_obs_dict
                    terminated = [all(t.values()) for t in terminated_dict]
                    truncated = truncated_list

                    step_count += 1
                    if step_count > configs.max_episode_steps:  # 防止死循环
                        break

                # --- 打印 episode 累积奖励 ---
                total_episode_reward = sum(episode_rewards.values())
                # print(
                    # f"Episode {i_episode + 1} 累积奖励: 各智能体 = {episode_rewards}, 总奖励 = {total_episode_reward:.3f}")

                # 记录得分 (假设 info 中存有累积得分)
                # 注意：XuanCe 的 VecEnv 返回的是列表形式的 info
                score = np.mean(list(info_list[0]["episode_score"].values()))
                all_scores.append(score)
                print(f"Episode {i_episode + 1} finished. Score: {score}")

            print(f"Mean Score: {np.mean(all_scores)}, Std: {np.std(all_scores)}")
            # ── 清理热力图 hook ───────────────────────────────────────────
            if visualizer is not None:
                visualizer.remove_hooks()
            # ─────────────────────────────────────────────────────────────
            test_envs.close()
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")


    Agent.finish()
