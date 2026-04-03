import argparse
import time
import numpy as np

# 导入 XuanCe 核心组件
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed

# 导入你的自定义环境类
from my_examples.masac.simple_search import SearchEnv


def parse_args():
    parser = argparse.ArgumentParser("Run Random Policy for SearchEnv")

    parser.add_argument("--env-id", type=str, default="masac_simple_search")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    # 1. 解析参数
    parser = parse_args()

    # 2. 读取 YAML 配置文件
    # 假设 yaml 文件在 masac_search_configs 文件夹下
    configs_dict = get_configs(file_dir=f"masac_search_configs/{parser.env_id}.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    # ====================================================
    # 关键步骤：注册自定义环境
    # XuanCe 会根据 configs.env_name ("Search") 来查找这个类
    # ====================================================
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv

    # 3. 设置随机种子
    set_seed(configs.seed)

    # 4. 创建环境
    # 注意：make_envs 返回的是一个 Vectorized Environment (并行环境包装器)
    # 即使 parallels=1，它也是一个列表形式的接口
    envs = make_envs(configs)

    # 获取环境基本信息
    num_envs = envs.num_envs
    agents = envs.agents  # 例如 ['searcher_0', 'searcher_1', ...]
    print(f"环境已启动: {num_envs} 个并行环境, 智能体列表: {agents}")
    print(f"动作空间: {envs.action_space}")

    # ====================================================
    # 随机策略循环
    # ====================================================
    n_episodes = 5  # 运行 5 个回合

    try:
        for episode in range(n_episodes):
            print(f"--- Episode {episode + 1} Start ---")

            # 重置环境
            obs, info = envs.reset()
            done = False
            step = 0

            while not done:
                # --- A. 生成随机动作 ---
                # XuanCe 的 VecEnv 要求动作是一个字典: {agent_name: action_array}
                # action_array 的形状必须是 (num_envs, action_dim)
                actions_dict = {}

                for agent_id in agents:
                    # 获取该智能体的动作空间
                    act_space = envs.action_space[agent_id]

                    # 为每个并行环境采样一个动作
                    # act_space.sample() 返回单个动作，我们需要把它堆叠成 (num_envs, ...)
                    if num_envs == 1:
                        # 维度扩展: (2,) -> (1, 2)
                        act = act_space.sample()[np.newaxis, :]
                    else:
                        act = np.array([act_space.sample() for _ in range(num_envs)])

                    actions_dict[agent_id] = act

                # --- B. 环境步进 ---
                next_obs, rewards, terminated, truncated, info = envs.step(actions_dict)

                # --- C. 渲染和调试信息 ---
                # 显示观测空间信息
                # if step == 0:  # 只在第一步显示
                #     print(f"观测空间形状: {next_obs[agents[0]].shape}")
                #     print(f"观测值范围: [{np.min(next_obs[agents[0]])}, {np.max(next_obs[agents[0]])}]")
                #
                # 你的 yaml 里配置了 render: true 和 render_mode: 'human'
                # 这里调用 render 会触发 SearchEnv 里的渲染逻辑
                envs.render(configs.render_mode)

                # 控制帧率 (FPS)
                time.sleep(1 / getattr(configs, 'fps', 10))  # 降低帧率便于观察

                # --- D. 检查结束条件 ---
                # 在向量化环境中，terminated 是一个布尔值列表或数组
                # 只要任意一个环境结束，或者达到最大步数，我们就认为演示结束
                step += 1
                if step >= configs.max_episode_steps:
                    done = True

                # 更新观测值
                obs = next_obs

            print(f"--- Episode {episode + 1} Finished ({step} steps) ---")
            # 显示回合统计信息
            if 'episode_score' in info:
                print(f"回合得分: {info['episode_score']}")

    except KeyboardInterrupt:
        print("用户强制停止")
    finally:
        envs.close()
        print("环境已关闭")
