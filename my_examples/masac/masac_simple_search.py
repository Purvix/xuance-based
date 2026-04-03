import argparse
import numpy as np
from copy import deepcopy
from operator import itemgetter

from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MASAC_Agents
from xuance.torch.policies.categorical_marl import MASAC_Policy


from simple_search import SearchEnv

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

    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv

    set_seed(configs.seed)
    envs = make_envs(configs)
    # print("Config object content:")
    # print(configs)
    # print("hidden_sizes in config:", getattr(configs, 'hidden_sizes', None))
    # print("fc_hidden_sizes in config:", getattr(configs, 'fc_hidden_sizes', None))


    Agent = MASAC_Agents(config=configs, envs=envs)

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
            test_scores = Agent.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agent.current_step}
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            test_envs = make_envs(configs)
            Agent.load_model(path=Agent.model_dir_load)
            # scores = Agent.test(test_episodes=configs.test_episode, test_envs=test_envs, close_envs=True)
            # print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            # print("Finish testing.")
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
                    print(f"--- Step {step_count} 观测 ---")
                    # 注意：XuanCe 的 VecEnv 返回的 obs_dict 可能被包裹在列表中，即 obs_dict[0]
                    current_obs = obs_dict[0] if isinstance(obs_dict, list) else obs_dict
                    for agent_id, obs in current_obs.items():
                        # 保留两位小数，方便查看
                        formatted_obs = [round(float(x), 2) for x in obs]
                        print(f"  {agent_id}: {formatted_obs}")

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
                    print_str = f"[Step {step_count:03d}] "
                    for agent in step_rewards.keys():
                        print_str += f"{agent}: 步奖励={step_rewards[agent]:+.2f}, 累计={episode_rewards[agent]:+.2f} | "
                    print(print_str)

                    # 可选：打印详细分解 (如果你需要看各项惩罚的具体数值，可以取消注释)
                    if "reward_details" in info_list[0]:
                        print(f"  详细 = {info_list[0]['reward_details']}")

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
            test_envs.close()
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")


    Agent.finish()
