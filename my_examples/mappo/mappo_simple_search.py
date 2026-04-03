import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import MAPPO_Agents

from simple_search_v1 import SearchEnv

def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: MAPPO for MPE.")
    parser.add_argument("--env-id", type=str, default="mappo_simple_search")
    # parser.add_argument("--test", type=int, default=0)
    # parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir=f"mappo_search_configs/{parser.env_id}.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = SearchEnv

    set_seed(configs.seed)  # Set the random seed.
    envs = make_envs(configs)  # Make the environment.
    Agents = MAPPO_Agents(config=configs, envs=envs)  # Create the Independent PPO agents.

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():  # Print the training information.
        print(f"{k}: {v}")

    if configs.benchmark:
        configs_test = deepcopy(configs)
        configs_test.parallels = configs_test.test_episode
        test_envs = make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agents.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)
        Agents.save_model(model_name="best_model.pth")
        best_scores_info = {"mean": np.mean(test_scores),
                            "std": np.std(test_scores),
                            "step": Agents.current_step}
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agents.train(eval_interval)
            test_scores = Agents.test(test_episodes=test_episode, test_envs=test_envs, close_envs=False)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {"mean": np.mean(test_scores),
                                    "std": np.std(test_scores),
                                    "step": Agents.current_step}
                # save best model
                Agents.save_model(model_name="best_model.pth")
        # end benchmarking
        print("Best Model Score: %.2f, std=%.2f" % (best_scores_info["mean"], best_scores_info["std"]))
    else:
        if configs.test:
            configs.parallels = configs.parallels
            test_envs = make_envs(configs)

            Agents.load_model(path=Agents.model_dir_load)
            # scores = Agents.test(test_episodes=configs.test_episode, test_envs=test_envs, close_envs=True)
            # 自定义渲染
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
                    policy_out = Agents.action(obs_dict=obs_dict, test_mode=True)
                    actions_dict = policy_out['actions']

                    # 环境步进
                    next_obs_dict, rewards, terminated_dict, truncated_list, info_list = test_envs.step(actions_dict)


                    # 累加奖励 (向量化环境下 rewards 是一个数组)
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

            test_envs.close()
            print(f"Mean Score: {np.mean(all_scores)}, Std: {np.std(all_scores)}")
            # print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agents.train(configs.running_steps // configs.parallels)
            Agents.save_model("final_train_model.pth")
            print("Finish training!")

    Agents.finish()
