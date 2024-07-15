"""
Multi-Agent Proximal Policy Optimization (MAPPO) with the Centralized Value Function
"""

import os

import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import PolicySpec
from ray.train import report
from ray.tune import Tuner, TuneConfig, PlacementGroupFactory
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from config import env_configs
from env import env_creator


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Centralized Critic Model for MAPPO
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.shared_network = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_shared")
        self.central_value_function = FullyConnectedNetwork(
            obs_space, action_space, 1, model_config, name + "_vf")

    def forward(self, input_dict, state, seq_lens):
        features, _ = self.shared_network(input_dict)
        self._value_out, _ = self.central_value_function(input_dict)
        return features, []

    def value_function(self):
        return torch.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)
register_env("InventoryManagementEnv", env_creator)


def tune_ppo(config):
    """
    Tune the PPO trainer
    """
    env_config_name = "constant_demand"
    env_config = env_configs[env_config_name]
    num_episodes = 100

    algo = PPOConfig() \
        .environment(env="InventoryManagementEnv", env_config=env_config) \
        .framework("torch") \
        .resources(num_gpus=0) \
        .env_runners(num_envs_per_env_runner=1) \
        .multi_agent(
            policies={
                f"policy_{m}": PolicySpec(
                    observation_space=env_creator(env_config).agent_observation_space,
                    action_space=env_creator(env_config).agent_action_space,
                    config={"model": {"custom_model": "centralized_critic"}},
                ) for m in range(env_config['num_stages'])
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"policy_{agent_id.split('_')[-1]}",
        ) \
        .training(gamma=1.0) \
        .evaluation(
            evaluation_interval=config["training_iteration"],
            evaluation_duration=num_episodes,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
        ) \
        .update_from_dict(config).build()

    for i in range(config["training_iteration"]):
        result = algo.train()
        report({"mean_episode_reward": result["env_runners"]["episode_reward_mean"]})

    results = algo.evaluate()
    episode_rewards = results['env_runners']['hist_stats']['episode_reward']
    episode_reward_mean = np.mean(episode_rewards)
    episode_reward_std = np.std(episode_rewards)
    print(f"env_config_name = {env_config_name}, num_episodes = {num_episodes}, "
          f"episode_reward_mean = {episode_reward_mean:.2f}, episode_reward_std = {episode_reward_std:.2f}")

    algo.stop()


if __name__ == '__main__':
    ray.init()

    resources_per_trial = PlacementGroupFactory([{"CPU": 4}] + [{"CPU": 2}] * 2)

    tuner = Tuner(
        tune.with_resources(tune_ppo, resources_per_trial),
        tune_config=TuneConfig(
            metric="mean_episode_reward",
            mode="max",
            num_samples=20,
        ),
        param_space={
            "model": {
                "fcnet_hiddens": tune.choice([[128, 128], [256, 256]]),
                "fcnet_activation": tune.choice(["relu"]),
            },
            "lr": tune.choice([1e-4, 5e-4, 1e-3]),
            "train_batch_size": tune.choice([500, 1000, 2000]),
            "sgd_minibatch_size": tune.choice([32, 64, 128]),
            "num_sgd_iter": tune.choice([5, 10]),
            "training_iteration": tune.choice([500, 800, 1000, 1500]),
        },
        run_config=ray.air.RunConfig(
            storage_path=os.path.join(os.getcwd(), "..", "results"),
        ),
    )

    result = tuner.fit()

    best_result = result.get_best_result(metric="mean_episode_reward", mode="max")
    best_config = best_result.config
    print(f"Best trial config: {pretty_print(best_config)}")

    ray.shutdown()
