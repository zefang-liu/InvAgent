"""
Fixed Policy Baselines
"""

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from config import env_configs
from env import env_creator


class FixedPolicy(Policy):
    """
    Fixed Policy
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.env = env_creator(config['env_config'])
        self.observation_space = self.env.agent_observation_space
        self.action_space = self.env.agent_action_space

    def compute_actions(
        self, obs_batch, state_batches, prev_action_batch=None, prev_reward_batch=None, info_batch=None,
        episodes=None, **kwargs):
        decoded_obs = [self.decode_obs(obs) for obs in obs_batch]
        parsed_obs = [self.env._parse_state(obs) for obs in decoded_obs]
        actions = [self.get_fixed_inventory_action(obs, policy_name='sale') for obs in parsed_obs]
        return actions, [], {}

    def decode_obs(self, obs):
        """
        Decode the one-hot encoded observation back to the original MultiDiscrete format

        :param obs: observation
        :return: decoded observation
        """
        original_obs = []
        index = 0
        for size in self.observation_space.nvec:
            one_hot_vector = obs[index:index + size]
            value = np.argmax(one_hot_vector)
            original_obs.append(value)
            index += size
        return np.array(original_obs)

    def get_fixed_inventory_action(self, obs: dict, policy_name: str, safety_ratio: float = 1.5) -> int:
        """
        Get the action from the fixed inventory policy

        :param obs: parsed observation
        :param policy_name: policy name
        :param safety_ratio: safety ratio for variability in demand and supply
        :return: action
        """
        if policy_name == 'production':
            desired_inventory = int(obs['prod_capacity'])
        elif policy_name == 'sale':
            desired_inventory = int(np.mean(obs['sales']) * obs['lead_time'] + obs['backlog'])
        else:
            raise KeyError(f'Unknown policy name: {policy_name}')
        action = desired_inventory - (obs['inventory'] + obs['upstream_backlog'] + np.sum(obs['deliveries']))
        action = min(max(0, action), self.env.max_production)
        return action

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


def evaluate_policy(env_config_name: str, env_config: dict, num_episodes: int = 10):
    """
    Evaluate a policy
    """
    ray.init()
    register_env("InventoryManagementEnv", env_creator)
    np.random.seed(0)

    env_instance = env_creator(env_config)
    agent_observation_space = env_instance.agent_observation_space
    agent_action_space = env_instance.agent_action_space

    config = PPOConfig() \
        .environment(env="InventoryManagementEnv", env_config=env_config) \
        .framework("torch") \
        .resources(num_gpus=0) \
        .env_runners(num_envs_per_env_runner=1) \
        .multi_agent(
            policies={"shared_policy": PolicySpec(
                policy_class=FixedPolicy,
                observation_space=agent_observation_space,
                action_space=agent_action_space,
            )},
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "shared_policy",
        ) \
        .training(gamma=1.0) \
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=num_episodes,
            evaluation_duration_unit="episodes",
        )

    algo = config.build()
    results = algo.evaluate()
    episode_rewards = results['env_runners']['hist_stats']['episode_reward']
    episode_reward_mean = np.mean(episode_rewards)
    episode_reward_std = np.std(episode_rewards)
    print(f"env_config_name = {env_config_name}, num_episodes = {num_episodes}, "
          f"episode_reward_mean = {episode_reward_mean:.2f}, episode_reward_std = {episode_reward_std:.2f}")

    ray.shutdown()


if __name__ == '__main__':
    for env_config_name, env_config in env_configs.items():
        evaluate_policy(env_config_name, env_config, num_episodes=100)
