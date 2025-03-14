import random
import numpy as np
from onpolicy.envs.bsk.make_bsk_env import make_BSK_Cluster_env, make_BSK_Walker_env
from gym import spaces


class BskEnv(object):
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.algorithm_name = args.algorithm_name
        self.constellation = args.constellation_type
        self.satellite_names = []
        for i in range(args.n_satellites):
            self.satellite_names.append(f"Satellite{i}")

        if self.constellation == "Cluster":
            self.env = make_BSK_Cluster_env(args, self.satellite_names)
        elif self.constellation == "Walker":
            self.env = make_BSK_Walker_env(args, self.satellite_names)
        else:
            NotImplementedError
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.num_agents == 1:
            self.action_space.append(self.env.action_space[0])
            self.observation_space.append(self.env.observation_space[0])
            self.share_observation_space.append(self.env.observation_space[0])
        else:

            for idx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[idx].n
                ))
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space[idx].low,
                    high=self.env.observation_space[idx].high,
                    shape=self.env.observation_space[idx].shape,
                    dtype=self.env.observation_space[idx].dtype
                ))
                # self.share_observation_space.append(spaces.Box(
                #     low=self.env.observation_space[idx].low,
                #     high=self.env.observation_space[idx].high,
                #     shape=self.env.observation_space[idx].shape,
                #     dtype=self.env.observation_space[idx].dtype
                # ))
            share_obs_dim = [self.env.observation_space[0].shape[0] *
                             self.num_agents]

            self.share_observation_space = [spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim), dtype=self.env.observation_space[0].dtype) for _ in range(self.num_agents)]

    def reset(self):
        obs_all = self.env.reset()
        obs = self._obs_wrapper(obs_all[0])

        return obs

    def step(self, action):
        # if self.algorithm_name == "c_ppo":
        #     obs, reward, terminated, truncated, info = self.env.step(
        #         tuple(action))
        # else:
        obs, reward, terminated, truncated, info = self.env.step(
            action.reshape(-1))
        done = False
        if terminated or truncated:
            done = True

        obs = self._obs_wrapper(obs)
        # reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            # global_reward = np.sum(reward)
            global_reward = reward
            reward = [[global_reward]] * self.num_agents

        done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)

        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[0][np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info):

        return info
