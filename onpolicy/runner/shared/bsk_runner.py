import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from collections import defaultdict
import pdb


def _t2n(x):
    return x.detach().cpu().numpy()


class BSKRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the BSK-RL. See parent class for details."""

    def __init__(self, config):
        super(BSKRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # Satellite names
        satellite_names = []
        for i in range(self.all_args.n_satellites):
            satellite_names.append(f"Satellite{i}")
        # Action names
        action_names = {}
        for i in range(self.all_args.n_act_image):
            action_names[i] = f"Image_Target_{i}"

        action_names[i + 1] = "Charge"
        action_names[i + 2] = "Downlink"
        action_names[i + 3] = "Desaturate"

        for episode in range(episodes):
            self.warmup()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # Track total actions for percentage calculation
            total_actions = 0
            battery_usage = {sat: [] for sat in satellite_names}
            memory_usage = {sat: [] for sat in satellite_names}

            # Initialize the nested dictionary
            action_frequencies = {
                sat: {action: 0 for action in action_names} for sat in satellite_names}
            epsiode_length = 0
            # battery_total_charge_amount = {sat: 0 for sat in self.envs.satellite_names}

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                # Track action counts for each satellite
                for k in range(self.n_rollout_threads):
                    for index, sat in enumerate(satellite_names):
                        action_frequencies[sat][actions[k][index][0]] += 1
                    total_actions += 1

                # Record battery and memory usage for each satellite
                for k in range(self.n_rollout_threads):
                    for i, sat in enumerate(satellite_names):
                        # Track battery for satellite `sat`
                        battery_usage[sat].append(obs[k][i][1].item())
                        # Track memory for satellite `sat`
                        memory_usage[sat].append(obs[k][i][0].item())

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads
            # total_num_steps = n_step

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, Speed {} steps/s.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "BSK":
                    env_infos = {}
                    # Log battery charge / action usage percentage for each satellite
                    for sat in satellite_names:
                        # Total actions for this satellite
                        sat_total_actions = sum(
                            action_frequencies[sat].values())
                        for action, count in action_frequencies[sat].items():
                            action_percentage = (count / sat_total_actions) * \
                                100 if sat_total_actions > 0 else 0
                            env_infos[f'{sat}/Action_{action_names[action]}_Usage'] = [
                                action_percentage]*self.n_rollout_threads

                        # mean Battery and Memory usage
                        mean_battery_usage = 1 - \
                            np.mean(battery_usage[sat]
                                    ) if battery_usage[sat] else 0
                        mean_memory_usage = np.mean(
                            memory_usage[sat]) if memory_usage[sat] else 0

                        env_infos[f'{sat}/mean_Battery_Usage'] = [
                            mean_battery_usage]*self.n_rollout_threads
                        env_infos[f'{sat}/mean_Memory_Usage'] = [
                            mean_memory_usage]*self.n_rollout_threads

                    # Print action counts for each satellite
                    print(f'Episode {episode} - Action Counts per Satellite:')
                    for sat in satellite_names:
                        print(f'  {sat}:')
                        for action, count in action_frequencies[sat].items():
                            print(f'    {action}: {count} times')
                # pdb.set_trace()
                train_infos["average_episode_rewards"] = np.mean(
                    self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(
                    train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            episode += 1

    def warmup(self):
        # reset env
        obs_all = self.envs.reset()
        obs = obs_all

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1)

        else:
            share_obs = obs
        # share_obs = obs
        # pdb.set_trace()
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(
                                                  self.buffer.obs[step]),
                                              np.concatenate(
                                                  self.buffer.rnn_states[step]),
                                              np.concatenate(
                                                  self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(
                    self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate(
                        (actions_env, uc_actions_env), axis=2)
            # actions_env = actions.reshape(len(actions),1,len(actions))

        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            # actions_env = np.squeeze(
            #     np.eye(self.envs.action_space[0].n)[actions], 2)
            # actions_env = actions.reshape(1, len(actions), len(actions))
            actions_env = [actions[idx, :, 0]
                           for idx in range(self.n_rollout_threads)]
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(
        ), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1)

        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads,
                             self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(
                                                                       eval_rnn_states),
                                                                   np.concatenate(
                                                                       eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                # eval_actions_env = np.squeeze(
                #     np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                eval_actions_env = eval_actions
            else:
                raise NotImplementedError

            # Obser reward and next obs
            # import pdb
            # pdb.set_trace()
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(
            np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(
            eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " +
              str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)
