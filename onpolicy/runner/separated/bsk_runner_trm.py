import time
import numpy as np
import torch
from onpolicy.runner.separated.base_runner import Runner
from collections import defaultdict
from itertools import chain
import pdb


def _t2n(x):
    return x.detach().cpu().numpy()


class BSKRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the BSK-RL. See parent class for details."""

    def __init__(self, config):
        super(BSKRunner, self).__init__(config)

    def run(self):
        # self.warmup()

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

        episode = 0
        n_step = 0
        train_interval = 2
        # for episode in range(episodes):
        while n_step < self.num_env_steps:
            self.warmup()
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].policy.lr_decay(
                    n_step, self.num_env_steps)

            # Track total actions for percentage calculation
            total_actions = 0
            battery_usage = {sat: [] for sat in satellite_names}
            memory_usage = {sat: [] for sat in satellite_names}

            # Initialize the nested dictionary
            action_frequencies = {
                sat: {action: 0 for action in action_names} for sat in satellite_names}
            epsiode_length = 0
            # battery_total_charge_amount = {sat: 0 for sat in self.envs.satellite_names}

            dones = np.array([False] * self.all_args.n_satellites)
            step = 0
            score = 0
            epsiode_length = 0
            # for step in range(self.episode_length):
            while not dones.any():
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                # pdb.set_trace()
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                score += rewards

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
                step += 1
                epsiode_length += self.n_rollout_threads
                n_step += self.n_rollout_threads

                if step >= self.episode_length:
                    break

            # compute return and update network
            if episode % train_interval == 0:
                self.compute()
                train_infos = self.train()

            # post process
            # total_num_steps = (episode + 1) * \
            #     self.episode_length * self.n_rollout_threads
            total_num_steps = n_step

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
                    env_infos[f'Episode length'] = [
                        (1-obs[0][0][-1].item())*100]

                    # Print action counts for each satellite
                    print(f'Episode {episode} - Action Counts per Satellite:')
                    for sat in satellite_names:
                        print(f'  {sat}:')
                        for action, count in action_frequencies[sat].items():
                            print(f'    {action}: {count} times')

                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update(
                            {"average_episode_rewards": np.mean(score)})

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                print(f'action taken in this episode: {epsiode_length}')
                print("average episode rewards is {}".format(np.mean(score)))
                print(
                    f'epsiode length or terminated at : {1-obs[0][0][-1].item()*100} of full duration')

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            episode += 1

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(
                list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(
                        self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate(
                            (action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                # action_env = np.squeeze(
                #     np.eye(self.envs.action_space[agent_id].n)[action], 1)
                # pdb.set_trace()
                action_env = [action[idx]
                              for idx in range(self.n_rollout_threads)]
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                   self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads,
                             self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:,
                                                                                                agent_id],
                                                                                eval_masks[:,
                                                                                           agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(
                            self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate(
                                (eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    # eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                    eval_action_env = eval_action
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
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

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(
                np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append(
                {'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " %
                  agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

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

        n_step = 0

        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents,
                                  self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            # Track total actions for percentage calculation
            total_actions = 0
            battery_usage = {sat: [] for sat in satellite_names}
            memory_usage = {sat: [] for sat in satellite_names}

            # Initialize the nested dictionary
            action_frequencies = {
                sat: {action: 0 for action in action_names} for sat in satellite_names}
            epsiode_length = 0
            dones = np.array([False] * self.all_args.n_satellites)
            score = 0
            step = 0
            while not dones.any():
                calc_start = time.time()

                actions = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                          rnn_states[:,
                                                                                     agent_id], masks[:, agent_id],
                                                                          deterministic=True)
                    action = _t2n(action)
                    actions.append(action)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                actions = np.array(actions).transpose(1, 0, 2)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32)

                score += rewards

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
                step += 1
                epsiode_length += self.n_rollout_threads
                n_step += self.n_rollout_threads

                if step >= self.episode_length:
                    break

            print(f'action taken in this episode: {epsiode_length}')
            print("average episode rewards is {}".format(np.mean(score)))
            print(
                f'epsiode length or terminated at : {1-obs[0][0][-1].item()*100} of full duration')
