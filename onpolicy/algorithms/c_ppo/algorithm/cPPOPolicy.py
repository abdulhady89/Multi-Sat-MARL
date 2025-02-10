import torch
from onpolicy.algorithms.c_ppo.algorithm.c_r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule


class cPPOPolicy:
    """
    Centralized Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for PPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads

        # self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.share_obs_space,
                             self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer,
                               episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer,
                               episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        selected_actions = []
        selected_log_probs = []

        for agent_id in range(self.num_agents):
            actions, action_log_probs, rnn_states_actor = self.actor(agent_id, cent_obs,
                                                                     rnn_states_actor,
                                                                     masks,
                                                                     available_actions,
                                                                     deterministic)

            for i in range(self.n_rollout_threads):
                selected_actions.append([actions[i+agent_id]])
                selected_log_probs.append([action_log_probs[i+agent_id]])

        cppo_actions = torch.tensor(selected_actions, device=self.device)
        cppo_log_probs = torch.tensor(selected_log_probs, device=self.device)

        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return values, cppo_actions, cppo_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        dist_entropies = []
        selected_log_probs = []
        action_logits_ls = []

        for agent_id in range(self.num_agents):
            action_log_probs, dist_entropy, action_logits = self.actor.evaluate_actions(agent_id, cent_obs,
                                                                                        rnn_states_actor,
                                                                                        action,
                                                                                        masks,
                                                                                        available_actions,
                                                                                        active_masks)
            # for i in range(self.n_rollout_threads):
            selected_log_probs.append(action_log_probs)
            dist_entropies.append(dist_entropy)
            action_logits_ls.append(action_logits)

        cppo_log_probs = torch.cat(
            selected_log_probs, dim=-1).mean(dim=-1).unsqueeze(1)
        cppo_dist_entropy = torch.stack(dist_entropies).mean()

        cppo_action_logits = action_logits_ls

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)

        return values, cppo_log_probs, cppo_dist_entropy, cppo_action_logits

    def act(self, cent_obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            cent_obs, rnn_states_actor, masks, available_actions, deterministic)

        return actions, rnn_states_actor
