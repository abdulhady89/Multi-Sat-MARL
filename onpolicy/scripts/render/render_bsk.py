#!/usr/bin/env python
import sys
import os
import wandb
import socket
import random
import time
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

from onpolicy.envs.bsk.bsk_Env import BskEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BSK":
                env = BskEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--cuda_device', type=int,
                        default=0, help="cuda device")
    parser.add_argument("--uniform_targets", type=int,
                        default=2000, help="Number of uniform targets.")
    parser.add_argument("--orbit_num", type=int, default=2,
                        help="Number of orbits.")
    parser.add_argument("--init_battery_level", type=float, default=100.0,
                        help="Initial stored battery level in percentage")
    parser.add_argument("--init_memory_percent", type=float, default=0.0,
                        help="Inital memory free space in percentage")
    parser.add_argument("--failure_penalty", type=float,
                        default=-100.0, help="")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True, share_policy is ",
              all_args.share_policy)
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False, share_policy is ", all_args.share_policy)
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "c_ppo":
        print("u are choosing to use c_ppo, we set use_recurrent_policy & use_naive_recurrent_policy to be False, share_policy is ", all_args.share_policy)
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False, share_policy is ",
              all_args.share_policy)
        all_args.use_centralized_V = False
        all_args.diversity_coef = 0.0
    elif all_args.algorithm_name == "happo":
        # can or cannot use recurrent network?
        print(
            f'using happo,without recurrent network, share_policy is {all_args.share_policy}')
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    else:
        raise NotImplementedError

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir ==
                ""), ("set model_dir first")
    assert all_args.n_rollout_threads == 1, (
        "only support to use 1 env to render.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    random_suffix = random.randint(1000, 9999)
    curr_run = (f'run_{all_args.experiment_name}'
                f'_sat{all_args.num_agents}'
                f'_tgt{all_args.uniform_targets}'
                f'_batt{all_args.battery_capacity}'
                f'_mem{all_args.memory_size}'
                f'_baud{all_args.baud_rate}'
                f'_seed{all_args.seed}'
                f'_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{random_suffix}'
                )
    run_dir = run_dir / curr_run

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              #   str(all_args.env_name) + "-" +
                              str(all_args.scenario_name) + "-" +
                              #   str(all_args.constellation_type) + "-" +
                              str(all_args.experiment_name))
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        if all_args.algorithm_name == "c_ppo":
            from onpolicy.runner.shared.bsk_runner_trm_cppo import BSKRunner as Runner
        else:
            from onpolicy.runner.shared.bsk_runner_trm import BSKRunner as Runner
    else:
        from onpolicy.runner.separated.bsk_runner_trm import BSKRunner as Runner

    runner = Runner(config)
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
