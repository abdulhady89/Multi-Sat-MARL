#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.bsk.bsk_Env import BskEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
import time
import random

"""Train script for BSK."""


def make_train_env(all_args):
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
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BSK":
                env = BskEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--cuda_device', type=int,
                        default=0, help="cuda device")
    # parser.add_argument('--num_agents', type=int,
    #                     default=4, help="number of agents")
    # parser.add_argument("--n_satellites", type=int,
    #                     default=4, help="Number of satellites in a constellation.")
    parser.add_argument("--share_reward", type=bool,
                        default=True,
                        help="by default true. If false, use different reward for each agent.")
    parser.add_argument("--uniform_targets", type=int,
                        default=500, help="Number of uniform targets.")
    parser.add_argument("--n_act_image", type=int, default=2,
                        help="Number of action images.")
    parser.add_argument("--n_obs_image", type=int, default=2,
                        help="Number of observation images.")
    parser.add_argument("--orbit_num", type=int, default=2,
                        help="Number of orbits.")
    # parser.add_argument("--battery_capacity", type=int, default=400,
    #                     help="Battery capacity size of the satellite (Wh).")
    parser.add_argument("--init_battery_level", type=float, default=100.0,
                        help="Initial stored battery level in percentage")
    # parser.add_argument("--memory_size", type=int, default=5e5,
    #                     help="Memory size of the satellite (Mbyte).")
    parser.add_argument("--init_memory_percent", type=float, default=0.0,
                        help="Inital memory free space in percentage")
    # parser.add_argument("--baud_rate", type=int, default=4.3,
    #                     help="control baud rate of S-Band sattellite comm. in Mbps")
    # parser.add_argument("--instr_baud_rate", type=int, default=500,
    #                     help="control baud rate of sattellite scanning instrument")
    parser.add_argument("--failure_penalty", type=float,
                        default=-100.0, help="")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "c_ppo":
        print("u are choosing to use mappo or c_ppo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
        all_args.diversity_coef = 0.0
    elif all_args.algorithm_name == "happo":
        # can or cannot use recurrent network?
        print("using", all_args.algorithm_name, 'without recurrent network')
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        all_args.share_policy = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:"+str(all_args.cuda_device))
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
                   0] + "/new_results") / all_args.scenario_name / all_args.constellation_type / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        random_suffix = random.randint(1000, 9999)
        curr_run = (f'run_trm{all_args.experiment_name}'
                    f'_tgt{all_args.uniform_targets}'
                    f'batt{all_args.battery_capacity}'
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
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
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
        from onpolicy.runner.shared.bsk_runner import BSKRunner as Runner
    else:
        from onpolicy.runner.separated.bsk_runner import BSKRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
