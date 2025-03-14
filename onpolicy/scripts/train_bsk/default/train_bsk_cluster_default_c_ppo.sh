#!/bin/sh
env="BSK"
scenario="hetero_batt" #"ideal","limited","random"
constellation="Cluster" #"Cluster" or "Walker"
num_agents=4
algo="c_ppo" #"mappo" "rmappo" "ippo"
exp="res"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, constellation is ${constellation}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python ../../train/train_bsk_trm.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --constellation_type ${constellation} --num_agents ${num_agents} --n_satellites ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 10 --episode_length 400 --num_env_steps 50000 \
    --ppo_epoch 20 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "xxx" \
    --use_value_active_masks False --eval_episodes 10 --n_act_image ${num_agents} --n_obs_image ${num_agents} --use_linear_lr_decay
done