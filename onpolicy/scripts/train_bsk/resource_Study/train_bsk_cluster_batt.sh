#!/bin/sh
env="BSK"
scenario="default" #"ideal","limited","random" "heterogeneous"
constellation="Cluster" #"Cluster" or "Walker"
num_agents=4
algo="happo" #"mappo" "ippo" "happo"
exp="batt" #"mem" "baud"
seed_max=3
# baud_list=(1.0 2.0)


echo "env is ${env}, scenario is ${scenario}, constellation is ${constellation}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

# for baud in {1.0..2.0}; 
for batt in 25 50 100; 
do
    for seed in `seq ${seed_max}`;
    do
        echo "seed is ${seed}:, batt is ${batt}"
        CUDA_VISIBLE_DEVICES=0 python ../../train/train_bsk.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
        --scenario_name ${scenario} --constellation_type ${constellation} --num_agents ${num_agents} --seed ${seed} --battery_capacity ${batt}\
        --n_training_threads 4 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps 100000 \
        --ppo_epoch 5 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "xxx" \
        --use_value_active_masks True --use_eval False --eval_episodes 10
    done
done

wait