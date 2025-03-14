#!/bin/sh
env="BSK"
scenario="default" #"ideal","limited","random"
constellation="Cluster" #"Cluster" or "Walker"
num_agents=4
algo="mappo" #"mappo" "rmappo" "ippo"
exp="render"
seed_max=1


echo "env is ${env}, scenario is ${scenario}, constellation is ${constellation}, ${num_agents} sats, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../render/render_bsk.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --constellation_type ${constellation} --num_agents ${num_agents} --n_satellites ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 400 --num_env_steps 50000 \
    --use_value_active_masks True --eval_episodes 10 --share_policy --n_act_image ${num_agents} --n_obs_image ${num_agents} --use_linear_lr_decay \
    --model_dir "../results/BSK/Cluster/default/mappo/run_res_sat4_tgt2000_batt400_mem500000.0_baud4.3_seed3_2025-02-25_20-43-40_3947/models"
done