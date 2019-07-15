
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
import random
import time
import argparse
import ray
from ray.tune import  run_experiments
from ray.tune.schedulers import PopulationBasedTraining
import ray.tune as tune

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=4)  # force pausing to happen for test
    else:
        ray.init()


    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_max",
        perturbation_interval=15,
        hyperparam_mutations={
        "train_batch_size":lambda: random.uniform(64, 150),
        # "exploration_final_eps": lambda: random.uniform(0.01,0.05),
        "lr":lambda: random.uniform(0.00001,0.000001),
        "target_network_update_freq": lambda: random.uniform(500,7000),
        # "hiddens": [[512, 256], [512, 512]]
        },

    )

    run_experiments(

        {
            "pbt": {
                "run": 'DQN',
                "env":'ECglass-v0',
                "stop": {
                    "timesteps_total": 4000000,
                    # "training_iteration": 1 if args.smoke_test else 99999
                },
                "num_samples": 5,

                "resources_per_trial": {
                    "cpu": 0.7,
                    "gpu": 0.15
                },
                "config": {
                    # "hiddens":tune.grid_search([[ 512, 256],[512,512]]),
                    "hiddens":[ 512, 256],
                    "learning_starts": 64,
                    "buffer_size": 1000000,
                    "exploration_fraction": 1,
                    "train_batch_size":64,
                    #"train_batch_size":tune.grid_search([64,100,150]),
                    "gamma":0,
                    # "exploration_final_eps":tune.grid_search([0.03,0.04,0.05]),
                    "exploration_final_eps": 0.05,
                    #"num_workers": 2,
                    # "lr": tune.grid_search([0.00001,0.000001]),
                    "lr":0.00001,
                    # "target_network_update_freq":tune.grid_search([500,3499,7000]),
                    "target_network_update_freq":500,
                    "timesteps_per_iteration": 3499,
                    "schedule_max_timesteps": 3000000

                },
                "checkpoint_freq": 5,
                "local_dir": "/media/raghu/6A3A-B7CD/ray_results/"
            }
        },
        scheduler=pbt)