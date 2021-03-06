
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
from ray.tune import grid_search, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler
import ray.tune as tune

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    ray.init()

    # asynchronous hyperband early stopping, configured with
    # `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_max",
        grace_period=2000,
        max_t=20000

    )

    run_experiments(
        {
            "asynchyperband_EC_dqn": {
                "run": 'DQN',
                "env":'ECglass-v0',
                "stop": {
                    "timesteps_total": 8000000,
                    # "training_iteration": 1 if args.smoke_test else 99999
                },
                "num_samples": 5,

                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 0.2
                },
                "config": {
                    "hiddens":tune.grid_search([[1024,512]]),

                    "learning_starts": 64,
                    "buffer_size": 1000000,
                    "exploration_fraction": 1,
                    "train_batch_size":tune.grid_search([150,200,250]),
                    "gamma":0,
                    "exploration_final_eps":tune.grid_search([0.05,0.03]),
                    #"exploration_final_eps": 0.05,
                    #"num_workers": 2,
                    "lr": tune.grid_search([0.00001,0.00005]),
                    "target_network_update_freq":tune.grid_search([3500,7000,10500]),
                    "timesteps_per_iteration": 3499,
                    "schedule_max_timesteps": 7000000

                },
                "checkpoint_freq": 1,
                "local_dir": "/media/raghu/6A3A-B7CD/ray_results"
            }
        },
        scheduler=ahb)