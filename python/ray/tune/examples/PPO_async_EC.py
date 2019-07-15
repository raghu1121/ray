
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
import random
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
        grace_period=10,
        max_t=20000,

    )

    run_experiments(
        {
            "asynchyperband_ppo": {
                "run": 'PPO',
                "env":'ECglass-v0',
                "stop": {
                    "timesteps_total": 4000000,

                },
                "num_samples": 5,

                "resources_per_trial": {
                    "cpu": 0.7,
                    "gpu": 0.15
                },
                "config": {
                    "gamma": 0,
                    "kl_coeff": 1.0,
                    "num_workers": 1,
                    "num_gpus": 1,
                    "model": {
                        "free_log_std": True
                    },

                    "lambda": lambda spec:  random.uniform(0.9, 1.0),
                    "clip_param": lambda spec: random.uniform(0.01, 0.5),
                    "lr": lambda spec: random.uniform(5e-4, 1e-6),
                    "num_sgd_iter": lambda spec:  random.randint(1, 30),
                    "sgd_minibatch_size": lambda spec:  random.randint(128, 16384),
                    "train_batch_size":lambda spec: random.randint(2000, 160000),


                },
                "checkpoint_freq": 5,
                "local_dir": "/media/raghu/6A3A-B7CD/ray_results"
            }
        },
        scheduler=ahb)