
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ray
from ray.tune import grid_search, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler


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
        reward_attr="episode_reward_mean",
        grace_period=5,
        max_t=100)

    run_experiments(
        {
            "asynchyperband_test": {
                "run": 'DQN',
                "env":'ECglass-v0',
                "stop": {
                    "training_iteration": 1 if args.smoke_test else 99999
                },
                "num_samples": 5,

                "resources_per_trial": {
                    "cpu": 1,
                    "gpu": 1
                },
                "config": {
                    "lr": grid_search([1e-3, 5e-4])

                },
            }
        },
        scheduler=ahb)