from util import DiscreteGymMDP, ContinuousGymMDP, simulate
from submission import (
    ModelBasedMonteCarlo,
    TabularQLearning,
    FunctionApproxQLearning,
    ConstrainedQLearning,
    fourier_feature_extractor,
)

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import sys, argparse, random, json


def moving_average(x, window):
    cum_sum = np.cumsum(x)
    ma = (cum_sum[window:] - cum_sum[:-window]) / window
    return ma


def plot_rewards(train_rewards, eval_rewards, save_path=None, show=True):
    plt.figure(figsize=(10, 5))
    window = 30
    train_ma = moving_average(train_rewards, window)
    eval_ma = moving_average(eval_rewards, window)
    t_len = len(train_rewards)
    e_len = len(eval_rewards)
    plt.scatter(range(t_len), train_rewards, alpha=0.5, c='tab:blue', linewidth=0, s=5)
    plt.plot(range(int(window / 2), t_len - int(window / 2)), train_ma, lw=2, c='b')
    plt.scatter(range(t_len, t_len + e_len), eval_rewards, alpha=0.5, c='tab:green', linewidth=0, s=5)
    plt.plot(range(t_len + int(window / 2), t_len + e_len - int(window / 2)), eval_ma, lw=2, c='darkgreen')
    plt.legend(['train rewards', 'train moving average', 'eval rewards', 'eval moving average'])
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward in Episode")

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


if __name__ == "__main__":
    """
    The main function called when train.py is run
    from the command line:

    > python train.py

    See the usage string for more details.

    > python train.py --help
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["value-iteration", "tabular", "function-approximation", "constrained"],
        help='model-based value iteration ("value-iteration"), tabular Q-learning ("tabular"),             function approximation Q-learning ("function-approximation"), or constrained Q-Learning ("constrained")',
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        help="Max speed constraint that only applies when doing function approximation",
    )
    args = parser.parse_args()

    if args.agent == "value-iteration":
        print("************************************************")
        print("Training agent with model-based value iteration to perform mountain car task!")
        print("************************************************")
        for i in range(1, 4):
            print("********************************************************")
            print(f"Trial {i} out of 3")
            print("********************************************************")
            mdp = DiscreteGymMDP(
                "MountainCar-v0",
                discount=0.999,
                low=[-1.2, -0.07],
                high=[0.6, 0.07],
                feature_bins=20,
                time_limit=1000,
            )
            rl = ModelBasedMonteCarlo(
                mdp.actions,
                mdp.discount,
                mdp.num_states,
                mdp.state_to_index,
                mdp.index_to_state,
                calc_val_iter_every=1e5,
                exploration_prob=0.50,
            )
            train_rewards = simulate(mdp, rl, train=True, num_trials=1000, verbose=True)
            print("Training complete! Running evaluation, writing weights to mcvi_weights.json and generating reward plot...")
            eval_rewards = simulate(mdp, rl, train=False, num_trials=500)

            serial_data = {
                str(idx): int(action)
                for idx, action in enumerate(rl.pi_actions)
                if action is not None
            }
            with open("mcvi_weights.json", "w") as f:
                json.dump(serial_data, f)
            plot_rewards(train_rewards, eval_rewards, f'mcvi_{i}.png')

    # Trained Discrete Agent
    elif args.agent == "tabular":
        print("********************************************************")
        print("Training agent with Tabular Q-Learning to perform mountain car task!")
        print("********************************************************")
        for i in range(1, 4):
            print("********************************************************")
            print(f"Trial {i} out of 3")
            print("********************************************************")
            mdp = DiscreteGymMDP(
                "MountainCar-v0",
                discount=0.999,
                low=[-1.2, -0.07],
                high=[0.6, 0.07],
                feature_bins=20,
                time_limit=1000,
            )
            rl = TabularQLearning(
                mdp.actions,
                mdp.discount,
                mdp.num_states,
                mdp.state_to_index,
                exploration_prob=0.15,
            )
            train_rewards = simulate(mdp, rl, train=True, num_trials=1000, verbose=True)
            print("Training complete! Running evaluation, writing weights to tabular_weights.npy and generating reward plot...")
            eval_rewards = simulate(mdp, rl, train=False, num_trials=500)
            np.save("tabular_weights", rl.q)

            plot_rewards(train_rewards, eval_rewards, f'tabular_{i}.png')

    # Trained Continuous Agent
    elif args.agent == "function-approximation":
        print("********************************************************")
        print(
            "Training agent with Function Approximation Q-Learning to perform mountain car task!"
        )
        print("********************************************************")
        for i in range(1, 4):
            print("********************************************************")
            print(f"Trial {i} out of 3")
            print("********************************************************")
            if args.max_speed is not None:
                gym.register(
                    id="CustomMountainCar-v0",
                    entry_point="custom_mountain_car:CustomMountainCarEnv",
                    max_episode_steps=1000,
                    reward_threshold=-110.0,
                )
                mdp = ContinuousGymMDP(
                    "CustomMountainCar-v0",
                    max_speed=args.max_speed,
                    discount=0.999,
                    time_limit=1000,
                )
            else:
                mdp = ContinuousGymMDP("MountainCar-v0", discount=0.999, time_limit=1000)
            rl = FunctionApproxQLearning(
                36,
                lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
                mdp.actions,
                mdp.discount,
                exploration_prob=0.2,
            )
            train_rewards = simulate(mdp, rl, train=True, num_trials=1000, verbose=True)
            print("Training complete! Running evaluation, writing weights to fapprox_weights.npy and generating reward plot...")
            eval_rewards = simulate(mdp, rl, train=False, num_trials=500)
            np.save("fapprox_weights", rl.w)

            plot_rewards(train_rewards, eval_rewards, f'fapprox_{i}.png')

    elif args.agent == "constrained":
        print("********************************************************")
        print(
            "Training agent with Constrained Q-Learning to perform mountain car task!"
        )
        print("********************************************************")
        gym.register(
            id="CustomMountainCar-v0",
            entry_point="custom_mountain_car:CustomMountainCarEnv",
            max_episode_steps=1000,
            reward_threshold=-110.0,
        )
        mdp = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, time_limit=1000)
        rl = ConstrainedQLearning(
            36,
            lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            mdp.env.force,
            mdp.env.gravity,
            exploration_prob=0.2,
        )
        train_rewards = simulate(mdp, rl, train=True, num_trials=1000, verbose=True)
        print("Training complete! Running evaluation, writing weights to constrainted_weights.npy and generating reward plot...")
        eval_rewards = simulate(mdp, rl, train=False, num_trials=500)
        np.save("constrained_weights", rl.w)

        plot_rewards(train_rewards, eval_rewards, 'constrained.png')
