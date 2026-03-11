from util import DiscreteGymMDP, ContinuousGymMDP, RandomAgent, simulate
from submission import (
    ModelBasedMonteCarlo,
    TabularQLearning,
    FunctionApproxQLearning,
    ConstrainedQLearning,
    fourier_feature_extractor,
)
import numpy as np
import gymnasium as gym
import sys, argparse, json

if __name__ == "__main__":
    """
    The main function called when mountaincar.py is run
    from the command line:

    > python mountaincar.py

    See the usage string for more details.

    > python mountaincar.py --help
    """
    # play.play(gym.make("MountainCar-v0", render_mode="human"), zoom=3)
    # TODO: Implement interactive mode for human play
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["naive", "value-iteration", "tabular", "function-approximation", "constrained"],
        help='naive ("naive"), model-based value iteration ("value-iteration"), tabular Q-learning ("tabular"),             function approximation Q-learning ("function-approximation"), or constrained Q-Learning ("constrained")',
    )
    args = parser.parse_args()

    # Naive Agent
    if args.agent == "naive":
        print("************************************************")
        print("Naive agent performing mountain car task!")
        print("************************************************")
        mdp = DiscreteGymMDP("MountainCar-v0", discount=0.999, time_limit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = RandomAgent(mdp.actions)
        simulate(mdp, rl, train=False, num_trials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Model-Based Value Iteration
    elif args.agent == "value-iteration":
        print("********************************************************")
        print("Agent trained with model-based value iteration performing mountain car task!")
        print("********************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            time_limit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = ModelBasedMonteCarlo(
            mdp.actions,
            mdp.discount,
            mdp.num_states,
            mdp.state_to_index,
            mdp.index_to_state,
            calc_val_iter_every=1e5,
            exploration_prob=0.15,
        )
        with open("mcvi_weights.json", "r") as f:
            data = json.load(f)
        rl.pi_actions = np.full(mdp.num_states, None, dtype=object)
        for key, action in data.items():
            try:
                state_idx = int(key)
            except ValueError:
                cleaned = key.strip()
                if cleaned.startswith("[") and cleaned.endswith("]"):
                    tokens = [int(tok) for tok in cleaned[1:-1].split()]
                    if len(tokens) == 1:
                        state_idx = mdp.state_to_index(tokens[0])
                    else:
                        state_idx = mdp.state_to_index(tuple(tokens))
                else:
                    raise
            rl.pi_actions[state_idx] = int(action)
        rl._sync_policy_indices()
        simulate(mdp, rl, train=False, num_trials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Tabular Q-Learning
    elif args.agent == "tabular":
        print("********************************************************")
        print("Agent trained with Tabular Q-Learning performing mountain car task!")
        print("********************************************************")
        mdp = DiscreteGymMDP(
            "MountainCar-v0",
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            feature_bins=20,
            time_limit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = TabularQLearning(
            mdp.actions,
            mdp.discount,
            mdp.num_states,
            mdp.state_to_index,
            exploration_prob=0.15,
        )
        loaded_q = np.load("tabular_weights.npy", allow_pickle=True)
        if loaded_q.dtype == object:
            legacy = loaded_q.item()
            rl.q = np.zeros((mdp.num_states, len(mdp.actions)))
            for (state, action), value in legacy.items():
                state_idx = mdp.state_to_index(state)
                action_idx = mdp.actions.index(action)
                rl.q[state_idx, action_idx] = value
        else:
            rl.q = loaded_q
        simulate(mdp, rl, train=False, num_trials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Function Approx Q-Learning
    elif args.agent == "function-approximation":
        print("********************************************************")
        print(
            "Agent trained with Function Approximation Q-Learning performing mountain car task!"
        )
        print("********************************************************")
        mdp = ContinuousGymMDP("MountainCar-v0", discount=0.999, time_limit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = FunctionApproxQLearning(
            36,
            lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            exploration_prob=0.2,
        )
        rl.w = np.load("fapprox_weights.npy", allow_pickle=True)
        simulate(mdp, rl, train=False, num_trials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Constrained Q-Learning
    elif args.agent == "constrained":
        print("********************************************************")
        print("Agent trained with Constrained Q-Learning performing mountain car task!")
        print("********************************************************")
        mdp = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, time_limit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = ConstrainedQLearning(
            36,
            lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
            mdp.actions,
            mdp.discount,
            mdp.env.force,
            mdp.env.gravity,
            exploration_prob=0.2,
        )
        rl.w = np.load("constrained_weights.npy", allow_pickle=True)
        simulate(mdp, rl, train=False, num_trials=1, verbose=False, demo=True)
        mdp.env.close()
