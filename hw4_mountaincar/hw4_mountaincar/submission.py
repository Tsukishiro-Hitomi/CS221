import math, random
from typing import List, Callable, Tuple, Any, Optional, Iterable
from einops import reduce, rearrange, einsum
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def value_iteration(
        transitions: np.ndarray,
        rewards: np.ndarray,
        discount: float,
        epsilon: float = 0.001,
        valid_actions: Optional[np.ndarray] = None,
        state_ids: Optional[Iterable[Any]] = None,
        action_ids: Optional[Iterable[Any]] = None,
    ):
    """
    Given transition probabilities and rewards, computes and returns the optimal policy.
    - transitions: np.ndarray with shape (num_states, num_actions, num_states)
    - rewards: np.ndarray with the same shape as transitions
    - epsilon: repeatedly update v until the v values for all states do not change by more than epsilon.
    - valid_actions: optional boolean mask of shape (num_states, num_actions) indicating available actions
    - state_ids: optional iterable mapping each index to a state identifier
    - action_ids: optional iterable mapping each action index to an action identifier
    - Returns: np.ndarray of shape (num_states,) storing the optimal action identifier per state (or None if no action is available).
    """

    transitions = np.asarray(transitions, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    num_states, num_actions, next_state_dim = transitions.shape

    if valid_actions is None:
        action_mask = np.ones((num_states, num_actions), dtype=bool)
    else:
        action_mask = np.asarray(valid_actions, dtype=bool)

    if state_ids is None:
        state_ids = np.arange(num_states)
    else:
        state_ids = np.array(list(state_ids), dtype=object)

    if action_ids is None:
        action_ids = np.arange(num_actions)
    else:
        action_ids = np.array(list(action_ids), dtype=object)

    tie_breaker = (np.arange(num_actions, dtype=np.float64) * 1e-12)[np.newaxis, :]
    
    # if some state is unaccessible, it's corresponding transiton probability is zero.
    transitions = transitions * valid_actions[..., np.newaxis]
    expected_rewards = np.sum(transitions * rewards, axis=2)

    def compute_q(v: np.ndarray) -> np.ndarray:
        """
        Computes the Q-value table based on the estimated value function `v`.

        - Returns: An np.ndarray of shape (num_states, num_actions) containing the Q-values.
        """
        # BEGIN_YOUR_CODE (our solution is 2 line(s) of code, but don't worry if you deviate from this)
        # since expected_rewards stay the same during iteration, we should precalculate it to reduce unnecessary calculations.
        q_values = discount * np.sum(transitions * v, axis=2) + expected_rewards
        return q_values
        # END_YOUR_CODE

    def compute_policy(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the optimal policy and corresponding values from a Q-value table.
        Ties are broken by selecting the action with the largest index.

        - Returns: A tuple of two np.ndarrays:
                   1. `best_actions`: An array of shape (num_states,) with the index of the optimal action for each state.
                   2. `best_values`: An array of shape (num_states,) with the Q-value corresponding to each best action.
        """
        # BEGIN_YOUR_CODE (our solution is 4 line(s) of code, but don't worry if you deviate from this)
        best_actions = np.argmax(q + tie_breaker, axis=1)
        best_values = np.max(q, axis=1)
        return (best_actions, best_values)
        # END_YOUR_CODE

    # Implement the value iteration algorithm.
    # Your goal is to create a vectorized NumPy implementation that mirrors the logic
    # from the dictionary-based example in lecture, using the helper functions above.
    #
    # Your code should initialize a value function, loop until the values converge,
    # and then compute and return the final policy.
    #
    # HINT: Recall that the value of any terminal state is always 0. A state is
    # considered terminal if there are no valid actions you can take from it.

    print('Running value iteration...')
    # BEGIN_YOUR_CODE (our solution is 18 line(s) of code, but don't worry if you deviate from this)
    pre_values = np.zeros(num_states)
    end_states = []
    for i in range(num_states):
        if np.sum(action_mask[i]) == 0:
            end_states.append(i)

    while True:
        q_values = compute_q(pre_values)
        actions, cur_values = compute_policy(q_values)
        cur_values[end_states] = 0
        diff = np.abs(pre_values - cur_values)
        if np.max(diff) < epsilon:
            break
        pre_values = cur_values
        
    best_actions_id = action_ids[actions]
    best_actions_id[end_states] = None
    return best_actions_id
    # END_YOUR_CODE


# Runs value iteration algorithm on the number line MDP
def run_vi_over_number_line(mdp: util.NumberLineMDP):
    num_states = mdp.num_states
    actions = np.array(list(mdp.actions), dtype=object)
    transitions = np.zeros((num_states, len(actions), num_states), dtype=np.float64)
    rewards = np.zeros_like(transitions)
    valid_actions = np.zeros((num_states, len(actions)), dtype=bool)
    for state_idx, state in enumerate(mdp.indexer.all_states()):
        if state in mdp.terminal_states:
            continue
        valid_actions[state_idx, :] = True
        for action_idx, action in enumerate(actions):
            forward_prob = 0.2 if action == 1 else 0.3
            backward_prob = 1.0 - forward_prob
            forward_state = state + 1
            backward_state = state - 1
            forward_reward = mdp.right_reward if forward_state == mdp.n else (mdp.left_reward if forward_state == -mdp.n else mdp.penalty)
            backward_reward = mdp.right_reward if backward_state == mdp.n else (mdp.left_reward if backward_state == -mdp.n else mdp.penalty)
            forward_idx = mdp.state_to_index(forward_state)
            backward_idx = mdp.state_to_index(backward_state)
            transitions[state_idx, action_idx, forward_idx] += forward_prob
            rewards[state_idx, action_idx, forward_idx] = forward_reward
            transitions[state_idx, action_idx, backward_idx] += backward_prob
            rewards[state_idx, action_idx, backward_idx] = backward_reward

    state_ids = np.array([mdp.index_to_state(i) for i in range(num_states)], dtype=object)
    pi = value_iteration(
        transitions,
        rewards,
        mdp.discount,
        valid_actions=valid_actions,
        state_ids=state_ids,
        action_ids=actions,
    )
    return state_ids, pi


############################################################
# Problem 3b
# Model-Based Monte Carlo
class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(
            self,
            actions: List[ActionT],
            discount: float,
            num_states: int,
            state_to_index: Callable[[StateT], int],
            index_to_state: Optional[Callable[[int], StateT]] = None,
            calc_val_iter_every: int = 10000,
            exploration_prob: float = 0.2,
        ) -> None:
        self.actions = list(actions)
        self.discount = discount
        self.num_states = int(num_states)
        self.state_to_index = state_to_index
        self.index_to_state = index_to_state or (lambda idx: idx)
        self.calc_val_iter_every = int(calc_val_iter_every)
        self.exploration_prob = exploration_prob
        self.num_iters = 0

        self.num_actions = len(self.actions)
        self.actions_array = np.array(self.actions, dtype=object)
        self.state_ids = np.array([self.index_to_state(i) for i in range(self.num_states)], dtype=object)

        self.transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float64)
        self.reward_sums = np.zeros_like(self.transition_counts)
        self.valid_actions = np.zeros((self.num_states, self.num_actions), dtype=bool)

        self.pi_actions = np.full(self.num_states, None, dtype=object)
        self.pi_indices = np.full(self.num_states, -1, dtype=int)

    def _sync_policy_indices(self):
        self.pi_indices[:] = -1
        valid_mask = self.pi_actions != None
        for idx in np.where(valid_mask)[0]:
            self.pi_indices[idx] = int(np.argmax(self.actions_array == self.pi_actions[idx]))

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # Should return random action if the given state does not yet have an action assigned in self.pi_actions.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4:  # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e6:  # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        state_idx = int(self.state_to_index(state))
        policy_idx = self.pi_indices[state_idx]

        # BEGIN_YOUR_CODE (our solution is 5 line(s) of code, but don't worry if you deviate from this)
        if policy_idx == -1 or (explore and random.random() < exploration_prob):
            return random.choice(self.actions)
        return self.actions[policy_idx]

        # END_YOUR_CODE

    # We will call this function with (s, a, r, s'), which is used to update counts and rewards.
    # For every self.calc_val_iter_every steps, runs value iteration after estimating the transition and reward tensors.
    def incorporate_feedback(self, state: StateT, action: ActionT, reward: int, next_state: StateT, terminal: bool):
        state_idx = int(self.state_to_index(state))
        matches = np.where(self.actions_array == action)[0]
        action_idx = int(matches[0])
        next_idx = int(self.state_to_index(next_state))

        self.transition_counts[state_idx, action_idx, next_idx] += 1.0
        self.reward_sums[state_idx, action_idx, next_idx] += reward
        self.valid_actions[state_idx, action_idx] = True

        if self.num_iters > 0 and self.num_iters % self.calc_val_iter_every == 0:
            # BEGIN_YOUR_CODE (our solution is 21 line(s) of code, but don't worry if you deviate from this)
            num_transitions = np.sum(self.transition_counts, axis=2)
            num_transitions_axis = num_transitions[..., np.newaxis]
            transitions = np.ones_like(self.transition_counts)
            transitions = np.divide(self.transition_counts, num_transitions_axis, \
                                    out=np.zeros_like(self.transition_counts, dtype=float), \
                                    where=num_transitions_axis != 0)
            
            rewards = np.divide(self.reward_sums, self.transition_counts, \
                                out=np.zeros_like(self.reward_sums, dtype=float), \
                                    where=self.transition_counts != 0)
            
            self.pi_actions = value_iteration(transitions=transitions, rewards=rewards, discount=self.discount, \
                                              valid_actions=self.valid_actions, action_ids=self.actions_array) 
            
            # END_YOUR_CODE
            self._sync_policy_indices()


############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
class TabularQLearning(util.RLAlgorithm):
    def __init__(
            self,
            actions: List[ActionT],
            discount: float,
            num_states: int,
            state_to_index: Callable[[StateT], int],
            exploration_prob: float = 0.2,
            initial_q: float = 0,
        ):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - num_states: total number of discrete states available
        - state_to_index: function mapping states to contiguous indices [0, num_states)
        - exploration_prob: the epsilon value indicating how frequently the policy returns a random action
        - initial_q: the value for initialising Q values.
        '''
        self.actions = list(actions)
        self.actions_array = np.array(self.actions, dtype=object)
        self.discount = discount
        self.num_states = int(num_states)
        self.state_to_index = state_to_index
        self.exploration_prob = exploration_prob
        self.q = np.full((self.num_states, len(self.actions)), initial_q, dtype=np.float64)
        self.num_iters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT: Use random.random() to sample from the uniform distribution [0, 1]
    def get_action(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4:  # explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5:  # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)
        state_idx = int(self.state_to_index(state))
        # BEGIN_YOUR_CODE (our solution is 4 line(s) of code, but don't worry if you deviate from this)
        if random.random() > exploration_prob or explore == False:
            action_idx = np.argmax(self.q[state_idx])
            return self.actions[action_idx]
        else:
            return random.choice(self.actions)
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def get_step_size(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    def incorporate_feedback(self, state: StateT, action: ActionT, reward: float, next_state: StateT, terminal: bool) -> None:
        state_idx = int(self.state_to_index(state))
        matches = np.where(self.actions_array == action)[0]
        action_idx = int(matches[0])
        # BEGIN_YOUR_CODE (our solution is 9 line(s) of code, but don't worry if you deviate from this)
        step_size = self.get_step_size()
        if terminal == True:
            sprime_opt_value = 0 
        else:
            next_state_idx = int(self.state_to_index(next_state))
            sprime_opt_value = np.max(self.q[next_state_idx])
        self.q[state_idx][action_idx] = (1 - step_size) * self.q[state_idx][action_idx] + step_size * (reward + self.discount * sprime_opt_value)
        # END_YOUR_CODE

############################################################
# Problem 4b: Fourier feature extractor

def fourier_feature_extractor(
        state: StateT,
        max_coeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), max_coeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((max_coeff+1)^(len(state)),).
    # HINT: refer to util.polynomial_feature_extractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # BEGIN_YOUR_CODE (our solution is 7 line(s) of code, but don't worry if you deviate from this)
    cur_poly_feat = state[0] * scale[0] * (np.arange(max_coeff + 1))
    for i in range(1, len(state)):
        new_poly_feat = state[i] * scale[i] * np.arange(max_coeff + 1)
        next_poly_feat = np.add.outer(cur_poly_feat, new_poly_feat)
        cur_poly_feat = next_poly_feat.flatten()
    features = np.cos(np.pi * cur_poly_feat)
    # END_YOUR_CODE
    return features

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, feature_dim: int, feature_extractor: Callable, actions: List[int],
                 discount: float, exploration_prob=0.2):
        '''
        - feature_dim: the dimensionality of the output of the feature extractor
        - feature_extractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - exploration_prob: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.feature_dim = feature_dim
        self.feature_extractor = feature_extractor
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.w = np.random.standard_normal(size=(feature_dim, len(actions)))
        self.num_iters = 0

    # Attention: Use the index funtion to get the correct action_idx instead of action itself (though its type is int)
    def get_q(self, state: np.ndarray, action: int) -> float:
        # BEGIN_YOUR_CODE (our solution is 3 line(s) of code, but don't worry if you deviate from this)
        action_idx = self.actions.index(action)
        weight = self.w[:, action_idx]
        feature = self.feature_extractor(state)
        return np.dot(weight, feature)
        # END_YOUR_CODE

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        # BEGIN_YOUR_CODE (our solution is 5 line(s) of code, but don't worry if you deviate from this)
        if explore == False or random.random() > exploration_prob:
            feature = self.feature_extractor(state)
            out = np.dot(self.w.T, feature)
            return self.actions[np.argmax(out)]
        return random.choice(self.actions)
        # END_YOUR_CODE

    # Call this function to get the step size to update the weights.
    def get_step_size(self) -> float:
        return 0.005 * (0.99)**(self.num_iters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # HINT 1: this part will look similar to 4a, but you are updating self.w
    # HINT 2: the increment term in the update rule is
    # step_size * (new_value - old_value) * features
    def incorporate_feedback(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool) -> None:
        # BEGIN_YOUR_CODE (our solution is 10 line(s) of code, but don't worry if you deviate from this)
        feature = self.feature_extractor(state)
        next_state_feature = self.feature_extractor(next_state)
        sprime_opt_value = np.max(self.w.T.dot(next_state_feature)) if terminal == False else 0.0
        target = self.discount * sprime_opt_value + reward
        prediction = self.get_q(state, action)
        step_size = self.get_step_size()
        # Attention: Use the index funtion to get the correct action_idx instead of action itself (though its type is int)
        action_idx = self.actions.index(action)
        self.w[:, action_idx] -= step_size * (prediction - target) * feature
        # END_YOUR_CODE

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, feature_dim: int, feature_extractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 exploration_prob=0.2):
        super().__init__(feature_dim, feature_extractor, actions,
                         discount, exploration_prob)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |exploration_prob|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.num_iters += 1
        exploration_prob = self.exploration_prob
        if self.num_iters < 2e4: # Always explore
            exploration_prob = 1.0
        elif self.num_iters > 1e5: # Lower the exploration probability by a logarithmic factor.
            exploration_prob = exploration_prob / math.log(self.num_iters - 100000 + 1)

        def compute_next_velocity(state: np.ndarray, action: int) ->float:
            cur_position = state[0]
            cur_velocity = state[1]
            diff = (action - 1) * self.force - np.cos(3 * cur_position) * self.gravity
            return cur_velocity + diff

        # BEGIN_YOUR_CODE (our solution is 18 line(s) of code, but don't worry if you deviate from this)
        valid_actions = []
        for action in self.actions:
            next_velocity = compute_next_velocity(state, action)
            if abs(next_velocity) < self.max_speed:
                valid_actions.append(action)

        if len(valid_actions) == 0:
            return None

        if explore == False or random.random() > exploration_prob:
            feature = self.feature_extractor(state)
            out = np.dot(self.w.T, feature)
            valid_out = []
            for action in valid_actions:
                action_index = self.actions.index(action)
                valid_out.append(out[action_index])
            return valid_actions[np.argmax(valid_out)]
        else:
            return random.choice(valid_actions)
        # END_YOUR_CODE

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, time_limit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, time_limit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_mdp_strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.force,
        mdp1.env.gravity,
        10000,
        exploration_prob=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourier_feature_extractor(s, max_coeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.force,
        mdp2.env.gravity,
        0.065,
        exploration_prob=0.2,
    )
    sample_krl_trajectories(mdp1, rl1)
    sample_krl_trajectories(mdp2, rl2)

def sample_krl_trajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_rl_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.start_state()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")
