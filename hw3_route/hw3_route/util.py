import heapq
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional

########################################################################################
# Abstract Interfaces for State, Search Problems, and Search Algorithms.


@dataclass(frozen=True, order=True)
class State:
    """
    A State consists of a string `location` and (possibly null) `memory`.
    Note that `memory` must be a "Hashable" data type (because we implement our search algorithm
    using a dict and use instances of the `State` class as keys for the values) -- for example:
        - any non-mutable primitive (str, int, float, etc.)
        - tuples
        - nested combinations of the above

    As you implement different types of search problems throughout the assignment,
    think of what `memory` should contain to enable efficient search!

    Usage:
        state = State(location="A", memory=("some_hashable_data_type", 123))
    """
    location: str
    memory: Optional[Hashable] = None

@dataclass(frozen=True)
class Step:
    """
    A Step represents a single transition in a search problem.

    Attributes:
        action: The action taken to reach the next state (e.g., "move_to_location")
        cost: The cost associated with taking this action
        state: The resulting state after taking this action
    """
    action: str
    cost: float
    state: State

class SearchProblem:
    # Return the start state.
    def start_state(self) -> State:
        raise NotImplementedError("Override me")

    # the various edges coming out of `state`. Note: it is valid for action
    # to be equivalent to location of a successor state.
    def successors(self, state: State) -> List[Step]:
        raise NotImplementedError("Override me")

    # Return whether `state` is an end state or not.
    def is_end(self, state: State) -> bool:
        raise NotImplementedError("Override me")


class SearchAlgorithm:
    def __init__(self):
        """
        A SearchAlgorithm is defined by the function `solve(problem: SearchProblem)`

        A call to `solve` sets the following instance variables:
            - self.actions: List of "actions" that takes one from the start state to a
                            valid end state, or None if no such action sequence exists.
                            > Note: For this assignment, an "action" is just the string
                                    "nextLocation" for a state, but in general, an
                                    action could be something like "up/down/left/right"

            - self.path_cost: Sum of the costs along the path, or None if no valid path.

            - self.num_states_explored: Number of States explored by the given search
                                      algorithm as it attempts to find a satisfying
                                      path. You can use this to gauge the efficiency of
                                      search heuristics, for example.

            - self.past_costs: Dictionary mapping each string location visited by the
                              SearchAlgorithm to the corresponding cost to get there
                              from the starting location.
        """
        self.actions: List[str] = None
        self.path_cost: float = None
        self.num_states_explored: int = 0
        self.past_costs: Dict[str, float] = {}

    def solve(self, problem: SearchProblem) -> None:
        raise NotImplementedError("Override me")


class Heuristic:
    # A Heuristic object is defined by a single function `evaluate(state)` that
    # returns an estimate of the cost of going from the specified `state` to an
    # end state. Used by A*.
    def evaluate(self, state: State) -> float:
        raise NotImplementedError("Override me")


########################################################################################
# Uniform Cost Search (Dijkstra's algorithm)


class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose: int = 0):
        super().__init__()
        self.verbose = verbose

    def solve(self, problem: SearchProblem) -> None:
        """
        Run Uniform Cost Search on the specified `problem` instance.

        Sets the following instance variables (see `SearchAlgorithm` docstring).
            - self.actions: List[str]
            - self.path_cost: float
            - self.num_states_explored: int
            - self.past_costs: Dict[str, float]

        *Hint*: Some of these variables might be really helpful for Problem 3!
        """
        self.actions: List[str] = []
        self.path_cost: float = None
        self.num_states_explored: int = 0
        self.past_costs: Dict[str, float] = {}

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}           # Map state -> previous state.

        # Add the start state
        start_state = problem.start_state()
        frontier.update(start_state, 0.0)

        while True:
            # Remove the state from the queue with the lowest past_cost (priority).
            state, past_cost = frontier.remove_min()
            if state is None and past_cost is None:
                if self.verbose >= 1:
                    print("Searched the entire search space!")
                return

            # Update tracking variables
            self.past_costs[state.location] = past_cost
            self.num_states_explored += 1
            if self.verbose >= 2:
                print(f"Exploring {state} with past_cost {past_cost}")

            # Check if we've reached an end state; if so, extract solution.
            if problem.is_end(state):
                self.actions = []
                while state != start_state:
                    action, prev_state = backpointers[state]
                    self.actions.append(action)
                    state = prev_state
                self.actions.reverse()
                self.path_cost = past_cost
                if self.verbose >= 1:
                    print(f"num_states_explored = {self.num_states_explored}")
                    print(f"path_cost = {self.path_cost}")
                    print(f"actions = {self.actions}")
                return

            # Expand from `state`, updating the frontier with each `new_state`
            for step in problem.successors(state):
                action = step.action
                cost = step.cost
                new_state = step.state
                if self.verbose >= 3:
                    print(f"\t{state} => {new_state} (Cost: {past_cost} + {cost})")

                if frontier.update(new_state, past_cost + cost):
                    # We found better way to go to `new_state` --> update backpointer!
                    backpointers[new_state] = (action, state)


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert `state` into the heap with priority `new_priority` if `state` isn't in
    # the heap or `new_priority` is smaller than the existing priority.
    #   > Return whether the priority queue was updated.
    def update(self, state: State, new_priority: float) -> bool:
        old_priority = self.priorities.get(state)
        if old_priority is None or new_priority < old_priority:
            self.priorities[state] = new_priority
            heapq.heappush(self.heap, (new_priority, state))
            return True
        return False

    # Returns (state with minimum priority, priority) or (None, None) if empty.
    def remove_min(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                # Outdated priority, skip
                continue
            self.priorities[state] = self.DONE
            return state, priority

        # Nothing left...
        return None, None
