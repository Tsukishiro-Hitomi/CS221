from dataclasses import dataclass
from typing import Any, List

from map_util import (
    CityMap,
    compute_distance,
    create_stanford_map,
    location_from_tag,
    make_tag,
)
from util import Heuristic, SearchProblem, State, Step, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`. We've also defined a class `Step`, similar to lecture,
# to help you think through successors, with fields called `action`, `cost`, and `state`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.
#   > Please read the docstring for `Step` in `util.py` for more details and code.
#   > Please read the docstrings for in `map_util.py`, especially for the CityMap class

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `start_location` to any location with the specified `end_tag`.
    """

    def __init__(self, start_location: str, end_tag: str, city_map: CityMap):
        self.start_location = start_location
        self.end_tag = end_tag
        self.city_map = city_map

    def start_state(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def successors(self, state: State) -> List[Step]:
        """
        Note we want to return a list of Step objects of the form:
            Step(action: str, cost: float, state: State)
        Our action space is the set of all named locations, where a named location
        string represents a transition from the current location to that new location.
        """
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def is_end(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def get_stanford_shortest_path_problem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `start_location`/`end_tag`.

    Run `python map_util.py > readable_stanford_map.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    city_map = create_stanford_map()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return ShortestPathProblem(start_location, end_tag, city_map)


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `start_location` to any location with the specified `end_tag` such that the path also
    traverses locations that cover the set of tags in `waypoint_tags`. Note that tags
    from the `start_location` count towards covering the set of tags.

    Hint: naively, your `memory` representation could be a list of all locations visited.
    However, that would be too large of a state space to search over! Think
    carefully about what `memory` should represent.
    """
    def __init__(
        self, start_location: str, waypoint_tags: List[str], end_tag: str, city_map: CityMap
    ):
        self.start_location = start_location
        self.end_tag = end_tag
        self.city_map = city_map

        # We want waypoint_tags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypoint_tags = tuple(sorted(waypoint_tags))

    def start_state(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def successors(self, state: State) -> List[Step]:
        # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def is_end(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def get_stanford_waypoints_shortest_path_problem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem with waypoints using the map of Stanford,
    specifying your own `start_location`/`waypoint_tags`/`end_tag`.

    Similar to Problem 2b, use `readable_stanford_map.txt` to identify potential
    locations and tags.
    """
    city_map = create_stanford_map()
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return WaypointsShortestPathProblem(start_location, waypoint_tags, end_tag, city_map)


########################################################################################
# Problem 4a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`new_search_problem`), such
# that running uniform cost search on `new_search_problem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.
# See util.py for the class definitions and methods of Heuristic and SearchProblem.


def a_star_reduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def start_state(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            raise Exception("Not implemented yet")
            # END_YOUR_CODE

        def successors(self, state: State) -> List[Step]:
            # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
            raise Exception("Not implemented yet")
            # END_YOUR_CODE

        def is_end(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            raise Exception("Not implemented yet")
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `compute_distance` defined in `map_util.py`
    """
    def __init__(self, end_tag: str, city_map: CityMap):
        self.end_tag = end_tag
        self.city_map = city_map

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `start_location` to any location with `end_tag`,
    ignoring all waypoints.
    """
    def __init__(self, end_tag: str, city_map: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired end_tag
        """
        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `start_location`.
        # Solving this reversed shortest path problem will give us our heuristic,
        # as it estimates the minimal cost of reaching an end state from each state
        class ReverseShortestPathProblem(SearchProblem):
            def start_state(self) -> State:
                """
                Return special "END" state
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                raise Exception("Not implemented yet")
                # END_YOUR_CODE

            def successors(
                self, state: State
            ) -> List[Step]:
                # If current location is the special "END" state,
                # return all the locations with the desired end_tag and cost 0
                # (i.e, we connect the special location "END" with cost 0 to all locations with end_tag)
                # Else, return all the successors of current location and their corresponding distances according to the city_map
                # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
                raise Exception("Not implemented yet")
                # END_YOUR_CODE

            def is_end(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False),
                UCS will exhaustively compute costs to *all* other states.
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                raise Exception("Not implemented yet")
                # END_YOUR_CODE

        # Call ucs.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`is_end` always returns False), will exhaustively
        # compute costs to *all* other states.

        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `end_tag`), we can retrieve `ucs.past_costs`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!

        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
