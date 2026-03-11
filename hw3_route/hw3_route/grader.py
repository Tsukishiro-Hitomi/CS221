#!/usr/bin/python3

import json
from typing import List, Optional
from math import radians

import grader_util
import util
from map_util import (
    CityMap,
    check_valid,
    create_grid_map,
    create_grid_map_with_custom_tags,
    create_stanford_map,
    get_total_cost,
    location_from_tag,
    make_grid_label,
    make_tag,
    RADIUS_EARTH,
)

grader = grader_util.Grader()
submission = grader.load("submission")

############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
    warnings.warn("Must be using Python 3.12 \n")


def extract_path(start_location: str, search: util.SearchAlgorithm) -> List[str]:
    """
    Assumes that `solve()` has already been called on the `SearchAlgorithm`.

    We extract a sequence of locations from `search.path` (see util.py to better
    understand exactly how this list gets populated).
    """
    return [start_location] + search.actions


def print_path(
    path: List[str],
    waypoint_tags: List[str],
    city_map: CityMap,
    out_path: Optional[str] = "path.json",
):
    done_waypoint_tags = set()
    for location in path:
        for tag in city_map.tags[location]:
            if tag in waypoint_tags:
                done_waypoint_tags.add(tag)
        tags_str = " ".join(city_map.tags[location])
        done_tags_str = " ".join(sorted(done_waypoint_tags))
        print(f"Location {location} tags:[{tags_str}]; done:[{done_tags_str}]")
    print(f"Total distance: {get_total_cost(path, city_map)}")

    # (Optional) Write path to file, for use with `visualize.py`
    if out_path is not None:
        with open(out_path, "w") as f:
            data = {"waypoint_tags": waypoint_tags, "path": path}
            json.dump(data, f, indent=2)


# Instantiate the Stanford Map as a constant --> just load once!
stanford_map = create_stanford_map()

########################################################################################
# Problem 1: Grid City

grader.add_manual_part("1a", max_points=2, description="minimum cost path")
grader.add_manual_part("1b", max_points=3, description="UCS basic behavior")
grader.add_manual_part("1c", max_points=3, description="UCS search behavior")

########################################################################################
# Problem 2a: Modeling the Shortest Path Problem.


def t_2a(
    city_map: CityMap,
    start_location: str,
    end_tag: str,
    expected_cost: Optional[float] = None,
):
    """
    Run UCS on a ShortestPathProblem, specified by
        (start_location, end_tag).
    Check that the cost of the minimum cost path is `expected_cost`.
    """
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(submission.ShortestPathProblem(start_location, end_tag, city_map))
    path = extract_path(start_location, ucs)
    grader.require_is_true(check_valid(path, city_map, start_location, end_tag, []))
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, get_total_cost(path, city_map))



grader.add_basic_part(
    "2a-1-basic",
    lambda: t_2a(
        city_map=create_grid_map(3, 5),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(2, 2)),
        expected_cost=4,
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path on small grid",
)

grader.add_basic_part(
    "2a-2-basic",
    lambda: t_2a(
        city_map=create_grid_map(30, 30),
        start_location=make_grid_label(20, 10),
        end_tag=make_tag("x", "5"),
        expected_cost=15,
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path with multiple end locations",
)

grader.add_hidden_part(
    "2a-3-hidden",
    lambda: t_2a(
        city_map=create_grid_map(100, 100),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(99, 99)),
    ),
    max_points=0.5,
    max_seconds=1,
    description="shortest path with larger grid",
)

# Problem 2a (continued): full Stanford map...
grader.add_basic_part(
    "2a-4-basic",
    lambda: t_2a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "gates"), stanford_map),
        end_tag=make_tag("landmark", "oval"),
        expected_cost=446.99724421432353,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-4)",
)

grader.add_basic_part(
    "2a-5-basic",
    lambda: t_2a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        end_tag=make_tag("landmark", "evgr_a"),
        expected_cost=660.9598696201658,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-5)",
)

grader.add_basic_part(
    "2a-6-basic",
    lambda: t_2a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        end_tag=make_tag("landmark", "bookstore"),
        expected_cost=1109.3271626156256,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic shortest path test case (2a-6)",
)

grader.add_hidden_part(
    "2a-7-hidden",
    lambda: t_2a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "hoover_tower"), stanford_map),
        end_tag=make_tag("landmark", "cantor_arts_center"),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden shortest path test case (2a-7)",
)

grader.add_hidden_part(
    "2a-8-hidden",
    lambda: t_2a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        end_tag=make_tag("landmark", "gates"),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden shortest path test case (2a-8)",
)

########################################################################################
# Problem 2b: Custom -- Plan a Route through Stanford


def t_2b_custom():
    """Given custom ShortestPathProblem, output path for visualization."""
    problem = submission.get_stanford_shortest_path_problem()
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extract_path(problem.start_location, ucs)
    print_path(path=path, waypoint_tags=[], city_map=stanford_map)
    grader.require_is_true(
        check_valid(path, stanford_map, problem.start_location, problem.end_tag, [])
    )


grader.add_basic_part(
    "2b-custom",
    t_2b_custom,
    max_points=0,
    max_seconds=10,
    description="customized shortest path through Stanford",
)
grader.add_manual_part("2b", max_points=3, description="customized shortest path through Stanford")


########################################################################################
# Problem 2c: Externalities
grader.add_manual_part("2c", max_points=3, description="externalities of algorithm")


########################################################################################
# Problem 3a: Modeling the Waypoints Shortest Path Problem.


def t_3ab(
    city_map: CityMap,
    start_location: str,
    end_tag: str,
    waypoint_tags: List[str],
    expected_cost: Optional[float] = None,
):
    """
    Run UCS on a WaypointsShortestPathProblem, specified by
        (start_location, waypoint_tags, end_tag).
    """
    ucs = util.UniformCostSearch(verbose=0)
    problem = submission.WaypointsShortestPathProblem(
        start_location,
        waypoint_tags,
        end_tag,
        city_map,
    )
    ucs.solve(problem)
    grader.require_is_true(ucs.path_cost is not None)
    path = extract_path(start_location, ucs)
    grader.require_is_true(
        check_valid(path, city_map, start_location, end_tag, waypoint_tags)
    )
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, get_total_cost(path, city_map))



grader.add_basic_part(
    "3a-1-basic",
    lambda: t_3ab(
        city_map=create_grid_map(3, 5),
        start_location=make_grid_label(0, 0),
        waypoint_tags=[make_tag("y", 4)],
        end_tag=make_tag("label", make_grid_label(2, 2)),
        expected_cost=8,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path on small grid with 1 waypoint",
)

grader.add_basic_part(
    "3a-2-basic",
    lambda: t_3ab(
        city_map=create_grid_map(30, 30),
        start_location=make_grid_label(20, 10),
        waypoint_tags=[make_tag("x", 5), make_tag("x", 7)],
        end_tag=make_tag("label", make_grid_label(3, 3)),
        expected_cost=24.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path on medium grid with 2 waypoints",
)

grader.add_basic_part(
    "3a-3-basic",
    lambda: t_3ab(
        city_map=create_grid_map_with_custom_tags(2, 2, {(0,0): [], (0,1): ["food", "fuel", "books"], (1,0): ["food"], (1,1): ["fuel"]}),
        start_location=make_grid_label(0, 0),
        waypoint_tags=[
            "food", "fuel", "books"
        ],
        end_tag=make_tag("label", make_grid_label(0, 1)),
        expected_cost=1.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path with 3 waypoints and some locations covering multiple waypoints",
)

grader.add_basic_part(
    "3a-4-basic",
    lambda: t_3ab(
        city_map=create_grid_map_with_custom_tags(2, 2, {(0,0): ["food"], (0,1): ["fuel"], (1,0): ["food"], (1,1): ["food", "fuel"]}),
        start_location=make_grid_label(0, 0),
        waypoint_tags=[
            "food", "fuel"
        ],
        end_tag=make_tag("label", make_grid_label(0, 1)),
        expected_cost=1.0,
    ),
    max_points=0.5,
    max_seconds=3,
    description="shortest path with 3 waypoints and start location covering some waypoints",
)

grader.add_hidden_part(
    "3a-5-hidden",
    lambda: t_3ab(
        city_map=create_grid_map(100, 100),
        start_location=make_grid_label(0, 0),
        waypoint_tags=[
            make_tag("x", 90),
            make_tag("x", 95),
            make_tag("label", make_grid_label(3, 99)),
            make_tag("label", make_grid_label(99, 3)),
        ],
        end_tag=make_tag("y", 95),
    ),
    max_points=1,
    max_seconds=3,
    description="shortest path with 4 waypoints and multiple end locations",
)

# Problem 3a (continued): full Stanford map...
grader.add_basic_part(
    "3a-6-basic",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "gates"), stanford_map),
        waypoint_tags=[make_tag("landmark", "hoover_tower")],
        end_tag=make_tag("landmark", "oval"),
        expected_cost=1108.3623108845995,
    ),
    max_points=0.5,
    max_seconds=3,
    description="basic waypoints test case (3a-4)",
)

grader.add_basic_part(
    "3a-7-basic",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "evgr_a"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "memorial_church"),
            make_tag("landmark", "tresidder"),
            make_tag("landmark", "gates"),
        ],
        end_tag=make_tag("landmark", "evgr_a"),
        expected_cost=3381.952714299139,
    ),
    max_points=0.5,
    max_seconds=3,
    description="basic waypoints test case (3a-5)",
)

grader.add_basic_part(
    "3a-8-basic",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "gates"),
            make_tag("landmark", "AOERC"),
            make_tag("landmark", "bookstore"),
            make_tag("landmark", "hoover_tower"),
        ],
        end_tag=make_tag("landmark", "green_library"),
        expected_cost=3946.478546309725,
    ),
    max_points=1,
    max_seconds=3,
    description="basic waypoints test case (3a-6)",
)

grader.add_hidden_part(
    "3a-9-hidden",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "oval"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "memorial_church"),
            make_tag("landmark", "hoover_tower"),
            make_tag("landmark", "bookstore"),
        ],
        end_tag=make_tag("landmark", "AOERC"),
    ),
    max_points=0.5,
    max_seconds=3,
    description="hidden waypoints test case (3a-7)",
)

grader.add_hidden_part(
    "3a-10-hidden",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "oval"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "memorial_church"),
            make_tag("landmark", "stanford_stadium"),
            make_tag("landmark", "rains"),
        ],
        end_tag=make_tag("landmark", "oval"),
    ),
    max_points=0.5,
    max_seconds=3,
    description="hidden waypoints test case (3a-8)",
)

grader.add_hidden_part(
    "3a-11-hidden",
    lambda: t_3ab(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "gates"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "lathrop_library"),
            make_tag("landmark", "green_library"),
            make_tag("landmark", "tresidder"),
            make_tag("landmark", "AOERC"),
        ],
        end_tag=make_tag("landmark", "evgr_a"),
    ),
    max_points=1,
    max_seconds=5,
    description="hidden waypoints test case (3a-9)",
)

########################################################################################
# Problem 3b: Maximum states with waypoints
grader.add_manual_part("3b", max_points=2, description="max states with waypoints")


########################################################################################
# Problem 3c: Custom -- Plan a Route with Unordered Waypoints through Stanford


def t_3c_custom():
    """Given custom WaypointsShortestPathProblem, output path for visualization."""
    problem = submission.get_stanford_waypoints_shortest_path_problem()
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(problem)
    path = extract_path(problem.start_location, ucs)
    print_path(path=path, waypoint_tags=problem.waypoint_tags, city_map=stanford_map)
    grader.require_is_true(
        check_valid(
            path,
            stanford_map,
            problem.start_location,
            problem.end_tag,
            problem.waypoint_tags,
        )
    )


grader.add_basic_part(
    "3c-custom",
    t_3c_custom,
    max_points=0,
    max_seconds=10,
    description="customized shortest path with waypoints through Stanford",
)
grader.add_manual_part("3c", max_points=3, description="customized shortest path with waypoints through Stanford")

########################################################################################
# Problem 3d: Ethical Considerations
grader.add_manual_part("3d", max_points=3, description="ethical considerations")


########################################################################################
# Problem 4a: A* to UCS reduction

class ZeroHeuristic(util.Heuristic):
    """Estimates the cost between locations as 0 distance."""
    def __init__(self, end_tag: str, city_map: CityMap):
        self.end_tag = end_tag
        self.city_map = city_map

    def evaluate(self, state: util.State) -> float:
        return 0.0

# Calculates distance only along the north south direction
class NorthSouthHeuristic(util.Heuristic):
    def __init__(self, end_tag: str, city_map: CityMap):
        self.end_tag = end_tag
        self.city_map = city_map
        self.end_geo_locations = [
            self.city_map.geo_locations[location]
            for location, tags in self.city_map.tags.items()
            if end_tag in tags
        ]

    def evaluate(self, state: util.State) -> float:
        current_geo_location = self.city_map.geo_locations[state.location]
        return min(
            RADIUS_EARTH * radians(abs(end_geo_location.latitude - current_geo_location.latitude))
            for end_geo_location in self.end_geo_locations
        )

def t_4a(
    city_map: CityMap,
    start_location: str,
    end_tag: str,
    expected_cost: Optional[float] = None,
    heuristic_cls: Optional[type[util.Heuristic]] = ZeroHeuristic,
):
    """
    Run UCS on the A* Reduction of a ShortestPathProblem, specified by
        (start_location, end_tag).
    """
    heuristic = heuristic_cls(end_tag, city_map)

    # Define the base_problem and corresponding reduction (using `zeroHeuristic`)
    base_problem = submission.ShortestPathProblem(start_location, end_tag, city_map)
    a_star_problem = submission.a_star_reduction(base_problem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(a_star_problem)
    path = extract_path(start_location, ucs)
    grader.require_is_true(check_valid(path, city_map, start_location, end_tag, []))
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, get_total_cost(path, city_map), tolerance=1e-2)



grader.add_basic_part(
    "4a-1-basic",
    lambda: t_4a(
        city_map=create_grid_map(3, 5),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(2, 2)),
        expected_cost=4,
    ),
    max_points=1,
    max_seconds=1,
    description="A* shortest path on small grid",
)

grader.add_basic_part(
    "4a-2-basic",
    lambda: t_4a(
        city_map=create_grid_map(30, 30),
        start_location=make_grid_label(20, 10),
        end_tag=make_tag("x", "5"),
        expected_cost=15,
    ),
    max_points=1,
    max_seconds=1,
    description="A* shortest path with multiple end locations",
)

grader.add_basic_part(
    "4a-3-basic",
    lambda: t_4a(
        city_map=stanford_map,
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        end_tag = make_tag("landmark", "lathrop_library"),
        heuristic_cls=NorthSouthHeuristic,
        expected_cost=1586.1140252970856,
    ),
    max_points=1,
    max_seconds=2,
    description="A* with nontrivial heuristic on Stanford map",
)

grader.add_hidden_part(
    "4a-4-hidden",
    lambda: t_4a(
        city_map=create_grid_map(100, 100),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(99, 99)),
    ),
    max_points=2,
    max_seconds=2,
    description="A* shortest path with larger grid",
)


########################################################################################
# Problem 4b: "straight-line" heuristic for A*


def t_4b_heuristic(
    city_map: CityMap,
    start_location: str,
    end_tag: str,
    expected_cost: Optional[float] = None,
):
    """Targeted test for `StraightLineHeuristic` to ensure correctness."""
    heuristic = submission.StraightLineHeuristic(end_tag, city_map)
    heuristic_cost = heuristic.evaluate(util.State(start_location))
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, heuristic_cost)
    else:
        # If no expected cost provided, just check that it doesn't crash and returns a valid number
        grader.require_is_numeric(heuristic_cost)



grader.add_basic_part(
    "4b-heuristic-1-basic",
    lambda: t_4b_heuristic(
        city_map=create_grid_map(3, 5),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(2, 2)),
        expected_cost=3.145067466556296,
    ),
    max_points=0.5,
    max_seconds=1,
    description="basic straight line heuristic unit test",
)

grader.add_hidden_part(
    "4b-heuristic-2-hidden",
    lambda: t_4b_heuristic(
        city_map=create_grid_map(100, 100),
        start_location=make_grid_label(0, 0),
        end_tag=make_tag("label", make_grid_label(99, 99)),
    ),
    max_points=0.5,
    max_seconds=1,
    description="hidden straight line heuristic unit test",
)


# Initialize a `StraightLineHeuristic` using `end_tag_4b` and the `stanford_map`
end_tag_4b = make_tag("landmark", "gates")
if grader.selectedPartName in [
    "4b-astar-1-basic",
    "4b-astar-2-basic",
    "4b-astar-3-hidden",
    "4b-astar-4-hidden",
    None,
]:
    try:
        stanford_straight_line_heuristic = submission.StraightLineHeuristic(
            end_tag_4b, stanford_map
        )
    except:
        stanford_straight_line_heuristic = None

def t_4b_aStar(
    start_location: str, heuristic: util.Heuristic, expected_cost: Optional[float] = None
):
    """Run UCS on the A* Reduction of a ShortestPathProblem, w/ `heuristic`"""
    base_problem = submission.ShortestPathProblem(start_location, end_tag_4b, stanford_map)
    a_star_problem = submission.a_star_reduction(base_problem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(a_star_problem)
    path = extract_path(start_location, ucs)
    grader.require_is_true(check_valid(path, stanford_map, start_location, end_tag_4b, []))
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, get_total_cost(path, stanford_map))



grader.add_basic_part(
    "4b-astar-1-basic",
    lambda: t_4b_aStar(
        start_location=location_from_tag(make_tag("landmark", "oval"), stanford_map),
        heuristic=stanford_straight_line_heuristic,
        expected_cost=446.9972442143235,
    ),
    max_points=0.5,
    max_seconds=2,
    description="basic straight line heuristic A* on Stanford map (4b-astar-1)",
)


grader.add_basic_part(
    "4b-astar-2-basic",
    lambda: t_4b_aStar(
        start_location=location_from_tag(make_tag("landmark", "rains"), stanford_map),
        heuristic=stanford_straight_line_heuristic,
        expected_cost=2005.4388573303765,
    ),
    max_points=1,
    max_seconds=2,
    description="basic straight line heuristic A* on Stanford map (4b-astar-2)",
)


grader.add_hidden_part(
    "4b-astar-3-hidden",
    lambda: t_4b_aStar(
        start_location=location_from_tag(make_tag("landmark", "bookstore"), stanford_map),
        heuristic=stanford_straight_line_heuristic,
    ),
    max_points=0.5,
    max_seconds=2,
    description="hidden straight line heuristic A* on Stanford map (4b-astar-3)",
)


grader.add_hidden_part(
    "4b-astar-4-hidden",
    lambda: t_4b_aStar(
        start_location=location_from_tag(make_tag("landmark", "evgr_a"), stanford_map),
        heuristic=stanford_straight_line_heuristic,
    ),
    max_points=1,
    max_seconds=2,
    description="hidden straight line heuristic A* on Stanford map (4b-astar-4)",
)


########################################################################################
# Problem 4c: "no waypoints" heuristic for A*


def t_4c_heuristic(
    start_location: str, end_tag: str, expected_cost: Optional[float] = None
):
    """Targeted test for `NoWaypointsHeuristic` -- uses the full Stanford map."""
    heuristic = submission.NoWaypointsHeuristic(end_tag, stanford_map)
    heuristic_cost = heuristic.evaluate(util.State(start_location))
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, heuristic_cost)
    else:
        # If no expected cost provided, just check that it doesn't crash and returns a valid number
        grader.require_is_numeric(heuristic_cost)



grader.add_basic_part(
    "4c-heuristic-1-basic",
    lambda: t_4c_heuristic(
        start_location=location_from_tag(make_tag("landmark", "oval"), stanford_map),
        end_tag=make_tag("landmark", "gates"),
        expected_cost=446.99724421432353,
    ),
    max_points=1,
    max_seconds=2,
    description="basic no waypoints heuristic unit test",
)

grader.add_hidden_part(
    "4c-heuristic-1-hidden",
    lambda: t_4c_heuristic(
        start_location=location_from_tag(make_tag("landmark", "bookstore"), stanford_map),
        end_tag=make_tag("amenity", "food"),
    ),
    max_points=1,
    max_seconds=2,
    description="hidden no waypoints heuristic unit test w/ multiple end locations",
)


# Initialize a `NoWaypointsHeuristic` using `end_tag_4c` and the `stanford_map`
end_tag_4c = make_tag("wheelchair", "yes")
if grader.selectedPartName in [
    "4c-astar-1-basic",
    "4c-astar-2-basic",
    "4c-astar-3-hidden",
    "4c-astar-3-hidden",
    None,
]:
    try:
        stanford_no_waypoints_heuristic = submission.NoWaypointsHeuristic(
            end_tag_4c, stanford_map
        )
    except:
        stanford_no_waypoints_heuristic = None

def t_4c_aStar(
    start_location: str,
    waypoint_tags: List[str],
    heuristic: util.Heuristic,
    expected_cost: Optional[float] = None,
):
    """Run UCS on the A* Reduction of a WaypointsShortestPathProblem, w/ `heuristic`"""
    base_problem = submission.WaypointsShortestPathProblem(
        start_location, waypoint_tags, end_tag_4c, stanford_map
    )
    a_star_problem = submission.a_star_reduction(base_problem, heuristic)

    # Solve the reduction via a call to `ucs.solve` (similar to prior tests)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(a_star_problem)
    path = extract_path(start_location, ucs)
    grader.require_is_true(
        check_valid(path, stanford_map, start_location, end_tag_4c, waypoint_tags)
    )
    if expected_cost is not None:
        grader.require_is_equal(expected_cost, get_total_cost(path, stanford_map))



grader.add_basic_part(
    "4c-astar-1-basic",
    lambda: t_4c_aStar(
        start_location=location_from_tag(make_tag("landmark", "oval"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "gates"),
            make_tag("landmark", "AOERC"),
            make_tag("landmark", "bookstore"),
            make_tag("landmark", "hoover_tower"),
        ],
        heuristic=stanford_no_waypoints_heuristic,
        expected_cost=2943.242598551967,
    ),
    max_points=1.5,
    max_seconds=2,
    description="basic no waypoints heuristic A* on Stanford map (4c-astar-1)",
)


grader.add_basic_part(
    "4c-astar-2-basic",
    lambda: t_4c_aStar(
        start_location=location_from_tag(make_tag("landmark", "AOERC"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "tresidder"),
            make_tag("landmark", "hoover_tower"),
            make_tag("amenity", "food"),
        ],
        heuristic=stanford_no_waypoints_heuristic,
        expected_cost=1677.3814380413373,
    ),
    max_points=1.5,
    max_seconds=2,
    description="basic no waypoints heuristic A* on Stanford map (4c-astar-2)",
)


grader.add_hidden_part(
    "4c-astar-3-hidden",
    lambda: t_4c_aStar(
        start_location=location_from_tag(make_tag("landmark", "tresidder"), stanford_map),
        waypoint_tags=[
            make_tag("landmark", "gates"),
            make_tag("amenity", "food"),
            make_tag("landmark", "rains"),
            make_tag("landmark", "stanford_stadium"),
            make_tag("bicycle", "yes"),
        ],
        heuristic=stanford_no_waypoints_heuristic,
    ),
    max_points=3,
    max_seconds=10,
    description="hidden no waypoints heuristic A* on Stanford map (4c-astar-3)",
)

grader.add_manual_part("5a", max_points=2, description="AI tutor dynamic programming")
grader.add_manual_part("5b", max_points=2, description="AI tutor beam search")

# Removed from Spring2025 per the Fixit doc's suggestion
# grader.add_manual_part("4d", max_points=2, description="example of n waypoint_tags")


if __name__ == "__main__":
    grader.grade()
