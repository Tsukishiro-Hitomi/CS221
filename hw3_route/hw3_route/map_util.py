import json
from collections import defaultdict
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Optional, Set, Tuple

import osmium
from osmium import osm

# Constants
RADIUS_EARTH = 6371000  # Radius of earth in meters (~ equivalent to 3956 miles).
UNIT_DELTA = 0.00001    # Denotes the change in latitude/longitude (in degrees) that
                        # equates to distance of ~1m.

########################################################################################
# Map Abstraction Overview & Useful Data Structures
#   > `GeoLocation` :: forms the atomic units of our abstraction; each `GeoLocation`
#                      object is uniquely specified as a pair of coordinates denoting
#                      latitude/longitude (in degrees).
#
#   > `CityMap` is the core structure defining the following:
#       + `locations` [str -> GeoLocation]: A dictionary mapping a unique label to a
#                                           specific GeoLocation.
#
#       + `tags` [str -> List[str]]: A dictionary mapping a location label (same keys
#                                    as above) to a list of meaningful "tags"
#                                    (e.g., amenity=park or landmark=hoover_tower).
#                                    These tags are parsed from OpenStreetMaps or
#                                    defined manually as "landmarks" in
#                                    `data/stanford-landmarks.json`.
#
#       + `distances` [str -> [str -> float]]: A nested dictionary mapping pairs of
#                                              locations to distances (e.g.,
#                                              `distances[label1][label2] = 21.3`).


@dataclass(frozen=True)
class GeoLocation:
    """A latitude/longitude of a physical location on Earth."""
    latitude: float
    longitude: float

    def __repr__(self):
        return f"{self.latitude},{self.longitude}"


class CityMap:
    """
    A city map consists of a set of *labeled* locations with associated tags, and
    connections between them.
    """
    def __init__(self) -> None:
        # Location label -> actual geolocation (latitude/longitude)
        # (e.g., self.geo_locations["0,1"] = GeoLocation(37.423576, -122.170087)
        self.geo_locations: Dict[str, GeoLocation] = {}

        # Location label -> list of tags 
        # (e.g., self.tags["0,1"] = ["amenity=building", "landmark=Gates"])
        self.tags: Dict[str, List[str]] = defaultdict(list)

        # Location label -> adjacent location label -> distance between the two
        # (e.g., self.distances["0,1"]["0,2"] = 21.3)
        self.distances: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_location(self, label: str, location: GeoLocation, tags: List[str]) -> None:
        """Add a location (denoted by `label`) to map with the provided set of tags."""
        assert label not in self.geo_locations, f"Location {label} already processed!"
        self.geo_locations[label] = location
        self.tags[label] = [make_tag("label", label)] + tags

    def add_connection(
        self, source: str, target: str, distance: Optional[float] = None
    ) -> None:
        """Adds a connection between source <--> target to `self.distances`."""
        if distance is None:
            distance = compute_distance(
                self.geo_locations[source], self.geo_locations[target]
            )
        self.distances[source][target] = distance
        self.distances[target][source] = distance


def add_landmarks(
    city_map: CityMap, landmark_path: str, tolerance_meters: float = 250.0
) -> None:
    """
    Add landmarks from `path` to `city_map`. A landmark (e.g., Gates Building) is
    associated with a `GeoLocation`.

    Landmarks are explicitly defined via the `landmark_path` file, which borrows
    latitude/longitude for various spots on Stanford Campus from Google Maps; these
    may not *exactly* line up with existing locations in the CityMap, so instead we map
    a given landmark onto the closest existing location (subject to a max tolerance).
    """
    with open(landmark_path) as f:
        landmarks = json.load(f)

    # Iterate through landmarks and map onto the closest location in `city_map`
    for item in landmarks:
        latitude_string, longitude_string = item["geo"].split(",")
        geo = GeoLocation(float(latitude_string), float(longitude_string))

        # Find the closest location by searching over all locations in `city_map`
        best_distance, best_label = min(
            (compute_distance(geo, existing_geo), existing_label)
            for existing_label, existing_geo in city_map.geo_locations.items()
        )

        if best_distance < tolerance_meters:
            for key in ["landmark", "amenity"]:
                if key in item:
                    city_map.tags[best_label].append(make_tag(key, item[key]))

########################################################################################
# Utility Functions


def make_tag(key: str, value: str) -> str:
    """Locations have string-valued tags which are created from (key, value) pairs."""
    return f"{key}={value}"


def location_from_tag(tag: str, city_map: CityMap) -> Optional[str]:
    possible_locations = sorted(
        [location for location, tags in city_map.tags.items() if tag in tags]
    )
    return possible_locations[0] if len(possible_locations) > 0 else None

def compute_distance(geo1: GeoLocation, geo2: GeoLocation) -> float:
    """
    Compute the distance (straight line) between two geolocations, specified as
    latitude/longitude. This function is analogous to finding the euclidean distance
    between points on a plane; however, because the Earth is spherical, we're using the
    *Haversine formula* to compute distance subject to the curved surface.

    You can read more about the Haversine formula here:
     > https://en.wikipedia.org/wiki/Haversine_formula

    Note :: For small distances (e.g., Stanford campus --> the greater Bay Area),
    factoring in the curvature of the earth might be a bit overkill!

    However, you could think about using this function to generalize to larger maps
    spanning much greater distances (possibly for fun future projects)!

    :param geo1: Source `GeoLocation`, with attributes for latitude/longitude.
    :param geo2: Target `GeoLocation`, with attributes for latitude/longitude.

    :return: Returns distance between geo1 and geo2 in meters.
    :rtype: float (distance)
    """
    lon1, lat1 = radians(geo1.longitude), radians(geo1.latitude)
    lon2, lat2 = radians(geo2.longitude), radians(geo2.latitude)

    # Haversine formula
    delta_lon, delta_lat = lon2 - lon1, lat2 - lat1
    haversine = (sin(delta_lat / 2) ** 2) + (cos(lat1) * cos(lat2)) * (
        sin(delta_lon / 2) ** 2
    )

    # Return distance d (factor in radius of earth in meters)
    return 2 * RADIUS_EARTH * asin(sqrt(haversine))


def check_valid(
    path: List[str],
    city_map: CityMap,
    start_location: str,
    end_tag: str,
    waypoint_tags: List[str],
) -> bool:
    """Check if a given solution/path is valid subject to the given CityMap instance."""
    # Check that path starts with `start_location`
    if path[0] != start_location:
        print(f"Invalid path: does not start with {start_location}")
        return False

    # Check that path ends with a location with `end_tag`
    if end_tag not in city_map.tags[path[-1]]:
        print(f"Invalid path: final location does not contain {end_tag}")
        return False

    # Check that adjacent locations are *connected* in the underlying CityMap instance
    for i in range(len(path) - 1):
        if path[i + 1] not in city_map.distances[path[i]]:
            print(f"Invalid path: {path[i]} is not connected to {path[i + 1]}")
            return False

    # Check that all waypoint_tags are represented
    done_tags = set(tag for location in path for tag in city_map.tags[location])
    diff_tags = set(waypoint_tags).difference(done_tags)
    if len(diff_tags) > 0:
        print(f"Invalid path: does not contain waypoints {diff_tags}")
        return False

    # Otherwise, we're good!
    return True


def get_total_cost(path: List[str], city_map: CityMap) -> float:
    """Return the total distance of the given path (assuming it's valid)."""
    cost = 0.0
    for i in range(len(path) - 1):
        cost += city_map.distances[path[i]][path[i + 1]]
    return cost


########################################################################################
# Data Processing Functions -- for creating simple programmatic maps, and loading maps
# from OpenStreetMap (OSM) data. Here are some useful acronyms that you may find useful
# as you read through the following code:
#
#   - `OSM` (OpenStreetMap): We use actual data from the OpenStreetMaps project
#                            (https://www.openstreetmap.org/). You can think of
#                            OpenStreetMaps as "Wikipedia" for Google Maps; lots
#                            of useful info!
#
#   - `*.pbf`: File format for OSM data; `pbf` = Protocolbuffer Binary Format; a file
#              format like xml/json that's used by OpenStreetMaps. You shouldn't need
#              to worry about this, as we provide utilities to read these files below.
#
#   - `osmium`: A Python package for dealing with `OSM` data. You will need to install
#               this as a dependency (via `requirements.txt` or `pip install osmium`).


def make_grid_label(x: int, y: int) -> str:
    """Function to create location label from (latitude, longitude) in degrees."""
    return f"{x},{y}"


def create_grid_map(width: int, height: int) -> CityMap:
    """Create a simple map width x height grid of locations."""
    city_map = CityMap()

    # A "simple" city is just a grid with distance ~1m between adjacent locations.
    for x, lat in enumerate([x * UNIT_DELTA for x in range(width)]):
        for y, lon in enumerate([y * UNIT_DELTA for y in range(height)]):
            # We label each location as just the grid index (x, y)
            city_map.add_location(
                make_grid_label(x, y),
                GeoLocation(lat, lon),
                tags=[make_tag("x", x), make_tag("y", y)],
            )
            if x > 0:
                city_map.add_connection(
                    make_grid_label(x - 1, y), make_grid_label(x, y), distance=1
                )
            if y > 0:
                city_map.add_connection(
                    make_grid_label(x, y - 1), make_grid_label(x, y), distance=1
                )

    return city_map

def create_grid_map_with_custom_tags(width: int, height: int, tags: Dict[Tuple[int,int], List[str]]) -> CityMap:
    """Create a simple map width x height grid of locations."""
    city_map = CityMap()

    # A "simple" city is just a grid with distance ~1m between adjacent locations.
    for x, lat in enumerate([x * UNIT_DELTA for x in range(width)]):
        for y, lon in enumerate([y * UNIT_DELTA for y in range(height)]):
            # We label each location as just the grid index (x, y)
            city_map.add_location(
                make_grid_label(x, y),
                GeoLocation(lat, lon),
                tags=[make_tag("x", x), make_tag("y", y)] + tags[(x,y)],
            )
            if x > 0:
                city_map.add_connection(
                    make_grid_label(x - 1, y), make_grid_label(x, y), distance=1
                )
            if y > 0:
                city_map.add_connection(
                    make_grid_label(x, y - 1), make_grid_label(x, y), distance=1
                )

    return city_map

def read_map(osm_path: str) -> CityMap:
    """
    Create a CityMap given a path to a OSM `.pbf` file; uses the osmium package to do
    any/all processing of discrete locations and connections between them.

    :param osm_path: Path to `.pbf` file defining a set of locations and connections.
    :return An initialized CityMap object, built using the OpenStreetMaps data.
    """
    # Note :: `osmium` defines a nice class called `SimpleHandler` to facilitate
    # reading `.pbf` files.
    #   > You can read more about this class/functionality here:
    #     https://docs.osmcode.org/pyosmium/latest/intro.html
    class MapCreationHandler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes: Dict[str, GeoLocation] = {}
            self.tags: Dict[str, List[str]] = defaultdict(list)
            self.edges: Set[str, str] = set()

        def node(self, n: osm.Node) -> None:
            """An `osm.Node` contains the actual tag attributes for a given node."""
            self.tags[str(n.id)] = [make_tag(tag.k, tag.v) for tag in n.tags]

        def way(self, w: osm.Way) -> None:
            """An `osm.Way` contains an ordered list of connected nodes."""

            # We only include "ways" that are accessible on foot
            #   =>> Reference: https://github.com/Tristramg/osm4routing2
            #                  See -> `src/osm4routing/categorize.rs#L96`
            path_type = w.tags.get("highway", None)
            if path_type is None or path_type in {
                "motorway",
                "motorway_link",
                "trunk",
                "trunk_link",
            }:
                return
            elif (
                w.tags.get("pedestrian", "n/a") == "no"
                or w.tags.get("foot", "n/a") == "no"
            ):
                return

            # Otherwise, iterate through all nodes along the "way"...
            way_nodes = w.nodes
            for source_idx in range(len(way_nodes) - 1):
                s, t = way_nodes[source_idx], way_nodes[source_idx + 1]
                s_label, t_label = str(s.ref), str(t.ref)
                s_loc = GeoLocation(s.location.lat, s.location.lon)
                t_loc = GeoLocation(t.location.lat, t.location.lon)

                # Assert that the locations aren't the same!
                assert s_loc != t_loc, "Source and Target are the same location!"

                # Add to trackers...
                self.nodes[s_label], self.nodes[t_label] = s_loc, t_loc
                self.edges.add((s_label, t_label))

    # Build nodes & edges via MapCreationHandler
    #   > Pass `location=True` to enforce embedded lat/lon geometries!
    map_creator = MapCreationHandler()
    map_creator.apply_file(osm_path, locations=True)

    # Build CityMap by iterating through the parsed nodes and connections
    city_map = CityMap()
    for node_label in map_creator.nodes:
        city_map.add_location(
            node_label, map_creator.nodes[node_label], tags=map_creator.tags[node_label]
        )

    # When adding connections, don't pass distance flag (automatically compute!)
    for src, tgt in map_creator.edges:
        city_map.add_connection(src, tgt)

    return city_map


def print_map(city_map: CityMap):
    """Display a dense overview of the provided map, with tags for each location."""
    for label in city_map.geo_locations:
        tags_str = " ".join(city_map.tags[label])
        print(f"{label} ({city_map.geo_locations[label]}): {tags_str}")
        for label2, distance in city_map.distances[label].items():
            print(f"  -> {label2} [distance = {distance}]")


def create_stanford_map() -> CityMap:
    city_map = read_map("data/stanford.pbf")
    add_landmarks(city_map, "data/stanford-landmarks.json")
    return city_map

def create_custom_map(map_file: str, landmarks_file: str) -> CityMap:
    """
    Create a CityMap given a path to an OSM `.pbf` file; uses the osmium package to do
    any/all processing of discrete locations and connections between them.
    
    :param map_file: Path to `.pbf` file defining a set of locations and connections.
    :param landmarks_file: Path to `.json` file defining a set of landmarks.
    
    For further details on the format of the `.pbf` and `.json` files, see the README.md file.
    """
    city_map = read_map(map_file)
    add_landmarks(city_map, landmarks_file)
    return city_map


if __name__ == "__main__":
    stanford_map = create_stanford_map()
    print_map(stanford_map)
