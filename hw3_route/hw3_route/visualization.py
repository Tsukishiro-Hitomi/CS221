import argparse
import json
from typing import List

import plotly.express as px
import plotly.graph_objects as go

from map_util import CityMap, add_landmarks, read_map


def plot_map(city_map: CityMap, path: List[str], waypoint_tags: List[str], map_name: str):
    """
    Plot the full map, highlighting the provided path.

    :param city_map: CityMap to plot.
    :param path: List of location labels of the path.
    :param waypoint_tags: List of tags that we care about hitting along the way.
    :param map_name: Display title for map visualization.
    """
    lat, lon = [], []

    # Convert `city_map.distances` to a list of (source, target) tuples...
    connections = [
        (source, target)
        for source in city_map.distances
        for target in city_map.distances[source]
    ]
    for source, target in connections:
        lat.append(city_map.geo_locations[source].latitude)
        lat.append(city_map.geo_locations[target].latitude)
        lat.append(None)
        lon.append(city_map.geo_locations[source].longitude)
        lon.append(city_map.geo_locations[target].longitude)
        lon.append(None)

    # Plot all states & connections
    fig = px.line_geo(lat=lat, lon=lon)

    # Plot path (represented by connections in `path`)
    if len(path) > 0:
        solution_lat, solution_lon = [], []

        # Get and convert `path` to (source, target) tuples to append to lat, lon lists
        connections = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for connection in connections:
            source, target = connection
            solution_lat.append(city_map.geo_locations[source].latitude)
            solution_lat.append(city_map.geo_locations[target].latitude)
            solution_lat.append(None)
            solution_lon.append(city_map.geo_locations[source].longitude)
            solution_lon.append(city_map.geo_locations[target].longitude)
            solution_lon.append(None)

        # Visualize path by adding a trace
        fig.add_trace(
            go.Scattergeo(
                lat=solution_lat,
                lon=solution_lon,
                mode="lines",
                line=dict(width=5, color="blue"),
                name="solution",
            )
        )

        # Plot the points
        for i, location in enumerate(path):
            tags = set(city_map.tags[location]).intersection(set(waypoint_tags))
            if i == 0 or i == len(path) - 1 or len(tags) > 0:
                for tag in city_map.tags[location]:
                    if tag.startswith("landmark="):
                        tags.add(tag)
            if len(tags) == 0:
                continue

            # Add descriptions as annotations for each point
            description = " ".join(sorted(tags))

            # Color the start node green, the end node red, intermediate gray
            if i == 0:
                color = "red"
            elif i == len(path) - 1:
                color = "green"
            else:
                color = "gray"

            waypoint_lat = [city_map.geo_locations[location].latitude]
            waypoint_lon = [city_map.geo_locations[location].longitude]

            fig.add_trace(
                go.Scattergeo(
                    lat=waypoint_lat,
                    lon=waypoint_lon,
                    mode="markers",
                    marker=dict(size=20, color=color),
                    name=description,
                )
            )

    # Plot city_map locations with special tags (e.g. landmarks, amenities)
    for location_id, lat_lon in city_map.geo_locations.items():
        tags = city_map.tags[location_id]
        for tag in tags:
            if "landmark" in tag:
                fig.add_trace(
                    go.Scattergeo(
                        locationmode="USA-states",
                        lon=[lat_lon.longitude],
                        lat=[lat_lon.latitude],
                        text=tag.split("landmark=")[1],
                        name=tag.split("landmark=")[1],
                        marker=dict(size=10, color="purple", line_width=3),
                    )
                )
            elif "amenity" in tag:
                fig.add_trace(
                    go.Scattergeo(
                        locationmode="USA-states",
                        lon=[lat_lon.longitude],
                        lat=[lat_lon.latitude],
                        text=tag.split("amenity=")[1],
                        name=tag.split("amenity=")[1],
                        marker=dict(size=10, color="blue", line_width=3),
                    )
                )

    # Final scaling, centering, and figure title
    mid_idx = len(lat) // 2
    fig.update_layout(title=map_name, title_x=0.5)
    fig.update_layout(
        geo=dict(projection_scale=20000, center=dict(lat=lat[mid_idx], lon=lon[mid_idx]))
    )
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-file", type=str, default="data/stanford.pbf", help="Map (.pbf)"
    )
    parser.add_argument(
        "--landmark-file",
        type=str,
        default="data/stanford-landmarks.json",
        help="Landmarks (.json)",
    )
    parser.add_argument(
        "--path-file",
        type=str,
        default="path.json",
        help="Path to visualize (.json), path should correspond to some map file",
    )
    args = parser.parse_args()

    # Create city_map and populate any relevant landmarks
    stanford_map_name = args.map_file.split("/")[-1].split("_")[0]
    stanford_city_map = read_map(args.map_file)
    add_landmarks(stanford_city_map, args.landmark_file)

    # (Optional) Read path to visualize from JSON file
    if args.path_file != 'None':
        with open(args.path_file) as f:
            data = json.load(f)
            parsed_path = data["path"]
            parsed_waypoint_tags = data["waypoint_tags"]
    else:
        parsed_path = []
        parsed_waypoint_tags = []

    # Run the visualization
    plot_map(
        city_map=stanford_city_map,
        path=parsed_path,
        waypoint_tags=parsed_waypoint_tags,
        map_name=stanford_map_name,
    )
