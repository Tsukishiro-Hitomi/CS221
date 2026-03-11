# Route

The following sections detail some general notes for working with `route`, including setup, and
various dependency requirements.

## Setup Instructions

### Step 1: Installing uv (Skip if you already did this for a previous homework)

We recommend using `uv` as it's much faster than pip and conda for managing Python environments and packages. `uv` will also automatically install Python 3.12 for you in the next step.

**What is uv?** `uv` is a modern, Rust-based package + project manager for Python. It keeps the familiar pip workflow but re-implements the engine for speed and reliability.

**Installing uv:** Please refer to the [official uv installation documentation](https://docs.astral.sh/uv/#installation) for the most up-to-date installation instructions for your platform.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Setting Up This Homework's Environment

This homework uses [OpenStreetMap](https://www.openstreetmap.org/) (OSM) data and visualizes maps in the browser. We need to install additional dependencies specific to this assignment.

In the `hw3_route` directory, run the following:

```bash
uv init .                             # Initialize project (creates pyproject.toml)
uv python pin 3.12                    # Pin Python version to 3.12
uv venv hw3_venv                      # Create virtual environment
source hw3_venv/bin/activate          # Activate your virtual env (or activate.fish for fish shell)
uv pip install -r requirements.txt    # Install hw3-specific dependencies
python grader.py                      # Run code using the activated venv's `python`
```

Alternatively, you can also fall back to the project's default environment and use `uv run`:
```bash
uv init .
uv python pin 3.12
uv venv                               # Creates the default environment (no name)
uv pip install -r requirements.txt    # Install into default venv
uv run grader.py                      # Run using the default environment
# NOTE: here, the command `python` may NOT point to this default environment and may thus fail.
```

This should work out of the box for all platforms (Linux, Mac OS, Windows).

### Troubleshooting

**Issue with installing `osmium` or other packages:**
The error messages can be quite long, but usually list near the beginning the reason for the issue. Common solutions might be having to manually install the `cmake` and `boost` dependencies first:

- **On macOS:** Please do this with homebrew: (1) install [homebrew](https://brew.sh/) here (if you do not already have it), and (2) `brew install cmake` and `brew install boost`
- **On Windows:** Please review these [instructions for downloading cmake](https://cmake.org/download/) and these [instructions for downloading boost](https://www.geeksforgeeks.org/how-to-install-c-boost-libraries-on-windows/)
- **Segmentation fault error:** If you're getting a `Segmentation fault (core dumped)` error, install a different osmium version with `uv pip uninstall osmium` and `uv pip install --no-binary :all: osmium`

**Module not found errors:**
Make sure you're running commands with `uv run python` to use the virtual environment.

**Other issues:**
We encourage you to come to Office Hours, or post on Ed. Please start early so you can troubleshoot any issues with ample time!

## Visualizing the Map

To visualize a particular map, you can use the following:

```bash
uv run python visualization.py --path-file None

# You can customize the map and the landmarks
uv run python visualization.py --map-file data/stanford.pbf --landmark-file data/stanford-landmarks.json --path-file None

# Visualize a particular solution path (requires running `grader.py` on question 1b/2c first!)
uv run python visualization.py --path-file path.json
```
