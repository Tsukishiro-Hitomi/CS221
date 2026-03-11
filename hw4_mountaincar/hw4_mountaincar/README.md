# Mountaincar

The following sections detail some general notes for working with `mountaincar`, including setup, and
various dependency requirements.

## Setup

**All platforms:**

```bash
# Download the homework zip and unzip into a folder
# In your hw directory
uv init .                            # Initialize project (creates pyproject.toml)
uv python pin 3.12                   # Pin Python version
uv venv                              # Create default virtual environment
uv pip install -r requirements.txt   # Add dependencies
uv run grader.py                     # Run the local grader (using the default venv)

# To use the `python` command inside this project,
# activate the (default) virtual environment:
source .venv/bin/activate
```

### Training agent with your RL Implementation

To train the mountaincar agent with your RL implmentations,

```bash
# MCValueIteration
uv run train.py --agent value-iteration

# Tabular Q-Learning
uv run train.py --agent tabular

# Function Approximation Q-Learning
uv run train.py --agent function-approximation

# Constrained Q-Learning
uv run train.py --agent constrained
```

This will save the resulting policy, Q values, or weights in your assignment directory.

## Visualizing the Trained Agent

To visualize the agent trained with your RL implementations,

```bash
# Agent trained with MCValueIteration
uv run mountaincar.py --agent value-iteration

# Agent trained with Tabular Q-Learning
uv run mountaincar.py --agent tabular

# Agent trained with Function Approximation Q-Learning
uv run mountaincar.py --agent function-approximation

# Agent trained with Constrained Q-Learning
uv run mountaincar.py --agent constrained
```
