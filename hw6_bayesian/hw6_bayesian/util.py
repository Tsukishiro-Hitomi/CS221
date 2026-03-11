from typing import List, Dict, Tuple, Optional, Iterator
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import math
import csv


class BayesianNode:
    """
    Represents a node (random variable) in a Bayesian network.
    
    Attributes:
        name: Unique identifier for this variable (e.g., 'Rain', 'Sprinkler')
        parents: List of parent nodes in the Bayesian network
        domain: Possible values/outcomes this variable can take (e.g., ['A', 'C', 'T', 'G'])
        conditional_prob_table: Conditional probability table (CPT) as a numpy array
                                Shape depends on number of parents and domain sizes
    
    Example:
        >>> # Create a node with no parents (root node)
        >>> rain = BayesianNode(name='Rain', domain=['true', 'false'], 
        ...                     conditional_prob_table=np.array([0.2, 0.8]))
        >>> 
        >>> # Create a node with parents
        >>> sprinkler = BayesianNode(name='Sprinkler', domain=['true', 'false'],
        ...                          parents=[rain],
        ...                          conditional_prob_table=np.array([[0.01, 0.99], [0.4, 0.6]]))
    """
    
    def __init__(
        self,
        name: str,
        domain: List[str],
        parents: Optional[List["BayesianNode"]] = None,
        conditional_prob_table: Optional[np.ndarray] = None
    ):
        """
        Initialize a Bayesian network node.
        
        Args:
            name: Unique identifier for this variable
            domain: List of possible values/outcomes (e.g., ['A', 'C', 'T', 'G'])
            parents: List of parent nodes (empty list or None if root node)
            conditional_prob_table: CPT as numpy array. If None, creates uniform distribution.
                                   Shape should be (*parent_domains, len(domain))
                                   
                                   For a node with 2 parents with domains of size 2 each,
                                   and this node's domain of size 3, the CPT shape is (2, 2, 3).
                                   The last dimension always corresponds to this node's domain.
        
        Raises:
            ValueError: If CPT shape doesn't match expected dimensions or probabilities don't sum to 1
        """
        self.name = name
        self.domain = domain
        self.parents = parents if parents is not None else []
        self.children: List["BayesianNode"] = []
        
        # Initialize or validate CPT
        if conditional_prob_table is None:
            # Create uniform distribution over domain
            if self.parents:
                cpt_shape = tuple([len(p.domain) for p in self.parents] + [len(domain)])
            else:
                # batch dimension is 1
                # the BayesianNetwork class will expand the batch dimension later if needed
                cpt_shape = (1, len(domain),)
            self.conditional_prob_table = np.ones(cpt_shape) / len(domain)
        else:
            self._validate_cpt(conditional_prob_table)
            self.conditional_prob_table = conditional_prob_table
    
    def _validate_cpt(self, cpt: np.ndarray) -> None:
        """
        Validate that the CPT has correct shape and probabilities sum to 1.
        
        Args:
            cpt: Conditional probability table to validate
            
        Raises:
            ValueError: If CPT is invalid
        """
        if self.parents:
            expected_shape = tuple([len(p.domain) for p in self.parents] + [len(self.domain)])
            if cpt.shape != expected_shape:
                raise ValueError(
                    f"CPT shape {cpt.shape} doesn't match expected shape {expected_shape} "
                    f"based on parent domains and this node's domain"
                )
        else:
            assert len(cpt.shape) == 2, "CPT must have 2 dimensions if no parents"
            assert cpt.shape[1] == len(self.domain), "CPT must have domain dimension equal to this node's domain"
        
        # Check that probabilities sum to 1 along the last axis (for each parent configuration)
        prob_sums = np.sum(cpt, axis=-1)
        if not np.allclose(prob_sums, 1.0):
            raise ValueError(
                f"CPT probabilities must sum to 1 along last axis. "
                f"Got sums: {prob_sums}"
            )
    
    def get_probability(
        self,
        value: str,
        parent_values: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Get P(this=value | parent_values) from the CPT.
        
        Args:
            value: The value of this variable to query
            parent_values: Dictionary mapping parent names to their values
                          (empty dict or None if no parents)
        
        Returns:
            Conditional probability as a float
            
        Raises:
            ValueError: If value not in domain or parent values are invalid
            
        Example:
            >>> rain = BayesianNode(name='Rain', domain=['true', 'false'],
            ...                     conditional_prob_table=np.array([0.2, 0.8]))
            >>> # Get P(Rain = 'true')
            >>> prob = rain.get_probability('true')  # Returns 0.2
            >>> 
            >>> # For a node with parents:
            >>> sprinkler = BayesianNode(name='Sprinkler', domain=['on', 'off'],
            ...                          parents=[rain],
            ...                          conditional_prob_table=np.array([[0.01, 0.99], [0.4, 0.6]]))
            >>> # Get P(Sprinkler = 'on' | Rain = 'true')
            >>> prob = sprinkler.get_probability('on', {'Rain': 'true'})  # Returns 0.01
        """
        if value not in self.domain:
            raise ValueError(f"Value '{value}' not in domain {self.domain}")
        
        value_idx = self.domain.index(value)
        
        if not self.parents:
            # No parents, just return marginal probability
            # Must be batched
            return self.conditional_prob_table[:, value_idx]

        if parent_values is None:
            raise ValueError(f"Missing parent assignments for node '{self.name}'")

        # Build indices for parent values using helper
        parent_indices = self.parent_assignment_indices(parent_values)

        # Index into CPT
        return self.conditional_prob_table[tuple(parent_indices + (value_idx,))]

    def __repr__(self) -> str:
        """String representation for debugging."""
        parent_names = [p.name for p in self.parents]
        return (
            f"BayesianNode(name='{self.name}', "
            f"domain={self.domain}, "
            f"parents={parent_names})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.parents:
            parent_str = ", ".join(p.name for p in self.parents)
            return f"{self.name} | {parent_str}"
        return self.name

    def parent_assignment_indices(self, assignment: Dict[str, str]) -> Tuple[int, ...]:
        """
        Convert an assignment dict into indices for the parents' values.
        """
        indices = []
        for parent in self.parents:
            if parent.name not in assignment:
                raise ValueError(f"Missing value for parent '{parent.name}'")
            value = assignment[parent.name]
            if value not in parent.domain:
                raise ValueError(
                    f"Parent value '{value}' not in parent domain {parent.domain}"
                )
            indices.append(parent.domain.index(value))
        return tuple(indices)

    def iter_parent_assignments(self) -> Iterator[Tuple[Tuple[int, ...], Dict[str, str]]]:
        """
        Yield index tuples and value dicts for every parent configuration.
        """
        if not self.parents:
            yield tuple(), {}
            return

        index_ranges = [range(len(parent.domain)) for parent in self.parents]
        for index_tuple in product(*index_ranges):
            assignment = {
                parent.name: parent.domain[idx]
                for parent, idx in zip(self.parents, index_tuple)
            }
            yield index_tuple, assignment


class BayesianNetwork:
    """
    Represents a Bayesian network.
    
    Attributes:
        nodes: List of Bayesian nodes in the network
        
    Example:
        >>> # Create a simple network: Rain -> Sprinkler -> WetGrass
        >>> rain = BayesianNode(name='Rain', domain=['true', 'false'],
        ...                     conditional_prob_table=np.array([0.2, 0.8]))
        >>> sprinkler = BayesianNode(name='Sprinkler', domain=['true', 'false'],
        ...                          parents=[rain],
        ...                          conditional_prob_table=np.array([[0.01, 0.99], [0.4, 0.6]]))
        >>> network = BayesianNetwork([rain, sprinkler])
        >>> # Get nodes in topological order
        >>> ordered_nodes = network.order
    """
    
    def __init__(self, nodes: List[BayesianNode], batch_size: int=1):
        """
        Initialize a Bayesian network.
        
        Args:
            nodes: List of all BayesianNode objects in the network
        """
        self.nodes = nodes
        self.batch_size = batch_size
        self._build_batch_dimension()
        # check node names are unique
        if len(nodes) != len(set(node.name for node in nodes)):
            raise ValueError("Node names must be unique.")
        self._build_children()
        self._build_names_to_nodes()
        self._build_order()

    def _build_batch_dimension(self) -> None:
        """
        Build the batch dimension for the network.
        """
        # set all nodes without parents to have a batch dimension
        for node in self.nodes:
            if not node.parents:
                # expand batch dimension at the front
                node.conditional_prob_table = np.repeat(node.conditional_prob_table, self.batch_size, axis=0)
    
    def _build_children(self) -> None:
        """
        Build the children attribute for each node based on parent relationships.
        This is computed automatically from the parent relationships.
        """
        # Initialize empty children lists
        for node in self.nodes:
            node.children = []
        
        # Populate children based on parent relationships
        for node in self.nodes:
            for parent in node.parents:
                parent.children.append(node)
    
    def _build_order(self):
        """
        Get a topological ordering of the nodes in the network.
        
        A topological ordering ensures that all parent nodes appear before their children.
        This ordering is useful for sampling (sample parents before children) and
        computing joint probabilities.
        
        Returns:
            List of nodes in topological order
            
        Note:
            If the network has cycles, this may not return all nodes or may behave unexpectedly.
            Bayesian networks should always be directed acyclic graphs (DAGs).
        """
        ordering = []
        visited = set()
        
        def visit(node: BayesianNode) -> None:
            """Depth-first search helper for topological sort."""
            if node in visited:
                return
            visited.add(node)
            
            # Visit all parents first
            for parent in node.parents:
                visit(parent)
            
            # Then add this node
            ordering.append(node)
        
        # Visit all nodes
        for node in self.nodes:
            visit(node)
        
        self.order = ordering

    def _build_names_to_nodes(self) -> None:
        """
        Build a dictionary mapping node names to nodes.
        """
        self.names_to_nodes = {node.name: node for node in self.nodes}
    
    def get_node_by_name(self, name: str) -> Optional[BayesianNode]:
        """
        Get a node by its name. Returns the first node with the given name.
        """
        return self.names_to_nodes[name]

def init_zero_conditional_probability_tables(network: BayesianNetwork) -> Dict[str, np.ndarray]:
    """
    Allocate zero-filled count tables matching each node's CPT shape.
    """
    return {
        node.name: np.zeros_like(node.conditional_prob_table, dtype=float)
        for node in network.nodes
    }

def normalize_counts(network: BayesianNetwork, counts: Dict[str, np.ndarray]) -> None:
    """
    Convert count tables into CPTs by normalizing the counts.
    """
    for node in network.nodes:
        count_table = counts[node.name]
        if not node.parents:
            totals = count_table.sum(axis=1, keepdims=True)
            normalized = np.full_like(count_table, 1.0 / len(node.domain), dtype=float)
            positive_rows = totals.squeeze(-1) > 0
            if np.any(positive_rows):
                normalized[positive_rows] = count_table[positive_rows] / totals[positive_rows]
            node.conditional_prob_table = normalized
            continue

        new_cpt = np.zeros_like(node.conditional_prob_table, dtype=float)
        for parent_indices, _ in node.iter_parent_assignments():
            child_counts = count_table[parent_indices]
            total = float(child_counts.sum())
            if total > 0:
                new_cpt[parent_indices] = child_counts / total
            else:
                new_cpt[parent_indices] = np.ones(len(node.domain), dtype=float) / len(node.domain)
        node.conditional_prob_table = new_cpt


def load_annotation_csv(path: str, include_labels: bool = True) -> List[Dict[str, List[str]]]:
    assignments: Dict[str, List[str]] = {}
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        sample_idx = 0
        for row in reader:
            if include_labels:
                if 'Y' not in assignments:
                    assignments['Y'] = []
                assignments['Y'].append(row['label'])
            for key, value in row.items():
                if key == 'label':
                    continue
                if key.startswith('annotator'):
                    suffix = key[len('annotator'):]
                    var_name = f'A_{suffix}'
                else:
                    var_name = key
                if var_name not in assignments:
                    assignments[var_name] = []
                assignments[var_name].append(value)
            sample_idx += 1
    return [assignments]


def plot_annotator_cpts(network: BayesianNetwork, path: str) -> None:
    """
    Plot the CPTs for the annotators.
    """
    annotator_nodes = [
        node for node in network.nodes
        if node.name.startswith("A_")
    ]
    if not annotator_nodes:
        raise ValueError("No annotator nodes found in the provided network.")

    annotator_nodes.sort(key=lambda node: node.name)

    parent_domain = None
    for node in annotator_nodes:
        parent_candidates = node.parents
        if not parent_candidates:
            raise ValueError(f"A_i node {node.name} has no Y parent.")
        this_parent = parent_candidates[0]
        if parent_domain is None:
            parent_domain = list(this_parent.domain)

    num_nodes = len(annotator_nodes)
    cols = min(3, num_nodes)
    rows = math.ceil(num_nodes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharey=True)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    x_positions = range(len(parent_domain))

    for axis, node in zip(axes, annotator_nodes):
        raw_cpt = np.array(node.conditional_prob_table, dtype=float)
        if raw_cpt.ndim == 3 and raw_cpt.shape[0] == network.batch_size:
            cpt = raw_cpt[0]
        else:
            cpt = raw_cpt

        if cpt.ndim != 2 or cpt.shape[0] != len(parent_domain):
            raise ValueError(
                f"Unexpected CPT shape {cpt.shape} for annotator node {node.name}."
            )

        num_child_vals = len(node.domain)
        bar_width = 0.7 / max(1, num_child_vals)
        offsets = np.linspace(-0.35 + bar_width / 2, 0.35 - bar_width / 2, num_child_vals)

        for offset, child_value, column in zip(offsets, node.domain, cpt.T):
            axis.bar(
                [x + offset for x in x_positions],
                column,
                width=bar_width,
                label=child_value.capitalize(),
            )

        axis.set_xticks(list(x_positions))
        axis.set_xticklabels([label.capitalize() for label in parent_domain])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(node.name)
        axis.set_ylabel("P(A_i | Y)")
        axis.legend(loc="upper center", fontsize="small")

    for axis in axes[num_nodes:]:
        axis.axis("off")

    fig.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def plot_label_cpt(network: BayesianNetwork, path: str) -> None:
    """
    Plot the CPT for the labels.
    """
    true_data_path = 'data/annotations.csv'
    labels_node = network.get_node_by_name('Y')
    raw_cpt = np.array(labels_node.conditional_prob_table, dtype=float)
    true_data = load_annotation_csv(true_data_path, include_labels=True)
    true_labels = true_data[0]['Y']
    true_good = np.array([label == 'good' for label in true_labels])
    estimated_good = raw_cpt[:, 0]
    
    # Split estimated_good scores by true label
    estimated_good_when_true = estimated_good[true_good]
    estimated_good_when_false = estimated_good[~true_good]
    
    # Create figure with two subplots (top and bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Top histogram: estimated_good scores where true_good = True
    # make sure limits are set to 0-1
    ax1.hist(estimated_good_when_true, bins=50, alpha=0.7, color='green', edgecolor='black', range=(-0.001, 1.001))
    ax1.set_xlabel('Estimated P(good)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Estimated P(good) for True Good Labels')
    ax1.grid(True, alpha=0.3)
    
    # Bottom histogram: estimated_good scores where true_good = False
    ax2.hist(estimated_good_when_false, bins=50, alpha=0.7, color='red', edgecolor='black', range=(-0.001, 1.001))
    ax2.set_xlabel('Estimated P(good)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Estimated P(good) for True Bad Labels')
    ax2.grid(True, alpha=0.3)

    # set xlim to 0-1
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
