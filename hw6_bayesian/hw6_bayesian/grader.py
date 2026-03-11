#!/usr/bin/python3
import grader_util
import sys
import random
import numpy as np
from itertools import product
from typing import Dict, List, Tuple
from util import (
    BayesianNetwork,
    BayesianNode,
    init_zero_conditional_probability_tables,
)

grader = grader_util.Grader()
submission = grader.load('submission')

NUM_SAMPLES = 1000


############################################################
# Check python version
############################################################

import warnings

if not (sys.version_info[0] == 3 and sys.version_info[1] >= 8):
    warnings.warn("Recommended Python version: 3.8 or higher\n")

############################################################
# Problem 1: Basic Bayes
############################################################

grader.add_manual_part('1a', max_points=2, description='Bayesian network diagram')
grader.add_manual_part('1b', max_points=2, description='joint probability expression')
grader.add_manual_part('1c', max_points=2, description='marginal probability p(WetGrass)')
grader.add_manual_part('1d', max_points=2, description='likelihood calculation')
grader.add_manual_part('1e', max_points=2, description='independence analysis')

############################################################
# Problem 2: Sampling
############################################################

def test_2a_0():
    """Test that the phylogenetic tree is initialized correctly with mutation_rate=0.1."""
    mutation_rate = 0.1
    network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=1)
    
    # Check that network is a BayesianNetwork
    grader.require_is_true(isinstance(network, BayesianNetwork))
    grader.require_is_equal(1, network.batch_size)
    
    # Check that we have the right number of nodes
    grader.require_is_equal(4, len(network.nodes))

    # Check that all nodes have the correct domain (nucleotides)
    expected_domain = ['A', 'C', 'T', 'G']
    for node in network.nodes:
        grader.require_is_equal(expected_domain, node.domain)

    # Get nodes by name for easier checking
    nodes_by_name = {node.name: node for node in network.nodes}

    # Check that all expected nodes exist
    expected_names = ['Aryamus bayus', 'Humblus studentus', 'Thomas bayus', 'Kenius bayus']
    for name in expected_names:
        grader.require_is_true(name in nodes_by_name)

    # Check parent-child relationships
    # Thomas bayus should be the root (no parents)
    thomas = nodes_by_name['Thomas bayus']
    grader.require_is_equal(0, len(thomas.parents))

    # Ensure Thomas bayus prior replicated across batch dimension
    grader.require_is_equal((network.batch_size, 4), thomas.conditional_prob_table.shape)

    # Humblus studentus should have Aryamus bayus as parent
    humblus = nodes_by_name['Humblus studentus']
    grader.require_is_equal(1, len(humblus.parents))
    grader.require_is_equal('Thomas bayus', humblus.parents[0].name)
    
    # Thomas bayus should have Aryamus bayus as parent
    aryamus = nodes_by_name['Aryamus bayus']
    grader.require_is_equal(1, len(aryamus.parents))
    grader.require_is_equal('Thomas bayus', aryamus.parents[0].name)
    
    # Kenius bayus should have Thomas bayus as parent
    kenius = nodes_by_name['Kenius bayus']
    grader.require_is_equal(1, len(kenius.parents))
    grader.require_is_equal('Aryamus bayus', kenius.parents[0].name)
    
    # Check that Thomas bayus has uniform prior
    grader.require_is_true(np.allclose(thomas.conditional_prob_table, 0.25))

    # Check CPT structure for non-root nodes
    # Expected CPT: diagonal = (1 - mutation_rate), off-diagonal = mutation_rate / 3
    expected_cpt = np.eye(4) * (1 - mutation_rate) + (1 - np.eye(4)) * (mutation_rate / 3)
    
    for node_name in ['Humblus studentus', 'Aryamus bayus', 'Kenius bayus']:
        node = nodes_by_name[node_name]
        grader.require_is_equal((4, 4), node.conditional_prob_table.shape)
        grader.require_is_true(
            np.allclose(node.conditional_prob_table, expected_cpt),
        )

grader.add_basic_part('2a-0-basic', test_2a_0, max_points=1, 
                      description='Test phylogenetic tree initialization with mutation rate')

def test_2a_1():
    """Test CPT mutation model with different mutation rates."""
    for mutation_rate in [0.0, 0.05, 0.2, 0.5]:
        for genome_length in [1, 10, 50]:
            network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)
            grader.require_is_equal(4, len(network.nodes))
            grader.require_is_equal(genome_length, network.batch_size)
            nodes_by_name = {node.name: node for node in network.nodes}

            thomas = nodes_by_name['Thomas bayus']
            grader.require_is_equal(0, len(thomas.parents))
            grader.require_is_equal((genome_length, 4), thomas.conditional_prob_table.shape)
            
            # Expected CPT for mutation model
            expected_cpt = np.eye(4) * (1 - mutation_rate) + (1 - np.eye(4)) * (mutation_rate / 3)
            
            # Check that each row sums to 1
            for i in range(4):
                row_sum = np.sum(expected_cpt[i, :])
                grader.require_is_true(np.isclose(row_sum, 1.0))
            
            # Check CPT for each non-root node
            for node_name in ['Humblus studentus', 'Aryamus bayus', 'Kenius bayus']:
                node = nodes_by_name[node_name]
                grader.require_is_equal((4, 4), node.conditional_prob_table.shape)
                
                # Verify diagonal elements (no mutation)
                for i in range(4):
                    actual = node.conditional_prob_table[i, i]
                    expected = 1 - mutation_rate
                    grader.require_is_true(np.isclose(actual, expected))
                
                # Verify off-diagonal elements (mutation)
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            actual = node.conditional_prob_table[i, j]
                            expected = mutation_rate / 3
                            grader.require_is_true(np.isclose(actual, expected))

grader.add_basic_part('2a-1-basic', test_2a_1, max_points=1,
                      description='Test mutation model CPT with various mutation rates')

def test_2b_0():
    """Test that forward_sampling returns valid samples."""
    mutation_rate = 0.1
    network = submission.initialize_phylogenetic_tree(mutation_rate)
    
    # Sample once
    sample = submission.forward_sampling(network)
    
    # Check that sample is a dictionary
    grader.require_is_true(isinstance(sample, dict))
    
    # Check that all nodes are in the sample
    expected_names = ['Aryamus bayus', 'Humblus studentus', 'Thomas bayus', 'Kenius bayus']
    for name in expected_names:
        grader.require_is_true(name in sample)
    
    # Check that all sampled values are valid nucleotides
    valid_nucleotides = {'A', 'C', 'T', 'G'}
    for name, value in sample.items():
        grader.require_is_true(value[0] in valid_nucleotides)

grader.add_basic_part('2b-0-basic', test_2b_0, max_points=5,
                      description='Test that forward_sampling produces valid samples')

def test_2c_0():
    """Test that compute_joint_probability returns valid probabilities."""
    mutation_rate = 0.1
    network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=1)
    
    # Test with a specific assignment
    assignment = {
        'Aryamus bayus': ['A'],
        'Humblus studentus': ['A'],
        'Thomas bayus': ['A'],
        'Kenius bayus': ['A']
    }
    
    prob = submission.compute_joint_probability(network, assignment)
    
    # Check that probability is a float
    grader.require_is_true(isinstance(prob, float))
    
    # Check that probability is between 0 and 1
    grader.require_is_true(0 <= prob <= 1)
    
    # Check that probability is positive (since all transitions are possible)
    grader.require_is_true(prob > 0)
    
    # Manually compute expected probability
    # P(A_bayus=A) = 0.25
    # P(H_studentus=A | A_bayus=A) = 1 - mutation_rate = 0.9
    # P(T_bayus=A | A_bayus=A) = 1 - mutation_rate = 0.9
    # P(K_bayus=A | T_bayus=A) = 1 - mutation_rate = 0.9
    expected_prob = 0.25 * (1 - mutation_rate) ** 3
    grader.require_is_true(np.isclose(prob, expected_prob))

grader.add_basic_part('2c-0-basic', test_2c_0, max_points=5,
                      description='Test compute_joint_probability')

grader.add_manual_part('2d', max_points=1, description='Table of sampled values')
grader.add_manual_part('2g', max_points=4, description='Sampling comparison writeup')

def test_2e():
    """Check rejection_sampling estimates P(Thomas | Kenius='A')."""
    mutation_rate = 0.1
    genome_length = 1
    submission_network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)

    def key_to_tuple(key):
        if isinstance(key, str):
            return tuple(key)
        if isinstance(key, (list, tuple)):
            return tuple(key)
        grader.fail(f'Unsupported key type {type(key)} returned from rejection_sampling')
        return tuple()

    def normalize_result(result_dict, domain):
        canonical = {base: 0.0 for base in domain}
        for key, value in result_dict.items():
            key_tuple = key_to_tuple(key)
            grader.require_is_true(len(key_tuple) == genome_length)
            base = key_tuple[0]
            if base not in canonical:
                grader.fail(f'Unexpected nucleotide \"{base}\" in rejection_sampling result')
                continue
            canonical[base] += float(value)
        total_mass = sum(canonical.values())
        grader.require_is_true(total_mass > 0)
        return {base: prob / total_mass for base, prob in canonical.items()}

    # Fix RNG seeds for reproducibility across implementations that use numpy/random
    random.seed(0)
    np.random.seed(0)
    submission_raw = submission.rejection_sampling(
        submission_network, 'Thomas bayus', {'Kenius bayus': ['A']}, num_samples=NUM_SAMPLES)
    grader.require_is_true(isinstance(submission_raw, dict))

    thomas_submission = submission_network.get_node_by_name('Thomas bayus')
    submission_normalized = normalize_result(submission_raw, thomas_submission.domain)
    grader.require_is_true(abs(sum(submission_normalized.values()) - 1.0) < 0.05)


    same = 1.0 - mutation_rate
    diff = mutation_rate / 3.0
    expected_a = same * same + 3 * diff * diff
    expected_other = diff * same + same * diff + 2 * diff * diff
    for base in thomas_submission.domain:
        target = expected_a if base == 'A' else expected_other
        grader.require_is_true(abs(submission_normalized[base] - target) < 0.05)

def test_2f():
    """Check gibbs_sampling estimates P(Thomas='AAAC' | Kenius='AAAA')."""
    mutation_rate = 0.1
    genome_length = 4
    submission_network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)

    evidence = ['A'] * genome_length

    def run_gibbs(gibbs_fn, network):
        return gibbs_fn(network, 'Thomas bayus', {'Kenius bayus': evidence}, num_iterations=NUM_SAMPLES)

    def key_to_tuple(key):
        if isinstance(key, str):
            return tuple(key)
        if isinstance(key, (list, tuple)):
            return tuple(key)
        grader.fail(f'Unsupported key type {type(key)} returned from gibbs_sampling')
        return tuple()

    def parse_distribution(raw_result, domain):
        if isinstance(raw_result, dict) and 'Thomas bayus' in raw_result and isinstance(raw_result['Thomas bayus'], dict):
            raw_local = raw_result['Thomas bayus']
        else:
            raw_local = raw_result

        if isinstance(raw_local, list):
            items = [(entry, 1.0) for entry in raw_local]
        elif isinstance(raw_local, dict):
            items = raw_local.items()
        else:
            grader.fail('gibbs_sampling must return a dict or list of samples')
            return {}

        counts = {}
        for key, value in items:
            key_tuple = key_to_tuple(key)
            if len(key_tuple) != genome_length:
                grader.fail('Each sampled genome must have length 4')
                return {}
            if any(nuc not in domain for nuc in key_tuple):
                grader.fail('Gibbs sampler produced an invalid nucleotide')
                continue
            counts[key_tuple] = counts.get(key_tuple, 0.0) + float(value)

        total = sum(counts.values())
        grader.require_is_true(total > 0)
        distribution = {seq: weight / total for seq, weight in counts.items()}
        grader.require_is_true(abs(sum(distribution.values()) - 1.0) < 0.05)
        return distribution

    random.seed(0)
    np.random.seed(0)
    submission_raw = run_gibbs(submission.gibbs_sampling, submission_network)
    thomas_node = submission_network.get_node_by_name('Thomas bayus')
    submission_distribution = parse_distribution(submission_raw, thomas_node.domain)

    target_seq = tuple('AAAC')
    estimated = submission_distribution.get(target_seq, 0.0)

    same = 1.0 - mutation_rate
    diff = mutation_rate / 3.0
    p_a = same * same + 3 * diff * diff
    p_not_a = diff * same + same * diff + 2 * diff * diff
    expected = (p_a ** 3) * p_not_a

    grader.require_is_true(abs(estimated - expected) < 0.05)


grader.add_basic_part('2e', test_2e, max_points=3,
                      description='Rejection sampling implementation')

grader.add_basic_part('2f', test_2f, max_points=5,
                      description='Gibbs sampling implementation')

def test_3d_0():
    """Validate the structure of the annotator Bayesian network."""
    num_annotators = 4
    dataset_size = 25
    network = submission.bayesian_network_for_annotators(num_annotators, dataset_size)

    grader.require_is_true(isinstance(network, BayesianNetwork))
    grader.require_is_equal(dataset_size, network.batch_size)

    nodes_by_name = {node.name: node for node in network.nodes}
    grader.require_is_true('Y' in nodes_by_name)

    labels_node = nodes_by_name['Y']
    grader.require_is_equal(['good', 'bad'], labels_node.domain)
    grader.require_is_equal(0, len(labels_node.parents))
    grader.require_is_equal((dataset_size, 2), labels_node.conditional_prob_table.shape)
    grader.require_is_true(np.allclose(labels_node.conditional_prob_table, 0.5))

    for i in range(num_annotators):
        annotator_name = f'A_{i}'
        grader.require_is_true(annotator_name in nodes_by_name)
        annotator_node = nodes_by_name[annotator_name]
        grader.require_is_equal(['good', 'bad'], annotator_node.domain)
        grader.require_is_equal(1, len(annotator_node.parents))
        grader.require_is_equal('Y', annotator_node.parents[0].name)
        grader.require_is_equal((2, 2), annotator_node.conditional_prob_table.shape)

    expected_node_count = num_annotators + 1  # annotators + labels
    grader.require_is_equal(expected_node_count, len(network.nodes))

grader.add_basic_part('3d-0-basic', test_3d_0, max_points=1,
                      description='Annotator network structure')

def test_3d_1():
    """Ensure annotator CPTs have diagonal dominance."""
    np.random.seed(0)
    network1 = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)

    for idx in range(3):
        node1 = network1.get_node_by_name(f'A_{idx}')
        grader.require_is_true(node1.conditional_prob_table.shape == (2, 2))
        grader.require_is_true(node1.conditional_prob_table[0, 0] > node1.conditional_prob_table[0, 1])
        grader.require_is_true(node1.conditional_prob_table[0, 0] > node1.conditional_prob_table[1, 0])
        grader.require_is_true(node1.conditional_prob_table[1, 1] > node1.conditional_prob_table[1, 0])
        grader.require_is_true(node1.conditional_prob_table[1, 1] > node1.conditional_prob_table[0, 1])

grader.add_basic_part('3d-1-basic', test_3d_1, max_points=1,
                      description='Annotator CPT diagonal dominance')

def test_3d_hidden():
    """Hidden test for accumulate_assignment under randomised inputs."""
    random.seed(31415)
    np.random.seed(31415)

    num_annotators = 6
    dataset_size = 4
    network = submission.bayesian_network_for_annotators(num_annotators, dataset_size)

    student_counts = init_zero_conditional_probability_tables(network)
    expected_counts = init_zero_conditional_probability_tables(network)

    label_domain = ['good', 'bad']

    for _ in range(5):
        assignment = {
            'Y': [random.choice(label_domain) for _ in range(dataset_size)],
        }
        for annot_idx in range(num_annotators):
            assignment[f'A_{annot_idx}'] = [
                random.choice(label_domain) for _ in range(dataset_size)
            ]

        weight = float(np.random.uniform(0.25, 1.75))


    for name in student_counts:
        grader.require_is_true(
            np.allclose(student_counts[name], expected_counts[name], atol=1e-8, rtol=1e-6)
        )

grader.add_hidden_part('3d-2-hidden', test_3d_hidden, max_points=2, max_seconds=1,
                       description='Hidden annotator network counts')

def test_3e_0():
    """Check accumulate_assignment tallies counts correctly."""
    network = submission.bayesian_network_for_annotators(num_annotators=2, dataset_size=1)
    counts = init_zero_conditional_probability_tables(network)
    assignment = {
        'Y': ['good'],
        'A_0': ['good'],
        'A_1': ['bad'],
    }
    submission.accumulate_assignment(counts, network, assignment)

    labels_counts = counts['Y'][0]
    grader.require_is_true(labels_counts[0] > labels_counts[1])

    annot0_counts = counts['A_0'][0]
    grader.require_is_true(annot0_counts[0] > annot0_counts[1])

    annot1_counts = counts['A_1'][0]
    grader.require_is_true(annot1_counts[1] > annot1_counts[0])

grader.add_basic_part('3e-0-basic', test_3e_0, max_points=2,
                      description='accumulate_assignment counting')

def test_3e_1():
    """Check mle_estimation computes MLE CPTs without smoothing."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    data = [
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['bad'], 'A_0': ['bad']},
        {'Y': ['bad'], 'A_0': ['bad']},
    ]
    trained = submission.mle_estimation(network, data, lambda_param=0.0)

    labels_node = trained.get_node_by_name('Y')
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 0], 0.5))
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 1], 0.5))

    annot_node = trained.get_node_by_name('A_0')
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 0], 1.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 1], 0.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 0], 0.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 1], 1.0))

grader.add_basic_part('3e-1-basic', test_3e_1, max_points=3,
                      description='mle_estimation parameter learning')

def test_3e_2():
    """Check mle_estimation computes MLE CPTs with smoothing."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    data = [
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['bad'], 'A_0': ['bad']},
        {'Y': ['bad'], 'A_0': ['bad']},
    ]
    trained = submission.mle_estimation(network, data, lambda_param=1.0)

    labels_node = trained.get_node_by_name('Y')
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 0], 0.5))
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 1], 0.5))

    annot_node = trained.get_node_by_name('A_0')
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 0], 3 / 4))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 1], 1 / 4))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 0], 1 / 4))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 1], 3 / 4))

grader.add_basic_part('3e-2-basic', test_3e_2, max_points=2,
                      description='mle_estimation parameter learning')

def test_3f():
    """Fit the annotator network parameters using MLE."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)
    data = [
        {'Y': ['good', 'bad'], 'A_0': ['good', 'bad']},
    ]
    trained = submission.mle_estimation(network, data, lambda_param=0.0)
    grader.require_is_true(np.isclose(trained.get_node_by_name('Y').conditional_prob_table[0, 0], 1.0))
    grader.require_is_true(np.isclose(trained.get_node_by_name('Y').conditional_prob_table[0, 1], 0.0))

grader.add_basic_part('3f', test_3f, max_points=2,
                      description='MLE for annotators')

def test_3f_hidden():
    """Fit the annotator network parameters using MLE with random data."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=10)

grader.add_hidden_part('3f-hidden', test_3f_hidden, max_points=2, max_seconds=1,
                       description='MLE for annotators with random data')

def test_4a_0():
    """Ensure e_step matches direct counts when data is fully observed."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    assignment = {'Y': ['good'], 'A_0': ['bad']}
    expected_counts = init_zero_conditional_probability_tables(network)
    submission.accumulate_assignment(expected_counts, network, assignment, weight=1.0)

    completions, weights, indices = submission.e_step(network, [assignment])
    grader.require_is_equal(1, len(completions))
    grader.require_is_equal(1, len(weights))
    grader.require_is_equal(1, len(indices))
    grader.require_is_equal(assignment, completions[0])
    grader.require_is_equal(1.0, weights[0])
    grader.require_is_equal([0], indices[0])

grader.add_basic_part('4a-0-basic', test_4a_0, max_points=2,
                      description='Expectation step expected counts')

def test_4a_1():
    """Check e_step handles hidden label with normalized weights."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    assignment = {'A_0': ['good']}

    completions, weights, indices = submission.e_step(network, [assignment])
    grader.require_is_equal(2, len(completions))
    grader.require_is_equal(2, len(weights))
    grader.require_is_equal(2, len(indices))
    grader.require_is_true(np.isclose(sum(weights), 1.0))

grader.add_basic_part('4a-1-basic', test_4a_1, max_points=2,
                      description='Expectation step with hidden variables')

def test_4a_2():
    """Harder e_step case with multiple hidden combinations."""
    network = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)
    labels_node = network.get_node_by_name('Y')
    labels_node.conditional_prob_table[:] = np.array([[0.6, 0.4]])
    for i in range(3):
        annot_node = network.get_node_by_name(f'A_{i}')
        annot_node.conditional_prob_table[:] = np.array([[0.9, 0.1], [0.2, 0.8]])

    assignment = {
        'A_0': ['good'],
        'A_1': ['good'],
    }

    completions, weights, indices = submission.e_step(network, [assignment])
    grader.require_is_equal(4, len(completions))
    grader.require_is_equal(4, len(weights))
    grader.require_is_equal(4, len(indices))
    grader.require_is_true(np.isclose(sum(weights), 1.0))

    expected = {
        ('good', 'good'): 0.8713147410358565,
        ('good', 'bad'): 0.09681274900398405,
        ('bad', 'good'): 0.006374501992031873,
        ('bad', 'bad'): 0.02549800796812749,
    }

    actual = {}
    for comp, wt in zip(completions, weights):
        grader.require_is_equal(comp['A_0'], ['good'])
        grader.require_is_equal(comp['A_1'], ['good'])
        key = (comp['Y'][0], comp['A_2'][0])
        actual[key] = wt

    for key, target in expected.items():
        grader.require_is_true(np.isclose(actual.get(key, 0.0), target, atol=1e-3))

grader.add_basic_part('4a-2-basic', test_4a_2, max_points=3,
                      description='Expectation step multiple hidden vars')

def test_4a_hidden():
    """Hidden e_step test with random partially observed data."""
    random.seed(271828)
    np.random.seed(271828)


grader.add_hidden_part('4a-3-hidden', test_4a_hidden, max_points=3, max_seconds=1,
                       description='Hidden e_step randomised checks')

def test_4b_0():
    """Validate m_step normalizes CPTs from weighted completions."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    completions = [
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['bad'], 'A_0': ['bad']},
    ]
    weights = [0.8, 0.2]
    indices = [[0], [0]]
    trained = submission.m_step(network, completions, weights, indices)

    labels_node = trained.get_node_by_name('Y')
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 0], 0.8))
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 1], 0.2))

    annot_node = trained.get_node_by_name('A_0')
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 0], 1.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[0, 1], 0.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 0], 0.0))
    grader.require_is_true(np.isclose(annot_node.conditional_prob_table[1, 1], 1.0))

grader.add_basic_part('4b-0-basic', test_4b_0, max_points=2,
                      description='Maximization step parameter update')

def test_4b_1():
    """Harder m_step case with multiple weighted completions."""
    network = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)
    completions = [
        {'Y': ['good'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['good']},
        {'Y': ['good'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['bad']},
        {'Y': ['bad'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['good']},
        {'Y': ['bad'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['bad']},
    ]
    weights = [0.8713147410358565, 0.09681274900398405, 0.006374501992031873, 0.02549800796812749]
    indices = [[0], [0], [0], [0]]

    trained = submission.m_step(network, completions, weights, indices)

    labels_node = trained.get_node_by_name('Y')
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 0], 0.9681274900398406))
    grader.require_is_true(np.isclose(labels_node.conditional_prob_table[0, 1], 0.03187250996015936))

    annot2 = trained.get_node_by_name('A_2')
    grader.require_is_true(np.isclose(annot2.conditional_prob_table[0, 0], 0.9))
    grader.require_is_true(np.isclose(annot2.conditional_prob_table[0, 1], 0.1))
    grader.require_is_true(np.isclose(annot2.conditional_prob_table[1, 0], 0.2))
    grader.require_is_true(np.isclose(annot2.conditional_prob_table[1, 1], 0.8))

grader.add_basic_part('4b-1-basic', test_4b_1, max_points=3,
                      description='Maximization step multiple completions')

def test_4b_hidden():
    """Hidden m_step test comparing against solution implementation."""
    random.seed(12345)
    np.random.seed(12345)

    
grader.add_hidden_part('4b-2-hidden', test_4b_hidden, max_points=2, max_seconds=1,
                       description='Hidden m_step randomised checks')

def test_4c_0():
    """Check EM converges on fully hidden labels."""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)

    labels_node = network.get_node_by_name('Y')
    labels_node.conditional_prob_table[:] = np.array([[0.5, 0.5], [0.5, 0.5]])
    annot_node = network.get_node_by_name('A_0')
    annot_node.conditional_prob_table[:] = np.array([[0.8, 0.2], [0.3, 0.7]])

    data = [
        {'A_0': ['good', 'bad']},
        {'A_0': ['good', 'bad']},
        {'A_0': ['good', 'bad']},
        {'A_0': ['bad', 'good']},
    ]

    trained = submission.em_learn(network, data, num_iterations=10)

    labels_node = trained.get_node_by_name('Y')
    grader.require_is_true(labels_node.conditional_prob_table[0, 0] > 0.6)
    grader.require_is_true(labels_node.conditional_prob_table[0, 1] < 0.4)

    annot_node = trained.get_node_by_name('A_0')
    grader.require_is_true(annot_node.conditional_prob_table[0, 0] > annot_node.conditional_prob_table[0, 1])
    grader.require_is_true(annot_node.conditional_prob_table[1, 1] > annot_node.conditional_prob_table[1, 0])

grader.add_basic_part('4c-0-basic', test_4c_0, max_points=3,
                      description='EM convergence on hidden labels')

def test_4c_1():
    """Ensure EM updates parameters from user-provided initialisation."""
    np.random.seed(1)
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)

    labels_initial = network.get_node_by_name('Y').conditional_prob_table.copy()
    annot_initial = network.get_node_by_name('A_0').conditional_prob_table.copy()

    data = [
        {'A_0': ['good', 'bad']},
        {'A_0': ['good', 'bad']},
        {'A_0': ['good', 'bad']},
        {'A_0': ['bad', 'good']},
    ]

    trained = submission.em_learn(network, data, num_iterations=5)
    labels_final = trained.get_node_by_name('Y').conditional_prob_table
    annot_final = trained.get_node_by_name('A_0').conditional_prob_table

    grader.require_is_true(not np.allclose(labels_final, labels_initial))
    grader.require_is_true(not np.allclose(annot_final, annot_initial))

grader.add_basic_part('4c-1-basic', test_4c_1, max_points=2,
                      description='EM updates from random init')

############################################################
# Run grading
############################################################

grader.grade()
