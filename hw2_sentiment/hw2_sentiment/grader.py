#!/usr/bin/python3
import grader_util
import util
import numpy as np
import torch
import torch.nn as nn
import sys

grader = grader_util.Grader()
submission = grader.load('submission')

############################################################
# Warm up einops to prevent timeout issues
############################################################
print("Einops warm-up started")

def warm_up_einops():
    """
    Warm up einops functions to prevent heavy initialization during tests.
    This helps avoid timeout issues in the grader.
    """
    try:
        from einops import reduce, rearrange, einsum

        # Create small test arrays to trigger einops initialization
        test_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        test_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Warm up reduce operations (most commonly used in submission)
        _ = reduce(test_array, "batch num_classes -> batch", "sum")
        _ = reduce(test_array, "batch num_classes -> batch 1", "sum")
        _ = reduce(test_array, "batch num_classes ->", "mean")
        _ = reduce(test_tensor, "words embed -> embed", "mean")

        # Warm up rearrange (if used)
        _ = rearrange(test_array, "batch num_classes -> num_classes batch")

        # Warm up einsum operations
        test_matrix_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        test_matrix_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        _ = einsum(test_matrix_a, test_matrix_b, "i j, j k -> i k")

        print("Einops warm-up completed successfully")

    except Exception as e:
        print(f"Warning: Einops warm-up failed: {e}")

# Perform warm-up
warm_up_einops()

############################################################
# Check python version
############################################################

import warnings

if not (sys.version_info[0] == 3 and sys.version_info[1] >= 8):
    warnings.warn("Recommended Python version: 3.8 or higher\n")

############################################################
# Problem 1: Building intuition for Bag-of-Words and linear classifier
############################################################

grader.add_manual_part('1a', max_points=1, description='feature representation')
grader.add_manual_part('1b', max_points=1, description='softmax computation')
grader.add_manual_part('1c', max_points=2, description='cross-entropy loss')
grader.add_manual_part('1d', max_points=2, description='gradient analysis')

############################################################
# Problem 2: Building intuition for embeddings and multilayer perceptron
############################################################

grader.add_manual_part('2a', max_points=2, description='embeddings via averaging')
grader.add_manual_part('2b', max_points=3, description='forward pass')
grader.add_manual_part('2c', max_points=3, description='backpropagation')
grader.add_manual_part('2d', max_points=2, description='embedding analysis')

############################################################
# Problem 3: NumPy linear classifier
############################################################

### 3a: build_vocabulary
def test3a0():
    examples = ["hello world", "hello python", "world peace", "hello there"]
    vocab = submission.build_vocabulary(examples)

    grader.require_is_true(vocab.size() > 0)
    grader.require_is_true(vocab.get_index("hello") >= 2)
    grader.require_is_true(vocab.get_index("world") >= 2)
    grader.require_is_true(vocab.get_index("nonexistent") == 1)

grader.add_basic_part('3a-0-basic', test3a0, max_seconds=2, description="build_vocabulary test")

### 3b: text_to_features
def test3b0():
    examples = ["hello world", "hello test", "world test"]
    vocab = submission.build_vocabulary(examples)
    text = "hello world"
    features = submission.text_to_features(text, vocab)

    grader.require_is_true(isinstance(features, np.ndarray))
    grader.require_is_equal(features.shape, (vocab.size(),))

    hello_idx = vocab.get_index("hello")
    world_idx = vocab.get_index("world")

    if hello_idx >= 0:
        grader.require_is_equal(features[hello_idx], 1.0)
    if world_idx >= 0:
        grader.require_is_equal(features[world_idx], 1.0)

    text_repeated = "hello hello world"
    features_repeated = submission.text_to_features(text_repeated, vocab)
    if hello_idx >= 0:
        grader.require_is_equal(features_repeated[hello_idx], 2.0)
    if world_idx >= 0:
        grader.require_is_equal(features_repeated[world_idx], 1.0)

    text_unknown = "hello unknown_word"
    features_unknown = submission.text_to_features(text_unknown, vocab)
    grader.require_is_equal(features_unknown.shape, (vocab.size(),))

    features_empty = submission.text_to_features("", vocab)
    grader.require_is_equal(features_empty.shape, (vocab.size(),))
    grader.require_is_true(np.sum(features_empty) == 0)

grader.add_basic_part('3b-0-basic', test3b0, max_seconds=2, description="text_to_features test")

### 3c: numpy_softmax
def test3c0():
    logits = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
    result = submission.numpy_softmax(logits)

    grader.require_is_equal(result.shape, (2, 3))

    row_sums = np.sum(result, axis=1)
    grader.require_is_true(np.allclose(row_sums, 1.0))
    grader.require_is_true(np.all(result > 0))
    grader.require_is_true(np.all(result <= 1.0))

grader.add_basic_part('3c-0-basic', test3c0, max_seconds=2, description="numpy_softmax test")

### 3d: numpy_cross_entropy_loss
def test3d0():
    predictions = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
    targets = np.array([[1, 0, 0], [0, 1, 0]])
    result = submission.numpy_cross_entropy_loss(predictions, targets)

    grader.require_is_true(isinstance(result, (float, np.floating)))

    grader.require_is_greater_than(0, result)

    perfect_pred = np.array([[1.0-1e-10, 1e-10, 1e-10], [1e-10, 1.0-1e-10, 1e-10]])
    perfect_loss = submission.numpy_cross_entropy_loss(perfect_pred, targets)
    grader.require_is_less_than(0.001, perfect_loss)

grader.add_basic_part('3d-0-basic', test3d0, max_points=2, max_seconds=2, description="numpy_cross_entropy_loss test")

### 3e: numpy_compute_gradients
def test3e0():
    batch_size = 2
    num_features = 3
    num_classes = 3

    features = np.random.randn(batch_size, num_features)
    predictions = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    targets = np.array([[1, 0, 0], [0, 0, 1]])

    grad_weights, grad_bias = submission.numpy_compute_gradients(features, predictions, targets)

    grader.require_is_equal(grad_weights.shape, (num_features, num_classes))
    grader.require_is_equal(grad_bias.shape, (1, num_classes))

    grader.require_is_true(np.any(np.abs(grad_weights) > 1e-6))

grader.add_basic_part('3e-0-basic', test3e0, max_seconds=2, description="numpy_compute_gradients test")

### 3f: predict_logistic_regression - basic test
def test3f0():
    np.random.seed(42)
    batch_size = 6
    num_features = 10
    num_classes = 3

    test_features = np.random.randn(batch_size, num_features)
    test_labels = np.eye(num_classes).repeat(batch_size // num_classes, axis=0)

    weights = np.random.randn(num_features, num_classes)
    bias = np.zeros((1, num_classes))

    accuracy = submission.predict_linear_classifier(test_features, test_labels, weights, bias)

    grader.require_is_true(isinstance(accuracy, (float, np.floating)))
    grader.require_is_true(0 <= accuracy <= 1)

grader.add_basic_part('3f-0-basic', test3f0, max_points=2, max_seconds=2, description="predict_linear_classifier test")

### 3g: train_linear_classifier - basic test
def test3g0():
    np.random.seed(42)
    batch_size = 6
    num_features = 10
    num_classes = 3

    train_features = np.random.randn(batch_size, num_features)
    train_labels = np.eye(num_classes).repeat(batch_size // num_classes, axis=0)
    val_features = np.random.randn(3, num_features)
    val_labels = np.eye(num_classes)

    weights, bias = submission.train_linear_classifier(
        train_features, train_labels, val_features, val_labels,
        num_epochs=5, lr=0.01
    )

    grader.require_is_equal(weights.shape, (num_features, num_classes))
    grader.require_is_equal(bias.shape, (1, num_classes))

    grader.require_is_true(np.any(np.abs(weights) > 1e-4))

grader.add_basic_part('3g-0-basic', test3g0, max_seconds=2, description="train_linear_classifier basic test")

### 3g: train_linear_classifier - gradient check
def test3g1():
    np.random.seed(42)

    batch_size = 4
    num_features = 3
    num_classes = 3

    train_features = np.random.randn(batch_size, num_features)
    train_labels = np.eye(num_classes).repeat(batch_size // num_classes + 1, axis=0)[:batch_size]
    val_features = np.random.randn(2, num_features)
    val_labels = np.eye(num_classes)[:2]

    weights, bias = submission.train_linear_classifier(
        train_features, train_labels, val_features, val_labels,
        num_epochs=1, lr=0.1
    )
    grader.require_is_equal(weights.shape, (num_features, num_classes))
    grader.require_is_true(np.any(np.abs(weights) > 1e-6))

grader.add_hidden_part('3g-1-hidden', test3g1, max_seconds=2, description="train_linear_classifier gradient test")

def test3g2():
    """Test NumPy implementation on real tweet data with accuracy requirement."""
    try:
        train_texts, train_labels = util.read_tweet_data('tweet.train')
        val_texts, val_labels = util.read_tweet_data('tweet.test')

        print(f"Training on {len(train_texts)} examples, testing on {len(val_texts)} examples")

        vocab = submission.build_vocabulary(train_texts)

        train_features = np.array([submission.text_to_features(text, vocab) for text in train_texts])
        val_features = np.array([submission.text_to_features(text, vocab) for text in val_texts])

        weights, bias = submission.train_linear_classifier(
            train_features, train_labels, val_features, val_labels,
            num_epochs=15, lr=0.2
        )

        accuracy = submission.predict_linear_classifier(val_features, val_labels, weights, bias)

        print(f"NumPy classifier test accuracy: {accuracy:.4f}")

        grader.require_is_equal(weights.shape[0], vocab.size())
        grader.require_is_equal(weights.shape[1], train_labels.shape[1])
        grader.require_is_true(np.any(np.abs(weights) > 1e-4))

        grader.require_is_greater_than(0.4, accuracy)

    except Exception as e:
        grader.print_exception()


grader.add_basic_part('3g-2-basic', test3g2, max_points=3, max_seconds=8,
                     description="Test NumPy implementation on real tweet data (accuracy ≥ 0.4)")

grader.add_manual_part('3h', max_points=3, description='exploring learning rates')

############################################################
# Problem 5: PyTorch MLP with embeddings
############################################################

### 4a: text_to_average_embedding
def test4a0():
    examples = ["hello world test", "hello test"]
    vocab = submission.build_vocabulary(examples)

    embedding_dim = 8
    embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)

    text = "hello world"
    result = submission.text_to_average_embedding(text, vocab, embedding_layer)

    grader.require_is_equal(result.shape, torch.Size([embedding_dim]))
    grader.require_is_true(torch.is_tensor(result))

    empty_result = submission.text_to_average_embedding("", vocab, embedding_layer)
    grader.require_is_equal(empty_result.shape, torch.Size([embedding_dim]))

grader.add_basic_part('4a-0-basic', test4a0, max_seconds=2, description="text_to_average_embedding test")

### 4b: extract_averaged_features
def test4b0():
    examples = ["hello world", "world peace", "hello peace"]
    vocab = submission.build_vocabulary(examples)

    embedding_dim = 10
    embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)

    texts = ["hello world", "peace world"]
    features = submission.extract_averaged_features(texts, vocab, embedding_layer)

    grader.require_is_equal(features.shape, torch.Size([2, embedding_dim]))
    grader.require_is_true(torch.is_tensor(features))

grader.add_basic_part('4b-0-basic', test4b0, max_seconds=2, description="extract_averaged_features test")

### 4c: MLPClassifier
def test4c0():
    embedding_dim, hidden_dim, output_dim = 16, 8, 3
    model = submission.MLPClassifier(embedding_dim, hidden_dim, output_dim)

    x = torch.randn(4, embedding_dim)
    output = model(x)

    grader.require_is_equal(output.shape, torch.Size([4, output_dim]))

    grader.require_is_true(hasattr(model, 'fc1'))
    grader.require_is_true(hasattr(model, 'fc2'))
    grader.require_is_true(hasattr(model, 'relu'))

grader.add_basic_part('4c-0-basic', test4c0, max_seconds=2, description="MLPClassifier test")

### 4d: torch_softmax, torch_cross_entropy_loss, update_parameter
def test4d0():
    logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
    result = submission.torch_softmax(logits)
    grader.require_is_equal(result.shape, torch.Size([2, 3]))
    row_sums = torch.sum(result, dim=1)
    grader.require_is_true(torch.allclose(row_sums, torch.ones(2)))

    predictions = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = submission.torch_cross_entropy_loss(predictions, targets)
    grader.require_is_equal(loss.shape, torch.Size([]))
    grader.require_is_true(loss.item() > 0)

    param = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    grad = torch.tensor([0.1, 0.2, 0.3])
    lr = 0.5
    original_param = param.clone()
    submission.update_parameter(param, grad, lr)
    expected = original_param - lr * grad
    grader.require_is_true(torch.allclose(param, expected))

grader.add_basic_part('4d-0-basic', test4d0, max_points=3, max_seconds=2, description="torch utility functions test")

### 4e: MLP predictor
def test4e0():
    examples = ["good happy", "bad sad", "great wonderful"]
    vocab = submission.build_vocabulary(examples)

    embedding_dim, hidden_dim, output_dim = 8, 4, 3
    embedding_layer = nn.Embedding(vocab.size(), embedding_dim, padding_idx=0)
    classifier = submission.MLPClassifier(embedding_dim, hidden_dim, output_dim)

    texts = ["good", "bad"]
    labels = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)

    accuracy = submission.predict_mlp(texts, labels, classifier, embedding_layer, vocab)

    grader.require_is_true(isinstance(accuracy, float))
    grader.require_is_true(0 <= accuracy <= 1)

grader.add_basic_part('4e-0-basic', test4e0, max_points=2, max_seconds=2, description="predict_mlp test")

### 4f: train_mlp_classifier - basic test
def test4f0():
    train_texts = ["happy good wonderful", "sad bad terrible", "love great amazing"]
    train_labels = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    val_texts = ["good", "bad"]
    val_labels = np.array([[0, 1, 0], [1, 0, 0]])

    try:
        classifier, embedding_layer, vocab = submission.train_mlp_classifier(
            train_texts, train_labels, val_texts, val_labels,
            num_epochs=3, lr=0.01, embedding_dim=8, hidden_dim=4,
            batch_size=2, num_classes=3
        )

        grader.require_is_true(isinstance(classifier, submission.MLPClassifier))
        grader.require_is_true(isinstance(embedding_layer, nn.Embedding))
        grader.require_is_true(hasattr(vocab, 'size'))

        grader.require_is_true(vocab.size() > 0)

    except Exception as e:
        grader.print_exception()

grader.add_basic_part('4f-0-basic', test4f0, max_seconds=2, description="train_mlp_classifier basic test")

### 4f: train_mlp_classifier - parameter shape test
def test4f1():
    train_texts = ["test one", "test two", "test three"]
    train_labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    val_texts = ["test"]
    val_labels = np.array([[1, 0, 0]])

    try:
        classifier, embedding_layer, vocab = submission.train_mlp_classifier(
            train_texts, train_labels, val_texts, val_labels,
            num_epochs=2, lr=0.1, embedding_dim=12, hidden_dim=6,
            batch_size=1, num_classes=3
        )

        grader.require_is_equal(embedding_layer.embedding_dim, 12)
        grader.require_is_equal(embedding_layer.num_embeddings, vocab.size())

        test_input = torch.randn(1, 12)
        output = classifier(test_input)
        grader.require_is_equal(output.shape, torch.Size([1, 3]))

    except Exception as e:
        grader.print_exception()

grader.add_hidden_part('4f-1-hidden', test4f1, max_seconds=2, description="train_mlp_classifier shape test")


def test4f2():
    """Test PyTorch embedding implementation on real tweet data with accuracy requirement."""
    try:
        train_texts, train_labels = util.read_tweet_data('tweet.train')
        val_texts, val_labels = util.read_tweet_data('tweet.test')

        print(f"Training on {len(train_texts)} examples, testing on {len(val_texts)} examples")

        classifier, embedding_layer, vocab = submission.train_mlp_classifier(
            train_texts, train_labels, val_texts, val_labels,
            num_epochs=15, lr=3e-6, embedding_dim=8, hidden_dim=16,
            batch_size=16, num_classes=train_labels.shape[1]
        )

        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)
        accuracy = submission.predict_mlp(val_texts, val_labels_tensor, classifier, embedding_layer, vocab)

        print(f"PyTorch embedding classifier test accuracy: {accuracy:.4f}")

        grader.require_is_true(isinstance(classifier, submission.MLPClassifier))
        grader.require_is_true(isinstance(embedding_layer, nn.Embedding))
        grader.require_is_true(vocab.size() > 0)

        grader.require_is_greater_than(0.55, accuracy)

    except Exception as e:
        grader.print_exception()


grader.add_basic_part('4f-2-basic', test4f2, max_points=3, max_seconds=40,
                      description="Test PyTorch embedding implementation on real tweet data (accuracy ≥ 0.55)")


############################################################
# Run grading
############################################################

grader.grade()