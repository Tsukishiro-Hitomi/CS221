#!/usr/bin/env python3

import collections
import grader_util
import random

grader = grader_util.Grader()
submission = grader.load('submission')

############################################################
# check python version

import sys
import warnings

if not (sys.version_info[0]==3 and sys.version_info[1]==12):
    warnings.warn("Must be using Python 3.12 \n")



############################################################
##### Problem 1 (Linear Algebra) ###########################
############################################################

grader.add_manual_part('1a', max_points=2, description='NumPy tutor session link')
grader.add_manual_part('1b', max_points=3, description='Matrix multiplication complexity')
grader.add_manual_part('1c', max_points=2, description='Einsum tutor session link')
grader.add_manual_part('1d', max_points=3, description='Einstein summation (written)')

# Problem 1e: linear_project
def test1e0():
    import numpy as np
    x = np.array([[1.0, 2.0, 3.0],
                  [0.0, -1.0, 4.0]])  # (B=2, Din=3)
    W = np.array([[1.0, 0.0],
                  [0.5, -1.0],
                  [2.0, 3.0]])        # (Din=3, Dout=2)
    b = np.array([0.1, -0.2])         # (Dout=2)
    expected = x @ W + b
    out = submission.linear_project(x, W, b)
    grader.require_is_equal(expected, out)


def test1e1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        B, Din, Dout = 4, 5, 3
        x = rng.standard_normal((B, Din))
        W = rng.standard_normal((Din, Dout))
        b = rng.standard_normal(Dout)
        expected = x @ W + b
        out = submission.linear_project(x, W, b)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1e-0-basic', test1e0, max_points=1, description='linear_project with bias (small deterministic)')
grader.add_hidden_part('1e-1-hidden', test1e1, max_points=2, description='linear_project randomized, with bias')

# Problem 1f: split_last_dim pattern string
def test1f0():
    import numpy as np
    from einops import rearrange
    x = np.arange(12, dtype=float).reshape(2, 6)  # (B=2, D=6)
    num_groups = 3
    expected = x.reshape(2, num_groups, 6 // num_groups)
    pattern = submission.split_last_dim_pattern()
    out = rearrange(x, pattern, g=num_groups)
    grader.require_is_equal(expected, out)


def test1f1():
    import numpy as np
    from einops import rearrange
    rng = np.random.default_rng(1)
    for _ in range(3):
        B, num_groups = 3, 4
        D = 20
        x = rng.standard_normal((B, D))
        expected = x.reshape(B, num_groups, D // num_groups)
        pattern = submission.split_last_dim_pattern()
        out = rearrange(x, pattern, g=num_groups)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1f-0-basic', test1f0, max_points=1, description='split_last_dim pattern applies correctly')
grader.add_hidden_part('1f-1-hidden', test1f1, max_points=2, description='split_last_dim pattern randomized')

# Problem 1g: normalized_inner_products
def test1g0():
    import numpy as np
    A = np.array([[[1., 0.], [0., 1.]]])  # (B=1, M=2, D=2)
    Bm = np.array([[[1., 2.], [3., 4.], [0., 1.]]])  # (B=1, N=3, D=2)
    expected = np.einsum('bmd,bnd->bmn', A, Bm)
    out = submission.normalized_inner_products(A, Bm, normalize=False)
    grader.require_is_equal(expected, out)
    # normalized
    D = A.shape[-1]
    expected_norm = expected / np.sqrt(D)
    out_norm = submission.normalized_inner_products(A, Bm, normalize=True)
    grader.require_is_equal(expected_norm, out_norm)


def test1g1():
    import numpy as np
    rng = np.random.default_rng(2)
    for _ in range(3):
        B, M, N, D = 2, 3, 4, 5
        A = rng.standard_normal((B, M, D))
        Bm = rng.standard_normal((B, N, D))
        exp = np.einsum('bmd,bnd->bmn', A, Bm)
        out = submission.normalized_inner_products(A, Bm, normalize=False)
        grader.require_is_equal(exp, out)
        expn = exp / np.sqrt(D)
        outn = submission.normalized_inner_products(A, Bm, normalize=True)
        grader.require_is_equal(expn, outn)


grader.add_basic_part('1g-0-basic', test1g0, max_points=1, description='normalized_inner_products small case + normalization')
grader.add_hidden_part('1g-1-hidden', test1g1, max_points=2, description='normalized_inner_products randomized')

# Problem 1h: mask_strictly_upper
def test1h0():
    import numpy as np
    B, L = 1, 4
    scores = np.arange(B * L * L, dtype=float).reshape(B, L, L)
    out = submission.mask_strictly_upper(scores.copy())
    expected = scores.copy()
    triu_rows, triu_cols = np.triu_indices(L, k=1)
    expected[:, triu_rows, triu_cols] = -np.inf
    # Check non-inf values first
    non_inf_mask = ~np.isinf(expected)
    grader.require_is_equal(expected[non_inf_mask], out[non_inf_mask])
    # Check inf values separately
    inf_mask = np.isinf(expected)
    grader.require_is_equal(np.all(np.isinf(out[inf_mask])), True)


def test1h1():
    import numpy as np
    rng = np.random.default_rng(3)
    for _ in range(3):
        B, L = 2, 5
        scores = rng.standard_normal((B, L, L))
        out = submission.mask_strictly_upper(scores.copy())
        expected = scores.copy()
        rr, cc = np.triu_indices(L, k=1)
        expected[:, rr, cc] = -np.inf
        # Check non-inf values first
        non_inf_mask = ~np.isinf(expected)
        grader.require_is_equal(expected[non_inf_mask], out[non_inf_mask])
        # Check inf values separately
        inf_mask = np.isinf(expected)
        grader.require_is_equal(np.all(np.isinf(out[inf_mask])), True)


grader.add_basic_part('1h-0-basic', test1h0, max_points=1, description='mask_strictly_upper sets upper triangle to -inf')
grader.add_hidden_part('1h-1-hidden', test1h1, max_points=2, description='mask_strictly_upper randomized')

# Problem 1i: prob_weighted_sum einsum string
def test1i0():
    import numpy as np
    from einops import einsum
    P = np.array([[0.25, 0.25, 0.5]])  # (B=1, N=3)
    V = np.array([[[1., 0.], [0., 1.], [2., 2.]]])  # (B=1, N=3, D=2)
    expected = einsum(P, V, 'b n, b n d -> b d')
    pattern = submission.prob_weighted_sum_einsum()
    out = einsum(P, V, pattern)
    grader.require_is_equal(expected, out)


def test1i1():
    import numpy as np
    from einops import einsum
    rng = np.random.default_rng(4)
    for _ in range(3):
        B, N, D = 2, 5, 3
        P = rng.random((B, N))
        P = P / P.sum(axis=1, keepdims=True)
        V = rng.standard_normal((B, N, D))
        expected = einsum(P, V, 'b n, b n d -> b d')
        pattern = submission.prob_weighted_sum_einsum()
        out = einsum(P, V, pattern)
        grader.require_is_equal(expected, out)


grader.add_basic_part('1i-0-basic', test1i0, max_points=1, description='prob_weighted_sum einsum string deterministic')
grader.add_hidden_part('1i-1-hidden', test1i1, max_points=2, description='prob_weighted_sum einsum string randomized')


############################################################
##### Problem 2 (Calculus & Gradients) #####################
############################################################

grader.add_manual_part('2a', max_points=2, description='Gradient warmup')
grader.add_manual_part('2c', max_points=3, description='Matrix multiplication gradient')

# Problem 2b: gradient_warmup implementation
def test2b0():
    import numpy as np
    w = np.array([1.0, -2.0, 3.0])
    c = np.array([0.0, 1.0, -1.0])
    expected = 2.0 * (w - c)
    out = submission.gradient_warmup(w, c)
    grader.require_is_equal(expected, out)


def test2b1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.integers(2, 8)
        w = rng.standard_normal(d)
        c = rng.standard_normal(d)
        expected = 2.0 * (w - c)
        out = submission.gradient_warmup(w, c)
        grader.require_is_equal(expected, out)


grader.add_basic_part('2b-0-basic', test2b0, max_points=1, description='gradient_warmup deterministic')
grader.add_hidden_part('2b-1-hidden', test2b1, max_points=2, description='gradient_warmup randomized')

# Problem 2d: matrix_grad implementation
def test2d0():
    import numpy as np
    A = np.array([[2., 1., 3.],
                  [4., 5., 6.]])  # (m=2, p=3)
    B = np.array([[7., 8.],
                  [9., 0.],
                  [1., 2.]])      # (p=3, n=2)
    grad_A, grad_B = submission.matrix_grad(A, B)
    # Expected gradients
    row_sum_B = B.sum(axis=1)  # (3,)
    col_sum_A = A.sum(axis=0)  # (3,)
    expected_grad_A = np.ones((A.shape[0], 1)) @ row_sum_B[None, :]
    expected_grad_B = col_sum_A[:, None] @ np.ones((1, B.shape[1]))
    grader.require_is_equal(expected_grad_A, grad_A)
    grader.require_is_equal(expected_grad_B, grad_B)


def test2d1():
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(3):
        m = rng.integers(2, 5)
        p = rng.integers(2, 5)
        n = rng.integers(2, 5)
        A = rng.standard_normal((m, p))
        B = rng.standard_normal((p, n))
        grad_A, grad_B = submission.matrix_grad(A, B)

        # Numeric check via central differences (function is linear; should match exactly)
        eps = 1e-6
        # Check a subset or all entries for small sizes
        num_grad_A = np.zeros_like(A)
        for i in range(m):
            for k in range(p):
                E = np.zeros_like(A)
                E[i, k] = eps
                s_plus = np.sum((A + E) @ B)
                s_minus = np.sum((A - E) @ B)
                num_grad_A[i, k] = (s_plus - s_minus) / (2 * eps)
        num_grad_B = np.zeros_like(B)
        for k in range(p):
            for j in range(n):
                E = np.zeros_like(B)
                E[k, j] = eps
                s_plus = np.sum(A @ (B + E))
                s_minus = np.sum(A @ (B - E))
                num_grad_B[k, j] = (s_plus - s_minus) / (2 * eps)
        grader.require_is_equal(num_grad_A, grad_A)
        grader.require_is_equal(num_grad_B, grad_B)

# Problem 2e: finite differences vs analytic gradient
def test2e0():
    import numpy as np
    rng = np.random.default_rng(42)
    n, d = 5, 4
    A = rng.standard_normal((n, d))
    b = rng.standard_normal(n)
    w = rng.standard_normal(d)
    g_analytic = submission.lsq_grad(w, A, b)
    g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=1e-5)
    grader.require_is_equal(g_analytic, g_numeric)


def test2e1():
    import numpy as np
    rng = np.random.default_rng(99)
    for _ in range(5):
        n = rng.integers(3, 8)
        d = rng.integers(3, 8)
        A = rng.standard_normal((n, d))
        b = rng.standard_normal(n)
        w = rng.standard_normal(d)
        g_analytic = submission.lsq_grad(w, A, b)
        g_numeric = submission.lsq_finite_diff_grad(w, A, b, epsilon=3e-6)
        grader.require_is_equal(g_analytic, g_numeric)

grader.add_basic_part('2d-0-basic', test2d0, max_points=1, description='matrix_grad deterministic')
grader.add_hidden_part('2d-1-hidden', test2d1, max_points=2, description='matrix_grad randomized + numeric check')

grader.add_basic_part('2e-0-basic', test2e0, max_points=1, description='finite differences matches analytic gradient (single case)')
grader.add_hidden_part('2e-1-hidden', test2e1, max_points=2, description='finite differences matches analytic gradient (random cases)')


############################################################
##### Problem 3 (Optimization) #############################
############################################################

grader.add_manual_part('3a', max_points=3, description='Weighted scalar quadratic minimizer')
grader.add_manual_part('3b', max_points=2, description='Gradient descent tutor session link')

# Problem 3c: gradient_descent_quadratic (code)
def test3c0():
    import numpy as np
    # Simple deterministic case: weighted average should be the minimizer
    x = np.array([0.0, 10.0])
    w = np.array([1.0, 3.0])
    theta_star = (w * x).sum() / w.sum()
    theta0 = 100.0
    lr = 0.25 / w.sum()  # stable stepsize (< 1/(2*sum w))
    theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=200)
    grader.require_is_equal(theta_star, theta)


def test3c1():
    import numpy as np
    rng = np.random.default_rng(5)
    for _ in range(5):
        n = rng.integers(2, 8)
        x = rng.standard_normal(n)
        w = rng.random(n) + 0.1  # strictly positive
        theta_star = (w * x).sum() / w.sum()
        theta0 = rng.standard_normal()
        lr = 0.25 / w.sum()
        theta = submission.gradient_descent_quadratic(x, w, theta0, lr, num_steps=300)
        grader.require_is_equal(theta_star, theta)


grader.add_basic_part('3c-0-basic', test3c0, max_points=1, description='gradient descent converges to weighted average (deterministic)')
grader.add_hidden_part('3c-1-hidden', test3c1, max_points=2, description='gradient descent converges on random instances')


############################################################
##### Problem 4 (Ethics in AI): written parts ##############
############################################################

grader.add_manual_part('4a', max_points=2, description='Ethics in AI part a')
grader.add_manual_part('4b', max_points=2, description='Ethics in AI part b')
grader.add_manual_part('4c', max_points=2, description='Ethics in AI part c')
grader.add_manual_part('4d', max_points=2, description='Ethics in AI part d')


############################################################
grader.grade()
