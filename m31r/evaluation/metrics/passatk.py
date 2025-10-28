# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

"""
Pass@k computation.

This implements the standard pass@k metric from the Codex/HumanEval paper.
The idea is simple: if you give the model K attempts to solve a problem,
what's the probability that at least one attempt actually works?

The formula uses exact combinatorics rather than sampling approximation
because we want deterministic, reproducible numbers. The math is:

    pass@k = 1 - C(n-c, k) / C(n, k)

Where:
    n = total number of attempts generated
    c = number of attempts that passed (compiled + tests passed)
    k = how many attempts we're "allowed" to use

If you've generated 10 samples and 3 passed, pass@1 tells you the chance
a single random sample works, pass@5 tells you the chance at least one of
five random samples works, and so on.

We use logarithmic combination to avoid integer overflow on large values.
"""

import math


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k using the unbiased estimator.

    This is the formula from the original Codex paper (Chen et al., 2021).
    It gives you the probability that at least one of k randomly chosen
    samples from n total samples is correct, given that c of the n
    samples are correct.

    We compute it as 1 - C(n-c, k) / C(n, k) using log-space arithmetic
    to handle large factorials without overflow.

    Special cases:
      - If c >= n, everything passed, so pass@k = 1.0
      - If k > n, we can't draw that many samples, so we clamp k to n.
      - If c == 0, nothing passed, so pass@k = 0.0
      - If n-c < k, there aren't enough failures to fill k slots, so pass@k = 1.0

    Args:
        n: Total number of generated samples.
        c: Number of correct samples (compiled + tests passed).
        k: Number of samples we get to pick from.

    Returns:
        Float between 0.0 and 1.0 representing the pass@k probability.
    """
    if n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        k = n
    if k <= 0:
        return 0.0
    if n - c < k:
        return 1.0

    # We compute C(n-c, k) / C(n, k) in log space to avoid overflow.
    #
    # log(C(n-c, k)) - log(C(n, k))
    # = sum(log(n-c-i) for i in 0..k-1) - sum(log(n-i) for i in 0..k-1)
    #
    # This works because C(a, b) = product((a-i)/(i+1), i=0..b-1)
    # and we're dividing two such products, so the (i+1) denominators cancel.
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)

    return 1.0 - math.exp(log_ratio)
