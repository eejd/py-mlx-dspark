"""Nucleus (top-p) / top-k truncation for lossless speculative *sampling*.

To keep speculative sampling lossless while honoring OpenAI's ``top_p`` / ``top_k``, the
truncation must be applied to the **target** distribution the accept test samples from — and,
so acceptance doesn't collapse, to the **draft** proposal too. This module provides one
primitive, :func:`truncate_probs`, used on both sides (draft q and target p) with the same
parameters. The output is then an exact sample from ``top-p/top-k(softmax(target_logits / T))``.

Identity fast-path: ``top_p >= 1.0`` and ``top_k <= 0`` returns the input untouched, so the
default temperature-sampling path (and greedy) is byte-for-byte unchanged.

No full-vocab sort when ``top_k`` is set. mlx's GPU partition/top-k is a full merge sort of the
row (~130 us at a 152k vocab, ~200 us at 248k-262k on an M4 Pro), and the plain top-k + top-p
path paid two of them per sampled token: ~0.4-0.6 ms, 2.7% of a Qwen3-4B-8bit decode step and
9% of a Qwen3-0.6B one under the model's own ``generation_config`` defaults. The k-th largest
value is the k-th largest of the per-group top-k's (every global top-k element is in its
group's top-k), and the nucleus cutoff after top-k only ever looks at those k values, so both
come from small block sorts instead. Same values, same order, same scan: the output is
bit-identical to the full-sort formulation (``tests/test_sampling.py`` pins it).
"""

from __future__ import annotations

import mlx.core as mx

_GROUPS = 256         # per-group rows of ~500-1000 elements: one block-sort launch, all groups
_SMALL_ROW = 8192     # a row this short is already a single cheap sort


def _topk_values(p: mx.array, k: int) -> mx.array | None:
    """The ``k`` largest values of each row of ``p`` (unsorted), via a two-stage grouped
    top-k — or ``None`` when the row is too short for grouping to beat one full sort."""
    v = p.shape[-1]
    if v <= _SMALL_ROW:
        return None
    g = _GROUPS
    # each group comfortably wider than k, and the g*k candidates well short of the row
    while g > 1 and (-(-v // g) < 2 * k or 4 * g * k > v):
        g //= 2
    if g < 2:
        return None
    c = -(-v // g)
    if c * g != v:
        # zero padding: probabilities are >= 0, so a pad can only tie a real zero, and then
        # the k-th largest value is 0 with or without it
        p = mx.pad(p, [(0, 0)] * (p.ndim - 1) + [(0, c * g - v)])
    cand = mx.topk(p.reshape(*p.shape[:-1], g, c), k, axis=-1)
    return mx.topk(cand.reshape(*p.shape[:-1], g * k), k, axis=-1)


def truncate_probs(probs: mx.array, top_p: float = 1.0, top_k: int = 0) -> mx.array:
    """Top-k then top-p (nucleus) truncate a probability distribution over the last axis,
    renormalized. ``probs`` sums to 1 per row and stays that way. Ties at the nucleus
    boundary are kept (standard nucleus behavior)."""
    top_p = 1.0 if top_p is None else float(top_p)
    top_k = 0 if not top_k else int(top_k)
    if top_p >= 1.0 and top_k <= 0:
        return probs

    p = probs
    v = p.shape[-1]
    top = None
    if 0 < top_k < v:
        # kth-largest value per row, keep >= it (ties at the k-th value all survive)
        top = _topk_values(p, top_k)
        if top is None:
            kth = mx.sort(p, axis=-1)[..., v - top_k][..., None]
        else:
            kth = mx.min(top, axis=-1, keepdims=True)
        p = mx.where(p >= kth, p, 0.0)

    if top_p < 1.0:
        # After top-k only the kept values can be in the nucleus, and the k largest values
        # are exactly the first k entries of the sorted masked row; extra copies of a tied
        # k-th value sort after them and equal the cutoff whenever they could matter. So the
        # cutoff comes from those k values alone.
        src = p if top is None else top
        sorted_desc = -mx.sort(-src, axis=-1)                     # descending values per row
        csum = mx.cumsum(sorted_desc, axis=-1)
        # a token is in the nucleus if the cumulative mass *before* it is still < top_p
        # (this always keeps at least the top-1 token)
        in_nucleus = (csum - sorted_desc) < top_p
        inf = mx.array(float("inf"), dtype=p.dtype)
        cutoff = mx.min(mx.where(in_nucleus, sorted_desc, inf), axis=-1, keepdims=True)
        p = mx.where(p >= cutoff, p, 0.0)

    return p / mx.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def sample_probs(probs: mx.array) -> mx.array:
    """Categorical sample from a (possibly truncated) probability row/batch of rows."""
    return mx.random.categorical(mx.log(probs + 1e-20))
