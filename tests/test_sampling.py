"""truncate_probs must stay bit-identical to the full-sort formulation it replaced: the grouped
top-k is a pure speedup, so any change in the truncated distribution is a bug."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_dspark.sampling import _topk_values, truncate_probs


def _reference(probs, top_p=1.0, top_k=0):
    """The original full-vocab-sort implementation, verbatim."""
    if top_p >= 1.0 and top_k <= 0:
        return probs
    p = probs
    v = p.shape[-1]
    if 0 < top_k < v:
        kth = mx.sort(p, axis=-1)[..., v - top_k][..., None]
        p = mx.where(p >= kth, p, 0.0)
    if top_p < 1.0:
        sorted_desc = -mx.sort(-p, axis=-1)
        csum = mx.cumsum(sorted_desc, axis=-1)
        in_nucleus = (csum - sorted_desc) < top_p
        inf = mx.array(float("inf"), dtype=p.dtype)
        cutoff = mx.min(mx.where(in_nucleus, sorted_desc, inf), axis=-1, keepdims=True)
        p = mx.where(p >= cutoff, p, 0.0)
    return p / mx.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def _rows(n, v, scale, seed):
    """bf16 logits rows: gaussian noise plus a few strong candidates (realistic heavy top),
    rounded to bf16 so exact ties occur at the top-k boundary."""
    mx.random.seed(seed)
    base = mx.random.normal((n, v)) * scale
    spikes = mx.random.randint(0, v, (n, 8))
    return base.at[mx.arange(n)[:, None], spikes].add(scale * 4).astype(mx.bfloat16)


CONFIGS = [(0.6, 0.95, 20), (1.0, 0.95, 64), (0.2, 1.0, 80), (1.0, 0.5, 20),
           (1.5, 0.9, 100), (1.0, 0.95, 1), (2.0, 0.99, 50), (1.0, 0.95, 200)]


@pytest.mark.parametrize("v", [151936, 50000, 32001])   # 32001: not divisible by the groups
def test_bit_identical_to_full_sort(v):
    for scale in (1.0, 4.0, 16.0):
        rows = _rows(4, v, scale, seed=v + int(scale))
        for t, tp, tk in CONFIGS:
            probs = mx.softmax(rows / t, axis=-1)
            assert mx.array_equal(truncate_probs(probs, tp, tk), _reference(probs, tp, tk)), \
                (v, scale, t, tp, tk)


def test_ties_at_kth_value_all_survive():
    # 30 tokens share the 20th-largest value exactly: all must be kept, as before
    v = 20000
    p = mx.zeros((1, v)).at[0, :10].add(0.05).at[0, 100:130].add(0.01)
    p = (p / p.sum()).astype(mx.bfloat16)
    out = truncate_probs(p, 1.0, 20)
    assert int((out > 0).sum()) == 40
    assert mx.array_equal(out, _reference(p, 1.0, 20))
    assert mx.array_equal(truncate_probs(p, 0.9, 20), _reference(p, 0.9, 20))


def test_fewer_than_k_nonzero():
    # a near-one-hot row: the k-th largest is 0, so top-k keeps the row as it was
    v = 20000
    p = mx.zeros((2, v)).at[:, 7].add(0.9).at[:, 9].add(0.1).astype(mx.bfloat16)
    for tp in (1.0, 0.95, 0.5):
        assert mx.array_equal(truncate_probs(p, tp, 20), _reference(p, tp, 20))


def test_grouped_topk_values_are_the_k_largest():
    rows = _rows(3, 151936, 4.0, seed=3).astype(mx.float32)
    top = _topk_values(rows, 20)
    want = mx.sort(rows, axis=-1)[..., -20:]
    assert mx.array_equal(mx.sort(top, axis=-1), want)


def test_small_rows_and_huge_k_fall_back_to_full_sort():
    assert _topk_values(mx.zeros((1, 4096)), 20) is None        # one cheap block sort
    assert _topk_values(mx.zeros((1, 151936)), 50000) is None   # candidates rival the row
    p = mx.softmax(_rows(2, 30000, 4.0, seed=5).astype(mx.float32), axis=-1)
    for tk in (5000, 29999, 30000, 40000):
        assert mx.array_equal(truncate_probs(p, 0.95, tk), _reference(p, 0.95, tk))
