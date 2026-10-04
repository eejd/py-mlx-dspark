"""Prefix-cache snapshots and RAM accounting against the REAL mlx-lm cache classes, on
whatever mlx-lm is installed. The fakes in test_prefix_cache.py mirror the pre-0.32 API
(``state`` + ``meta_state``), so they kept passing while mlx-lm 0.32.0 — where ``state``
carries everything and ``meta_state`` is gone — crashed every checkpoint snapshot in
``serve`` (issue #40)."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache

from mlx_dspark.prefix_cache import PrefixCache, _cache_ram_bytes

H, D = 2, 64


def _mk():
    # a hybrid's recurrent layer, a full-attention layer, a sliding-window layer
    return [ArraysCache(2), KVCache(), RotatingKVCache(max_size=64)]


def _feed(cache, n, seed):
    """Advance every layer by ``n`` tokens with deterministic contents."""
    mx.random.seed(seed)
    for c in cache:
        if isinstance(c, ArraysCache):
            c[0] = mx.random.normal((1, 3, 16))            # conv window
            c[1] = mx.random.normal((1, 4, 8, 8))          # recurrent state
        else:
            c.update_and_fetch(mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)))
    mx.eval([a for c in cache for a in _arrays(c)])


def _arrays(c):
    """A layer's live contents, the same on every mlx-lm: recurrent arrays, or KV rows."""
    if isinstance(c, ArraysCache):
        return [a for a in c.cache if a is not None]
    quantized = isinstance(c.keys, tuple)           # QuantizedKVCache: (packed, scales, biases)
    ks, vs = (c.keys, c.values) if quantized else ((c.keys,), (c.values,))
    return [x[..., : c.offset, :] for x in (*ks, *vs)]


def _same(a, b) -> bool:
    xs, ys = _arrays(a), _arrays(b)
    return len(xs) == len(ys) and all(mx.array_equal(x, y) for x, y in zip(xs, ys))


def test_checkpoint_round_trips_real_mlx_lm_caches():
    pc = PrefixCache(_mk, None, min_reuse=4, checkpoint=True)
    prompt = list(range(20))
    cache, _, reuse = pc.acquire(prompt)
    assert reuse == 0
    _feed(cache, 20, seed=0)
    pc.checkpoint(cache, None, 20, prompt)          # mlx-lm 0.32.0: AttributeError here (#40)
    got, _, reuse = pc.acquire(prompt + [1, 2, 3])
    assert reuse == 20
    assert [c.offset for c in got[1:]] == [20, 20]
    assert all(_same(g, c) for g, c in zip(got, cache))
    _feed(got, 5, seed=1)                           # the borrower mutates its copy...
    again, _, reuse = pc.acquire(prompt + [4])
    assert reuse == 20
    assert all(_same(g, c) for g, c in zip(again, cache))   # ...the snapshot is untouched


def test_rung_partial_restore_with_real_caches():
    pc = PrefixCache(lambda: [KVCache(), ArraysCache(2)], None, min_reuse=4, checkpoint=True)
    prompt = list(range(21))
    cache, _, _ = pc.acquire(prompt)
    _feed(cache, 8, seed=2)
    at_rung = [mx.array(a) for a in cache[1].cache]
    pc.rung(cache, 8)                               # interior mark mid-prefill
    _feed(cache, 12, seed=3)                        # ... prefill continues to 20
    pc.checkpoint(cache, None, 20, prompt)
    got, _, reuse = pc.acquire(prompt[:10] + [777] * 15)
    assert reuse == 8 and pc.partial_hits == 1
    assert got[0].offset == 8                       # trimmable layer: boundary trimmed back
    assert mx.array_equal(got[0].keys[..., :8, :], cache[0].keys[..., :8, :])
    assert all(mx.array_equal(a, b) for a, b in zip(got[1].cache, at_rung))


def test_ram_accounting_counts_every_layer_kind():
    cache = _mk() + [QuantizedKVCache(group_size=64, bits=8)]
    _feed(cache, 20, seed=4)
    # recurrent state was 0 bytes under mlx-lm 0.32.0's (cache, left_padding, lengths) state,
    # and quantized KV (tuples of arrays) was 0 on every version
    assert _cache_ram_bytes(cache[:1]) == sum(a.nbytes for a in cache[0].cache)
    assert _cache_ram_bytes(cache[3:]) > 0
    assert _cache_ram_bytes(cache) >= sum(a.nbytes for c in cache[:3] for a in _arrays(c))
