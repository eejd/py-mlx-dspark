"""Model-free tests for the small-M MMA verify kernel (small_m_qmm.py).

The kernel is fp-tie class BY DESIGN (different fp32 accumulation order than
``quantized_matmul``), so unlike wide_gemm there is no bit-identity to assert; the
load-bearing checks are (1) numerics stay within a few bf16 ulps at every window width,
(2) dispatch NEVER touches anything outside the window (M outside its width's range, non-bf16
input, unlisted instances) — those must return the stock kernel's exact bits — and
(3) the context patch composes with ``wide_matmul`` and always restores.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dspark.small_m_qmm import (
    M4_MAX,
    M4_MIN,
    M8_MAX,
    M8_MIN,
    _orig_call,
    active,
    eligible,
    ids_for_shapes,
    measure_shapes,
    shape_key,
    small_m_matmul,
)
from mlx_dspark.wide_gemm import wide_matmul

K, N = 512, 4096  # smallest eligible shape: N >= 4096, K % 512 == 0


def _qlinear(out_features=N, in_features=K, bits=4, group_size=64, bias=False):
    lin = nn.Linear(in_features, out_features, bias=bias)
    return nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)


class _Tiny(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.proj = _qlinear(**kw)


def test_eligibility_gates():
    assert eligible(_qlinear())
    assert eligible(_qlinear(bits=8))                          # 8-bit unpack variant
    assert not eligible(_qlinear(bits=2))                      # no 2-bit kernel
    assert not eligible(_qlinear(group_size=32))               # gs64 only
    # any K % 64 / N % 8 layout, both widths (the per-shape race decides whether it pays)
    assert eligible(_qlinear(out_features=1024))
    assert eligible(_qlinear(in_features=K + 128))
    assert eligible(_qlinear(bits=8, out_features=1024, in_features=K + 64))
    assert not eligible(_qlinear(out_features=1020))           # N % 8 != 0
    assert not eligible(nn.Linear(K, N))                       # not quantized


def test_kernel_numerics_within_window():
    model = _Tiny()
    mod = model.proj
    ids = ids_for_shapes([shape_key(mod)], model)
    assert ids == frozenset({id(mod)})
    for m in range(M4_MIN, M4_MAX + 1):                # 4-bit window: 5-16
        x = (mx.random.normal((1, m, K)) * 0.1).astype(mx.bfloat16)
        ref = _orig_call(mod, x).astype(mx.float32)
        with small_m_matmul(ids):
            got = mod(x).astype(mx.float32)
        diff = mx.max(mx.abs(ref - got)).item()
        scale = max(mx.max(mx.abs(ref)).item(), 1.0)
        assert diff <= 0.02 * scale, f"M={m}: |d|={diff} vs max|y|={scale}"
        assert got.shape == ref.shape


def test_kernel_numerics_8bit():
    model = _Tiny(bits=8)
    mod = model.proj
    ids = ids_for_shapes([shape_key(mod)], model)
    assert ids == frozenset({id(mod)})
    for m in range(M8_MIN, M8_MAX + 1):                # 8-bit window: 6-16
        x = (mx.random.normal((m, K)) * 0.1).astype(mx.bfloat16)
        ref = _orig_call(mod, x).astype(mx.float32)
        with small_m_matmul(ids):
            got = mod(x).astype(mx.float32)
        diff = mx.max(mx.abs(ref - got)).item()
        scale = max(mx.max(mx.abs(ref)).item(), 1.0)
        assert diff <= 0.02 * scale, f"8-bit M={m}: |d|={diff} vs max|y|={scale}"


def test_shape_key_separates_bits():
    assert shape_key(_qlinear()) != shape_key(_qlinear(bits=8))


def test_dispatch_outside_window_is_stock_bits():
    for bits, outside in ((4, (1, 2, M4_MIN - 1, M4_MAX + 1, 32)),
                          (8, (1, 2, M8_MIN - 1, M8_MAX + 1, 32))):
        model = _Tiny(bits=bits)
        ids = frozenset({id(model.proj)})
        for m in outside:
            x = (mx.random.normal((m, K)) * 0.1).astype(mx.bfloat16)
            ref = _orig_call(model.proj, x)
            with small_m_matmul(ids):
                got = model.proj(x)
            assert mx.array_equal(ref, got).item(), f"{bits}-bit M={m} must be the stock path"


def test_skinny_rows_are_row_count_invariant():
    """A row's result does not depend on how many rows ride with it: 5-, 9- and 16-row calls
    (one and two 8-row tiles) agree on the shared rows bit for bit, both widths — so a verify
    row is the same number whatever the cap."""
    from mlx_dspark import skinny_qmm

    for bits in (4, 8):
        mod = _qlinear(bits=bits)
        x = (mx.random.normal((16, K)) * 0.1).astype(mx.bfloat16)
        full = skinny_qmm.qmm(x, mod["weight"], mod["scales"], mod["biases"], bits)
        for m in (5, 9):
            part = skinny_qmm.qmm(x[:m], mod["weight"], mod["scales"], mod["biases"], bits)
            assert mx.array_equal(full[:m], part).item(), (bits, m)


def test_non_bf16_and_unlisted_instances_fall_back():
    model = _Tiny()
    other = _qlinear()
    ids = frozenset({id(model.proj)})
    x16 = (mx.random.normal((M8_MIN, K)) * 0.1).astype(mx.float16)
    with small_m_matmul(ids):
        assert mx.array_equal(_orig_call(model.proj, x16), model.proj(x16)).item()
        xbf = (mx.random.normal((M8_MIN, K)) * 0.1).astype(mx.bfloat16)
        assert mx.array_equal(_orig_call(other, xbf), other(xbf)).item()


def test_bias_applies_on_kernel_path():
    mod = _qlinear(bias=True)
    model = _Tiny()
    model.proj = mod
    ids = ids_for_shapes([shape_key(mod)], model)
    x = (mx.random.normal((M8_MIN, K)) * 0.1).astype(mx.bfloat16)
    ref = _orig_call(mod, x).astype(mx.float32)
    with small_m_matmul(ids):
        got = mod(x).astype(mx.float32)
    assert mx.max(mx.abs(ref - got)).item() <= 0.02 * max(mx.max(mx.abs(ref)).item(), 1.0)


def test_disabled_is_a_noop():
    with small_m_matmul(None):
        assert not active()
    with small_m_matmul(frozenset()):
        assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_patch_restores_on_exception():
    with pytest.raises(RuntimeError), small_m_matmul(frozenset({1})):
        raise RuntimeError("boom")
    assert not active()


def test_nests_with_wide_matmul():
    """The prefill wide-GEMM context runs INSIDE a decode loop that holds the small-M
    patch; on exit it must restore the small-M patch (the entry value), not the
    import-time original — otherwise the kernel silently disappears after the first
    prefill of every generation."""
    ids = frozenset({1})
    with small_m_matmul(ids):
        assert active()
        with wide_matmul(512):
            assert not active()          # wide patch holds the class during prefill
        assert active()                  # and hands it back after
    assert not active()
    assert nn.QuantizedLinear.__call__ is _orig_call


def test_measure_shapes_returns_verified_subset():
    model = _Tiny()
    shapes = measure_shapes(model)
    assert isinstance(shapes, list)
    for s in shapes:
        assert s == shape_key(model.proj)
    # an ineligible model yields nothing
    small = _Tiny(out_features=1020)
    assert measure_shapes(small) == []


# ---- threadgroup ceilings (M1/M2, PR #39) ----------------------------------------------


class _Ceiling:
    """The real kernel behind an M1/M2-style driver: launches wider than ``limit`` threads
    raise mlx's refusal (an M2 Pro refuses this kernel at 1024 threads and again at 512)."""

    def __init__(self, real, limit):
        self.real, self.limit, self.refused = real, limit, 0

    def __call__(self, **kw):
        t = kw["threadgroup"][0]
        if t > self.limit:
            self.refused += 1
            raise ValueError(f"[metal::Device] Thread group size ({t}) is greater than the "
                             f"maximum allowed threads per threadgroup ({self.limit}).")
        return self.real(**kw)


@pytest.fixture
def ceiling(monkeypatch):
    from mlx_dspark import skinny_qmm as sk

    real = sk._kernel()

    def install(limit):
        fake = _Ceiling(real, limit)
        monkeypatch.setattr(sk, "_kernel", lambda: fake)
        return fake

    yield install
    sk.set_s_cap(sk.S_MAX)


class _ThreeWidths(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _qlinear(out_features=64)       # n < 1024: S=32, 1024 threads
        self.kv = _qlinear(out_features=1024)       # 1024 <= n < 4096: S=16, 512 threads
        self.up = _qlinear(out_features=4096)       # wide: S=8, 256 threads


def test_threadgroup_refusal_lowers_the_cap_instead_of_aborting(ceiling):
    from mlx_dspark import skinny_qmm as sk

    fake = ceiling(448)
    model = _ThreeWidths()
    shapes = measure_shapes(model)                  # raised and aborted the load before
    assert fake.refused > 0
    assert 1 <= sk.s_cap() <= 8                     # 32 and 16 refused at 448 threads
    x = (mx.random.normal((8, K)) * 0.1).astype(mx.bfloat16)
    for mod in (model.gate, model.kv, model.up):
        assert sk.geometry(int(mod["weight"].shape[0]))[1] * 32 <= 448
        mx.eval(sk.call(mod, x, 8))                 # the dispatch geometry launches too
    assert set(shapes) <= {shape_key(m) for m in (model.gate, model.kv, model.up)}


def test_no_launchable_geometry_turns_the_kernel_off(ceiling):
    ceiling(16)                                     # below a single simdgroup
    assert measure_shapes(_ThreeWidths()) == []


def test_unrefused_gpus_keep_the_full_geometry():
    from mlx_dspark import skinny_qmm as sk

    q = _qlinear(out_features=64)
    try:                                            # this machine's own driver first
        mx.eval(sk.qmm(mx.zeros((8, K), dtype=mx.bfloat16), q["weight"], q["scales"],
                       q["biases"], 4))
    except ValueError as e:
        if sk.is_threadgroup_refusal(e):
            pytest.skip(f"this GPU refuses 1024-thread launches: {e}")
        raise
    measure_shapes(_ThreeWidths())
    assert sk.s_cap() == sk.S_MAX
    assert [sk.geometry(n)[1] for n in (64, 1024, 4096)] == [32, 16, 8]
