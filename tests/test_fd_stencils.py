"""Finite-difference stencil orders 2/4/6 on uniform grids."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from moju.monitor.fd_stencils import diff1_uniform, diff2_uniform, validate_fd_order


def test_validate_fd_order():
    assert validate_fd_order(2) == 2
    with pytest.raises(ValueError):
        validate_fd_order(3)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_diff1_sine_converges(order):
    # Interior accuracy on sin(x); need enough points for the stencil.
    n = 64
    L = 2.0 * jnp.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = float(L / n)
    f = jnp.sin(x)
    df = diff1_uniform(f, h, order=order, axis=0)
    # Compare interior away from boundaries
    pad = order
    err = float(jnp.max(jnp.abs(df[pad:-pad] - jnp.cos(x[pad:-pad]))))
    assert err < 5e-3


@pytest.mark.parametrize("order", [2, 4, 6])
def test_diff2_sine(order):
    n = 80
    L = 2.0 * jnp.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = float(L / n)
    f = jnp.sin(x)
    d2 = diff2_uniform(f, h, order=order, axis=0)
    pad = order + 1
    err = float(jnp.max(jnp.abs(d2[pad:-pad] + jnp.sin(x[pad:-pad]))))
    assert err < 5e-3


def test_higher_order_more_accurate_on_smooth():
    n = 48
    L = 2.0 * jnp.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = float(L / n)
    f = jnp.sin(2.0 * x)
    exact = 2.0 * jnp.cos(2.0 * x)
    e2 = float(jnp.max(jnp.abs(diff1_uniform(f, h, order=2)[6:-6] - exact[6:-6])))
    e6 = float(jnp.max(jnp.abs(diff1_uniform(f, h, order=6)[6:-6] - exact[6:-6])))
    assert e6 < e2
