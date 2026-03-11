"""Pytest configuration and shared fixtures for moju tests."""

import pytest
import jax.numpy as jnp


@pytest.fixture
def rtol():
    """Relative tolerance for float comparisons (JAX default)."""
    return 1e-5


@pytest.fixture
def atol():
    """Absolute tolerance for float comparisons."""
    return 1e-8
