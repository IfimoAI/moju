import numpy as np
import pytest

import jax.numpy as jnp

from moju.piratio import Laws


torch = pytest.importorskip("torch", reason="torch is required for torch interop tests")
jax2torch = pytest.importorskip(
    "jax2torch", reason="jax2torch is required for torch interop tests"
)

from moju.torch_interop import wrap_law_torch


def test_wrap_law_torch_matches_jax_result():
    """
    wrap_law_torch should produce the same residuals as the underlying JAX law.
    """
    # Small batch of 2D velocity gradients (Jacobian) from Torch.
    u_grad_torch = torch.randn(8, 2, 2, dtype=torch.float32)

    # JAX ground truth.
    u_grad_jax = jnp.asarray(u_grad_torch.detach().numpy())
    expected = Laws.mass_incompressible(u_grad_jax)

    # Torch-wrapped version.
    mass_incompressible_torch = wrap_law_torch(Laws.mass_incompressible)
    result_torch = mass_incompressible_torch(u_grad_torch)

    assert isinstance(result_torch, torch.Tensor)
    assert result_torch.shape == expected.shape

    np.testing.assert_allclose(
        result_torch.detach().cpu().numpy(),
        np.array(expected),
        rtol=1e-5,
        atol=1e-6,
    )


def test_wrap_law_torch_is_differentiable_in_torch():
    """
    The wrapped law should participate in PyTorch autograd.
    """
    u_grad_torch = torch.randn(8, 2, 2, dtype=torch.float32, requires_grad=True)
    mass_incompressible_torch = wrap_law_torch(Laws.mass_incompressible)

    residual = mass_incompressible_torch(u_grad_torch)
    loss = (residual ** 2).mean()
    loss.backward()

    assert u_grad_torch.grad is not None
    assert torch.isfinite(u_grad_torch.grad).all()

