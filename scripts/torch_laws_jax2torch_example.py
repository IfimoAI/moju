"""
Minimal example: using moju JAX laws from PyTorch via jax2torch.

Run (after installing extras):

    pip install moju[torch]  # installs torch + jax2torch
    python scripts/torch_laws_jax2torch_example.py
"""

import torch

from moju.piratio import Laws
from moju.torch_interop import wrap_law_torch


def main() -> None:
    # A small batch of 2D velocity gradients (Jacobian matrices) from Torch.
    # Shape: (batch, d, d) with d=2.
    u_grad = torch.randn(16, 2, 2, dtype=torch.float32, device="cpu", requires_grad=True)

    # Wrap the JAX law so it can be called from Torch.
    mass_incompressible_torch = wrap_law_torch(Laws.mass_incompressible)

    # Forward pass: residual of div(u) = 0
    residual = mass_incompressible_torch(u_grad)  # shape: (batch,)

    # Simple scalar loss: mean squared residual
    loss = (residual ** 2).mean()
    loss.backward()

    print(f"Residual shape (Torch): {residual.shape}")
    print(f"Loss: {loss.item():.6f}")
    # Gradients now live on the Torch tensor u_grad
    print(f"Gradient on u_grad has NaNs? {torch.isnan(u_grad.grad).any().item()}")


if __name__ == "__main__":
    main()

