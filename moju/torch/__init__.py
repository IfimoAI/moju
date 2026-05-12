"""
``moju.torch`` — full-parity PyTorch interface for Moju.

Install with ``pip install moju[torch]`` to pull in the required dependencies
(``torch >= 2.0`` and ``jax2torch``).

Quick start
-----------
>>> from moju.torch import TorchResidualEngine
>>> from moju.piratio import NondimScales
>>>
>>> scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)
>>> engine = TorchResidualEngine(
...     laws=[{"name": "momentum_navier_stokes"}],
...     constants={"re": 1000.0},
...     scales=scales,
... )
>>> loss = engine.training_loss(state, apply_nondim=True)
>>> loss.backward()

Exported symbols
----------------
TorchResidualEngine
    Full-parity PyTorch engine with identical feature set to
    :class:`moju.monitor.ResidualEngine`.

dimensional_to_nd_torch / nd_to_dimensional_torch
    Torch-native nondimensionalisation (stays on autograd tape).

r_eff_scalar_torch / build_loss_torch
    Robust R_eff training loss matching JAX ``build_loss`` exactly.

wrap_law_torch
    Low-level JAX-law-to-PyTorch wrapper (re-exported from
    :mod:`moju.torch_interop` for convenience).
"""
from moju.torch._nondim import dimensional_to_nd_torch, nd_to_dimensional_torch
from moju.torch._r_eff import r_eff_scalar_torch, build_loss_torch
from moju.torch._engine import TorchResidualEngine
from moju.torch_interop import wrap_law_torch

__all__ = [
    "TorchResidualEngine",
    "dimensional_to_nd_torch",
    "nd_to_dimensional_torch",
    "r_eff_scalar_torch",
    "build_loss_torch",
    "wrap_law_torch",
]
