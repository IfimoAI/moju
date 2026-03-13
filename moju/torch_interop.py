from __future__ import annotations

"""
Thin interop helpers for using moju JAX laws from PyTorch.

moju is JAX-first. These utilities are optional glue so that users who
already live in the PyTorch ecosystem (or torch-based frameworks such as
physicsNemo) can call JAX-residual functions from ``moju.piratio.Laws``
without rewriting them in Torch.

Usage
-----

    from moju.piratio import Laws
    from moju.torch_interop import wrap_law_torch

    mass_incompressible_torch = wrap_law_torch(Laws.mass_incompressible)

    # In PyTorch code:
    u_grad = torch.randn(32, 2, 2, device=\"cuda\", dtype=torch.float32)
    residual = mass_incompressible_torch(u_grad)  # torch.Tensor
    loss = (residual ** 2).mean()
    loss.backward()

This module deliberately keeps the surface area tiny: we export a single
\"blessed\" helper instead of introducing separate *_torch variants for
every law. Users can wrap whichever residuals they need.
"""

from typing import Callable

import jax

try:  # pragma: no cover - import error path is trivial
    from jax2torch import jax2torch as _jax2torch

    _HAS_JAX2TORCH = True
except Exception:  # pragma: no cover - missing optional dependency
    _HAS_JAX2TORCH = False


def wrap_law_torch(jax_law_fn: Callable) -> Callable:
    """
    Wrap a JAX law function so it can be called from PyTorch via jax2torch.

    The returned callable:

    - Accepts and returns ``torch.Tensor`` objects.
    - Participates in PyTorch autograd (gradients are computed via JAX under
      the hood and converted back to Torch).
    - Does *not* modify the original JAX function.

    Parameters
    ----------
    jax_law_fn:
        A JAX function, typically one of ``moju.piratio.Laws.*``. It should
        accept JAX arrays (``jax.numpy.ndarray``) and return a JAX array.

    Returns
    -------
    Callable
        A PyTorch-callable function wrapping ``jax_law_fn``.

    Notes
    -----
    - This helper requires the optional ``jax2torch`` (and ``torch``)
      dependency to be installed. If they are missing, an ImportError is
      raised with a short message.
    - We apply ``jax.jit`` by default to take advantage of XLA compilation
      on the JAX side.
    """
    if not _HAS_JAX2TORCH:
        raise ImportError(
            "jax2torch is not installed. Install it (for example via "
            "'pip install jax2torch torch' or use the moju[torch] extra) "
            "before calling wrap_law_torch."
        )

    # JIT compile the JAX function once; jax2torch will take care of the
    # Torch-facing autograd integration.
    jitted = jax.jit(jax_law_fn)
    return _jax2torch(jitted)

