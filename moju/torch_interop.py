from __future__ import annotations

"""
Low-level JAX → PyTorch interop helper.

This module exposes :func:`wrap_law_torch`, the single primitive that
bridges an arbitrary JAX-based law / model / group function into PyTorch's
autograd ecosystem.  It is intentionally kept minimal.

For the full PyTorch-first interface — including :class:`TorchResidualEngine`,
nondimensionalisation, R_eff loss, derived-state chain, group inference,
Path-B FD fill, and constitutive audits — install ``moju[torch]`` and use::

    from moju.torch import TorchResidualEngine

:func:`wrap_law_torch` is re-exported from :mod:`moju.torch` for convenience.

Usage
-----

    from moju.piratio import Laws
    from moju.torch_interop import wrap_law_torch

    mass_incompressible_torch = wrap_law_torch(Laws.mass_incompressible)

    # In PyTorch code:
    u_grad = torch.randn(32, 2, 2, device=\"cpu\", dtype=torch.float32)
    residual = mass_incompressible_torch(u_grad)  # torch.Tensor
    loss = (residual ** 2).mean()
    loss.backward()

Note: ``jax2torch`` is most stable on CPU.  Tensors on other devices are
automatically moved to CPU before calling the JAX function and moved back to
the original device on return.  For full GPU support, prefer the native
torch reimplementations in :mod:`moju.torch`.
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

