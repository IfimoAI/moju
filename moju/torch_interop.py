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

Tensors are handed to JAX through DLPack. A CPU tensor stays on CPU. A CUDA tensor
stays on that GPU when this JAX process has a GPU at the same index, which needs
a CUDA jaxlib (for example ``jax[cuda12]``). Otherwise the tensor is copied to
CPU for the JAX call, the result is copied back, and a one-time warning is
emitted. Apple MPS always uses that CPU fallback. Dict and tuple arguments and
return values are preserved.
"""

import threading
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

_SLOT = object()
_CPU_FALLBACK_WARNED = False


def _import_torch():
    try:
        import torch
    except ImportError as err:  # pragma: no cover - missing optional dependency
        raise ImportError(
            "torch is not installed. Install it (for example via "
            "'pip install torch' or the moju[torch] extra) before calling wrap_law_torch."
        ) from err
    return torch


def _is_array(value: Any) -> bool:
    return isinstance(value, (jax.Array, jnp.ndarray))


def _jax_device_placement(device: Any) -> str:
    """
    ``"native"`` when JAX can take this torch device through DLPack.

    CUDA uses ``"native"`` only when ``jax.devices("gpu")`` has a device at that
    index. Any other non-CPU device, including Apple MPS, is ``"cpu_fallback"``.
    """
    kind = getattr(device, "type", None)
    if kind == "cpu":
        return "native"
    if kind == "cuda":
        index = 0 if getattr(device, "index", None) is None else int(device.index)
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = ()
        if index < len(gpus):
            return "native"
    _warn_cpu_fallback()
    return "cpu_fallback"


def _warn_cpu_fallback() -> None:
    global _CPU_FALLBACK_WARNED
    if _CPU_FALLBACK_WARNED:
        return
    _CPU_FALLBACK_WARNED = True
    warnings.warn(
        "wrap_law_torch copied a non-CPU tensor to CPU because JAX has no matching GPU. "
        "Staying on GPU requires a CUDA jaxlib, for example jax[cuda12].",
        UserWarning,
        stacklevel=3,
    )


def _require_one_device(devices: Sequence[Any]) -> None:
    identities = {(getattr(d, "type", None), getattr(d, "index", None)) for d in devices}
    if len(identities) > 1:
        raise ValueError("wrap_law_torch received tensors on more than one device")


def _torch_to_jax(torch_tensor: Any, *, placement: str):
    if placement == "cpu_fallback" and getattr(torch_tensor.device, "type", "cpu") != "cpu":
        held = torch_tensor.detach().to("cpu").contiguous()
    else:
        held = torch_tensor.detach().contiguous()
    return held, jax.dlpack.from_dlpack(held)


def _jax_to_torch(torch: Any, value: Any, device: Any):
    if not _is_array(value):
        value = jnp.asarray(value)
    # DLPack keeps a GPU JAX array on that GPU; `.to` only copies the CPU fallback.
    return torch.from_dlpack(jnp.asarray(value)).to(device)


def _invoke(
    jitted: Callable,
    in_spec: Any,
    template: Sequence[Any],
    tensor_pos: Sequence[int],
    jax_tensors: Tuple[Any, ...],
):
    rebuilt = list(template)
    for pos, val in zip(tensor_pos, jax_tensors):
        rebuilt[pos] = val
    args, kwargs = tree_unflatten(in_spec, rebuilt)
    return jitted(*args, **kwargs)


def _assemble(meta: dict, torch_leaves: Sequence[Any]) -> Any:
    leaves = list(meta["leaves"])
    for slot, tensor in zip(meta["tensor_slots"], torch_leaves):
        leaves[slot] = tensor
    return tree_unflatten(meta["out_spec"], leaves)


def wrap_law_torch(jax_law_fn: Callable) -> Callable:
    """
    Wrap a JAX function so it can be called from PyTorch.

    The returned callable:

    - Accepts and returns ``torch.Tensor`` objects, including tensors nested in
      dicts and tuples.
    - Participates in PyTorch autograd (gradients are computed with ``jax.vjp``
      and converted back through DLPack).
    - Does *not* modify the original JAX function.

    Parameters
    ----------
    jax_law_fn:
        A JAX function, typically one of ``moju.piratio.Laws.*``. It should
        accept JAX arrays (``jax.numpy.ndarray``) and return a JAX array or a
        pytree of arrays.

    Returns
    -------
    Callable
        A PyTorch-callable function wrapping ``jax_law_fn``.

    Notes
    -----
    - This helper requires the optional ``torch`` dependency. If it is missing,
      an ImportError is raised with a short message.
    - CPU tensors are evaluated by JAX on CPU. A CUDA tensor stays on that GPU
      when ``jax.devices("gpu")`` has a device at the same index. That requires
      a CUDA jaxlib, for example ``jax[cuda12]``, which is not part of the
      ``moju[torch]`` extra. Otherwise, and for Apple MPS, the tensor is copied
      to CPU, evaluated there, copied back, and a warning is issued once per
      process. Tensors on different devices in one call raise ``ValueError``.
    - We apply ``jax.jit`` by default to take advantage of XLA compilation
      on the JAX side.
    """
    torch = _import_torch()
    jitted = jax.jit(jax_law_fn)
    call: Dict[str, Any] = {}
    lock = threading.Lock()

    class _DlpackBridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *tensors):  # type: ignore[override]
            held = []
            jax_tensors = []
            placement = call["placement"]
            for tensor in tensors:
                kept, jax_arr = _torch_to_jax(tensor, placement=placement)
                held.append(kept)
                jax_tensors.append(jax_arr)
            jax_tensors_t = tuple(jax_tensors)
            needs = call["needs"]
            if any(needs):
                out, vjp = jax.vjp(
                    lambda *xs: _invoke(jitted, call["in_spec"], call["template"], call["tensor_pos"], xs),
                    *jax_tensors_t,
                )
            else:
                out = _invoke(jitted, call["in_spec"], call["template"], call["tensor_pos"], jax_tensors_t)
                vjp = None
            leaves, out_spec = tree_flatten(out)
            target = call["devices"][0] if call["devices"] else torch.device("cpu")
            returned = []
            tensor_slots = []
            shapes = []
            dtypes = []
            for i, leaf in enumerate(leaves):
                if not _is_array(leaf):
                    continue
                tensor_slots.append(i)
                shapes.append(tuple(getattr(leaf, "shape", ())))
                dtypes.append(leaf.dtype)
                returned.append(_jax_to_torch(torch, leaf, target))
            ctx.vjp = vjp
            ctx.out_spec = out_spec
            ctx.tensor_slots = tensor_slots
            ctx.shapes = shapes
            ctx.dtypes = dtypes
            ctx.needs = list(needs)
            ctx.devices = list(call["devices"])
            ctx.n_in = len(tensors)
            ctx.n_leaves = len(leaves)
            ctx.placement = placement
            ctx.keep = held
            call["meta"] = {
                "leaves": leaves,
                "out_spec": out_spec,
                "tensor_slots": tensor_slots,
            }
            if len(returned) == 1:
                return returned[0]
            return tuple(returned)

        @staticmethod
        def backward(ctx, *grad_out):  # type: ignore[override]
            if ctx.vjp is None:
                return (None,) * ctx.n_in
            cot_leaves: List[Any] = [None] * ctx.n_leaves
            for slot, grad in zip(ctx.tensor_slots, grad_out):
                if grad is None:
                    cot_leaves[slot] = jnp.zeros(ctx.shapes[slot], dtype=ctx.dtypes[slot])
                else:
                    _, cot_leaves[slot] = _torch_to_jax(grad, placement=ctx.placement)
            in_grads = ctx.vjp(tree_unflatten(ctx.out_spec, cot_leaves))
            out_grads = []
            for grad, need, device in zip(in_grads, ctx.needs, ctx.devices):
                if not need or grad is None:
                    out_grads.append(None)
                else:
                    out_grads.append(_jax_to_torch(torch, grad, device))
            return tuple(out_grads)

    def wrapped(*args: Any, **kwargs: Any):
        flat, in_spec = tree_flatten((args, kwargs))
        tensor_pos: List[int] = []
        tensors = []
        devices = []
        needs = []
        template: List[Any] = []
        for i, leaf in enumerate(flat):
            if isinstance(leaf, torch.Tensor):
                tensor_pos.append(i)
                tensors.append(leaf)
                devices.append(leaf.device)
                needs.append(bool(leaf.requires_grad and torch.is_floating_point(leaf)))
                template.append(_SLOT)
            else:
                template.append(leaf)
        if not tensors:
            return _assemble_raw(torch, jitted(*args, **kwargs), torch.device("cpu"))
        _require_one_device(devices)
        placement = _jax_device_placement(devices[0])
        with lock:
            call["in_spec"] = in_spec
            call["template"] = template
            call["tensor_pos"] = tensor_pos
            call["needs"] = needs
            call["devices"] = devices
            call["placement"] = placement
            raw = _DlpackBridge.apply(*tensors)
            meta = call["meta"]
        if len(meta["tensor_slots"]) == 1:
            replayed = [raw]
        else:
            replayed = list(raw)
        return _assemble(meta, replayed)

    return wrapped


def _assemble_raw(torch: Any, value: Any, device: Any) -> Any:
    leaves, spec = tree_flatten(value)
    out = [_jax_to_torch(torch, leaf, device) if _is_array(leaf) else leaf for leaf in leaves]
    return tree_unflatten(spec, out)
