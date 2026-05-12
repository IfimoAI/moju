"""
TorchResidualEngine — full-parity PyTorch wrapper for Moju's physics audit.

Provides the same interface as :class:`moju.monitor.ResidualEngine` but
accepts and returns ``torch.Tensor`` values with full autograd compatibility.

Training path (differentiable):
- ``dimensional_to_nd_torch`` — nondimensionalisation
- ``apply_derived_state_chain_torch`` — intermediate quantities (alpha, Pe, …)
- user-supplied callables (user_fns)
- ``Groups.*`` wrapped via ``wrap_law_torch`` — group inference (Re, Pr, …)
- Path-B FD fill via ``fill_path_b_derivatives_torch``
- ``Laws.*`` wrapped via ``wrap_law_torch`` — physics residuals
- Constitutive audits (Models.* + torch balance fns)
- ``build_loss_torch`` — R_eff loss identical to JAX ``build_loss``

Eval path (no grad needed):
- ``audit()`` — delegates to JAX ``ResidualEngine`` + ``moju.monitor.audit``
- ``visualize()`` — delegates to ``moju.monitor.visualize``

CPU constraint: ``jax2torch`` requires CPU tensors.  Tensors on other devices
are automatically moved to CPU for JAX law evaluation and moved back to their
original device on return.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from moju.piratio.laws import Laws
from moju.piratio.groups import Groups
from moju.piratio.models import Models
from moju.piratio.nondim import NondimScales
from moju.monitor.law_group_inference import (
    law_parameter_names,
    group_parameter_names,
    implied_group_specs_for_laws,
)
from moju.torch_interop import wrap_law_torch
from moju.torch._nondim import dimensional_to_nd_torch, nd_to_dimensional_torch
from moju.torch._derived import apply_derived_state_chain_torch
from moju.torch._r_eff import r_eff_scalar_torch, build_loss_torch
from moju.torch._path_b import fill_path_b_derivatives_torch
from moju.torch._closure import (
    compute_implied_delta_torch,
    compute_ref_delta_torch,
    _to_tensor,
)
from moju.torch._implied_diagnostics import merge_law_implied_audit_specs_torch


def _positional_param_names(fn: Any) -> List[str]:
    sig = inspect.signature(fn)
    return [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]




def _device_of(state: Dict[str, Any]) -> Optional[torch.device]:
    for v in state.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return None


def _to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state.items()
    }


def _restore_device(state: Dict[str, Any], device: Optional[torch.device]) -> Dict[str, Any]:
    if device is None:
        return state
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in state.items()
    }


class TorchResidualEngine:
    """
    Full-parity PyTorch wrapper for Moju's physics residual engine.

    Parameters
    ----------
    laws:
        List of law spec dicts, e.g. ``[{"name": "momentum_navier_stokes"}]``.
        Follows the same format as :class:`moju.monitor.ResidualEngine`.
    constants:
        Scalar constants merged into every law / group / model call
        (e.g. ``{"re": 1000.0, "L": 0.1}``).
    scales:
        :class:`~moju.piratio.NondimScales` instance used by
        ``apply_nondim=True`` and automatically populated ``derived_state_chain``
        steps when they reference scale quantities.
    constitutive_audit:
        Additional user-supplied constitutive audit specs.  Law-linked implied
        rows are auto-prepended when ``law_implied_audits=True``.
    derived_state_chain:
        List of JSON DSL steps ``{"output_key": str, "expr": dict}`` that
        compute intermediate quantities before law evaluation.
    user_fns:
        Dict of ``key -> callable`` for state quantities computed by the user
        (e.g. ``{"k": lambda T: k0 * T**0.5}``). Callables are called with
        current torch tensor values for their parameter names.
    law_implied_audits:
        Auto-prepend implied constitutive audit rows for supported laws
        (default ``True``).
    path_b_fill:
        Auto-fill missing spatial derivatives via ``torch.gradient`` before
        law evaluation (default ``False``).
    best_effort:
        If ``True`` (default), skip laws / audits whose required state keys
        are missing.  If ``False``, raise ``KeyError`` on any missing key.

    Examples
    --------
    >>> from moju.torch import TorchResidualEngine
    >>> from moju.piratio import NondimScales
    >>> scales = NondimScales(L_ref=0.1, U_ref=1.0, rho_ref=1000.0)
    >>> engine = TorchResidualEngine(
    ...     laws=[{"name": "momentum_navier_stokes"}],
    ...     constants={"re": 1000.0},
    ...     scales=scales,
    ... )
    >>> loss = engine.training_loss(state_phys, apply_nondim=True)
    >>> loss.backward()
    """

    def __init__(
        self,
        laws: List[Dict[str, Any]],
        *,
        constants: Optional[Dict[str, Any]] = None,
        scales: Optional[NondimScales] = None,
        constitutive_audit: Optional[List[Dict[str, Any]]] = None,
        derived_state_chain: Optional[List[Dict[str, Any]]] = None,
        user_fns: Optional[Dict[str, Callable[..., Any]]] = None,
        law_implied_audits: bool = True,
        path_b_fill: bool = False,
        best_effort: bool = True,
    ) -> None:
        self._laws_spec: List[Dict[str, Any]] = list(laws)
        self._constants: Dict[str, Any] = dict(constants or {})
        self._scales = scales
        self._derived_state_chain: List[Dict[str, Any]] = list(derived_state_chain or [])
        self._user_fns: Dict[str, Callable[..., Any]] = dict(user_fns or {})
        self._path_b_fill = path_b_fill
        self._best_effort = best_effort

        # ------------------------------------------------------------------
        # Wrap Laws.* — one per unique law name.
        # We do NOT use functools.partial or closures here.  Instead we wrap
        # the raw JAX function so jax2torch can inspect its real signature
        # (with named positional parameters).  In compute_residuals_torch we
        # pass ALL parameters — tensor fields from state AND scalar constants
        # converted to 0-d torch tensors.  This avoids the "multiple values
        # for argument" error that occurs when jax2torch expands *args-style
        # signatures through bound.apply_defaults().
        # ------------------------------------------------------------------
        self._wrapped_laws: Dict[str, Callable] = {}
        self._law_all_params: Dict[str, List[str]] = {}
        for spec in self._laws_spec:
            name = str(spec["name"])
            if name in self._wrapped_laws:
                continue
            fn = getattr(Laws, name)
            all_params = law_parameter_names(name)
            self._wrapped_laws[name] = wrap_law_torch(fn)
            self._law_all_params[name] = all_params

        # ------------------------------------------------------------------
        # Group inference — topological order, wrap each group function.
        # Same strategy as for laws: wrap the raw function, pass ALL params
        # (including scalars from constants as 0-d tensors) in the pipeline.
        # ------------------------------------------------------------------
        law_names_list = [str(s["name"]) for s in self._laws_spec]
        group_specs = implied_group_specs_for_laws(law_names_list)
        self._group_compute_plan: List[Dict[str, Any]] = []
        for gspec in group_specs:
            gname = str(gspec["name"])
            output_key = str(gspec.get("output_key", gname))
            state_map: Dict[str, str] = dict(gspec.get("state_map") or {})
            fn = getattr(Groups, gname)
            all_params = group_parameter_names(gname)
            self._group_compute_plan.append({
                "output_key": output_key,
                "wrapped_fn": wrap_law_torch(fn),
                "all_params": all_params,
                "state_map": state_map,
            })

        # ------------------------------------------------------------------
        # Constitutive audit specs — law-implied + user-supplied
        # ------------------------------------------------------------------
        implied_specs = merge_law_implied_audit_specs_torch(
            self._laws_spec, enabled=law_implied_audits
        )
        user_specs = list(constitutive_audit or [])
        self._audit_specs: List[Dict[str, Any]] = implied_specs + user_specs

        # Wrap Models.* for each unique model name referenced by audits.
        # Same strategy: wrap raw function, pass all args as tensors.
        self._wrapped_models: Dict[str, Callable] = {}
        self._model_all_params: Dict[str, List[str]] = {}
        for aspec in self._audit_specs:
            mname = str(aspec["name"])
            if mname in self._wrapped_models:
                continue
            if not hasattr(Models, mname):
                continue
            fn = getattr(Models, mname)
            all_params = _positional_param_names(fn)
            self._wrapped_models[mname] = wrap_law_torch(fn)
            self._model_all_params[mname] = all_params

    # ------------------------------------------------------------------
    # Primary compute method
    # ------------------------------------------------------------------

    def compute_residuals_torch(
        self,
        state: Dict[str, Any],
        *,
        state_ref: Optional[Dict[str, Any]] = None,
        apply_nondim: bool = False,
        run_mode: str = "training",
    ) -> Dict[str, Any]:
        """
        Compute physics residuals from a (possibly dimensional) state dict.

        Execution order mirrors JAX :meth:`ResidualEngine.compute_residuals`:

        1. Optional :func:`dimensional_to_nd_torch`
        2. ``user_fns`` materialisation
        3. :func:`apply_derived_state_chain_torch`
        4. Group inference (wrapped ``Groups.*``)
        5. Optional Path-B FD fill
        6. Law residuals (wrapped ``Laws.*``)
        7. Constitutive implied-delta audits
        8. ``data/`` key differences when *state_ref* provided (eval mode)

        Parameters
        ----------
        state:
            Input state dict (torch.Tensor values or Python scalars).
        state_ref:
            Optional reference state for ``ref_delta`` audits (eval mode).
        apply_nondim:
            Apply :func:`dimensional_to_nd_torch` before processing.
            Requires ``scales`` to be set on the engine.
        run_mode:
            ``"training"`` (default) or ``"eval"``.  ``"eval"`` enables
            ``ref_delta`` and ``data/`` computation.

        Returns
        -------
        Dict[str, Any]
            ``{"laws": {...}, "constitutive": {...}, "data": {...}}``
        """
        if run_mode not in ("training", "eval"):
            raise ValueError("run_mode must be 'training' or 'eval'")

        # Detect original device to restore outputs
        orig_device = _device_of(state)
        # Move to CPU for JAX-bridged computations
        state = _to_cpu(dict(state))

        # 1. Optional nondim
        if apply_nondim:
            if self._scales is None:
                raise ValueError(
                    "scales must be provided on TorchResidualEngine to use apply_nondim=True"
                )
            state = dimensional_to_nd_torch(state, self._scales, warn_unknown=False)

        # 2. user_fns materialisation
        state = self._materialise_user_fns(state)

        # 3. Derived state chain
        if self._derived_state_chain:
            state, chain_warns = apply_derived_state_chain_torch(
                state, self._constants, self._derived_state_chain
            )
            for w in chain_warns:
                warnings.warn(f"TorchResidualEngine derived_state: {w}", UserWarning, stacklevel=2)

        # 4. Group inference (compute Re, Pr, fo, etc.)
        merged = {**self._constants, **state}
        for plan in self._group_compute_plan:
            out_key = plan["output_key"]
            if out_key in merged:
                continue  # already provided
            all_params = plan["all_params"]
            state_map = plan["state_map"]
            # Build resolved param keys (state_map[p] or p directly)
            resolved_keys = [state_map.get(p, p) for p in all_params]
            if self._best_effort and any(k not in merged for k in resolved_keys):
                continue
            args = [_to_tensor(merged[k]) for k in resolved_keys]
            try:
                result = plan["wrapped_fn"](*args)
                merged[out_key] = result
                state[out_key] = result
            except Exception as exc:  # noqa: BLE001
                if not self._best_effort:
                    raise
                warnings.warn(
                    f"TorchResidualEngine group {out_key}: {exc}", UserWarning, stacklevel=2
                )

        # 5. Path-B FD fill
        if self._path_b_fill:
            state, pb_warns = fill_path_b_derivatives_torch(
                state, laws_spec=self._laws_spec, constants=self._constants
            )
            for w in pb_warns:
                warnings.warn(f"TorchResidualEngine path_b: {w}", UserWarning, stacklevel=2)
            merged = {**self._constants, **state}

        # 6. Law residuals — pass ALL params as tensors (scalars become 0-d)
        law_residuals: Dict[str, Any] = {}
        for spec in self._laws_spec:
            name = str(spec["name"])
            all_params = self._law_all_params[name]
            # Merge spec-level constants (highest priority)
            spec_consts = spec.get("constants") or {}
            effective_merged = {**merged, **spec_consts}
            if self._best_effort and any(k not in effective_merged for k in all_params):
                continue
            args = []
            try:
                for k in all_params:
                    args.append(_to_tensor(effective_merged[k]))
                result = self._wrapped_laws[name](*args)
                law_residuals[name] = _restore_device({"r": result}, orig_device)["r"]
            except Exception as exc:  # noqa: BLE001
                if not self._best_effort:
                    raise
                warnings.warn(
                    f"TorchResidualEngine law {name}: {exc}", UserWarning, stacklevel=2
                )

        # 7. Constitutive audits — pass ALL model params as tensors
        constitutive_residuals: Dict[str, Any] = {}
        for aspec in self._audit_specs:
            mname = str(aspec["name"])
            if mname not in self._wrapped_models:
                continue
            basename = str(aspec.get("residual_basename") or mname)
            state_map: Dict[str, str] = dict(aspec.get("state_map") or {})
            fn_wrapped = self._wrapped_models[mname]
            # Use all model function params (not just those in state_map)
            # so the wrapped fn with its real signature receives all args
            all_model_params = self._model_all_params[mname]

            # Implied delta
            result = compute_implied_delta_torch(
                fn_wrapped=fn_wrapped,
                arg_names=all_model_params,
                state_map=state_map,
                state_pred=merged,
                constants=self._constants,
                implied_balance_fn_torch=aspec.get("implied_balance_fn_torch"),
                implied_fn_torch=aspec.get("implied_fn_torch"),
                output_key=aspec.get("output_key"),
                implied_delta_ref_key=aspec.get("implied_delta_ref_key"),
            )
            if result is not None:
                r = _restore_device({"r": result}, orig_device)["r"]
                constitutive_residuals[f"{basename}/implied_delta"] = r

            # Ref delta (eval mode only)
            if run_mode == "eval" and state_ref is not None and aspec.get("include_ref_delta", True):
                ref_merged = {**self._constants, **_to_cpu(state_ref)}
                ref_result = compute_ref_delta_torch(
                    fn_wrapped=fn_wrapped,
                    arg_names=all_model_params,
                    output_key=str(aspec.get("output_key") or mname),
                    state_map=state_map,
                    state_pred=merged,
                    state_ref=ref_merged,
                    constants=self._constants,
                    ref_delta_ref_key=aspec.get("ref_delta_ref_key"),
                )
                if ref_result is not None:
                    r = _restore_device({"r": ref_result}, orig_device)["r"]
                    constitutive_residuals[f"{basename}/ref_delta"] = r

        # 8. Data comparison (eval mode)
        data_residuals: Dict[str, Any] = {}
        if run_mode == "eval" and state_ref is not None:
            ref_cpu = _to_cpu(state_ref)
            common = set(state.keys()) & set(ref_cpu.keys())
            for k in common:
                try:
                    diff = _to_tensor(ref_cpu[k]) - _to_tensor(state[k])
                    data_residuals[k] = _restore_device({"r": diff}, orig_device)["r"]
                except Exception:  # noqa: BLE001
                    pass

        out: Dict[str, Any] = {}
        if law_residuals:
            out["laws"] = law_residuals
        if constitutive_residuals:
            out["constitutive"] = constitutive_residuals
        if data_residuals:
            out["data"] = data_residuals
        return out

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def build_loss(
        self,
        residual_dict: Dict[str, Any],
        *,
        law_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Weighted R_eff loss over ``laws/`` keys.

        Matches :func:`moju.monitor.auditor.build_loss` exactly.
        """
        return build_loss_torch(residual_dict, law_weights=law_weights)

    def training_loss(
        self,
        state: Dict[str, Any],
        *,
        apply_nondim: bool = False,
        law_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Single-call training loss: nondim → residuals → R_eff loss.

        Returns a scalar ``torch.Tensor`` with full autograd support.

        Parameters
        ----------
        state:
            Physical or nondimensional state dict.
        apply_nondim:
            Apply ``dimensional_to_nd_torch`` before law evaluation.
        law_weights:
            Optional per-law weights passed to :func:`build_loss_torch`.
        """
        residuals = self.compute_residuals_torch(state, apply_nondim=apply_nondim)
        return self.build_loss(residuals, law_weights=law_weights)

    def audit(
        self,
        state: Dict[str, Any],
        *,
        state_ref: Optional[Dict[str, Any]] = None,
        apply_nondim: bool = False,
        r_ref: Optional[Dict[str, float]] = None,
        export_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full Moju audit via the JAX ``ResidualEngine`` (eval mode).

        Detaches all tensors, converts to numpy, runs the JAX engine, and
        calls :func:`moju.monitor.audit`.  Not differentiable; intended for
        post-training analysis.

        Returns
        -------
        Dict[str, Any]
            Full report: ``overall_admissibility_score``,
            ``overall_admissibility_level``, ``per_key``, ``per_category``,
            ``log``, ``last_residual_dict``.
        """
        import numpy as np
        from moju.monitor import ResidualEngine
        from moju.monitor.auditor import audit as jax_audit

        def _to_numpy(d: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = v
            return out

        state_np = _to_numpy(state)
        if apply_nondim and self._scales is not None:
            from moju.piratio.nondim import dimensional_to_nd
            state_np = dimensional_to_nd(state_np, self._scales, warn_unknown=False)

        state_ref_np = _to_numpy(state_ref) if state_ref is not None else None

        jax_engine = ResidualEngine(
            laws=self._laws_spec,
            constants=self._constants,
            derived_state_chain=self._derived_state_chain if self._derived_state_chain else None,
        )
        residual_dict = jax_engine.compute_residuals(
            state_pred=state_np,
            state_ref=state_ref_np,
            run_mode="eval",
        )
        log = jax_engine.log
        report = jax_audit(
            log,
            r_ref=r_ref,
            export_dir=export_dir,
            last_residual_dict=residual_dict,
        )
        report["log"] = log
        report["last_residual_dict"] = residual_dict
        return report

    def visualize(self, log: List[Dict[str, Any]], **kwargs: Any) -> Any:
        """
        Plotly visualisation of Moju audit log.

        Delegates to :func:`moju.monitor.visualize`.  Call :meth:`audit`
        first to obtain a ``log`` list, then pass ``report["log"]`` here.
        """
        from moju.monitor.auditor import visualize as jax_visualize
        return jax_visualize(log, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _materialise_user_fns(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call user_fns for any key not yet in state."""
        if not self._user_fns:
            return state
        state = dict(state)
        merged = {**self._constants, **state}
        for key, fn in self._user_fns.items():
            if key in state:
                continue
            sig_params = list(inspect.signature(fn).parameters.keys())
            if any(p not in merged for p in sig_params):
                continue
            args = [merged[p] for p in sig_params]
            try:
                state[key] = fn(*args)
                merged[key] = state[key]
            except Exception as exc:  # noqa: BLE001
                if not self._best_effort:
                    raise
                warnings.warn(
                    f"TorchResidualEngine user_fn {key!r}: {exc}",
                    UserWarning,
                    stacklevel=3,
                )
        return state
