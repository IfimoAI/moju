"""
Public extension registry for Moju.

Register user functions so specs can refer to them by name, exactly like built-ins:

- :func:`register_model` / :func:`register_group`: ``AuditSpec(name=...)`` and group specs resolve
  user constitutive models and dimensionless groups. Third-party packages can also publish them via
  the ``moju.models`` and ``moju.groups`` entry-point groups (loaded lazily on the first lookup miss).
- :func:`register_law_scale_recipe`: term-balance ``scale_k`` for a custom law in ``law_scale_mode="auto"``.
- :func:`register_law_implied_check`: law-linked constitutive (implied-property) check for a custom law.
- :func:`register_law_time_scale`: time-scale convention for a custom law in dimensional mode.

Registered models and groups must be JAX-traceable (they are also wrapped for
:class:`moju.torch.TorchResidualEngine` through ``jax2torch``).
"""

from __future__ import annotations

from typing import Any, Callable, List

from moju.monitor.closure_registry import GROUP_FNS, MODEL_FNS, get_group_fn, get_model_fn


def register_model(name: str, fn: Callable[..., Any], *, overwrite: bool = False) -> None:
    """
    Register a constitutive model under ``name`` (resolvable wherever ``Models.<name>`` is).

    Registering a built-in ``Models`` name raises ``ValueError`` unless ``overwrite=True``.
    Signatures must use explicit named parameters (no ``*args`` / ``**kwargs``).
    """
    MODEL_FNS.register(name, fn, overwrite=overwrite)


def register_group(name: str, fn: Callable[..., Any], *, overwrite: bool = False) -> None:
    """Register a dimensionless group under ``name`` (resolvable wherever ``Groups.<name>`` is)."""
    GROUP_FNS.register(name, fn, overwrite=overwrite)


def unregister_model(name: str) -> None:
    MODEL_FNS.unregister(name)


def unregister_group(name: str) -> None:
    GROUP_FNS.unregister(name)


def list_registered_models() -> List[str]:
    """Names registered by users or entry points (built-ins excluded)."""
    return MODEL_FNS.user_names()


def list_registered_groups() -> List[str]:
    return GROUP_FNS.user_names()


def register_law_scale_recipe(law_name: str, fn: Callable[..., float], *, overwrite: bool = False) -> None:
    from moju.monitor.law_scale_recipes import register_law_scale_recipe as _r

    _r(law_name, fn, overwrite=overwrite)


def unregister_law_scale_recipe(law_name: str) -> None:
    from moju.monitor.law_scale_recipes import unregister_law_scale_recipe as _u

    _u(law_name)


def register_law_implied_check(law_name: str, check: Any) -> None:
    from moju.monitor.law_implied_diagnostics import register_law_implied_check as _r

    _r(law_name, check)


def unregister_law_implied_checks(law_name: str) -> None:
    from moju.monitor.law_implied_diagnostics import unregister_law_implied_checks as _u

    _u(law_name)


def register_law_time_scale(law_name: str, time_scale: Any, *, overwrite: bool = False) -> None:
    from moju.monitor.nondim_inference import register_law_time_scale as _r

    _r(law_name, time_scale, overwrite=overwrite)


def unregister_law_time_scale(law_name: str) -> None:
    from moju.monitor.nondim_inference import unregister_law_time_scale as _u

    _u(law_name)


__all__ = [
    "get_group_fn",
    "get_model_fn",
    "list_registered_groups",
    "list_registered_models",
    "register_group",
    "register_law_implied_check",
    "register_law_scale_recipe",
    "register_law_time_scale",
    "register_model",
    "unregister_group",
    "unregister_law_implied_checks",
    "unregister_law_scale_recipe",
    "unregister_law_time_scale",
    "unregister_model",
]
