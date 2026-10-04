"""
Public extension registry for Moju.

Register user functions so specs can refer to them by name, exactly like built-ins:

- :func:`register_model` / :func:`register_group`: ``AuditSpec(name=...)`` and group specs resolve
  user constitutive models and dimensionless groups. Third-party packages can also publish them via
  the ``moju.models`` and ``moju.groups`` entry-point groups (loaded lazily on the first lookup miss).
- :func:`register_law`: a governing law by name, with ``required_keys`` and optional scale, time-scale,
  and implied-check hooks. Third-party packages publish laws through the ``moju.laws`` entry-point group.
- :func:`register_law_scale_recipe`: term-balance ``scale_k`` for a custom law in ``law_scale_mode="auto"``.
- :func:`register_law_implied_check`: law-linked constitutive (implied-property) check for a custom law.
- :func:`register_law_time_scale`: time-scale convention for a custom law in dimensional mode.

Registered callables must be JAX-traceable (they are also wrapped for
:class:`moju.torch.TorchResidualEngine` through a DLPack handoff).
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from moju.monitor.closure_registry import (
    GROUP_FNS,
    MODEL_FNS,
    get_group_fn,
    get_model_fn,
)
from moju.piratio.laws import Laws


def register_model(
    name: str, fn: Callable[..., Any], *, overwrite: bool = False
) -> None:
    """
    Register a constitutive model under ``name`` (resolvable wherever ``Models.<name>`` is).

    Registering a built-in ``Models`` name raises ``ValueError`` unless ``overwrite=True``.
    Signatures must use explicit named parameters (no ``*args`` / ``**kwargs``).
    """
    MODEL_FNS.register(name, fn, overwrite=overwrite)


def register_group(
    name: str, fn: Callable[..., Any], *, overwrite: bool = False
) -> None:
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


def register_law_scale_recipe(
    law_name: str, fn: Callable[..., float], *, overwrite: bool = False
) -> None:
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


def register_law_time_scale(
    law_name: str, time_scale: Any, *, overwrite: bool = False
) -> None:
    from moju.monitor.nondim_inference import register_law_time_scale as _r

    _r(law_name, time_scale, overwrite=overwrite)


def unregister_law_time_scale(law_name: str) -> None:
    from moju.monitor.nondim_inference import unregister_law_time_scale as _u

    _u(law_name)


@dataclass(frozen=True)
class RegisteredLaw:
    """
    Payload for a ``moju.laws`` entry point, or the fields stored by :func:`register_law`.

    ``required_keys`` must equal the callable's positional parameter names. They are the default
    identity ``state_map`` when a spec omits one.
    """

    fn: Callable[..., Any]
    required_keys: Tuple[str, ...]
    scale_recipe: Optional[Callable[..., Any]] = None
    time_scale: Any = None
    implied_check: Any = None


@dataclass(frozen=True)
class LawRecord:
    """Resolved law: callable, required keys, optional hooks, and where the law came from."""

    name: str
    fn: Callable[..., Any]
    required_keys: Tuple[str, ...]
    source: str
    scale_recipe: Optional[Callable[..., Any]] = None
    time_scale: Any = None
    implied_check: Any = None
    package: Optional[str] = None
    version: Optional[str] = None

    def to_source_dict(self) -> Dict[str, str]:
        out = {"source": self.source}
        if self.package:
            out["package"] = self.package
        if self.version:
            out["version"] = self.version
        return out


def _positional_parameter_names(fn: Callable[..., Any]) -> Tuple[str, ...]:
    sig = inspect.signature(fn)
    names: List[str] = []
    label = getattr(fn, "__name__", repr(fn))
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(str(p.name))
        elif p.kind == p.KEYWORD_ONLY:
            raise TypeError(
                f"law callable {label!r} must not use keyword-only parameters"
            )
        else:
            raise TypeError(f"law callable {label!r} must not use *args or **kwargs")
    return tuple(names)


def _normalize_required_keys(
    fn: Callable[..., Any], required_keys: Iterable[str]
) -> Tuple[str, ...]:
    keys = tuple(str(k) for k in required_keys)
    params = _positional_parameter_names(fn)
    if keys != params:
        raise ValueError(
            f"required_keys {keys} must match the law callable's positional parameters {params}"
        )
    return keys


def _validate_law_hooks(
    name: str,
    *,
    scale_recipe: Optional[Callable[..., Any]],
    time_scale: Any,
    implied_check: Any,
) -> Tuple[Optional[Callable[..., Any]], Any, Any]:
    if scale_recipe is not None and not callable(scale_recipe):
        raise TypeError(f"scale_recipe for law {name!r} must be callable")
    if time_scale is not None:
        from moju.monitor.nondim_inference import _validate_time_scale_hint

        time_scale = _validate_time_scale_hint(name, time_scale)
    if implied_check is not None:
        from moju.monitor.law_implied_diagnostics import LawImpliedCheck

        if not isinstance(implied_check, LawImpliedCheck):
            raise TypeError(f"implied_check for law {name!r} must be a LawImpliedCheck")
    return scale_recipe, time_scale, implied_check


class _LawTable:
    """Built-in ``Laws.*``, then :func:`register_law`, then ``moju.laws`` entry points."""

    def __init__(self) -> None:
        self._builtin: Dict[str, LawRecord] = {}
        for name in dir(Laws):
            if name.startswith("_"):
                continue
            fn = getattr(Laws, name)
            if not callable(fn):
                continue
            try:
                keys = _positional_parameter_names(fn)
            except (TypeError, ValueError):
                continue
            self._builtin[name] = LawRecord(
                name=name,
                fn=fn,
                required_keys=keys,
                source="builtin",
            )
        self._user: Dict[str, LawRecord] = {}
        self._entry_points_loaded = False

    def _distribution(self, ep: Any) -> Tuple[Optional[str], Optional[str]]:
        dist = getattr(ep, "dist", None)
        if dist is None:
            return None, None
        package = getattr(dist, "name", None)
        version = getattr(dist, "version", None)
        return (str(package) if package else None, str(version) if version else None)

    def _record_from_loaded(self, name: str, loaded: Any, ep: Any) -> LawRecord:
        package, version = self._distribution(ep)
        if isinstance(loaded, RegisteredLaw):
            keys = _normalize_required_keys(loaded.fn, loaded.required_keys)
            scale_recipe, time_scale, implied_check = _validate_law_hooks(
                name,
                scale_recipe=loaded.scale_recipe,
                time_scale=loaded.time_scale,
                implied_check=loaded.implied_check,
            )
            fn = loaded.fn
        elif callable(loaded):
            keys = _positional_parameter_names(loaded)
            scale_recipe, time_scale, implied_check = None, None, None
            fn = loaded
        else:
            raise TypeError(
                f"moju.laws entry point {name!r} must load a callable or a RegisteredLaw, "
                f"got {type(loaded).__name__}"
            )
        return LawRecord(
            name=name,
            fn=fn,
            required_keys=keys,
            source="entry_point",
            scale_recipe=scale_recipe,
            time_scale=time_scale,
            implied_check=implied_check,
            package=package,
            version=version,
        )

    def load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            return
        try:
            eps = entry_points()
            selected = (
                eps.select(group="moju.laws")
                if hasattr(eps, "select")
                else eps.get("moju.laws", [])
            )
        except Exception:  # noqa: BLE001
            return
        for ep in selected:
            if ep.name in self._builtin or ep.name in self._user:
                warnings.warn(
                    f"moju: entry point {ep.name!r} from group 'moju.laws' collides with an existing "
                    "law and was skipped",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            try:
                self._user[ep.name] = self._record_from_loaded(ep.name, ep.load(), ep)
            except Exception as err:  # noqa: BLE001
                warnings.warn(
                    f"moju: failed to load entry point {ep.name!r} from group 'moju.laws': {err}",
                    UserWarning,
                    stacklevel=3,
                )

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        required_keys: Iterable[str],
        scale_recipe: Optional[Callable[..., Any]] = None,
        time_scale: Any = None,
        implied_check: Any = None,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registry name must be a non-empty string")
        if not callable(fn):
            raise TypeError(f"law {name!r} must be callable")
        if not overwrite and (name in self._builtin or name in self._user):
            raise ValueError(
                f"{name!r} is already registered; pass overwrite=True to replace it"
            )
        keys = _normalize_required_keys(fn, required_keys)
        scale_recipe, time_scale, implied_check = _validate_law_hooks(
            name,
            scale_recipe=scale_recipe,
            time_scale=time_scale,
            implied_check=implied_check,
        )
        self._user[name] = LawRecord(
            name=name,
            fn=fn,
            required_keys=keys,
            source="registered",
            scale_recipe=scale_recipe,
            time_scale=time_scale,
            implied_check=implied_check,
        )

    def unregister(self, name: str) -> None:
        if name in self._builtin and name not in self._user:
            raise ValueError(f"{name!r} is a built-in and cannot be unregistered")
        self._user.pop(name, None)

    def get(self, name: str) -> LawRecord:
        if name in self._user:
            return self._user[name]
        if name in self._builtin:
            return self._builtin[name]
        self.load_entry_points()
        try:
            return self._user[name]
        except KeyError:
            raise KeyError(
                f"Unknown law {name!r}: not in Laws.* or the moju registry"
            ) from None

    def names(self) -> List[str]:
        self.load_entry_points()
        return sorted(set(self._builtin) | set(self._user))


_LAWS = _LawTable()


def register_law(
    name: str,
    fn: Callable[..., Any],
    *,
    required_keys: Iterable[str],
    scale_recipe: Optional[Callable[..., Any]] = None,
    time_scale: Any = None,
    implied_check: Any = None,
    overwrite: bool = False,
) -> None:
    """
    Register a governing law under ``name``.

    ``required_keys`` must match ``fn``'s positional parameter names. A spec that names this law and
    omits ``state_map`` uses those keys as an identity map. ``scale_recipe``, ``time_scale``, and
    ``implied_check`` follow the same rules as :func:`register_law_scale_recipe`,
    :func:`register_law_time_scale`, and :func:`register_law_implied_check`, and they are removed with
    :func:`unregister_law`.

    Registering a built-in ``Laws`` name, or a name already registered, raises ``ValueError`` unless
    ``overwrite=True``.
    """
    _LAWS.register(
        name,
        fn,
        required_keys=required_keys,
        scale_recipe=scale_recipe,
        time_scale=time_scale,
        implied_check=implied_check,
        overwrite=overwrite,
    )


def unregister_law(name: str) -> None:
    """Remove a user or entry-point law. Built-in ``Laws.*`` names cannot be unregistered."""
    _LAWS.unregister(name)


def get_law(name: str) -> LawRecord:
    """Built-in, registered, or entry-point law. Raises ``KeyError`` when ``name`` is unknown."""
    return _LAWS.get(name)


def law_record_or_none(name: str) -> Optional[LawRecord]:
    try:
        return get_law(name)
    except KeyError:
        return None


def list_laws() -> List[str]:
    """Built-in ``Laws.*`` names plus laws from :func:`register_law` and ``moju.laws`` entry points."""
    return _LAWS.names()


def resolve_law_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Copy a law spec, filling a missing ``state_map`` from the named law's ``required_keys``.

    An explicit ``state_map`` is kept as written, including a partial map. Omitted arguments
    are still resolved by parameter name from state and constants. A spec ``fn`` is left unchanged.
    """
    out = dict(spec)
    if out.get("fn") is not None or "name" not in out:
        return out
    record = law_record_or_none(str(out["name"]))
    if record is None:
        return out
    if not out.get("state_map"):
        out["state_map"] = {key: key for key in record.required_keys}
    return out


def resolve_law_specs(
    specs: Optional[Iterable[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    return [resolve_law_spec(spec) for spec in (specs or [])]


def law_sources_for_specs(
    specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Law name to ``{source, package?, version?}`` for a configured spec list."""
    out: Dict[str, Dict[str, str]] = {}
    for spec in specs:
        name = str(spec.get("name") or "")
        if not name:
            continue
        if spec.get("fn") is not None:
            out[name] = {"source": "spec_fn"}
            continue
        record = law_record_or_none(name)
        if record is not None:
            out[name] = record.to_source_dict()
    return out


__all__ = [
    "LawRecord",
    "RegisteredLaw",
    "get_group_fn",
    "get_law",
    "get_model_fn",
    "law_sources_for_specs",
    "list_laws",
    "list_registered_groups",
    "list_registered_models",
    "register_group",
    "register_law",
    "register_law_implied_check",
    "register_law_scale_recipe",
    "register_law_time_scale",
    "register_model",
    "resolve_law_spec",
    "resolve_law_specs",
    "unregister_group",
    "unregister_law",
    "unregister_law_implied_checks",
    "unregister_law_scale_recipe",
    "unregister_law_time_scale",
    "unregister_model",
]
