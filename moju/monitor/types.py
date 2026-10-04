"""
Stable public types for Moju specs and reports.

All engine entry points accept either these frozen dataclasses or the equivalent plain dicts, so
existing dict-based configurations keep working. Reports returned by :func:`moju.monitor.audit` and
:func:`moju.monitor.evaluate` follow :class:`AuditReport` and carry ``schema_version``
(:data:`REPORT_SCHEMA_VERSION`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, TypedDict, Union

REPORT_SCHEMA_VERSION = "1.0"

ScoringMetric = Literal["rms", "worst_point"]
_SCORING_METRICS = ("rms", "worst_point")

KEY_BASES = (
    "length",
    "time",
    "velocity",
    "temperature",
    "temperature_offset",
    "pressure",
    "density",
    "concentration",
    "field",
    "energy",
    "dimensionless",
)

DerivativeOrigin = Literal["supplied", "finite_difference", "spectral"]


@dataclass(frozen=True)
class Scoring:
    """
    Scoring declaration for a custom law, custom constitutive check, or bound check.

    - ``metric``: ``"rms"`` (average compliance) or ``"worst_point"`` (``max |r|``).
    - ``scale``: positive ``scale_k`` for ``R_norm = v_eff / scale_k`` (same units as the residual).
    - ``dimensionless``: the residual is already a nondimensional fraction; scored on the default
      ``1e-2`` gauge shared by built-in implied/ref deltas so tiers line up.

    Provide ``scale`` or ``dimensionless=True``.
    """

    metric: ScoringMetric = "rms"
    scale: Optional[float] = None
    dimensionless: bool = False

    def __post_init__(self) -> None:
        if self.metric not in _SCORING_METRICS:
            raise ValueError(f"Scoring.metric must be one of {_SCORING_METRICS}, got {self.metric!r}")
        if self.scale is not None:
            s = float(self.scale)
            if not s > 0.0:
                raise ValueError(f"Scoring.scale must be positive, got {self.scale!r}")
        if self.scale is None and not self.dimensionless:
            raise ValueError("Scoring needs a positive scale or dimensionless=True")

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "scale": self.scale, "dimensionless": bool(self.dimensionless)}

    @staticmethod
    def coerce(value: Any) -> Optional["Scoring"]:
        """Accept ``None``, a :class:`Scoring`, or a dict with the same fields."""
        if value is None or isinstance(value, Scoring):
            return value
        if isinstance(value, Mapping):
            return Scoring(
                metric=value.get("metric", "rms"),
                scale=value.get("scale"),
                dimensionless=bool(value.get("dimensionless", False)),
            )
        raise TypeError(f"scoring must be a Scoring or dict, got {type(value).__name__}")


@dataclass(frozen=True)
class KeyDeclaration:
    """
    Physical meaning of a state key for dimensional (SI) Path B input.

    The key is ``d^(time_order+space_order) Q / dt^time_order dx^space_order`` of a base quantity
    ``Q``; nondimensionalisation multiplies by ``t_ref**time_order * L_ref**space_order / ref(Q)``.
    Example: ``y_tt`` (second time derivative of a length) is ``KeyDeclaration("length", time_order=2)``.

    ``temperature`` subtracts ``T0`` for the undifferentiated field; ``temperature_offset`` is a
    temperature difference (no ``T0`` shift).
    """

    base: str
    time_order: int = 0
    space_order: int = 0

    def __post_init__(self) -> None:
        if self.base not in KEY_BASES:
            raise ValueError(f"KeyDeclaration.base must be one of {KEY_BASES}, got {self.base!r}")
        if int(self.time_order) < 0 or int(self.space_order) < 0:
            raise ValueError("KeyDeclaration orders must be nonnegative")

    @property
    def derivative_order(self) -> int:
        return int(self.time_order) + int(self.space_order)

    def to_dict(self) -> Dict[str, Any]:
        return {"base": self.base, "time_order": int(self.time_order), "space_order": int(self.space_order)}

    @staticmethod
    def coerce(value: Any) -> "KeyDeclaration":
        if isinstance(value, KeyDeclaration):
            return value
        if isinstance(value, Mapping):
            return KeyDeclaration(
                base=value["base"],
                time_order=int(value.get("time_order", 0)),
                space_order=int(value.get("space_order", 0)),
            )
        if isinstance(value, str):
            return KeyDeclaration(base=value)
        raise TypeError(f"state declaration must be a KeyDeclaration, dict, or base name; got {value!r}")


@dataclass(frozen=True)
class LawSpec:
    """
    Governing-law spec. ``fn`` is optional for a built-in ``Laws.<name>`` or a name from
    :func:`moju.registry.register_law`. Omit ``state_map`` to use that law's ``required_keys`` as an
    identity map.

    ``time_scale`` (dimensional mode): one of ``"convective"``, ``"fourier"``, ``"mass_fourier"``,
    ``"wave"``, a dict ``{"t_ref": float}``, or a callable ``(constants, scales) -> t_ref``.
    ``scale_recipe`` (auto ``scale_k``): ``fn(merged, constants, law_spec, nondim_scales) -> float``.
    """

    name: str
    state_map: Dict[str, str]
    fn: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)
    scoring: Optional[Scoring] = None
    time_scale: Any = None
    scale_recipe: Optional[Callable[..., float]] = field(default=None, repr=False, compare=False)

    def to_engine_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "state_map": dict(self.state_map)}
        if self.fn is not None:
            d["fn"] = self.fn
        if self.scoring is not None:
            d["scoring"] = self.scoring
        if self.time_scale is not None:
            d["time_scale"] = self.time_scale
        if self.scale_recipe is not None:
            d["scale_recipe"] = self.scale_recipe
        return d


@dataclass(frozen=True)
class GroupSpec:
    """Dimensionless-group spec writing ``Groups.<name>`` (or ``fn``) output to ``output_key``."""

    name: str
    output_key: str
    state_map: Dict[str, str]
    fn: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)

    def to_engine_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "output_key": self.output_key, "state_map": dict(self.state_map)}
        if self.fn is not None:
            d["fn"] = self.fn
        return d


@dataclass(frozen=True)
class ConstitutiveCustomSpec:
    """
    Free-form constitutive residual ``fn(merged_state, constants) -> array``, logged under
    ``constitutive/custom/<name>``. Declare ``scoring`` to put it on the tier calibration.
    """

    name: str
    fn: Callable[[Dict[str, Any], Dict[str, Any]], Any] = field(repr=False, compare=False)
    scoring: Optional[Scoring] = None

    def to_engine_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "fn": self.fn}
        if self.scoring is not None:
            d["scoring"] = self.scoring
        return d


BoundValue = Union[float, int, Callable[[Dict[str, Any], Dict[str, Any]], Any]]


@dataclass(frozen=True)
class BoundCheck:
    """
    Inequality check ``lower <= v <= upper`` scored as a constitutive check.

    ``v`` comes from ``value_key`` (state/constants) or ``value_fn(state, constants)``. Bounds are
    floats or callables ``(state, constants) -> array``. The residual
    ``(relu(lower - v) + relu(v - upper)) / scale`` is zero wherever the bound holds; ``scale`` is the
    physical size of a meaningful violation. Logged as ``constitutive/bound/<name>/violation`` and
    scored worst-point on the default gauge, so a violation of 0.1 % of ``scale`` at one point sits at
    the High-tier boundary.
    """

    name: str
    scale: float
    value_key: Optional[str] = None
    value_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = field(
        default=None, repr=False, compare=False
    )
    lower: Optional[BoundValue] = field(default=None, compare=False)
    upper: Optional[BoundValue] = field(default=None, compare=False)
    scoring: Scoring = field(default_factory=lambda: Scoring("worst_point", dimensionless=True))

    def __post_init__(self) -> None:
        if not str(self.name).strip() or "/" in str(self.name):
            raise ValueError(f"BoundCheck.name must be non-empty without '/', got {self.name!r}")
        if not float(self.scale) > 0.0:
            raise ValueError(f"BoundCheck {self.name!r}: scale must be positive")
        if (self.value_key is None) == (self.value_fn is None):
            raise ValueError(f"BoundCheck {self.name!r}: provide exactly one of value_key and value_fn")
        if self.lower is None and self.upper is None:
            raise ValueError(f"BoundCheck {self.name!r}: provide lower and/or upper")

    def to_engine_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scale": float(self.scale),
            "value_key": self.value_key,
            "value_fn": self.value_fn,
            "lower": self.lower,
            "upper": self.upper,
            "scoring": self.scoring,
        }

    @staticmethod
    def coerce(value: Any) -> "BoundCheck":
        if isinstance(value, BoundCheck):
            return value
        if isinstance(value, Mapping):
            kw = dict(value)
            sc = Scoring.coerce(kw.pop("scoring", None))
            if sc is not None:
                kw["scoring"] = sc
            return BoundCheck(**kw)
        raise TypeError(f"bound check must be a BoundCheck or dict, got {type(value).__name__}")


def spec_to_engine_dict(spec: Any) -> Dict[str, Any]:
    """Normalise a typed spec (or plain dict) to the engine dict form."""
    if isinstance(spec, Mapping):
        return dict(spec)
    to_engine = getattr(spec, "to_engine_dict", None)
    if callable(to_engine):
        return to_engine()
    from moju.monitor.config import AuditSpec, audit_spec_to_engine_dict

    if isinstance(spec, AuditSpec):
        return audit_spec_to_engine_dict(spec)
    raise TypeError(f"Unsupported spec type {type(spec).__name__}; use a dict or a moju.monitor.types spec")


def specs_to_engine_dicts(specs: Any) -> List[Dict[str, Any]]:
    return [spec_to_engine_dict(s) for s in (specs or [])]


class PerKeyReport(TypedDict, total=False):
    rms: float
    r_max: float
    r_norm: float
    admissibility_score: float
    admissibility_level: str
    admissibility_metric: str
    score_for_admissibility: float
    scale_source: str


class AuditReport(TypedDict, total=False):
    """Report returned by :func:`moju.monitor.audit` (schema :data:`REPORT_SCHEMA_VERSION`)."""

    schema_version: str
    moju_version: str
    tier_definition: Dict[str, Any]
    per_key: Dict[str, PerKeyReport]
    per_category: Dict[str, float]
    overall_admissibility_score: float
    overall_admissibility_level: str
    monitor_run_mode: Optional[str]
    constitutive_closure_summary: Any
    audit_meta: Dict[str, Any]
    derivative_provenance: Dict[str, str]
    law_sources: Dict[str, Dict[str, str]]
    execution: str


def report_json_schema() -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) for the JSON-serialisable part of :class:`AuditReport`."""
    number = {"type": ["number", "null"]}
    per_key = {
        "type": "object",
        "properties": {
            "rms": number,
            "r_max": number,
            "r_norm": number,
            "admissibility_score": number,
            "admissibility_level": {"type": "string"},
            "admissibility_metric": {"type": "string"},
            "score_for_admissibility": number,
            "scale_source": {"type": "string"},
        },
        "required": ["r_norm", "admissibility_score", "admissibility_level"],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/IfimoAI/moju/report-schema/{REPORT_SCHEMA_VERSION}",
        "title": "Moju audit report",
        "type": "object",
        "properties": {
            "schema_version": {"const": REPORT_SCHEMA_VERSION},
            "moju_version": {"type": "string"},
            "tier_definition": {
                "type": "object",
                "properties": {
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "min_score": {"type": "number"}},
                            "required": ["name", "min_score"],
                        },
                    },
                    "meaning": {"type": "string"},
                },
                "required": ["tiers", "meaning"],
            },
            "per_key": {"type": "object", "additionalProperties": per_key},
            "per_category": {"type": "object", "additionalProperties": number},
            "overall_admissibility_score": number,
            "overall_admissibility_level": {"type": "string"},
            "monitor_run_mode": {"type": ["string", "null"]},
            "derivative_provenance": {
                "type": "object",
                "additionalProperties": {"enum": ["supplied", "finite_difference", "spectral"]},
            },
            "law_sources": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "source": {"enum": ["builtin", "registered", "entry_point", "spec_fn"]},
                        "package": {"type": "string"},
                        "version": {"type": "string"},
                    },
                    "required": ["source"],
                },
            },
            "execution": {"type": "string"},
        },
        "required": [
            "schema_version",
            "moju_version",
            "tier_definition",
            "per_key",
            "overall_admissibility_score",
            "overall_admissibility_level",
        ],
    }
