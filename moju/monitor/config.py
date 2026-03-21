from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class AuditSpec:
    """
    Typed config for a Model/Group audit.

    - name: Models.<name> or Groups.<name>
    - output_key: state key for F output (expects derivative keys d_<output_key>_dx/dt for chain closures)
    - state_map: function arg name -> state key
    - predicted_spatial/temporal: state keys (not arg names) that vary in x/t
    - invariance_pi_constant (scaling only, Path A): second forward with constants scaled so
      the audited Group stays fixed; residual on invariance_compare_keys; flat key
      scaling/<name>/pi_constant. Requires built-in recipe and keys in engine.constants.
    """

    name: str
    output_key: str
    state_map: Dict[str, str]
    predicted_spatial: List[str] = field(default_factory=list)
    predicted_temporal: List[str] = field(default_factory=list)
    # Closure evaluation mode for chain_dx/chain_dt.
    # - pointwise: return pointwise residual array (current behavior)
    # - weak: return weighted integrated RMS (noise-robust)
    closure_mode: str = "pointwise"
    # Weak-form hook: canonical axis -> state key holding quadrature weights.
    # Examples: {"x": "w_x"} or {"t": "w_t"}.
    quadrature_weights: Dict[str, str] = field(default_factory=dict)
    # π-constant closure (Path A only): perturb constants along a built-in recipe so the
    # audited Group stays fixed; compare merged state keys between baseline and scaled runs.
    invariance_pi_constant: bool = False
    invariance_compare_keys: List[str] = field(default_factory=list)
    invariance_scale_c: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AuditSpec":
        return AuditSpec(
            name=d["name"],
            output_key=d["output_key"],
            state_map=dict(d.get("state_map") or {}),
            predicted_spatial=list(d.get("predicted_spatial") or []),
            predicted_temporal=list(d.get("predicted_temporal") or []),
            closure_mode=str(d.get("closure_mode") or "pointwise"),
            quadrature_weights=dict(d.get("quadrature_weights") or {}),
            invariance_pi_constant=bool(d.get("invariance_pi_constant", False)),
            invariance_compare_keys=list(d.get("invariance_compare_keys") or []),
            invariance_scale_c=float(d.get("invariance_scale_c", 10.0)),
        )


@dataclass(frozen=True)
class MonitorConfig:
    constants: Dict[str, Any] = field(default_factory=dict)
    laws: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    constitutive_audit: List[AuditSpec] = field(default_factory=list)
    scaling_audit: List[AuditSpec] = field(default_factory=list)
    constitutive_custom: List[Dict[str, Any]] = field(default_factory=list)
    scaling_custom: List[Dict[str, Any]] = field(default_factory=list)

    # Used for default inference when predicted_* omitted.
    primary_fields: List[str] = field(
        default_factory=lambda: ["T", "u", "v", "w", "p", "rho"]
    )

    # Optional Path A state builder (callable is not JSON-serializable; excluded from to_dict)
    state_builder: Optional[Callable[..., Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constants": dict(self.constants),
            "laws": list(self.laws),
            "groups": list(self.groups),
            "constitutive_audit": [s.to_dict() for s in self.constitutive_audit],
            "scaling_audit": [s.to_dict() for s in self.scaling_audit],
            "constitutive_custom": list(self.constitutive_custom),
            "scaling_custom": list(self.scaling_custom),
            "primary_fields": list(self.primary_fields),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MonitorConfig":
        return MonitorConfig(
            constants=dict(d.get("constants") or {}),
            laws=list(d.get("laws") or []),
            groups=list(d.get("groups") or []),
            constitutive_audit=[AuditSpec.from_dict(x) for x in (d.get("constitutive_audit") or [])],
            scaling_audit=[AuditSpec.from_dict(x) for x in (d.get("scaling_audit") or [])],
            constitutive_custom=list(d.get("constitutive_custom") or []),
            scaling_custom=list(d.get("scaling_custom") or []),
            primary_fields=list(d.get("primary_fields") or ["T", "u", "v", "w", "p", "rho"]),
        )

