from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class AuditSpec:
    """
    Typed config for a Model/Group audit.

    - name: Models.<name> or Groups.<name>
    - output_key: state key for F output (used by ref_delta and implied_delta evaluation)
    - state_map: function arg name -> state key
    - invariance_pi_constant (scaling only, Path A): second forward with constants scaled so
      the audited Group stays fixed; residual on invariance_compare_keys; flat key
      scaling/<name>/pi_constant. Requires built-in recipe and keys in engine.constants.
    - implied_value_key (optional): state/constants key holding implied constitutive value;
      residual constitutive/<name>/implied_delta is always a **nondimensional** discrepancy vs
      ``F(pred args)`` (see ``moju.monitor.closure_registry``). Mutually exclusive with
      implied_fn. Omitted if key missing (same as other closures returning None).
    - implied_fn (optional, Python only): (merged_state, constants) -> array or None; not
      serialized in to_dict(). Use audit_spec_to_engine_dict() when building ResidualEngine.
    """

    name: str
    output_key: str
    state_map: Dict[str, str]
    implied_value_key: Optional[str] = None
    implied_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = field(
        default=None, repr=False, compare=False
    )
    # Optional residual subdirectory for flat log keys (law-linked implied audits).
    residual_basename: Optional[str] = None
    # When False, skip F(pred)-F(ref) even if state_ref is set.
    include_ref_delta: bool = True
    # Optional reference tensor key for implied/ref ND discrepancy denominator (|ref|); else symmetric scale.
    implied_delta_ref_key: Optional[str] = None
    ref_delta_ref_key: Optional[str] = None
    # π-constant closure (Path A only): perturb constants along a built-in recipe so the
    # audited Group stays fixed; compare merged state keys between baseline and scaled runs.
    invariance_pi_constant: bool = False
    invariance_compare_keys: List[str] = field(default_factory=list)
    invariance_scale_c: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        # implied_fn omitted (not JSON-serializable); use audit_spec_to_engine_dict for engine.
        return {
            "name": self.name,
            "output_key": self.output_key,
            "state_map": dict(self.state_map),
            "invariance_pi_constant": self.invariance_pi_constant,
            "invariance_compare_keys": list(self.invariance_compare_keys),
            "invariance_scale_c": self.invariance_scale_c,
            "implied_value_key": self.implied_value_key,
            "residual_basename": self.residual_basename,
            "include_ref_delta": self.include_ref_delta,
            "implied_delta_ref_key": self.implied_delta_ref_key,
            "ref_delta_ref_key": self.ref_delta_ref_key,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AuditSpec":
        removed_keys = {
            "predicted_spatial",
            "predicted_temporal",
            "chain_output",
            "closure_mode",
            "quadrature_weights",
            "chain_spatial_axes",
        }
        legacy = sorted(k for k in removed_keys if k in (d or {}))
        if legacy:
            raise ValueError(
                "AuditSpec no longer supports chain-closure configuration. "
                f"Remove legacy keys {legacy} from your config/spec. "
                "Moju clean-out removed chain_dx/dy/dz/dt and all related fields."
            )
        return AuditSpec(
            name=d["name"],
            output_key=d["output_key"],
            state_map=dict(d.get("state_map") or {}),
            invariance_pi_constant=bool(d.get("invariance_pi_constant", False)),
            invariance_compare_keys=list(d.get("invariance_compare_keys") or []),
            invariance_scale_c=float(d.get("invariance_scale_c", 10.0)),
            implied_value_key=(d.get("implied_value_key") or None),
            implied_fn=d.get("implied_fn"),
            residual_basename=(d.get("residual_basename") or None),
            include_ref_delta=bool(d.get("include_ref_delta", True)),
            implied_delta_ref_key=(d.get("implied_delta_ref_key") or None),
            ref_delta_ref_key=(d.get("ref_delta_ref_key") or None),
        )


def audit_spec_to_engine_dict(spec: AuditSpec) -> Dict[str, Any]:
    """Like AuditSpec.to_dict() but attaches implied_fn for in-memory ResidualEngine specs."""
    d = spec.to_dict()
    if spec.implied_fn is not None:
        d["implied_fn"] = spec.implied_fn
    # to_dict already includes residual_basename (may be None) and include_ref_delta
    return d


@dataclass(frozen=True)
class MonitorConfig:
    constants: Dict[str, Any] = field(default_factory=dict)
    laws: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    # When True (default), prepend auto implied_delta rows from :mod:`moju.monitor.law_implied_diagnostics`
    # for each selected law (e.g. Fourier -> thermal_diffusivity vs α implied from T_t, T_laplacian).
    law_implied_audits: bool = True
    constitutive_audit: List[AuditSpec] = field(default_factory=list)
    scaling_audit: List[AuditSpec] = field(default_factory=list)
    constitutive_custom: List[Dict[str, Any]] = field(default_factory=list)
    # Ordered steps: each {"output_key": str, "expr": dict} evaluated before groups / FD / laws.
    derived_state_chain: List[Dict[str, Any]] = field(default_factory=list)

    # Default field names for Studio / NPZ hints.
    primary_fields: List[str] = field(
        default_factory=lambda: ["T", "u", "v", "w", "p", "rho"]
    )

    # Optional Path A state builder (callable is not JSON-serializable; excluded from to_dict)
    state_builder: Optional[Callable[..., Dict[str, Any]]] = None

    # Opt-in: when building an eval engine via ``build_residual_engine_for_pi_constant_eval``,
    # append π-constant scaling_audit rows from :mod:`moju.monitor.law_group_defaults`.
    pi_constant_law_defaults_enabled: bool = False
    pi_constant_default_c: float = 10.0
    pi_constant_law_group_overrides: Dict[str, List[str]] = field(default_factory=dict)
    pi_constant_extra_groups: List[str] = field(default_factory=list)
    pi_constant_default_compare_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constants": dict(self.constants),
            "laws": list(self.laws),
            "groups": list(self.groups),
            "law_implied_audits": bool(self.law_implied_audits),
            "constitutive_audit": [s.to_dict() for s in self.constitutive_audit],
            "scaling_audit": [s.to_dict() for s in self.scaling_audit],
            "constitutive_custom": list(self.constitutive_custom),
            "derived_state_chain": list(self.derived_state_chain),
            "primary_fields": list(self.primary_fields),
            "pi_constant_law_defaults_enabled": bool(self.pi_constant_law_defaults_enabled),
            "pi_constant_default_c": float(self.pi_constant_default_c),
            "pi_constant_law_group_overrides": {
                k: list(v) for k, v in self.pi_constant_law_group_overrides.items()
            },
            "pi_constant_extra_groups": list(self.pi_constant_extra_groups),
            "pi_constant_default_compare_keys": list(self.pi_constant_default_compare_keys),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MonitorConfig":
        legacy_sc = d.get("scaling_custom") or []
        if legacy_sc:
            raise ValueError(
                "MonitorConfig no longer supports scaling_custom; remove it from JSON "
                "and use scaling_audit (Groups.*) with ref_delta and optional π-constant only."
            )
        ov = d.get("pi_constant_law_group_overrides") or {}
        if not isinstance(ov, dict):
            raise ValueError("pi_constant_law_group_overrides must be a dict[str, list[str]]")
        overrides: Dict[str, List[str]] = {
            str(k): list(v) if isinstance(v, (list, tuple)) else [str(v)] for k, v in ov.items()
        }
        return MonitorConfig(
            constants=dict(d.get("constants") or {}),
            laws=list(d.get("laws") or []),
            groups=list(d.get("groups") or []),
            law_implied_audits=bool(d.get("law_implied_audits", True)),
            constitutive_audit=[AuditSpec.from_dict(x) for x in (d.get("constitutive_audit") or [])],
            scaling_audit=[AuditSpec.from_dict(x) for x in (d.get("scaling_audit") or [])],
            constitutive_custom=list(d.get("constitutive_custom") or []),
            derived_state_chain=list(d.get("derived_state_chain") or []),
            primary_fields=list(d.get("primary_fields") or ["T", "u", "v", "w", "p", "rho"]),
            pi_constant_law_defaults_enabled=bool(d.get("pi_constant_law_defaults_enabled", False)),
            pi_constant_default_c=float(d.get("pi_constant_default_c", 10.0)),
            pi_constant_law_group_overrides=overrides,
            pi_constant_extra_groups=list(d.get("pi_constant_extra_groups") or []),
            pi_constant_default_compare_keys=list(d.get("pi_constant_default_compare_keys") or []),
        )

