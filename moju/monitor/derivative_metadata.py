"""
Optional per-key derivative metadata and consistency checks (warnings only).

Metadata never changes admissibility scores or tiers. Warnings are appended to the
monitor log entry and surfaced on the audit report under ``diagnostic_warnings``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DERIVATIVE_SOURCES = (
    "solver",
    "finite_difference",
    "spectral",
    "autodiff",
    "supplied",
    "unknown",
)


@dataclass
class DerivativeKeyMeta:
    """Optional provenance for a supplied or filled derivative key."""

    source: str = "unknown"
    method: Optional[str] = None
    stencil_order: Optional[int] = None
    spacing: Optional[float] = None
    evaluation_time: Optional[float] = None
    time_stencil: Optional[Sequence[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.time_stencil is not None:
            d["time_stencil"] = [float(x) for x in self.time_stencil]
        return {k: v for k, v in d.items() if v is not None}


def normalize_derivative_metadata(
    meta: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Accept ``{key: DerivativeKeyMeta|dict}`` and return plain dicts.

    Unknown keys are kept; invalid entries are skipped with no raise (caller may warn).
    """
    if not meta:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, val in meta.items():
        if isinstance(val, DerivativeKeyMeta):
            out[str(key)] = val.to_dict()
        elif isinstance(val, Mapping):
            d = dict(val)
            src = str(d.get("source", "unknown"))
            if src not in DERIVATIVE_SOURCES:
                d["source"] = "unknown"
                d["source_raw"] = src
            else:
                d["source"] = src
            out[str(key)] = d
    return out


def _is_time_derivative_key(key: str) -> bool:
    return key.endswith("_t") or key.endswith("_tt")


def check_derivative_consistency(
    laws_spec: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    derivative_keys_by_law: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """
    Return human-readable warnings (empty if consistent).

    Checks:
    - Keys used by the same law with differing ``evaluation_time``.
    - Time-derivative keys (``*_t``, ``*_tt``) that lack ``time_stencil`` and
      ``evaluation_time`` when any metadata is provided for that key.
    """
    warnings: List[str] = []
    if not metadata:
        return warnings

    for spec in laws_spec:
        law = str(spec.get("name") or "")
        sm = spec.get("state_map") or {}
        if derivative_keys_by_law and law in derivative_keys_by_law:
            keys = list(derivative_keys_by_law[law])
        else:
            keys = [str(v) for v in sm.values()]
        present = [k for k in keys if k in metadata]
        if not present:
            continue

        times = []
        for k in present:
            et = metadata[k].get("evaluation_time")
            if et is not None:
                try:
                    times.append((k, float(et)))
                except (TypeError, ValueError):
                    warnings.append(
                        f"derivative_metadata: law {law!r} key {k!r} has non-numeric evaluation_time"
                    )
        if len(times) >= 2:
            t0 = times[0][1]
            mismatched = [(k, t) for k, t in times[1:] if abs(t - t0) > 1e-12 * max(1.0, abs(t0))]
            if mismatched:
                detail = ", ".join(f"{k}@{t}" for k, t in times)
                warnings.append(
                    f"derivative_consistency: law {law!r} mixes evaluation times ({detail}). "
                    "All terms in a residual should be evaluated at the same instant."
                )

        for k in present:
            if not _is_time_derivative_key(k):
                continue
            m = metadata[k]
            has_stencil = m.get("time_stencil") is not None
            has_time = m.get("evaluation_time") is not None
            if not has_stencil and not has_time:
                warnings.append(
                    f"derivative_consistency: time derivative {k!r} used by law {law!r} "
                    "has metadata but no time_stencil or evaluation_time."
                )
            elif not has_stencil:
                warnings.append(
                    f"derivative_consistency: time derivative {k!r} used by law {law!r} "
                    "lacks a time_stencil (evaluation_time alone does not document the FD/stencil)."
                )
    return warnings


def merge_provenance_with_metadata(
    provenance: Mapping[str, str],
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Build an extended provenance record: mode labels plus optional per-key metadata.

    Shape::

        {
          "modes": {key: "supplied"|"finite_difference"|"spectral", ...},
          "metadata": {key: {...}, ...},
        }
    """
    return {
        "modes": dict(provenance),
        "metadata": {k: dict(v) for k, v in metadata.items()},
    }
