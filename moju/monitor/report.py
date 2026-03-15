"""PDF export for Physics Admissibility Report. Requires reportlab (pip install moju[report])."""

from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional, Tuple


try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

# Section headers for key prefixes
_CATEGORY_HEADERS: Dict[str, str] = {
    "laws": "Governing Laws",
    "groups": "Scaling & Similarity",
    "models": "Constitutive Relations",
    "data": "Data",
}


def _group_keys_by_category(per_key: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    """Group per_key by prefix (laws/, groups/, models/, data/) and return (section_header, [(key, data), ...])."""
    buckets: Dict[str, List[Tuple[str, Any]]] = {
        "laws": [],
        "groups": [],
        "models": [],
        "data": [],
    }
    for key, data in per_key.items():
        if "/" in key:
            prefix, rest = key.split("/", 1)
            if prefix in buckets:
                buckets[prefix].append((rest.replace("_", " ").title(), data))
    result: List[Tuple[str, List[Tuple[str, Any]]]] = []
    for cat in ("laws", "groups", "models", "data"):
        if buckets[cat]:
            result.append((_CATEGORY_HEADERS[cat], buckets[cat]))
    return result


def _residual_dict_to_json_serializable(residual_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert residual dict (JAX/NumPy arrays) to JSON-serializable lists."""
    try:
        import jax
    except ImportError:
        jax = None
    import numpy as np

    out: Dict[str, Any] = {}
    for key, val in residual_dict.items():
        if isinstance(val, dict):
            out[key] = _residual_dict_to_json_serializable(val)
        elif hasattr(val, "tolist"):
            try:
                arr = val
                if jax is not None and hasattr(arr, "__jax_array__"):
                    arr = jax.device_get(arr)
                out[key] = np.asarray(arr).tolist()
            except Exception:
                out[key] = str(val)
        else:
            out[key] = val
    return out


def write_residuals_json(residual_dict: Dict[str, Any], path: str) -> None:
    """Write residual dict to a JSON file (arrays converted to lists)."""
    data = _residual_dict_to_json_serializable(residual_dict)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_audit_pdf(
    report: Dict[str, Any],
    path: str,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> None:
    """
    Write a Physics Admissibility Report PDF to the given path.

    :param report: Dict from audit() with per_key, overall_admissibility_score, overall_admissibility_level.
    :param path: Output file path (e.g. .pdf).
    :param model_name: Optional model name for the header.
    :param model_id: Optional model ID for the header.
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF export. Install with: pip install moju[report] or pip install reportlab"
        )

    per_key = report.get("per_key", {})
    overall_score = report.get("overall_admissibility_score", 0.0)
    overall_level = report.get("overall_admissibility_level", "Non-Admissible")

    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = styles["Normal"]

    story: List[Any] = []

    # Title
    story.append(Paragraph("Physics Admissibility Report", title_style))
    story.append(Spacer(1, 6))

    # Optional model name, ID, date
    meta_parts: List[str] = []
    if model_name:
        meta_parts.append(f"Model: {model_name}")
    if model_id:
        meta_parts.append(f"Model ID: {model_id}")
    meta_parts.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if meta_parts:
        story.append(Paragraph(" &nbsp; &nbsp; ".join(meta_parts), body_style))
        story.append(Spacer(1, 12))

    # Overall score and level
    story.append(Paragraph("Overall", heading_style))
    score_text = f"Admissibility score: {overall_score:.2f} &nbsp; — &nbsp; {overall_level}"
    story.append(Paragraph(score_text, body_style))
    story.append(Spacer(1, 16))

    # Per-key metrics by category
    grouped = _group_keys_by_category(per_key)
    for section_header, items in grouped:
        story.append(Paragraph(section_header, heading_style))
        table_data = [["Metric", "Score", "Level"]]
        for label, data in items:
            score = data.get("admissibility_score", 0.0)
            level = data.get("admissibility_level", "Non-Admissible")
            table_data.append([label, f"{score:.2f}", level])
        t = Table(table_data, colWidths=[2.5 * inch, 1 * inch, 2.2 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("TOPPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

    doc.build(story)
