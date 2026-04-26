"""PDF export for Physics Admissibility Report. Requires reportlab (pip install moju[report])."""

from __future__ import annotations

import datetime
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from moju.monitor.visualize_labels import format_admissibility_pct, pretty_residual_key


try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

# Section headers for first path segment of log keys (laws/..., constitutive/..., scaling/..., data/...)
_CATEGORY_HEADERS: Dict[str, str] = {
    "laws": "Governing Laws",
    "constitutive": "Constitutive relations",
    "scaling": "Scaling and similarity",
    "data": "Data",
    "groups": "Scaling and similarity (legacy key)",
    "models": "Constitutive (legacy key)",
}


def _group_keys_by_category(per_key: Dict[str, Any]) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    buckets: Dict[str, List[Tuple[str, Any]]] = {
        "laws": [],
        "constitutive": [],
        "scaling": [],
        "data": [],
        "groups": [],
        "models": [],
    }
    for key, data in per_key.items():
        if "/" in key:
            prefix, _rest = key.split("/", 1)
            if prefix in buckets:
                buckets[prefix].append((pretty_residual_key(key), data))
    order = ("laws", "constitutive", "scaling", "data", "groups", "models")
    result: List[Tuple[str, List[Tuple[str, Any]]]] = []
    for cat in order:
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


_FOOTER_LEFT = "Moju is developed by Ifimo Lab at Ifimo Analytics"
_FOOTER_RIGHT = "This report is a heuristic and not a certification."


def _footer_canvas(canvas: Any, doc: Any) -> None:
    """Draw footer on each page: credit (left), disclaimer (right)."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    y = 0.5 * inch
    canvas.drawString(doc.leftMargin, y, _FOOTER_LEFT)
    x_right = doc.pagesize[0] - doc.rightMargin
    canvas.drawRightString(x_right, y, _FOOTER_RIGHT)
    canvas.restoreState()


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
    per_category = report.get("per_category", {})
    overall_score = report.get("overall_admissibility_score", 0.0)
    overall_level = report.get("overall_admissibility_level", "Non-Admissible")
    training_doc = report.get("monitor_run_mode") == "training"

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

    story.append(
        Paragraph(
            "<i>These metrics are consistency indicators for declared laws, constitutive closures, "
            "on the evaluated sample. They are not a certification of "
            "physical correctness.</i>",
            body_style,
        )
    )
    story.append(Spacer(1, 12))
    if training_doc:
        story.append(
            Paragraph(
                "<i>Training-style summary: reference (data/) section is "
                "omitted. Use <b>compute_residuals(..., run_mode='eval')</b> with <b>state_ref</b> "
                "for ground-truth comparison in the report.</i>",
                body_style,
            )
        )
        story.append(Spacer(1, 8))

    # Overall score and level (omitted when non-finite, e.g. eval mode)
    try:
        _ov = float(overall_score)
    except (TypeError, ValueError):
        _ov = float("nan")
    if math.isfinite(_ov):
        story.append(Paragraph("Overall", heading_style))
        score_text = f"Admissibility score: {format_admissibility_pct(_ov)} &nbsp; — &nbsp; {overall_level}"
        story.append(Paragraph(score_text, body_style))
        story.append(Spacer(1, 16))
    else:
        story.append(
            Paragraph(
                "<i>No single overall admissibility score (eval-style log). See category summary below.</i>",
                body_style,
            )
        )
        story.append(Spacer(1, 16))

    if per_category:
        story.append(Paragraph("Category summary", heading_style))
        table_data = [["Category", "Score"]]
        cat_rows: List[Tuple[str, str]] = [
            ("laws", "Governing laws"),
            ("constitutive", "Constitutive"),
        ]
        if not training_doc:
            cat_rows.append(("data", "Data / ground truth"))
        for key, label in cat_rows:
            if key in per_category:
                table_data.append([label, format_admissibility_pct(float(per_category[key]))])
        t = Table(table_data, colWidths=[3.0 * inch, 1.2 * inch])
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
        story.append(
            Paragraph(
                "<i>Category scores are geometric means of the per-metric admissibility scores in that section.</i>",
                body_style,
            )
        )
        story.append(Spacer(1, 16))

    # Per-key metrics by category
    per_key_filtered = dict(per_key)
    if training_doc:
        per_key_filtered = {
            k: v
            for k, v in per_key_filtered.items()
            if not (k.startswith("scaling/") or k.startswith("data/"))
        }
    grouped = _group_keys_by_category(per_key_filtered)
    for section_header, items in grouped:
        story.append(Paragraph(section_header, heading_style))
        table_data = [["Metric", "Score", "Level"]]
        for label, data in items:
            score = data.get("admissibility_score", 0.0)
            level = data.get("admissibility_level", "Non-Admissible")
            table_data.append([label, format_admissibility_pct(float(score)), level])
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

    doc.build(
        story,
        onFirstPage=_footer_canvas,
        onLaterPages=_footer_canvas,
    )
