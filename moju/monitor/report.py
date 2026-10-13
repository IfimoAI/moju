"""PDF export for Physics Admissibility Report. Requires reportlab (included in core moju)."""

from __future__ import annotations

import datetime
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

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


def _implied_scoring_pdf_note(data: Mapping[str, Any]) -> str:
    """Short PDF cell note for opt-in implied scoring + raw worst-point."""
    if data.get("implied_metric") is None and not data.get("implied_noise_normalization"):
        return ""
    parts = [str(data.get("admissibility_metric") or data.get("implied_metric") or "")]
    if data.get("r_max_raw") is not None:
        try:
            parts.append(f"raw max|δ|={float(data['r_max_raw']):.3g}")
        except (TypeError, ValueError):
            pass
    return "; ".join(p for p in parts if p)


def _not_identifiable_pdf_notes(report: Dict[str, Any]) -> List[str]:
    """
    HTML note lines for PDF: Not identifiable implied rows with excluded fraction + reason.

    “Not identifiable” means the constitutive check could not be evaluated, not that it passed.
    """
    id_map = report.get("implied_identifiability") or {}
    notes: List[str] = []
    if not isinstance(id_map, dict):
        return notes
    for bn, meta in id_map.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("status") != "not_identifiable" and meta.get("identifiable") is not False:
            continue
        excl = meta.get("excluded_fraction")
        reason = meta.get("reason") or (
            "Constitutive viscosity check could not be evaluated. "
            "This does not mean the check passed."
        )
        try:
            excl_s = f"{100.0 * float(excl):.1f}%" if excl is not None else "n/a"
        except (TypeError, ValueError):
            excl_s = "n/a"
        label = pretty_residual_key(f"constitutive/{bn}/implied_delta")
        notes.append(
            f"<b>{label}</b>: Not identifiable (excluded fraction {excl_s}). {reason}"
        )
    return notes


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


_FOOTER_LEFT = "Moju is developed by Ifimo Lab, a DBA of Ifimo Global Solutions LLC"
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
            "reportlab is required for PDF export. Install with: pip install moju or pip install reportlab"
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
    if report.get("moju_version"):
        meta_parts.append(f"Moju {report['moju_version']}")
    if report.get("schema_version"):
        meta_parts.append(f"Report schema {report['schema_version']}")
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

    audit_meta = report.get("audit_meta") or {}
    if audit_meta:
        story.append(Paragraph("Scoring calibration", heading_style))
        plain_summary = audit_meta.get("plain_summary")
        if plain_summary:
            story.append(Paragraph(str(plain_summary), body_style))
        plain_sections = audit_meta.get("plain_sections") or {}
        for section_key in ("scaling", "nondim", "pipeline"):
            block = plain_sections.get(section_key)
            if block and block.strip():
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>{section_key.title()}</b>", body_style))
                for line in str(block).splitlines():
                    line = line.strip()
                    if line.startswith("- "):
                        story.append(Paragraph(line, body_style))
                    elif line:
                        story.append(Paragraph(line, body_style))
        sc = audit_meta.get("scale_calibration") or {}
        per_key_cal = sc.get("per_key") or []
        law_rows = [r for r in per_key_cal if str(r.get("key", "")).startswith("laws/")]
        table_rows = law_rows[:12] if law_rows else per_key_cal[:12]
        if table_rows:
            story.append(Spacer(1, 8))
            table_data = [["Metric", "scale_k", "Calibration"]]
            for row in table_rows:
                key = str(row.get("key", ""))
                try:
                    sk = float(row.get("scale_k", float("nan")))
                    sk_s = f"{sk:.4g}" if math.isfinite(sk) else "N/A"
                except (TypeError, ValueError):
                    sk_s = "N/A"
                cal = str(row.get("plain") or row.get("scale_source") or "")
                table_data.append([key, sk_s, cal])
            t = Table(table_data, colWidths=[2.2 * inch, 0.9 * inch, 3.4 * inch])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("TOPPADDING", (0, 0), (-1, 0), 6),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(t)
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
                "<i>Category scores: geometric mean for Governing and Data; "
                "minimum worst-point admissibility for Constitutive when closure keys are present.</i>",
                body_style,
            )
        )
        story.append(Spacer(1, 16))

    provenance = report.get("derivative_provenance") or {}
    if provenance:
        story.append(Paragraph("Derivative provenance", heading_style))
        prov_rows = [["Derivative key", "Origin"]] + [
            [str(k), str(v).replace("_", " ")] for k, v in sorted(provenance.items())
        ]
        t = Table(prov_rows, colWidths=[3.0 * inch, 2.0 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 16))

    diag_warn = report.get("diagnostic_warnings") or []
    if diag_warn:
        story.append(Paragraph("Diagnostic warnings", heading_style))
        story.append(
            Paragraph(
                "These warnings do not change admissibility tiers.",
                body_style,
            )
        )
        for w in diag_warn:
            story.append(Paragraph(f"• {w}", body_style))
        story.append(Spacer(1, 12))

    term_bd = report.get("term_breakdown") or {}
    if term_bd:
        story.append(Paragraph("Per-term residual breakdown", heading_style))
        story.append(
            Paragraph(
                "RMS and max of each PDE term, residual, residual / largest term, "
                "and the law scale_k. Diagnostics only — tiers are unchanged.",
                body_style,
            )
        )
        for law_name, bd in sorted(term_bd.items()):
            story.append(Paragraph(str(law_name), heading_style))
            rows = [["Quantity", "RMS", "Max"]]
            for tname, st in (bd.get("terms") or {}).items():
                rows.append(
                    [
                        str(tname),
                        f"{float(st.get('rms', float('nan'))):.4e}",
                        f"{float(st.get('max', float('nan'))):.4e}",
                    ]
                )
            resid = bd.get("residual") or {}
            rows.append(
                [
                    "residual",
                    f"{float(resid.get('rms', float('nan'))):.4e}",
                    f"{float(resid.get('max', float('nan'))):.4e}",
                ]
            )
            t = Table(rows, colWidths=[2.4 * inch, 1.4 * inch, 1.4 * inch])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            story.append(t)
            story.append(
                Paragraph(
                    f"residual / largest term = "
                    f"{bd.get('residual_over_largest_term')}; "
                    f"scale_k = {bd.get('scale_k')} ({bd.get('scale_source')})",
                    body_style,
                )
            )
            story.append(Spacer(1, 10))

    der_err = report.get("derivative_error") or {}
    if der_err.get("per_law"):
        story.append(Paragraph("Derivative-error estimates", heading_style))
        story.append(
            Paragraph(
                f"Flag threshold (error/residual) = {der_err.get('threshold')}. "
                "Flags do not change tiers.",
                body_style,
            )
        )
        rows = [["Law", "err RMS", "resid RMS", "ratio", "flagged"]]
        for law_name, info in sorted((der_err.get("per_law") or {}).items()):
            rows.append(
                [
                    str(law_name),
                    f"{float(info.get('estimated_derivative_error_rms', float('nan'))):.3e}",
                    f"{float(info.get('residual_rms', float('nan'))):.3e}",
                    f"{float(info.get('error_over_residual', float('nan'))):.3e}",
                    "yes"
                    if info.get("residual_may_be_dominated_by_derivative_error")
                    else "no",
                ]
            )
        t = Table(rows, colWidths=[1.8 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 0.7 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    ev_mask = report.get("eval_mask") or {}
    if ev_mask:
        story.append(Paragraph("Evaluation mask", heading_style))
        story.append(
            Paragraph(
                f"kind={ev_mask.get('kind')}; kept {ev_mask.get('n_kept')} / "
                f"{ev_mask.get('n_points')} points "
                f"(fraction {ev_mask.get('fraction_kept')}).",
                body_style,
            )
        )
        story.append(Spacer(1, 12))

    closure_summary = report.get("constitutive_closure_summary")
    if closure_summary:
        story.append(Paragraph("Constitutive closure summary", heading_style))
        story.append(Paragraph(str(closure_summary), body_style))
        story.append(Spacer(1, 16))

    not_ident_notes = _not_identifiable_pdf_notes(report)
    if not_ident_notes:
        story.append(Paragraph("Implied property identifiability", heading_style))
        story.append(
            Paragraph(
                '"Not identifiable" means the constitutive check could not be evaluated '
                "(too few gated points); it is listed below and in the per-key table, "
                "but omitted from the overall rollup so it is never treated as passing.",
                body_style,
            )
        )
        for note in not_ident_notes:
            story.append(Paragraph(note, body_style))
        story.append(Spacer(1, 12))

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
        table_data = [["Metric", "Score", "Level", "Scoring note"]]
        for label, data in items:
            score = data.get("admissibility_score", 0.0)
            level = data.get("admissibility_level", "Non-Admissible")
            score_s = format_admissibility_pct(float(score)) if level != "Not identifiable" else "N/A"
            if level == "Not identifiable" and data.get("excluded_fraction") is not None:
                try:
                    excl_pct = 100.0 * float(data["excluded_fraction"])
                    level = f"Not identifiable ({excl_pct:.1f}% excl.)"
                except (TypeError, ValueError):
                    pass
            note = _implied_scoring_pdf_note(data if isinstance(data, Mapping) else {})
            table_data.append([label, score_s, level, note])
        t = Table(table_data, colWidths=[2.0 * inch, 0.9 * inch, 1.6 * inch, 1.8 * inch])
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

    tier_def = report.get("tier_definition") or {}
    tiers = tier_def.get("tiers") or []
    if tiers:
        story.append(Paragraph("Tier definition", heading_style))
        cut_rows = [["Tier", "Minimum score"]] + [
            [str(t_.get("name", "")), format_admissibility_pct(float(t_.get("min_score", 0.0)))] for t_ in tiers
        ]
        t = Table(cut_rows, colWidths=[3.0 * inch, 1.5 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        story.append(t)
        if tier_def.get("meaning"):
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"<i>{tier_def['meaning']}</i>", body_style))

    doc.build(
        story,
        onFirstPage=_footer_canvas,
        onLaterPages=_footer_canvas,
    )
