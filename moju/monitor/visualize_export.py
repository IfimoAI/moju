"""
Export pipeline for Moju Plotly dashboards.

Supports three artefact types:

- ``export_dashboard_html`` — single self-contained HTML.  When the input is a
  ``dash-tabs`` payload, a small tab-navigator UI is emitted so the file is
  fully usable offline.
- ``export_dashboard_png`` — static PNG (requires ``kaleido``); produces a
  vertically-stacked sheet for ``dash-tabs`` payloads.
- ``export_dashboard_pdf`` — invokes :func:`moju.monitor.report.write_audit_pdf`
  and decorates the result with rendered PNG cards.  Falls back gracefully
  when ``kaleido`` is unavailable.

The functions accept either a single :class:`plotly.graph_objects.Figure` or
the dict returned by :func:`moju.monitor.visualize_plotly.build_plotly_monitor_dash_payload`.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_dash_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("mode") == "dash-tabs" and isinstance(obj.get("tabs"), dict)


def _iter_tab_figures(payload: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return [(str(k), v) for k, v in payload["tabs"].items() if v is not None]


def _ensure_path(path: PathLike, *, suffix: Optional[str] = None) -> Path:
    p = Path(path)
    if suffix and p.suffix.lower() != suffix.lower():
        p = p.with_suffix(suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _kaleido_install_hint() -> str:
    return (
        "PNG/SVG export requires kaleido.  Install with `pip install -U kaleido` "
        "(included in core `pip install moju`)."
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
:root {{
  --bg: #f8fafc;
  --fg: #0f172a;
  --muted: #64748b;
  --tab-bg: #ffffff;
  --tab-active: #1d4ed8;
  --tab-border: #e2e8f0;
  --shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 4px 12px rgba(15, 23, 42, 0.06);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: var(--bg);
  color: var(--fg);
}}
header {{
  padding: 18px 28px;
  border-bottom: 1px solid var(--tab-border);
  background: #ffffff;
}}
header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
header p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
nav {{
  display: flex;
  gap: 0;
  padding: 0 28px;
  border-bottom: 1px solid var(--tab-border);
  background: #ffffff;
}}
nav button {{
  border: none;
  background: transparent;
  padding: 12px 18px;
  font: inherit;
  color: var(--muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
}}
nav button.active {{
  color: var(--tab-active);
  border-bottom-color: var(--tab-active);
}}
section.tab-panel {{
  padding: 22px 28px;
  display: none;
}}
section.tab-panel.active {{
  display: block;
}}
.card {{
  background: #ffffff;
  border: 1px solid var(--tab-border);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 16px;
}}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Moju enterprise dashboard · {tab_count} tab{tab_plural}</p>
</header>
<nav>{nav_html}</nav>
{panels_html}
<script>
const buttons = document.querySelectorAll('nav button');
const panels = document.querySelectorAll('section.tab-panel');
buttons.forEach(btn => btn.addEventListener('click', () => {{
  buttons.forEach(b => b.classList.toggle('active', b === btn));
  const target = btn.dataset.target;
  panels.forEach(p => p.classList.toggle('active', p.id === target));
}}));
</script>
</body>
</html>
"""


def _figure_to_html(fig: Any, *, full_html: bool = False, div_id: Optional[str] = None) -> str:
    import plotly.io as pio

    return pio.to_html(
        fig,
        full_html=full_html,
        include_plotlyjs="cdn",
        div_id=div_id,
        config={"responsive": True, "displaylogo": False},
    )


def export_dashboard_html(
    fig_or_payload: Any,
    path: PathLike,
    *,
    title: str = "Moju audit dashboard",
) -> Path:
    """Write a self-contained HTML file (with tab navigation for ``dash-tabs`` payloads)."""
    out = _ensure_path(path, suffix=".html")
    if _is_dash_payload(fig_or_payload):
        tabs = _iter_tab_figures(fig_or_payload)
        if not tabs:
            raise ValueError("Dash-tabs payload has no tabs.")
        nav_parts: List[str] = []
        panel_parts: List[str] = []
        for i, (name, fig) in enumerate(tabs):
            tab_id = f"tab-{i}"
            active = " active" if i == 0 else ""
            label = name.replace("_", " ").title()
            nav_parts.append(
                f'<button class="{active.strip() or "tab-btn"}" data-target="{tab_id}">{label}</button>'
            )
            div_id = f"plot-{i}"
            fig_html = _figure_to_html(fig, full_html=False, div_id=div_id)
            panel_parts.append(
                f'<section class="tab-panel{active}" id="{tab_id}"><div class="card">{fig_html}</div></section>'
            )
        html = _HTML_TEMPLATE.format(
            title=title,
            tab_count=len(tabs),
            tab_plural="" if len(tabs) == 1 else "s",
            nav_html="\n".join(nav_parts),
            panels_html="\n".join(panel_parts),
        )
        out.write_text(html, encoding="utf-8")
        return out
    # Single Plotly figure
    out.write_text(_figure_to_html(fig_or_payload, full_html=True), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# PNG / SVG (raster)
# ---------------------------------------------------------------------------


def _figure_to_image_bytes(fig: Any, *, fmt: str = "png", scale: float = 2.0, width: int = 1280, height: int = 720) -> bytes:
    try:
        import plotly.io as pio  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Plotly is required for image export: {exc}") from exc
    try:
        return fig.to_image(format=fmt, scale=scale, width=width, height=height)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{_kaleido_install_hint()}\n\nUnderlying error: {exc}") from exc


def export_dashboard_png(
    fig_or_payload: Any,
    path: PathLike,
    *,
    scale: float = 2.0,
    width: int = 1280,
    height: int = 720,
) -> List[Path]:
    """
    Write one or more PNG files; returns the list of written paths.

    For ``dash-tabs`` payloads, each tab is written to ``<path>_<tab>.png``
    and the list of paths is returned.  For a single figure, returns
    ``[path]``.
    """
    out_base = _ensure_path(path, suffix=".png")
    if _is_dash_payload(fig_or_payload):
        written: List[Path] = []
        stem = out_base.with_suffix("")
        for name, fig in _iter_tab_figures(fig_or_payload):
            target = stem.with_name(f"{stem.name}_{name}.png")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_figure_to_image_bytes(fig, fmt="png", scale=scale, width=width, height=height))
            written.append(target)
        return written
    out_base.write_bytes(_figure_to_image_bytes(fig_or_payload, fmt="png", scale=scale, width=width, height=height))
    return [out_base]


def export_dashboard_svg(
    fig_or_payload: Any,
    path: PathLike,
    *,
    scale: float = 1.0,
    width: int = 1280,
    height: int = 720,
) -> List[Path]:
    """Same as :func:`export_dashboard_png` but emits SVG vector files."""
    out_base = _ensure_path(path, suffix=".svg")
    if _is_dash_payload(fig_or_payload):
        written: List[Path] = []
        stem = out_base.with_suffix("")
        for name, fig in _iter_tab_figures(fig_or_payload):
            target = stem.with_name(f"{stem.name}_{name}.svg")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_figure_to_image_bytes(fig, fmt="svg", scale=scale, width=width, height=height))
            written.append(target)
        return written
    out_base.write_bytes(_figure_to_image_bytes(fig_or_payload, fmt="svg", scale=scale, width=width, height=height))
    return [out_base]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def export_dashboard_pdf(
    report_dict: Dict[str, Any],
    path: PathLike,
    *,
    fig_or_payload: Optional[Any] = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Path:
    """
    Write a PDF report by delegating to :func:`moju.monitor.report.write_audit_pdf`.

    When ``fig_or_payload`` is supplied and ``kaleido`` is installed, each
    rendered Plotly card is embedded after the standard report tables for a
    fully visual brief.

    Requires ReportLab (included in core ``moju``).
    """
    out = _ensure_path(path, suffix=".pdf")
    try:
        from moju.monitor.report import write_audit_pdf
    except ImportError as exc:
        raise ImportError(
            "moju.monitor.report.write_audit_pdf requires ReportLab; install via `pip install moju`."
        ) from exc

    # 1. Always write the textual PDF first via the existing pipeline.
    write_audit_pdf(report_dict, str(out), model_name=model_name, model_id=model_id)

    # 2. Optionally append rendered images of the dashboard tabs (best-effort).
    if fig_or_payload is None:
        return out
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas as rcanvas
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore
    except Exception:  # noqa: BLE001 - merging is optional
        return out

    images: List[Tuple[str, bytes]] = []
    try:
        if _is_dash_payload(fig_or_payload):
            for name, fig in _iter_tab_figures(fig_or_payload):
                try:
                    images.append((name, _figure_to_image_bytes(fig, fmt="png", scale=2.0, width=1280, height=720)))
                except Exception:  # noqa: BLE001
                    continue
        else:
            images.append(("dashboard", _figure_to_image_bytes(fig_or_payload, fmt="png", scale=2.0, width=1280, height=720)))
    except Exception:  # noqa: BLE001
        return out
    if not images:
        return out

    # 3. Build supplementary canvas with one image per page and append.
    suppl = io.BytesIO()
    c = rcanvas.Canvas(suppl, pagesize=letter)
    width_pt, height_pt = letter
    margin = 0.5 * inch
    max_w = width_pt - 2 * margin
    max_h = height_pt - 2 * margin - 0.4 * inch
    from reportlab.lib.utils import ImageReader

    for name, png_bytes in images:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, height_pt - margin, name.replace("_", " ").title())
        img = ImageReader(io.BytesIO(png_bytes))
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        draw_w = iw * scale
        draw_h = ih * scale
        c.drawImage(
            img,
            (width_pt - draw_w) / 2.0,
            (height_pt - draw_h) / 2.0 - 0.2 * inch,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            anchor="c",
        )
        c.showPage()
    c.save()
    suppl.seek(0)

    try:
        base = PdfReader(str(out))
        extra = PdfReader(suppl)
        writer = PdfWriter()
        for page in base.pages:
            writer.add_page(page)
        for page in extra.pages:
            writer.add_page(page)
        with open(out, "wb") as fh:
            writer.write(fh)
    except Exception:  # noqa: BLE001
        # If merging fails, leave the textual PDF intact.
        pass
    return out


__all__ = [
    "export_dashboard_html",
    "export_dashboard_png",
    "export_dashboard_svg",
    "export_dashboard_pdf",
]
