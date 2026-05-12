"""
Upload state, configure physics via Studio allowlists (or expert JSON), run audit + visualize.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import streamlit as st

from apps.moju_studio.studio_streamlit_extras import (
    as_fragment,
    pipeline_status,
    status_complete,
    status_update,
    studio_sidebar_branding_and_nav,
    toast,
)
from apps.moju_studio.config_forms import (
    merge_simple_config_with_json_override,
    path_b_grid_from_options,
    preflight_checklist_with_dependency_plan,
    reindex_log_entries,
)
from apps.moju_studio.studio_auto_config import (
    STUDIO_GROUP_NAMES_EFFECTIVE,
    STUDIO_LAW_NAMES,
    STUDIO_MODEL_NAMES,
    build_studio_auto_fragment,
)
from apps.moju_studio.studio_law_fd_hints import format_laws_fd_help
from apps.moju_studio.studio_core import (
    audit_report_to_jsonable,
    dependency_plan_for_path_b_run,
    flatten_residuals,
    generate_python_snippet,
    list_registered_law_names,
    make_session_state_builder,
    monitor_config_from_merged_dict,
    validate_studio_pi_gating,
)
from apps.moju_studio.studio_dependency_planner import (
    format_planner_preflight_warning,
    plan_markdown_for_display,
)
from apps.moju_studio.studio_model_derived_registry import enrich_fragment_from_model_audits
from moju.monitor.path_b_derivatives import PathBGridConfig
from apps.moju_studio.studio_io import (
    constants_json_to_dict,
    load_state_bundle_bytes,
    merge_monitor_config_fragment,
    parse_monitor_config_json,
    validate_non_empty_state,
)

_STATE_UPLOAD_TYPES = ["npz", "npy", "h5", "hdf5", "nc", "nc4"]
from apps.moju_studio.studio_plots import plotly_pred_minus_ref, plotly_residual_or_state
from moju.monitor import ResidualEngine, audit, build_monitor_visualize_bundle, visualize
from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals
from moju.monitor.visualize_labels import format_admissibility_pct
from moju.monitor.visualize_plotly import (
    build_plotly_category_admissibility_bar_figure,
    build_plotly_law_rnorm_final_bar_figure,
    build_plotly_spatial_rnorm_heatmap_card,
    format_admissibility_status_label,
)

# Sidebar → Path B finite-difference grid (shared by Run tab and Config dependency preview).
_FD_SPATIAL_OPTIONS = ["Auto (infer from arrays)", "1D", "2D", "3D"]
_FD_SPATIAL_LABEL_TO_SD: Dict[str, Any] = {
    "Auto (infer from arrays)": "auto",
    "1D": 1,
    "2D": 2,
    "3D": 3,
}

_STUDIO_HEATMAP_COLORSCALES = ("Jet", "Turbo", "Viridis")


def _fd_grid_kw_from_sidebar_session() -> Dict[str, Any]:
    lab = st.session_state.get("studio_fd_spatial_label", _FD_SPATIAL_OPTIONS[0])
    sd = _FD_SPATIAL_LABEL_TO_SD.get(lab, "auto")
    steady = st.session_state.get("studio_fd_time_label", "Steady") == "Steady"
    layout = st.session_state.get("studio_fd_layout", "meshgrid")
    if layout not in ("meshgrid", "separable"):
        layout = "meshgrid"
    return {"spatial_dimension": sd, "steady": steady, "layout": layout}


def _fd_expected_field_shapes_md() -> str:
    """
    Human-readable expected shapes for a scalar field ``T`` and coords, from sidebar FD settings.
    Matches Path B / ``PathBGridConfig`` conventions for FD + law fill.
    """
    lab = st.session_state.get("studio_fd_spatial_label", _FD_SPATIAL_OPTIONS[0])
    sd = _FD_SPATIAL_LABEL_TO_SD.get(lab, "auto")
    steady = st.session_state.get("studio_fd_time_label", "Steady") == "Steady"
    layout = str(st.session_state.get("studio_fd_layout", "meshgrid"))
    if layout not in ("meshgrid", "separable"):
        layout = "meshgrid"
    time_lbl = "Steady" if steady else "Transient"
    lay_lbl = "Meshgrid" if layout == "meshgrid" else "Separable"

    lines: List[str] = [
        f"**Sidebar:** *{lab}* · *{time_lbl}* · *{lay_lbl}*  ",
        "",
        "Suggested layouts for a **scalar** field **`T`** (e.g. Fourier conduction) and coords:",
    ]

    def add_block(dim_name: str, d: int) -> None:
        if steady:
            if layout == "separable":
                if d == 1:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x)` · `x(n_x)` — 1D coord vector (not `(n_x,1)`)."
                    )
                elif d == 2:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x, n_y)` · `x(n_x)` · `y(n_y)`."
                    )
                else:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x, n_y, n_z)` · `x(n_x)` · `y(n_y)` · `z(n_z)`."
                    )
            else:
                if d == 1:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x)` · `x` same shape as `T` **or** same **total size** "
                        f"(e.g. `x(n_x, 1)`)."
                    )
                elif d == 2:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x, n_y)` · `x`,`y` same shape as `T` (or rectilinear rules in engine)."
                    )
                else:
                    lines.append(
                        f"- **{dim_name}:** `T(n_x, n_y, n_z)` · `x`,`y`,`z` same shape as `T` when full mesh arrays."
                    )
        else:
            if layout == "separable":
                if d == 1:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x)`** · **`t(n_t)`** · **`x(n_x)`** — time is the **leading** axis."
                    )
                elif d == 2:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x, n_y)`** · **`t(n_t)`** · **`x(n_x)`** · **`y(n_y)`**."
                    )
                else:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x, n_y, n_z)`** · **`t(n_t)`** · **`x(n_x)`** · **`y(n_y)`** · **`z(n_z)`**."
                    )
            else:
                if d == 1:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x)`** · **`t(n_t)`** · `x` length `n_x` or aligned with each row `T[i, :]`."
                    )
                elif d == 2:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x, n_y)`** · **`t(n_t)`** · spatial coords match `T[0, …]` shape."
                    )
                else:
                    lines.append(
                        f"- **{dim_name}:** **`T(n_t, n_x, n_y, n_z)`** · **`t(n_t)`** · coords match spatial slice."
                    )

    if sd == "auto":
        lines.append(
            "- **Auto (spatial):** Moju infers dimension from `T`. Examples: steady 1D-style `T(n_x)`, `(n_x,1)`, `(1,n_x)`; "
            "for **Transient**, use explicit **1D / 2D / 3D** if `T` is flat — you usually need **`T(n_t, n_x, …)`** with **`t(n_t)`**, not one long `(N,)` vector."
        )
        lines.append("")
        lines.append("*If you treat the grid as 1D:*")
        add_block("1D", 1)
    elif sd == 1:
        add_block("1D", 1)
    elif sd == 2:
        add_block("2D", 2)
    else:
        add_block("3D", 3)

    lines.append("")
    lines.append(
        "**Vector fields** (e.g. `u`): same leading layout + trailing axis `[…, d]` for components."
    )
    return "\n".join(lines)


def _fd_expected_field_one_liner() -> str:
    """Short hint for captions (scalar T)."""
    lab = st.session_state.get("studio_fd_spatial_label", _FD_SPATIAL_OPTIONS[0])
    sd = _FD_SPATIAL_LABEL_TO_SD.get(lab, "auto")
    steady = st.session_state.get("studio_fd_time_label", "Steady") == "Steady"
    layout = str(st.session_state.get("studio_fd_layout", "meshgrid"))
    if layout not in ("meshgrid", "separable"):
        layout = "meshgrid"
    if not steady:
        if sd == 1:
            return "**Heads-up:** Transient **1D** → use **`T(n_t, n_x)`**, **`t(n_t)`**, **`x(n_x)`** (not a single flat `(N,)`). Open expander above for full detail."
        if sd == 2:
            return "**Heads-up:** Transient **2D** → **`T(n_t, n_x, n_y)`**, **`t(n_t)`**, coords per layout. See expander above."
        if sd == 3:
            return "**Heads-up:** Transient **3D** → **`T(n_t, n_x, n_y, n_z)`**, **`t(n_t)`**, … See expander above."
        return "**Heads-up:** **Transient** → time is the **first** axis of `T`; use **`t(n_t)`**. Pick **1D/2D/3D** and see expander above."
    if sd == 1 and layout == "separable":
        return "**Heads-up:** Separable **1D** → **`T(n_x)`**, **`x(n_x)`** (1D vector). See expander above."
    return "Expected **`T`** / coord shapes follow **sidebar → Path B — FD grid**. Open the expander above."


st.set_page_config(page_title="Moju Studio — Audit", layout="wide", page_icon="🔬")


@st.dialog("Clear session log")
def _dialog_clear_viz_log() -> None:
    st.markdown("Remove all accumulated steps used for multi-step **visualize**.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear log", type="primary"):
            st.session_state["viz_log"] = []
            toast("Session log cleared", icon="✅")
            st.rerun()
    with c2:
        if st.button("Cancel"):
            st.rerun()


with st.sidebar:
    studio_sidebar_branding_and_nav()
    st.divider()
    st.markdown("##### Path B — FD grid")
    st.caption(
        "Finite differences (**Compute state derivatives (finite difference)** on Run) and **Dependency preview** use these settings. "
        "*(API: `auto_path_b_derivatives`.)*"
    )
    st.selectbox(
        "Spatial data",
        options=_FD_SPATIAL_OPTIONS,
        index=0,
        key="studio_fd_spatial_label",
        help="Which mesh coordinates NPZ should include for spatial FD hints (e.g. 1D+Steady: `x` only; 3D adds `y`,`z`).",
    )
    st.radio(
        "Time",
        options=["Steady", "Transient"],
        horizontal=True,
        index=0,
        key="studio_fd_time_label",
        help="Transient: `t` is required when filling ∂/∂t-style law inputs or time stacks per engine rules.",
    )
    _fd_layout_labels = {
        "meshgrid": "Meshgrid (coords same shape as fields)",
        "separable": "Separable (1D x / y / z vectors)",
    }
    st.selectbox(
        "Grid layout",
        options=["meshgrid", "separable"],
        index=0,
        key="studio_fd_layout",
        format_func=lambda v: _fd_layout_labels.get(str(v), str(v)),
        help="Meshgrid: each coord array matches field shape (e.g. T and x both (nx,) or (nx,ny)). "
        "Separable: x length nx, field (nx,), (nx,ny), or (nx,ny,nz); spacing via 1D coords.",
    )
    st.caption(
        "**Laplacian / FD:** For 1D data use **Spatial data → 1D** (or Auto with `(N,)` / `(N,1)` / `(1,N)` fields). "
        "**Meshgrid:** `T` and `x` same shape *or* same total size. **Separable:** 1D `x` length `nx` and `T` shape `(nx,)`, "
        "`(nx,ny)`, … After **Run**, open **Path B FD messages** if `T_laplacian` is missing or laws fail."
    )
    st.divider()
    st.checkbox(
        "Append next run to session log",
        key="sb_append_log",
        help="Chain multiple runs into one Plotly timeline (same browser session).",
    )
    if st.button("Clear session log…"):
        _dialog_clear_viz_log()
    st.selectbox(
        "Heatmap colorscale (monitor dashboard)",
        options=list(_STUDIO_HEATMAP_COLORSCALES),
        index=0,
        key="sb_heatmap_cs",
        help="Applied to governing-law and constitutive R_norm heatmaps in the Plotly monitor dashboard.",
    )
    st.selectbox(
        "Spatial heatmap axis",
        options=["x", "y", "z"],
        index=0,
        key="sb_spatial_axis",
        help="Coordinate for spatial R_norm slices (requires that key in state_pred, e.g. y for 2D/3D).",
    )


DEFAULT_CONFIG_FRAGMENT = """{
  "laws": [],
  "groups": [],
  "law_implied_audits": true,
  "constitutive_audit": [],
  "derived_state_chain": []
}
"""

DEMO_LAPLACE = """{
  "laws": [
    {
      "name": "laplace_equation",
      "state_map": {
        "phi_laplacian": "phi_laplacian"
      }
    }
  ],
  "groups": [],
  "law_implied_audits": true,
  "constitutive_audit": [],
  "derived_state_chain": []
}
"""

DEMO_CONSTANTS = """{}
"""


def _parse_float_dict(raw: str, label: str) -> Dict[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    d = json.loads(raw)
    if not isinstance(d, dict):
        raise ValueError(f"{label} must be a JSON object")
    out: Dict[str, float] = {}
    for k, v in d.items():
        out[str(k)] = float(v)
    return out


def _studio_monitor_spatial_bundle() -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Spatial panels + Plotly heatmap colorscale from sidebar session state."""
    eng = st.session_state.get("last_engine")
    log_entry = eng.log[-1] if eng and getattr(eng, "log", None) else None
    log_idx = (len(eng.log) - 1) if eng and getattr(eng, "log", None) and len(eng.log) > 0 else None
    first_rms = (eng.log[0].get("rms") if eng and eng.log else None) or {}
    coord = str(st.session_state.get("sb_spatial_axis", "x"))
    prefer_last_t = st.session_state.get("studio_fd_time_label", "Steady") != "Steady"
    r_ref = st.session_state.get("last_r_ref") or {}
    if not isinstance(r_ref, dict):
        r_ref = {}
    pred: Dict[str, Any] = {}
    sp = st.session_state.get("state_pred")
    if sp is not None:
        pred.update(dict(sp))
    if log_entry:
        snap = log_entry.get("coord_snapshot")
        if isinstance(snap, Mapping):
            for k in ("x", "y", "z", "t"):
                if k in snap and snap[k] is not None and k not in pred:
                    pred[k] = snap[k]
    law_panel, implied_panel = build_spatial_rnorm_panels_from_residuals(
        st.session_state.get("last_residuals"),
        pred,
        coord_key=coord,
        prefer_last_t=prefer_last_t,
        log_entry=log_entry,
        first_rms=first_rms,
        r_ref=r_ref,
        log_step_index=log_idx,
        normalize_spatial=False,
    )
    cs = str(st.session_state.get("sb_heatmap_cs", "Jet"))
    return law_panel, implied_panel, cs


@as_fragment
def studio_redraw_plotly_fragment() -> None:
    if not st.session_state.get("viz_rms_keys"):
        return
    with st.expander("Redraw Plotly dashboard (subset of keys)", expanded=False):
        vk = st.multiselect(
            "Keys to include",
            options=st.session_state["viz_rms_keys"],
            default=st.session_state["viz_rms_keys"][: min(16, len(st.session_state["viz_rms_keys"]))],
            key="viz_redraw_keys",
        )
        if st.button("Redraw dashboard", key="viz_redraw_btn"):
            try:
                _law_panel, _rnorm_panel, _hm_cs = _studio_monitor_spatial_bundle()
                _pref_t = st.session_state.get("studio_fd_time_label", "Steady") != "Steady"
                _sax = str(st.session_state.get("sb_spatial_axis", "x"))
                fig2 = visualize(
                    st.session_state.get("viz_log") or [],
                    keys=list(vk) if vk else None,
                    backend="plotly",
                    r_ref=st.session_state.get("last_r_ref") or None,
                    max_legend_keys=int(st.session_state.get("last_max_leg", 16)),
                    mode="training",
                    spatial_law_panel=_law_panel,
                    spatial_rnorm_panel=_rnorm_panel,
                    residuals=st.session_state.get("last_residuals"),
                    state_pred=st.session_state.get("state_pred"),
                    spatial_coord_key=_sax,
                    spatial_prefer_last_t=_pref_t,
                    spatial_heatmap_colorscale=_hm_cs,
                    engine=st.session_state.get("last_engine"),
                )
                if fig2 is not None:
                    try:
                        st.plotly_chart(
                            fig2,
                            use_container_width=True,
                            key="plotly_redraw",
                            on_select="rerun",
                            selection_mode="points",
                        )
                    except TypeError:
                        st.plotly_chart(fig2, use_container_width=True, key="plotly_redraw")
                toast("Dashboard redrawn", icon="📊")
            except Exception as ex:  # noqa: BLE001
                st.warning(str(ex))


_SPATIAL_VIEW_OPTIONS: Dict[str, str] = {
    "Auto (line / heatmap / histogram)": "auto",
    "3D surface (2D field + x, y)": "surface3d",
    "3D volume (3D field + x, y, z)": "volume3d",
}

_ARRAY_PLOT_OPTIONS: Dict[str, str] = {
    "Auto": "auto",
    "Line (time slice)": "line",
    "Heatmap (2D)": "heatmap2d",
}


@as_fragment
def studio_spatial_fragment() -> None:
    st.subheader("Explore a residual or state array")
    res = st.session_state.get("last_residuals")
    pred = st.session_state.get("state_pred")
    ref = st.session_state.get("state_ref")
    if not res and not pred:
        st.info("Run an audit on the **Run** tab first, then use **Dashboard** for summaries.")
        return
    flat = flatten_residuals(res) if res else {}
    keys = sorted(set(flat.keys()) | set((pred or {}).keys()))
    _hm_sp = str(st.session_state.get("sb_heatmap_cs", "Jet"))
    _sv_label = st.selectbox(
        "Spatial plot type",
        options=list(_SPATIAL_VIEW_OPTIONS.keys()),
        index=0,
        key="sp_spatial_view_label",
        help="3D surface: 2D slice (after time index) with 1D x and y in state_pred. "
        "3D volume: 3D array with shape (len(x), len(y), len(z)) and 1D coords.",
    )
    _spatial_view = _SPATIAL_VIEW_OPTIONS[_sv_label]
    _ap_label = st.selectbox(
        "Array plot (for 2D / time–space fields)",
        options=list(_ARRAY_PLOT_OPTIONS.keys()),
        index=0,
        key="sp_array_plot_label",
        help="**Heatmap (2D)** keeps the full 2D array (e.g. time × space) instead of slicing to a 1D line.",
    )
    _array_plot = _ARRAY_PLOT_OPTIONS[_ap_label]
    mode = st.radio("View", ["Single array", "pred − ref (shared keys)"], horizontal=True, key="sp_mode")
    if mode.startswith("pred"):
        if not ref:
            st.warning("Upload `state_ref` on the Data tab.")
        else:
            common = sorted(set(pred.keys()) & set(ref.keys()))
            if not common:
                st.warning("No shared keys between pred and ref.")
            else:
                choice = st.selectbox("Shared key", common, key="sp_key_pr")
                t_idx = None
                pa = jnp.asarray(pred[choice])
                ra = jnp.asarray(ref[choice])
                shape = tuple(int(x) for x in pa.shape)
                st.caption(f"shape pred = {shape}, ref = {tuple(int(x) for x in ra.shape)}")
                t_arr = (pred or {}).get("t")
                t_idx = None
                if _array_plot != "heatmap2d" and len(shape) >= 1 and shape[0] > 1 and t_arr is not None:
                    t_idx = st.slider("Time index (pred-ref)", 0, shape[0] - 1, shape[0] - 1, key="sp_t2")
                x_arr = (pred or {}).get("x")
                y_arr = (pred or {}).get("y")
                z_arr = (pred or {}).get("z")
                t_flat = np.asarray(jnp.asarray(t_arr)).ravel() if t_arr is not None else None
                fig = plotly_pred_minus_ref(
                    pa,
                    ra,
                    title=choice,
                    time_index=t_idx,
                    time_axis=0,
                    t_coord=t_flat,
                    heatmap_colorscale=_hm_sp,
                    spatial_view=_spatial_view,  # type: ignore[arg-type]
                    array_plot=_array_plot,  # type: ignore[arg-type]
                    x=np.asarray(jnp.asarray(x_arr)).ravel() if x_arr is not None else None,
                    y=np.asarray(jnp.asarray(y_arr)).ravel() if y_arr is not None else None,
                    z_coord=np.asarray(jnp.asarray(z_arr)).ravel() if z_arr is not None else None,
                )
                _pc_spatial(fig, "plotly_spatial_pr")
    else:
        choice = st.selectbox("Array key", keys, key="sp_key_single")
        arr = flat.get(choice) if choice in flat else (pred or {}).get(choice)
        if arr is None:
            st.warning("Key not found.")
        else:
            shape = tuple(int(x) for x in jnp.asarray(arr).shape)
            st.caption(f"shape = {shape}")
            t_idx = None
            t_arr = (pred or {}).get("t")
            if _array_plot != "heatmap2d" and len(shape) >= 1 and shape[0] > 1 and t_arr is not None:
                t_idx = st.slider("Time index", 0, shape[0] - 1, shape[0] - 1, key="sp_t1")
            x_arr = (pred or {}).get("x")
            y_arr = (pred or {}).get("y")
            z_arr = (pred or {}).get("z")
            t_flat = np.asarray(jnp.asarray(t_arr)).ravel() if t_arr is not None else None
            fig = plotly_residual_or_state(
                arr,
                title=choice,
                x=np.asarray(jnp.asarray(x_arr)).ravel() if x_arr is not None else None,
                y=np.asarray(jnp.asarray(y_arr)).ravel() if y_arr is not None else None,
                z_coord=np.asarray(jnp.asarray(z_arr)).ravel() if z_arr is not None else None,
                time_index=t_idx,
                time_axis=0,
                t_coord=t_flat,
                heatmap_colorscale=_hm_sp,
                spatial_view=_spatial_view,  # type: ignore[arg-type]
                array_plot=_array_plot,  # type: ignore[arg-type]
            )
            _pc_spatial(fig, "plotly_spatial_single")


def _pc_spatial(fig: Any, key: str) -> None:
    """Plotly chart; use ``on_select`` when supported for lasso/point feedback reruns."""
    try:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _plotly_dashboard_card(fig: Any, key: str) -> None:
    """Plotly chart for Dashboard cards (same interaction as spatial plots)."""
    if fig is None:
        return
    try:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


st.title("Audit workspace")
tab_data, tab_cfg, tab_run, tab_dash, tab_divergence, tab_space, tab_export = st.tabs(
    ["Data", "Config", "Run", "Dashboard", "Constitutive divergence", "Spatial / time", "Export"]
)

with tab_data:
    st.subheader("State prediction (`state_pred`)")
    with st.expander("Expected shapes for `T` and coords (from sidebar **Path B — FD grid**)", expanded=False):
        st.markdown(_fd_expected_field_shapes_md())
    st.caption(_fd_expected_field_one_liner())
    st.caption(
        "Formats: **.npz** (multi-key), **.npy** (single array — set key name below), "
        "**.h5** / **.hdf5**, **.nc** / **.nc4** (install `moju[studio-science]` for HDF5/NetCDF)."
    )
    pred_npy_key = st.text_input(
        "NPY array key name (used only for `.npy` uploads)",
        value="field",
        key="studio_pred_npy_key",
        help="Name of the state field in Moju’s dict, e.g. T or u.",
    )
    pred_science_sel = st.text_input(
        "HDF5 / NetCDF selection (optional)",
        value="",
        key="studio_pred_science_sel",
        help="Comma-separated dataset paths (HDF5) or variable names (NetCDF). Leave empty to load all numeric arrays (capped at 512).",
    )
    up = st.file_uploader(
        "Upload `state_pred` (.npz, .npy, .h5, .hdf5, .nc, .nc4)",
        type=_STATE_UPLOAD_TYPES,
    )
    if up is not None:
        try:
            pred = load_state_bundle_bytes(
                up.getvalue(),
                filename=up.name or "state.npz",
                npy_key=pred_npy_key,
                science_selection=pred_science_sel,
            )
        except (ImportError, ValueError) as e:
            st.error(str(e))
        else:
            st.session_state["state_pred"] = pred
            st.success(
                f"Loaded {len(pred)} arrays: {', '.join(sorted(pred.keys())[:20])}{'…' if len(pred) > 20 else ''}"
            )
            toast("state_pred loaded", icon="📁")
    elif "state_pred" in st.session_state:
        st.caption(f"Using cached state ({len(st.session_state['state_pred'])} keys).")
    else:
        st.warning("Upload a state file to define `state_pred`.")

    _pred_view = st.session_state.get("state_pred") or {}
    if _pred_view:
        rows = []
        for k in sorted(_pred_view.keys()):
            v = _pred_view[k]
            sh = getattr(v, "shape", None)
            dt = getattr(v, "dtype", None)
            rows.append(
                {
                    "key": k,
                    "shape": str(tuple(sh)) if sh is not None else "",
                    "dtype": str(dt) if dt is not None else "",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Optional reference (`state_ref`)")
    ref_npy_key = st.text_input(
        "NPY key for `state_ref` (only for `.npy`)",
        value="field",
        key="studio_ref_npy_key",
    )
    ref_science_sel = st.text_input(
        "HDF5 / NetCDF selection for `state_ref` (optional)",
        value="",
        key="studio_ref_science_sel",
    )
    up_ref = st.file_uploader(
        "state_ref (.npz, .npy, .h5, .hdf5, .nc, .nc4)",
        type=_STATE_UPLOAD_TYPES,
    )
    if up_ref is not None:
        try:
            st.session_state["state_ref"] = load_state_bundle_bytes(
                up_ref.getvalue(),
                filename=up_ref.name or "ref.npz",
                npy_key=ref_npy_key,
                science_selection=ref_science_sel,
            )
        except (ImportError, ValueError) as e:
            st.error(str(e))
        else:
            st.success("state_ref loaded.")
            toast("state_ref loaded", icon="📁")
    elif st.session_state.get("state_ref"):
        st.caption("Using cached state_ref.")

    st.subheader("Collocation (Path A / optional)")
    st.caption("Path A shim passes this dict to `compute_residuals(..., collocation=...)`.")
    col_json = st.text_area("Collocation JSON", value="{}", height=80)
    try:
        st.session_state["collocation"] = json.loads(col_json or "{}")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")

with tab_cfg:
    st.markdown(
        "### Simplified physics setup\n"
        "- Name **`state_pred`** keys to match **law / model / group argument names** "
        "(e.g. `k_solid`, `T`, `rho`). Constants go in **Constants JSON**.\n"
        "- **Models** → constitutive audits; **Groups** → both `groups` and scaling audits (auto).\n"
        "- **Finite-difference options** default **on** on the Run tab (grid coords `x`… in `state_pred`)."
    )

    st.subheader("Constants")
    cjson = st.text_area("Constants JSON (merged into config)", value=DEMO_CONSTANTS, height=100)
    try:
        st.session_state["constants_dict"] = constants_json_to_dict(cjson)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(str(e))
        st.session_state["constants_dict"] = {}

    expert = st.checkbox(
        "Expert: edit full MonitorConfig JSON (disables auto builder below)",
        value=False,
        key="cfg_expert_mode",
    )

    if expert:
        preset = st.selectbox(
            "Template",
            ["Empty audits", "Demo: Laplace law (needs phi_laplacian in NPZ)"],
            key="cfg_expert_preset",
        )
        base_json = DEMO_LAPLACE if preset.startswith("Demo") else DEFAULT_CONFIG_FRAGMENT
        st.text_area(
            "MonitorConfig fragment (JSON)",
            value=base_json,
            height=260,
            key="config_fragment_raw",
        )
        st.caption(
            "Law FD: derived inputs (e.g. `T_laplacian`) need **primitives** + **x** (… in `state_pred`) "
            "with Run tab **Compute state derivatives (finite difference)** + **Compute law derivatives (finite difference)**. "
            "See `moju.monitor.law_fd_recipes`. *(API: `auto_path_b_derivatives`, `fill_law_fd`.)* "
            "**Constitutive `implied_delta` / `ref_delta`:** always nondimensional symmetric discrepancy "
            "`(F−\\tilde F)/(ε+|F|+|\\tilde F|)` (or divide by `ε+|ref|` if `*_ref_key` / `{output_key}_ref` is set). "
            "**Fourier / `fo`:** implied **`Groups.fo`** needs **`alpha`**, **`t`**, **`L`**. **`t`** = mesh **time coordinate** in "
            "**`state_pred`** (match sidebar **Path B — FD grid** `key_t`, default `t`; aliases `time`/`coords_t`), not Constants unless you want a scalar broadcast. "
            "**`thermal_diffusivity`** auto-fills **`alpha`** from `k`,`rho`,`cp` when selected. README → Fourier."
        )
    else:
        st.multiselect(
            "Laws (FD-supported subset)",
            options=list(STUDIO_LAW_NAMES),
            default=[],
            key="st_auto_laws",
            help="Governing laws; arguments use identity state_map (NPZ key = law argument name).",
        )
        st.multiselect(
            "Models → constitutive audits",
            options=list(STUDIO_MODEL_NAMES),
            default=[],
            key="st_auto_models",
            help="Each model becomes one constitutive_audit row with automatic chain keys from your NPZ.",
        )
        st.multiselect(
            "Groups → dimensionless groups (for laws)",
            options=list(STUDIO_GROUP_NAMES_EFFECTIVE),
            default=[],
            key="st_auto_groups",
            help="Each group builds a `groups` entry so Moju can compute e.g. `fo`, `re` before laws.",
        )
        st.text_area(
            "Optional JSON override (merge into auto fragment: laws, groups, audits, primary_fields, derived_state_chain)",
            value="{}",
            height=100,
            key="sf_json_override",
        )
        st.caption(
            "**Fourier:** **`t`** = time coord array in NPZ (`key_t`), not Constants by default. **thermal_diffusivity** fills **`alpha`** from `k`,`rho`,`cp`. Registry: `studio_model_derived_registry.py`. README § Fourier."
        )

        _laws_sel = st.session_state.get("st_auto_laws") or []
        if _laws_sel:
            with st.expander("Law FD prerequisites (selected laws)", expanded=False):
                st.markdown(format_laws_fd_help(list(_laws_sel)))

    with st.expander("Dependency preview (NPZ + constants + FD)", expanded=False):
        pred_keys_preview = set((st.session_state.get("state_pred") or {}).keys())
        cdict = st.session_state.get("constants_dict") or {}
        ck = set(cdict.keys())
        try:
            if expert:
                fr = parse_monitor_config_json(st.session_state.get("config_fragment_raw") or "{}")
            else:
                fr = build_studio_auto_fragment(
                    law_names=list(st.session_state.get("st_auto_laws") or []),
                    model_names=list(st.session_state.get("st_auto_models") or []),
                    group_names=list(st.session_state.get("st_auto_groups") or []),
                    pred_keys=pred_keys_preview,
                    constant_keys=ck,
                )
                override = st.session_state.get("sf_json_override") or "{}"
                fr = merge_simple_config_with_json_override(fr, override)
            fr = merge_monitor_config_fragment(fr, {"constants": cdict})
            fr = enrich_fragment_from_model_audits(fr)
            preview_grid_kw = dict(_fd_grid_kw_from_sidebar_session())
            if st.session_state.get("run_fd_customize"):
                preview_grid_kw["key_x"] = st.session_state.get("run_grid_key_x", "x")
                preview_grid_kw["key_y"] = st.session_state.get("run_grid_key_y", "y")
                preview_grid_kw["key_z"] = st.session_state.get("run_grid_key_z", "z")
                preview_grid_kw["key_t"] = st.session_state.get("run_grid_key_t", "t")
            preview_grid = path_b_grid_from_options(**preview_grid_kw)
            st.caption(
                "Assumes **Compute state derivatives (finite difference)** and **Compute law derivatives (finite difference)** are ON. "
                "Spatial data, time (steady/transient), and grid layout match the **sidebar** (*Path B — FD grid*). "
                "Built-in **aliases** (e.g. temperature→T) are noted when relevant. "
                "*(API: `auto_path_b_derivatives`, `fill_law_fd`.)*"
            )
            st.markdown(
                plan_markdown_for_display(
                    fr,
                    pred_keys=pred_keys_preview,
                    constant_keys=ck,
                    auto_path_b_derivatives=True,
                    fill_law_fd=True,
                    path_b_grid=preview_grid,
                )
            )
        except Exception as e:  # noqa: BLE001
            st.warning(str(e))

    with st.expander("Studio allowlists (reference)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_area("Laws", "\n".join(STUDIO_LAW_NAMES), height=200, disabled=True)
        with c2:
            st.text_area("Models", "\n".join(STUDIO_MODEL_NAMES), height=200, disabled=True)
        with c3:
            st.text_area("Groups", "\n".join(STUDIO_GROUP_NAMES_EFFECTIVE), height=200, disabled=True)

with tab_run:
    st.caption(
        "Use the **sidebar** for **Path B — FD grid** (spatial dimension, time, layout), session log, and dashboard mode. "
        "Submit the form below to run the pipeline (single batch update)."
    )
    with st.form("audit_run_form"):
        path_mode = st.radio(
            "Execution path",
            [
                "Path B — pass uploaded `state_pred` (default)",
                "Path A — NPZ `state_builder` shim (π-constant needs a recomputing builder)",
            ],
            horizontal=True,
            index=0,
        )
        path_b = path_mode.startswith("Path B")
        cfd1, cfd2 = st.columns(2)
        with cfd1:
            auto_fd = st.checkbox(
                "Compute state derivatives (finite difference)",
                value=True,
                help="Fills missing audit d_* derivatives on the Path B grid (needs mesh coordinates in state_pred). "
                "API: auto_path_b_derivatives.",
            )
        with cfd2:
            fill_law = st.checkbox(
                "Compute law derivatives (finite difference)",
                value=True,
                help="Requires state derivatives above; fills registered Laws.* inputs (e.g. Laplacians) on the grid. "
                "API: fill_law_fd.",
            )
        st.caption(
            "**Spatial data**, **Time**, and **Grid layout** are in the **sidebar** (*Path B — FD grid*). "
            "Example: **1D + Steady + Meshgrid** → NPZ needs `x` (and fields like `T`); `y`/`z`/`t` not required for spatial FD hints. "
            "**Separable** uses 1D `x` (and `y`/`z`) vectors vs field shape `(nx,)`, `(nx,ny)`, …"
        )
        use_custom_grid = st.checkbox(
            "Customize coordinate key names (PathBGridConfig key_x / key_y / key_z / key_t)",
            value=False,
            key="run_fd_customize",
        )
        grid_kw: Dict[str, Any] = {}
        if auto_fd:
            grid_kw.update(_fd_grid_kw_from_sidebar_session())
        if auto_fd and use_custom_grid:
            st.caption("Override NPZ axis names if your mesh uses other keys (defaults: `x`,`y`,`z`,`t`).")
            grid_kw["key_x"] = st.text_input("key_x", value="x", key="run_grid_key_x")
            grid_kw["key_y"] = st.text_input("key_y", value="y", key="run_grid_key_y")
            grid_kw["key_z"] = st.text_input("key_z", value="z", key="run_grid_key_z")
            grid_kw["key_t"] = st.text_input("key_t", value="t", key="run_grid_key_t")

        run_clicked = st.form_submit_button("Run compute_residuals + audit", type="primary")

    if run_clicked:
        pred = st.session_state.get("state_pred")
        ok, msg = validate_non_empty_state(pred or {})
        if not ok:
            st.error(msg)
        else:
            try:
                r_ref = _parse_float_dict(
                    str(st.session_state.get("dashboard_r_ref_json", "")), "r_ref"
                )
                weights = _parse_float_dict(
                    str(st.session_state.get("dashboard_weights_json", "")), "weights"
                )
            except (json.JSONDecodeError, ValueError) as e:
                st.error(str(e))
                st.stop()

            if fill_law and not auto_fd:
                st.error(
                    "Compute law derivatives (finite difference) requires Compute state derivatives "
                    "(finite difference): enable state FD or turn off law FD. "
                    "(API: fill_law_fd needs auto_path_b_derivatives.)"
                )
                st.stop()

            expert_cfg = bool(st.session_state.get("cfg_expert_mode", False))
            append_log = bool(st.session_state.get("sb_append_log", False))

            try:
                with pipeline_status("Running Moju audit pipeline…") as pstat:
                    status_update(pstat, "Building MonitorConfig…")
                    if expert_cfg:
                        frag_d = parse_monitor_config_json(st.session_state.get("config_fragment_raw") or "{}")
                    else:
                        pred_keys = set((pred or {}).keys())
                        cdict_run = st.session_state.get("constants_dict") or {}
                        const_keys = set(cdict_run.keys())
                        laws_sel = list(st.session_state.get("st_auto_laws") or [])
                        models_sel = list(st.session_state.get("st_auto_models") or [])
                        groups_sel = list(st.session_state.get("st_auto_groups") or [])
                        try:
                            frag_d = build_studio_auto_fragment(
                                law_names=laws_sel,
                                model_names=models_sel,
                                group_names=groups_sel,
                                pred_keys=pred_keys,
                                constant_keys=const_keys,
                            )
                        except ValueError as e:
                            st.error(str(e))
                            st.stop()
                        override = st.session_state.get("sf_json_override") or "{}"
                        frag_d = merge_simple_config_with_json_override(frag_d, override)

                    frag_d = merge_monitor_config_fragment(
                        frag_d, {"constants": st.session_state.get("constants_dict") or {}}
                    )
                    frag_d = enrich_fragment_from_model_audits(frag_d)
                    sb = None
                    if not path_b:
                        custom_sb = st.session_state.get("studio_recomputing_state_builder")
                        sb = (
                            custom_sb
                            if custom_sb is not None
                            else make_session_state_builder(pred)
                        )
                    try:
                        validate_studio_pi_gating(
                            use_path_b=path_b,
                            scaling_audit_specs=[],
                            state_builder=sb,
                        )
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                    cfg = monitor_config_from_merged_dict(frag_d, state_builder=sb)

                    fd_arg: Any = False
                    plan_grid_for_plan = PathBGridConfig()
                    if auto_fd:
                        fd_arg = path_b_grid_from_options(**grid_kw)
                        plan_grid_for_plan = fd_arg
                    pred_keys_run = set((pred or {}).keys())
                    dep_plan = dependency_plan_for_path_b_run(
                        cfg,
                        pred_keys_run,
                        auto_path_b_derivatives=bool(auto_fd),
                        fill_law_fd=bool(fill_law),
                        path_b_grid=plan_grid_for_plan,
                    )

                    status_update(pstat, "Computing residuals…")
                    t0 = time.perf_counter()
                    engine = ResidualEngine(config=cfg)
                    ref = st.session_state.get("state_ref")
                    col = st.session_state.get("collocation") or {}

                    if path_b and pred is not None and dep_plan.has_blocking_gaps():
                        st.warning(
                            "**Preflight (dependency planner)**\n\n"
                            + format_planner_preflight_warning(dep_plan)
                        )
                        with st.expander("Dependency detail", expanded=False):
                            st.markdown(dep_plan.to_markdown())

                    run_mode = "eval" if ref is not None else "training"
                    if path_b:
                        residuals = engine.compute_residuals(
                            pred,
                            ref,
                            auto_path_b_derivatives=fd_arg,
                            fill_law_fd=fill_law,
                            run_mode=run_mode,
                        )
                    else:
                        residuals = engine.compute_residuals(
                            None,
                            model=0,
                            params=0,
                            collocation=col,
                            auto_path_b_derivatives=fd_arg,
                            fill_law_fd=fill_law,
                            run_mode=run_mode,
                        )

                    elapsed = time.perf_counter() - t0

                    prev_log = list(st.session_state.get("viz_log") or [])
                    if append_log and prev_log:
                        viz_log = reindex_log_entries(prev_log, engine.log)
                    else:
                        viz_log = list(engine.log)
                    st.session_state["viz_log"] = viz_log

                    status_update(pstat, "Running audit()…")
                    rep = audit(
                        viz_log,
                        r_ref=r_ref or None,
                        weights=weights or None,
                    )

                    st.session_state["last_engine"] = engine
                    st.session_state["last_residuals"] = residuals
                    st.session_state["last_report"] = rep
                    st.session_state["last_cfg"] = cfg
                    st.session_state["last_path_b"] = path_b
                    st.session_state["run_elapsed_s"] = elapsed
                    st.session_state["last_r_ref"] = r_ref or {}
                    st.session_state["last_weights"] = weights or {}
                    st.session_state["last_max_leg"] = int(
                        st.session_state.get("dashboard_max_leg", 16)
                    )
                    rms_keys = sorted((engine.log[-1].get("rms") or {}).keys())
                    st.session_state["viz_rms_keys"] = rms_keys

                    req_s = sorted(engine.required_state_keys())
                    req_d: list[str] = []
                    pred_k_list = list((pred or {}).keys())
                    chk = preflight_checklist_with_dependency_plan(
                        req_s,
                        req_d,
                        pred_k_list,
                        dep_plan.to_markdown(),
                        available_keys=sorted(dep_plan.effective_available_keys),
                    )
                    st.session_state["last_preflight_text"] = chk
                    st.session_state["last_omitted"] = engine.log[-1].get("omitted") or []
                    st.session_state["last_inferred"] = engine.log[-1].get("inferred") or []
                    st.session_state["last_preflight_planner_blocking"] = dep_plan.has_blocking_gaps()
                    st.session_state["last_preflight_planner_summary"] = (
                        format_planner_preflight_warning(dep_plan)
                        if dep_plan.has_blocking_gaps()
                        else ""
                    )
                    st.session_state["last_preflight_derivable_law_fd"] = list(
                        dep_plan.derivable_law_fd_if_enabled
                    )
                    st.session_state["last_preflight_chk"] = chk
                    st.session_state["last_dep_plan_blocking"] = dep_plan.has_blocking_gaps()
                    st.session_state["last_dep_plan_md"] = dep_plan.to_markdown()

                    status_complete(pstat, f"Done in {elapsed:.3f}s — log steps: {len(viz_log)}")

                toast(f"Audit finished in {elapsed:.2f}s", icon="✅")

            except Exception as ex:  # noqa: BLE001
                st.exception(ex)

    if st.session_state.get("last_report"):
        st.caption("Latest run: open the **Dashboard** tab for charts, preflight download, and tables.")

with tab_dash:
    rep = st.session_state.get("last_report")
    if rep:
        path_b_hdr = bool(st.session_state.get("last_path_b", True))
        dash_title = (
            "Physics Admissibility Audit (model evaluation) (Physics Residuals)"
            if path_b_hdr
            else "Model Audit (Physics Residuals)"
        )
        st.markdown(f"### {dash_title}")
        ov_raw = rep.get("overall_admissibility_score")
        try:
            ov_hdr = float(ov_raw) if ov_raw is not None else float("nan")
        except (TypeError, ValueError):
            ov_hdr = float("nan")
        _rm_rep = rep.get("monitor_run_mode")
        if _rm_rep == "eval" or not math.isfinite(ov_hdr):
            st.caption(
                "No single overall score in eval mode; see **Admissibility** per-category below."
            )
        else:
            pct_hdr = format_admissibility_pct(ov_hdr) if math.isfinite(ov_hdr) else "N/A"
            st.caption(
                f"Overall admissibility (final): {pct_hdr} - {format_admissibility_status_label(ov_hdr)}"
            )
        _law_panel_main, _rnorm_panel_main, _hm_cs_main = _studio_monitor_spatial_bundle()
        _pref_t_dash = st.session_state.get("studio_fd_time_label", "Steady") != "Steady"
        _sax_dash = str(st.session_state.get("sb_spatial_axis", "x"))
        bundle = build_monitor_visualize_bundle(
            st.session_state.get("viz_log") or [],
            keys=None,
            r_ref=st.session_state.get("last_r_ref") or None,
            max_legend_keys=int(st.session_state.get("last_max_leg", 16)),
            spatial_law_panel=_law_panel_main,
            spatial_rnorm_panel=_rnorm_panel_main,
            mode="training",
            residuals=st.session_state.get("last_residuals"),
            state_pred=st.session_state.get("state_pred"),
            spatial_coord_key=_sax_dash,
            spatial_prefer_last_t=_pref_t_dash,
            engine=st.session_state.get("last_engine"),
        )
        if bundle:
            bc1, bc2 = st.columns(2)
            with bc1:
                with st.container(border=True):
                    try:
                        _plotly_dashboard_card(
                            build_plotly_law_rnorm_final_bar_figure(bundle),
                            "dash_card_law_rnorm",
                        )
                    except Exception as ex:  # noqa: BLE001
                        st.warning(str(ex))
            with bc2:
                with st.container(border=True):
                    try:
                        _plotly_dashboard_card(
                            build_plotly_category_admissibility_bar_figure(bundle),
                            "dash_card_cat_adm",
                        )
                    except Exception as ex:  # noqa: BLE001
                        st.warning(str(ex))
            hc1, hc2 = st.columns(2)
            with hc1:
                with st.container(border=True):
                    try:
                        _plotly_dashboard_card(
                            build_plotly_spatial_rnorm_heatmap_card(
                                bundle.get("spatial"),
                                colorscale=_hm_cs_main,
                                card_title="Law |residual| (spatial)",
                            ),
                            "dash_card_law_hm",
                        )
                    except Exception as ex:  # noqa: BLE001
                        st.warning(str(ex))
            with hc2:
                with st.container(border=True):
                    try:
                        _plotly_dashboard_card(
                            build_plotly_spatial_rnorm_heatmap_card(
                                bundle.get("spatial_rnorm"),
                                colorscale=_hm_cs_main,
                                card_title="Constitutive |residual| (spatial)",
                            ),
                            "dash_card_const_hm",
                        )
                    except Exception as ex:  # noqa: BLE001
                        st.warning(str(ex))
        else:
            st.caption("No visualization bundle (empty session log).")
    else:
        st.info("Run **compute_residuals + audit** on the **Run** tab to load dashboard charts.")

    if rep:
        st.divider()
        st.caption(f"Last run wall time: {st.session_state.get('run_elapsed_s', 0):.3f}s")
        engine = st.session_state.get("last_engine")

        chk = st.session_state.get("last_preflight_chk", "")
        st.download_button(
            "Download preflight checklist (.txt)",
            data=chk,
            file_name="preflight_checklist.txt",
            mime="text/plain",
            key="dl_preflight_persistent",
        )
        dep_blk = bool(st.session_state.get("last_dep_plan_blocking"))
        plan_summary = st.session_state.get("last_preflight_planner_summary") or ""
        if dep_blk and plan_summary:
            st.warning(
                "**Preflight (dependency planner)**\n\n"
                + plan_summary
                + " Closures may be omitted — check log `omitted` / `inferred`."
            )
        elif dep_blk:
            st.warning(
                "Preflight: dependency planner reports blocking gaps — open **Required keys detail** "
                "or **Last run — dependency planner**. Closures may be omitted — check log `omitted` / `inferred`."
            )
        with st.expander("Required keys detail"):
            st.text(chk)
        dep_md = st.session_state.get("last_dep_plan_md")
        if dep_md:
            with st.expander("Last run — dependency planner", expanded=False):
                st.markdown(dep_md)

        om = st.session_state.get("last_omitted") or []
        inf = st.session_state.get("last_inferred") or []
        if om:
            st.info("Omitted: " + "; ".join(om[:12]))
        if inf:
            laplace_hints = [s for s in inf if "laplacian" in s.lower() or "law_fd" in s.lower()]
            if laplace_hints:
                st.warning(
                    "**Laplacian / law-FD:** some fills may have failed — check **Path B FD messages** below "
                    f"({len(laplace_hints)} related line(s))."
                )
            st.info("Inferred (first 12): " + "; ".join(inf[:12]))
            with st.expander("Path B FD messages (full log)", expanded=bool(laplace_hints)):
                st.caption(
                    "From `compute_residuals` (finite differences + law recipe fill). "
                    "Typical Laplacian fixes: match **sidebar** 1D/2D/3D to data, use **Meshgrid** unless data is separable, "
                    "align `T`/`x` shapes or sizes."
                )
                st.markdown("\n".join(f"- `{s}`" for s in inf))

        st.subheader("Admissibility")
        ov = rep.get("overall_admissibility_score")
        pc_raw = rep.get("per_category") or {}
        pc_fmt: Dict[str, str] = {}
        for k, v in pc_raw.items():
            try:
                pc_fmt[str(k)] = format_admissibility_pct(float(v))
            except (TypeError, ValueError):
                pc_fmt[str(k)] = "N/A"
        try:
            ov_f = float(ov) if ov is not None else float("nan")
        except (TypeError, ValueError):
            ov_f = float("nan")
        adm_payload: Dict[str, Any] = {"per_category": pc_fmt}
        if rep.get("monitor_run_mode") == "eval" or not math.isfinite(ov_f):
            adm_payload["overall"] = "N/A"
        else:
            adm_payload["overall"] = format_admissibility_pct(ov_f)
        st.json(adm_payload)

        st.subheader("RMS (last step)")
        rms = (engine.log[-1].get("rms") if engine else {}) or {}
        rms_rows = [{"key": k, "rms": v} for k, v in sorted(rms.items())]
        try:
            st.dataframe(
                rms_rows,
                use_container_width=True,
                height=min(400, 40 + 28 * len(rms)),
                column_config={
                    "key": st.column_config.TextColumn("Residual key", width="large"),
                    "rms": st.column_config.NumberColumn("RMS", format="%.6e", help="Root mean square"),
                },
                hide_index=True,
            )
        except Exception:  # noqa: BLE001
            st.dataframe(
                rms_rows,
                use_container_width=True,
                height=min(400, 40 + 28 * len(rms)),
            )

    with st.container(border=True):
        st.subheader("Audit / visualization options (next run)")
        st.text_area(
            "Optional r_ref (JSON object: residual_key → float scale)",
            height=60,
            key="dashboard_r_ref_json",
        )
        st.text_area(
            "Optional audit weights (JSON object: key → float)",
            height=60,
            key="dashboard_weights_json",
        )
        st.number_input(
            "visualize max_legend_keys",
            min_value=1,
            max_value=64,
            value=16,
            key="dashboard_max_leg",
        )

    studio_redraw_plotly_fragment()

with tab_divergence:
    st.subheader("Where the model and the law-implied closure diverge")
    st.caption(
        "Built from the engine's `closure_debug` sidecar (raw `pred` / `implied` / balance terms). "
        "Auto-selects the constitutive basename with the worst final-step R_norm; pick another from the dropdown to drill in."
    )
    res_div = st.session_state.get("last_residuals")
    if not res_div or not isinstance(res_div, dict) or not res_div.get("closure_debug"):
        st.info("No `closure_debug` sidecar yet — run an audit with at least one constitutive_audit row.")
    else:
        from moju.monitor.visualize_constitutive import (
            build_constitutive_divergence_card,
            build_constitutive_divergence_dashboard,
            list_constitutive_basenames,
        )

        _div_log = st.session_state.get("viz_log") or []
        _div_bundle = build_monitor_visualize_bundle(
            _div_log,
            keys=None,
            r_ref=st.session_state.get("last_r_ref") or None,
            max_legend_keys=int(st.session_state.get("last_max_leg", 16)),
            mode="training",
            residuals=res_div,
            state_pred=st.session_state.get("state_pred"),
            engine=st.session_state.get("last_engine"),
        )
        if not _div_bundle:
            st.warning("Could not build visualization bundle.")
        else:
            _div_basenames = list_constitutive_basenames(_div_bundle)
            if not _div_basenames:
                st.info("No constitutive divergence data available.")
            else:
                _div_choice = st.selectbox(
                    "Constitutive basename",
                    options=_div_basenames,
                    index=0,
                    key="divergence_basename",
                    help="Sorted by worst final-step R_norm first.",
                )
                _div_mode = st.radio(
                    "View",
                    options=["dashboard (2×2)", "spatial", "scatter", "distribution", "hotspot"],
                    horizontal=True,
                    key="divergence_mode",
                )
                try:
                    if _div_mode.startswith("dashboard"):
                        fig_div = build_constitutive_divergence_dashboard(
                            _div_bundle, residual_basename=_div_choice
                        )
                    else:
                        fig_div = build_constitutive_divergence_card(
                            _div_bundle, residual_basename=_div_choice, mode=_div_mode
                        )
                    _plotly_dashboard_card(fig_div, "divergence_card")
                except Exception as ex:  # noqa: BLE001
                    st.warning(str(ex))


with tab_space:
    st.caption("Sliders and plot updates are isolated in a **fragment** so the rest of the app may not rerun.")
    studio_spatial_fragment()

with tab_export:
    rep = st.session_state.get("last_report")
    cfg = st.session_state.get("last_cfg")
    path_b = st.session_state.get("last_path_b", True)
    residuals = st.session_state.get("last_residuals")
    if rep is None:
        st.info("Run an audit on **Run** first; summaries and charts are on **Dashboard**.")
    else:
        j = audit_report_to_jsonable(rep)
        st.download_button(
            "Download audit_report.json",
            data=json.dumps(j, indent=2),
            file_name="audit_report.json",
            mime="application/json",
        )
        if cfg is not None:
            cfg_d = cfg.to_dict()
            st.download_button(
                "Download monitor_config.json",
                data=json.dumps(cfg_d, indent=2, default=str),
                file_name="monitor_config.json",
                mime="application/json",
            )
            st.code(generate_python_snippet(cfg, path_b=path_b), language="python")

        st.subheader("PDF report (optional)")
        st.caption("Requires `pip install moju[report]` (reportlab). Produces a ZIP with PDF (+ optional residuals JSON).")
        mn = st.text_input("model_name (PDF metadata)", value="")
        mid = st.text_input("model_id (PDF metadata)", value="")
        if st.button("Generate PDF bundle"):
            try:
                flat_r = flatten_residuals(residuals) if residuals else {}
                td = tempfile.mkdtemp()
                lw = st.session_state.get("last_weights") or {}
                audit(
                    st.session_state.get("viz_log") or [],
                    r_ref=st.session_state.get("last_r_ref") or None,
                    weights=lw if lw else None,
                    export_dir=td,
                    save_residuals=True,
                    last_residual_dict=flat_r,
                    model_name=mn or None,
                    model_id=mid or None,
                )
                from pathlib import Path as P

                zips = sorted(P(td).glob("*.zip"))
                if zips:
                    data = zips[0].read_bytes()
                    st.session_state["pdf_zip_bytes"] = data
                    st.session_state["pdf_zip_name"] = zips[0].name
                    st.success("Bundle created.")
                    toast("PDF bundle ready to download", icon="📄")
                else:
                    st.error("ZIP not found after export.")
            except ImportError as ie:
                st.warning(str(ie))
            except Exception as ex:  # noqa: BLE001
                st.exception(ex)

        zb = st.session_state.get("pdf_zip_bytes")
        zn = st.session_state.get("pdf_zip_name", "audit_bundle.zip")
        if zb:
            st.download_button("Download audit ZIP", data=zb, file_name=zn, mime="application/zip")
