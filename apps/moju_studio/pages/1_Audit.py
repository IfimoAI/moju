"""
Upload state, configure audits via forms or JSON, run residuals + audit + visualize.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import streamlit as st

from apps.moju_studio.studio_streamlit_extras import (
    as_fragment,
    cached_registry_names,
    pipeline_status,
    status_complete,
    status_update,
    toast,
)
from apps.moju_studio.config_forms import (
    build_audit_spec_dict,
    build_group_spec,
    build_law_spec,
    group_parameter_names,
    law_parameter_names,
    merge_simple_config_with_json_override,
    model_parameter_names,
    path_b_grid_from_options,
    preflight_checklist_text,
    reindex_log_entries,
    scaling_fn_parameter_names,
)
from apps.moju_studio.studio_core import (
    audit_report_to_jsonable,
    flatten_residuals,
    generate_python_snippet,
    list_registered_law_names,
    make_session_state_builder,
    monitor_config_from_merged_dict,
    preflight_engine,
    validate_studio_pi_gating,
)
from apps.moju_studio.studio_io import (
    constants_json_to_dict,
    load_state_bundle_bytes,
    merge_monitor_config_fragment,
    parse_monitor_config_json,
    validate_non_empty_state,
)

_STATE_UPLOAD_TYPES = ["npz", "npy", "h5", "hdf5", "nc", "nc4"]
from apps.moju_studio.studio_plots import plotly_pred_minus_ref, plotly_residual_or_state
from moju.monitor import ResidualEngine, audit, visualize
from moju.monitor.pi_constant_recipes import list_pi_constant_group_names

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
    st.markdown("### Moju Studio")
    try:
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/1_Audit.py", label="Audit", icon="🔬")
        st.page_link("pages/2_Quick_start.py", label="Quick start", icon="📘")
        st.page_link("pages/3_Help.py", label="Help and UX", icon="❓")
    except Exception:  # noqa: BLE001
        st.caption("Use the app **pages** menu to navigate.")
    st.divider()
    st.checkbox(
        "Append next run to session log",
        key="sb_append_log",
        help="Chain multiple runs into one Plotly timeline (same browser session).",
    )
    if st.button("Clear session log…"):
        _dialog_clear_viz_log()


DEFAULT_CONFIG_FRAGMENT = """{
  "laws": [],
  "groups": [],
  "constitutive_audit": [],
  "scaling_audit": []
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
  "constitutive_audit": [],
  "scaling_audit": []
}
"""

DEMO_CONSTANTS = """{}
"""

DEFAULT_PRIMARY = ["T", "u", "v", "w", "p", "rho"]


def _apply_pi_c_to_scaling_audit_dict(d: Dict[str, Any], c: float) -> Dict[str, Any]:
    out = {**d}
    audits = []
    for spec in d.get("scaling_audit") or []:
        spec = dict(spec)
        if spec.get("invariance_pi_constant"):
            spec["invariance_scale_c"] = float(c)
        audits.append(spec)
    out["scaling_audit"] = audits
    return out


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


def _key_options(pred: Dict[str, Any], constants: Dict[str, Any]) -> List[str]:
    return sorted(set(pred.keys()) | set(constants.keys()))


def _collect_simple_fragment() -> Dict[str, Any]:
    """Read form widgets from st.session_state (must match keys used in Config tab)."""
    laws: List[Dict[str, Any]] = []
    n_laws = int(st.session_state.get("sf_n_laws", 0))
    for i in range(n_laws):
        nm = st.session_state.get(f"sf_law_name_{i}")
        if not nm:
            continue
        try:
            args = law_parameter_names(str(nm))
        except Exception:  # noqa: BLE001
            continue
        sm: Dict[str, str] = {}
        ok = True
        for a in args:
            v = st.session_state.get(f"sf_law_{i}_{a}")
            if v is None:
                ok = False
                break
            sm[a] = str(v)
        if ok and sm:
            laws.append(build_law_spec(str(nm), sm))

    groups: List[Dict[str, Any]] = []
    n_grp = int(st.session_state.get("sf_n_groups", 0))
    for i in range(n_grp):
        nm = st.session_state.get(f"sf_grp_name_{i}")
        out_k = st.session_state.get(f"sf_grp_out_{i}")
        if not nm or not out_k:
            continue
        try:
            args = group_parameter_names(str(nm))
        except Exception:  # noqa: BLE001
            continue
        sm = {}
        ok = True
        for a in args:
            v = st.session_state.get(f"sf_grp_{i}_{a}")
            if v is None:
                ok = False
                break
            sm[a] = str(v)
        if ok and sm:
            groups.append(build_group_spec(str(nm), str(out_k), sm))

    ca: List[Dict[str, Any]] = []
    n_c = int(st.session_state.get("sf_n_caudit", 0))
    for i in range(n_c):
        nm = st.session_state.get(f"sf_ca_name_{i}")
        out_k = st.session_state.get(f"sf_ca_out_{i}")
        if not nm or not out_k:
            continue
        try:
            args = model_parameter_names(str(nm))
        except Exception:  # noqa: BLE001
            continue
        sm = {}
        ok = True
        for a in args:
            v = st.session_state.get(f"sf_ca_{i}_{a}")
            if v is None:
                ok = False
                break
            sm[a] = str(v)
        if not (ok and sm):
            continue
        ps = st.session_state.get(f"sf_ca_ps_{i}") or []
        pt = st.session_state.get(f"sf_ca_pt_{i}") or []
        if isinstance(ps, str):
            ps = [ps] if ps else []
        if isinstance(pt, str):
            pt = [pt] if pt else []
        cm = st.session_state.get(f"sf_ca_cm_{i}") or "pointwise"
        axes = st.session_state.get(f"sf_ca_axes_{i}") or ["x"]
        if isinstance(axes, str):
            axes = [axes]
        ivk = st.session_state.get(f"sf_ca_ivk_{i}") or ""
        ivk = str(ivk).strip() or None
        qw_raw = (st.session_state.get(f"sf_ca_qw_{i}") or "").strip()
        qw: Dict[str, str] = {}
        if qw_raw:
            qobj = json.loads(qw_raw)
            if isinstance(qobj, dict):
                qw = {str(k): str(v) for k, v in qobj.items()}
        ca.append(
            build_audit_spec_dict(
                category="constitutive",
                name=str(nm),
                output_key=str(out_k),
                state_map=sm,
                predicted_spatial=list(ps),
                predicted_temporal=list(pt),
                closure_mode=str(cm),
                quadrature_weights=qw,
                chain_spatial_axes=list(axes),
                implied_value_key=ivk,
            )
        )

    sa: List[Dict[str, Any]] = []
    n_s = int(st.session_state.get("sf_n_saudit", 0))
    pi_names = set(list_pi_constant_group_names())
    for i in range(n_s):
        nm = st.session_state.get(f"sf_sa_name_{i}")
        out_k = st.session_state.get(f"sf_sa_out_{i}")
        if not nm or not out_k:
            continue
        try:
            args = scaling_fn_parameter_names(str(nm))
        except Exception:  # noqa: BLE001
            continue
        sm = {}
        ok = True
        for a in args:
            v = st.session_state.get(f"sf_sa_{i}_{a}")
            if v is None:
                ok = False
                break
            sm[a] = str(v)
        if not (ok and sm):
            continue
        ps = st.session_state.get(f"sf_sa_ps_{i}") or []
        pt = st.session_state.get(f"sf_sa_pt_{i}") or []
        if isinstance(ps, str):
            ps = [ps] if ps else []
        if isinstance(pt, str):
            pt = [pt] if pt else []
        cm = st.session_state.get(f"sf_sa_cm_{i}") or "pointwise"
        axes = st.session_state.get(f"sf_sa_axes_{i}") or ["x"]
        if isinstance(axes, str):
            axes = [axes]
        ivk = st.session_state.get(f"sf_sa_ivk_{i}") or ""
        ivk = str(ivk).strip() or None
        qw_raw = (st.session_state.get(f"sf_sa_qw_{i}") or "").strip()
        qw = {}
        if qw_raw:
            qobj = json.loads(qw_raw)
            if isinstance(qobj, dict):
                qw = {str(k): str(v) for k, v in qobj.items()}
        use_pi = bool(st.session_state.get(f"sf_sa_pi_{i}", False)) and str(nm) in pi_names
        cmp_keys = st.session_state.get(f"sf_sa_cmp_{i}") or []
        if isinstance(cmp_keys, str):
            cmp_keys = [cmp_keys] if cmp_keys else []
        sa.append(
            build_audit_spec_dict(
                category="scaling",
                name=str(nm),
                output_key=str(out_k),
                state_map=sm,
                predicted_spatial=list(ps),
                predicted_temporal=list(pt),
                closure_mode=str(cm),
                quadrature_weights=qw,
                chain_spatial_axes=list(axes),
                implied_value_key=ivk,
                invariance_pi_constant=use_pi,
                invariance_compare_keys=list(cmp_keys) if use_pi else [],
                invariance_scale_c=float(st.session_state.get("sf_pi_c_global", 10.0)),
            )
        )

    pf = st.session_state.get("sf_primary_fields")
    if pf is None:
        pf = list(DEFAULT_PRIMARY)
    if isinstance(pf, str):
        pf = [pf]

    return {
        "laws": laws,
        "groups": groups,
        "constitutive_audit": ca,
        "scaling_audit": sa,
        "primary_fields": list(pf),
    }


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
                fig2 = visualize(
                    st.session_state.get("viz_log") or [],
                    keys=list(vk) if vk else None,
                    backend="plotly",
                    r_ref=st.session_state.get("last_r_ref") or None,
                    max_legend_keys=int(st.session_state.get("last_max_leg", 16)),
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


@as_fragment
def studio_spatial_fragment() -> None:
    st.subheader("Explore a residual or state array")
    res = st.session_state.get("last_residuals")
    pred = st.session_state.get("state_pred")
    ref = st.session_state.get("state_ref")
    if not res and not pred:
        st.info("Run an audit on the **Run** tab first.")
        return
    flat = flatten_residuals(res) if res else {}
    keys = sorted(set(flat.keys()) | set((pred or {}).keys()))
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
                if len(shape) >= 1 and shape[0] > 1:
                    t_idx = st.slider("Time / leading-axis index", 0, shape[0] - 1, 0, key="sp_t2")
                fig = plotly_pred_minus_ref(
                    pa,
                    ra,
                    title=choice,
                    time_index=t_idx,
                    time_axis=0,
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
            if len(shape) >= 1 and shape[0] > 1:
                t_idx = st.slider("Time / leading-axis index", 0, shape[0] - 1, 0, key="sp_t1")
            fig = plotly_residual_or_state(arr, title=choice, time_index=t_idx, time_axis=0)
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


st.title("Audit workspace")
tab_data, tab_cfg, tab_run, tab_space, tab_export = st.tabs(
    ["Data", "Config", "Run", "Spatial / time", "Export"]
)

with tab_data:
    st.subheader("State prediction (`state_pred`)")
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
    law_names_t, models_t, gnames_t = cached_registry_names()
    law_names, models, gnames = list(law_names_t), list(models_t), list(gnames_t)

    use_simple = st.checkbox(
        "Use simple (form) builder for laws / groups / audits",
        value=True,
        key="cfg_use_simple",
    )

    pred = st.session_state.get("state_pred") or {}
    cdict = st.session_state.get("constants_dict") or {}
    opts = _key_options(pred, {k: None for k in cdict.keys()})

    st.subheader("Constants")
    cjson = st.text_area("Constants JSON (merged into config)", value=DEMO_CONSTANTS, height=100)
    try:
        st.session_state["constants_dict"] = constants_json_to_dict(cjson)
        cdict = st.session_state["constants_dict"]
        opts = _key_options(pred, {k: None for k in cdict.keys()})
    except (json.JSONDecodeError, ValueError) as e:
        st.error(str(e))
        st.session_state["constants_dict"] = {}

    opts_pf = sorted(set(DEFAULT_PRIMARY) | set(opts))
    default_pf = [x for x in DEFAULT_PRIMARY if x in opts_pf] or opts_pf[: min(6, len(opts_pf))]
    st.multiselect(
        "primary_fields (monitor inference hints)",
        options=opts_pf,
        default=default_pf,
        key="sf_primary_fields",
    )

    st.subheader("π-constant scale `c`")
    st.caption("Applied to every scaling audit with π enabled (Path A only).")
    st.slider(
        "invariance_scale_c",
        min_value=1.01,
        max_value=100.0,
        value=10.0,
        step=0.01,
        key="sf_pi_c_global",
    )
    pi_c = float(st.session_state.get("sf_pi_c_global", 10.0))

    if use_simple:
        st.subheader("Laws")
        st.number_input("Number of laws", 0, 8, 0, step=1, key="sf_n_laws")
        n_laws = int(st.session_state.get("sf_n_laws", 0))
        for i in range(n_laws):
            st.markdown(f"**Law {i + 1}**")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.selectbox("name", law_names, key=f"sf_law_name_{i}")
            nm = st.session_state.get(f"sf_law_name_{i}", law_names[0] if law_names else "")
            if nm and opts:
                for a in law_parameter_names(str(nm)):
                    st.selectbox(f"state key for `{a}`", opts, key=f"sf_law_{i}_{a}")

        st.subheader("Groups (dimensionless helpers)")
        st.number_input("Number of group specs", 0, 8, 0, step=1, key="sf_n_groups")
        n_grp = int(st.session_state.get("sf_n_groups", 0))
        for i in range(n_grp):
            st.markdown(f"**Group {i + 1}**")
            st.selectbox("Groups.* name", gnames, key=f"sf_grp_name_{i}")
            st.text_input("output_key (state field name)", value="Re", key=f"sf_grp_out_{i}")
            nm = st.session_state.get(f"sf_grp_name_{i}", gnames[0] if gnames else "")
            if nm and opts:
                for a in group_parameter_names(str(nm)):
                    st.selectbox(f"state key for `{a}`", opts, key=f"sf_grp_{i}_{a}")

        st.subheader("Constitutive audits")
        st.number_input("Number of constitutive audits", 0, 8, 0, step=1, key="sf_n_caudit")
        n_c = int(st.session_state.get("sf_n_caudit", 0))
        for i in range(n_c):
            st.markdown(f"**Constitutive audit {i + 1}**")
            st.selectbox("Models.* name", models, key=f"sf_ca_name_{i}")
            st.text_input("output_key", value="out", key=f"sf_ca_out_{i}")
            nm = st.session_state.get(f"sf_ca_name_{i}", models[0] if models else "")
            if nm and opts:
                for a in model_parameter_names(str(nm)):
                    st.selectbox(f"state key for `{a}`", opts, key=f"sf_ca_{i}_{a}")
            st.multiselect("predicted_spatial (state keys)", opts, key=f"sf_ca_ps_{i}")
            st.multiselect("predicted_temporal (state keys)", opts, key=f"sf_ca_pt_{i}")
            st.radio("closure_mode", ["pointwise", "weak"], horizontal=True, key=f"sf_ca_cm_{i}")
            st.multiselect("chain_spatial_axes", ["x", "y", "z"], default=["x"], key=f"sf_ca_axes_{i}")
            st.text_input("implied_value_key (optional)", value="", key=f"sf_ca_ivk_{i}")
            st.text_input('quadrature_weights JSON e.g. {"x":"w_x"}', value="", key=f"sf_ca_qw_{i}")

        st.subheader("Scaling audits")
        st.number_input("Number of scaling audits", 0, 8, 0, step=1, key="sf_n_saudit")
        n_s = int(st.session_state.get("sf_n_saudit", 0))
        for i in range(n_s):
            st.markdown(f"**Scaling audit {i + 1}**")
            st.selectbox("Groups.* name", gnames, key=f"sf_sa_name_{i}")
            st.text_input("output_key", value="Re", key=f"sf_sa_out_{i}")
            nm = st.session_state.get(f"sf_sa_name_{i}", gnames[0] if gnames else "")
            if nm and opts:
                for a in scaling_fn_parameter_names(str(nm)):
                    st.selectbox(f"state key for `{a}`", opts, key=f"sf_sa_{i}_{a}")
            st.multiselect("predicted_spatial", opts, key=f"sf_sa_ps_{i}")
            st.multiselect("predicted_temporal", opts, key=f"sf_sa_pt_{i}")
            st.radio("closure_mode", ["pointwise", "weak"], horizontal=True, key=f"sf_sa_cm_{i}", index=0)
            st.multiselect("chain_spatial_axes", ["x", "y", "z"], default=["x"], key=f"sf_sa_axes_{i}")
            st.text_input("implied_value_key (optional)", value="", key=f"sf_sa_ivk_{i}")
            st.text_input('quadrature_weights JSON', value="", key=f"sf_sa_qw_{i}")
            pi_ok = str(nm) in set(list_pi_constant_group_names())
            st.checkbox(
                "Enable π-constant invariance (Path A only; needs recipe + compare keys)",
                value=False,
                key=f"sf_sa_pi_{i}",
                disabled=not pi_ok,
            )
            if pi_ok:
                st.multiselect(
                    "invariance_compare_keys",
                    opts,
                    key=f"sf_sa_cmp_{i}",
                )

        st.subheader("Optional JSON override")
        st.caption("Non-empty lists in this JSON replace the corresponding form section.")
        st.text_area(
            "Override JSON (laws, groups, constitutive_audit, scaling_audit, primary_fields, constants)",
            value="{}",
            height=120,
            key="sf_json_override",
        )
    else:
        preset = st.selectbox(
            "Config template",
            ["Custom only", "Empty audits", "Demo: Laplace law (needs phi_laplacian in NPZ)"],
        )
        base_json = DEFAULT_CONFIG_FRAGMENT
        if preset == "Empty audits":
            base_json = DEFAULT_CONFIG_FRAGMENT
        elif preset == "Demo: Laplace law (needs phi_laplacian in NPZ)":
            base_json = DEMO_LAPLACE
        st.text_area(
            "MonitorConfig fragment (JSON)",
            value=base_json,
            height=220,
            key="config_fragment_raw",
        )

    with st.expander("Registry hints"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_area("Laws", "\n".join(law_names_t[:50]), height=160, disabled=True)
        with c2:
            st.text_area("Models", "\n".join(models_t[:50]), height=160, disabled=True)
        with c3:
            st.text_area("Groups", "\n".join(gnames_t[:50]), height=160, disabled=True)

with tab_run:
    st.caption(
        "Use the **sidebar** for session log append/clear. Submit the form below to run the pipeline "
        "(single batch update)."
    )
    with st.form("audit_run_form"):
        path_mode = st.radio(
            "Execution path",
            [
                "Path B — pass uploaded `state_pred`",
                "Path A — shim (default NPZ `state_builder`; π-constant needs a recomputing builder)",
            ],
            horizontal=True,
        )
        path_b = path_mode.startswith("Path B")
        cfd1, cfd2 = st.columns(2)
        with cfd1:
            auto_fd = st.checkbox("auto_path_b_derivatives", value=False)
        with cfd2:
            fill_law = st.checkbox("fill_law_fd (needs auto_path_b_derivatives)", value=False)
        use_custom_grid = st.checkbox("Customize PathBGridConfig", value=False)
        grid_kw: Dict[str, Any] = {}
        if auto_fd and use_custom_grid:
            st.caption("Path B grid")
            grid_kw["layout"] = st.selectbox("layout", ["meshgrid", "separable"], index=0)
            sd = st.selectbox("spatial_dimension", ["auto", "1", "2", "3"], index=0)
            grid_kw["spatial_dimension"] = "auto" if sd == "auto" else int(sd)
            grid_kw["steady"] = st.checkbox("steady", value=True)
            grid_kw["key_x"] = st.text_input("key_x", value="x")
            grid_kw["key_y"] = st.text_input("key_y", value="y")
            grid_kw["key_z"] = st.text_input("key_z", value="z")
            grid_kw["key_t"] = st.text_input("key_t", value="t")

        st.subheader("Audit / visualize options")
        r_ref_json = st.text_area(
            "Optional r_ref (JSON object: residual_key → float scale)", value="", height=60, key="form_r_ref"
        )
        weights_json = st.text_area(
            "Optional audit weights (JSON object: key → float)", value="", height=60, key="form_weights"
        )
        max_leg = st.number_input("visualize max_legend_keys", min_value=1, max_value=64, value=16)
        run_clicked = st.form_submit_button("Run compute_residuals + audit", type="primary")

    if run_clicked:
        pred = st.session_state.get("state_pred")
        ok, msg = validate_non_empty_state(pred or {})
        if not ok:
            st.error(msg)
        else:
            try:
                r_ref = _parse_float_dict(r_ref_json, "r_ref")
                weights = _parse_float_dict(weights_json, "weights")
            except (json.JSONDecodeError, ValueError) as e:
                st.error(str(e))
                st.stop()

            use_simple_cfg = bool(st.session_state.get("cfg_use_simple", True))
            pi_c_run = float(st.session_state.get("sf_pi_c_global", 10.0))
            append_log = bool(st.session_state.get("sb_append_log", False))

            try:
                with pipeline_status("Running Moju audit pipeline…") as pstat:
                    status_update(pstat, "Building MonitorConfig…")
                    if use_simple_cfg:
                        simple = _collect_simple_fragment()
                        override = st.session_state.get("sf_json_override") or "{}"
                        frag_d = merge_simple_config_with_json_override(simple, override)
                    else:
                        frag_d = parse_monitor_config_json(st.session_state.get("config_fragment_raw") or "{}")

                    frag_d = merge_monitor_config_fragment(
                        frag_d, {"constants": st.session_state.get("constants_dict") or {}}
                    )
                    frag_d = _apply_pi_c_to_scaling_audit_dict(frag_d, pi_c_run)
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
                            scaling_audit_specs=list(frag_d.get("scaling_audit") or []),
                            state_builder=sb,
                        )
                    except ValueError as e:
                        st.error(str(e))
                        st.stop()

                    cfg = monitor_config_from_merged_dict(frag_d, state_builder=sb)

                    fd_arg: Any = False
                    if auto_fd:
                        fd_arg = path_b_grid_from_options(**grid_kw) if use_custom_grid else True

                    status_update(pstat, "Computing residuals…")
                    t0 = time.perf_counter()
                    engine = ResidualEngine(config=cfg)
                    ref = st.session_state.get("state_ref")
                    col = st.session_state.get("collocation") or {}

                    if path_b:
                        residuals = engine.compute_residuals(
                            pred,
                            ref,
                            auto_path_b_derivatives=fd_arg,
                            fill_law_fd=fill_law,
                        )
                    else:
                        residuals = engine.compute_residuals(
                            None,
                            model=0,
                            params=0,
                            collocation=col,
                            auto_path_b_derivatives=fd_arg,
                            fill_law_fd=fill_law,
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
                    st.session_state["last_max_leg"] = int(max_leg)
                    rms_keys = sorted((engine.log[-1].get("rms") or {}).keys())
                    st.session_state["viz_rms_keys"] = rms_keys

                    miss_s, miss_d = preflight_engine(engine, set(pred.keys()))
                    req_s = sorted(engine.required_state_keys())
                    req_d = sorted(engine.required_derivative_keys())
                    chk = preflight_checklist_text(req_s, req_d, pred.keys())
                    st.session_state["last_preflight_text"] = chk
                    st.session_state["last_omitted"] = engine.log[-1].get("omitted") or []
                    st.session_state["last_inferred"] = engine.log[-1].get("inferred") or []
                    st.session_state["last_miss_s"] = miss_s
                    st.session_state["last_miss_d"] = miss_d
                    st.session_state["last_preflight_chk"] = chk

                    status_complete(pstat, f"Done in {elapsed:.3f}s — log steps: {len(viz_log)}")

                toast(f"Audit finished in {elapsed:.2f}s", icon="✅")

            except Exception as ex:  # noqa: BLE001
                st.exception(ex)

    if st.session_state.get("last_report"):
        st.divider()
        st.caption(f"Last run wall time: {st.session_state.get('run_elapsed_s', 0):.3f}s")
        rep = st.session_state["last_report"]
        engine = st.session_state.get("last_engine")
        pred = st.session_state.get("state_pred") or {}

        chk = st.session_state.get("last_preflight_chk", "")
        st.download_button(
            "Download preflight checklist (.txt)",
            data=chk,
            file_name="preflight_checklist.txt",
            mime="text/plain",
            key="dl_preflight_persistent",
        )
        miss_s = st.session_state.get("last_miss_s") or []
        miss_d = st.session_state.get("last_miss_d") or []
        if miss_s or miss_d:
            st.warning(
                "Preflight: missing keys vs engine requirements "
                f"(state: {miss_s or 'none'}; derivatives: {miss_d or 'none'}). "
                "Closures may be omitted — check log `omitted` / `inferred`."
            )
        with st.expander("Required keys detail"):
            st.text(chk)

        om = st.session_state.get("last_omitted") or []
        inf = st.session_state.get("last_inferred") or []
        if om:
            st.info("Omitted: " + "; ".join(om[:12]))
        if inf:
            st.info("Inferred: " + "; ".join(inf[:12]))

        st.subheader("Admissibility")
        st.json(
            {
                "overall": rep.get("overall_admissibility_score"),
                "per_category": rep.get("per_category"),
            }
        )

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

        st.subheader("Monitor dashboard (Plotly)")
        try:
            fig = visualize(
                st.session_state.get("viz_log") or [],
                keys=None,
                backend="plotly",
                r_ref=st.session_state.get("last_r_ref") or None,
                max_legend_keys=int(st.session_state.get("last_max_leg", 16)),
            )
            if fig is not None:
                try:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="plotly_main_dashboard",
                        on_select="rerun",
                        selection_mode="points",
                    )
                except TypeError:
                    st.plotly_chart(fig, use_container_width=True, key="plotly_main_dashboard")
            else:
                st.caption("Plotly visualize returned None (install plotly).")
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Plotly dashboard skipped: {ex}")

    studio_redraw_plotly_fragment()

with tab_space:
    st.caption("Sliders and plot updates are isolated in a **fragment** so the rest of the app may not rerun.")
    studio_spatial_fragment()

with tab_export:
    rep = st.session_state.get("last_report")
    cfg = st.session_state.get("last_cfg")
    path_b = st.session_state.get("last_path_b", True)
    residuals = st.session_state.get("last_residuals")
    if rep is None:
        st.info("Run an audit first.")
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
