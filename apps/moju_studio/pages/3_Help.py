"""
Help: Streamlit UX features used in Moju Studio.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="Moju Studio — Help", layout="wide", page_icon="❓")

st.title("Help and Streamlit UX")
st.markdown(
    """
This app uses several Streamlit patterns (requires **streamlit >= 1.33** in `moju[studio]`):

| Feature | Where it helps |
|--------|----------------|
| **`st.sidebar` + `st.page_link`** | Global navigation, session log append, clear-log entry point. |
| **`st.dialog`** | Confirm **Clear session log** without cluttering the main layout. |
| **`st.form` + `st.form_submit_button`** | Batch all **Run** options and submit once (fewer partial reruns). |
| **`st.status`** (with spinner fallback) | Long **JAX** runs show step labels: config, residuals, audit. |
| **`st.toast`** | Lightweight confirmations (upload, audit done, PDF ready). |
| **`@st.fragment`** | **Spatial / time** plots and **Redraw dashboard** update locally. |
| **`st.column_config`** | RMS table: formatted numeric column + wide text for residual keys. |
| **`on_select` on `st.plotly_chart`** | When supported, point/lasso selection triggers a targeted rerun. |
| **`.streamlit/config.toml`** (repo root) | Theme, `maxUploadSize`, `fastReruns`. |

### Tips

- If the main dashboard feels slow to interact with, use the **Spatial / time** tab first; it runs in a **fragment**.
- **Results** (admissibility, RMS, main Plotly figure) render whenever `last_report` exists, so they persist across reruns after a successful run.
- **π-constant** scale-invariance: requires **Path A** (not Path B), non-empty **`invariance_compare_keys`**, and a **`state_builder` that recomputes `state_pred` from `constants`** when those constants are scaled. The default **NPZ Path A shim** does *not* recompute tensors from `constants`, so Studio **blocks** π-constant runs in that mode. Advanced: assign a recomputing builder to `st.session_state["studio_recomputing_state_builder"]`, or use the Python API.

### Limitations

- `implied_fn` and custom Python closures are not exposed in the GUI (by design).
- Streamlit Cloud + JAX may need extra setup; local or VM is recommended.
"""
)

try:
    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/1_Audit.py", label="Audit", icon="🔬")
    st.page_link("pages/2_Quick_start.py", label="Quick start", icon="📘")
except Exception:  # noqa: BLE001
    pass
