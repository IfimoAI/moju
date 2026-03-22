"""
Quick start for Moju Studio.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="Moju Studio — Quick start", layout="wide", page_icon="📘")

st.title("Quick start")
st.markdown(
    """
### 1. Install (from repo root)

```bash
pip install -e ".[studio,viz]"
```

For PDF export: `pip install -e ".[studio,viz,report]"`.

### 2. Run

```bash
streamlit run apps/moju_studio/Home.py
```

Theme and server options load from **`.streamlit/config.toml`** at the repository root when you start Streamlit from that root.

### 3. Minimal audit (Path B)

1. Open **Audit** in the sidebar.
2. **Data** tab: upload state as `.npz`, `.npy`, or (with `pip install -e ".[studio-science]"`) HDF5 / NetCDF — keys must match your physics state (e.g. `phi_laplacian` for the Laplace demo).
3. **Config** tab: enable **simple builder**, set **Number of laws** to `1`, pick `laplace_equation`, map `phi_laplacian` → `phi_laplacian`.  
   Or disable simple mode and paste the demo Laplace JSON.
4. **Run** tab: submit the form (**Path B**, leave FD off unless you know your grid).  
   Results and Plotly dashboard stay visible after the run (not only on submit frame).
5. **Spatial / time** tab: explore arrays in an isolated **fragment** (sliders may not rerun the whole app).

### 4. Session log (multi-step)

Use the **sidebar**: check **Append next run to session log**, run again; **Clear session log** opens a confirmation **dialog**.

### 5. Where to go next

- **Help and UX** page — Streamlit features used in this app.
- Repository **README** and `moju.monitor` docstrings for `ResidualEngine`, `audit`, `visualize`.
"""
)

try:
    st.page_link("Home.py", label="Home", icon="🏠")
    st.page_link("pages/1_Audit.py", label="Audit", icon="🔬")
except Exception:  # noqa: BLE001
    pass
