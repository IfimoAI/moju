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

from apps.moju_studio.studio_streamlit_extras import studio_sidebar_branding_and_nav

st.set_page_config(page_title="Moju Studio — Quick start", layout="wide", page_icon="📘")

with st.sidebar:
    studio_sidebar_branding_and_nav()

st.title("Quick start")
st.markdown(
    """
### 1. Install (from repo root)

```bash
pip install -e ".[studio]"
```

PDF export is included in core `moju` (ReportLab); Studio adds Streamlit and HDF5/NetCDF upload support.

### 2. Run

```bash
streamlit run apps/moju_studio/Home.py
```

Theme and server options load from **`.streamlit/config.toml`** at the repository root when you start Streamlit from that root.

### 3. Minimal audit (Path B)

1. Open **Audit** in the sidebar.
2. **Data** tab: upload state as `.npz`, `.npy`, or (with `pip install -e ".[studio]"`) HDF5 / NetCDF — use keys that match Moju argument names (e.g. `phi_laplacian` for `laplace_equation`; include grid coords `x`, … for FD).
3. **Config** tab: set **Constants JSON** if needed; under **Laws** pick e.g. `laplace_equation`. Add **Models** / **Groups** if you want constitutive or scaling audits (each Group creates both a dimensionless helper and a scaling audit). Use **Expert** only if you need full JSON control.
4. **Run** tab: submit (**Path B**). **Finite differences** default **on**; turn off only if you supply all `d_*` derivatives yourself.  
   Results and Plotly dashboard stay visible after the run (not only on submit frame).
5. **Spatial / time** tab: explore arrays in an isolated **fragment** (sliders may not rerun the whole app).

### 4. Session log (multi-step)

Use the **sidebar**: check **Append next run to session log**, run again; **Clear session log** opens a confirmation **dialog**.

### 5. Where to go next

- **Help and UX** page — Streamlit features used in this app.
- Repository **README** and `moju.monitor` docstrings for `ResidualEngine`, `audit`, `visualize`.
"""
)
