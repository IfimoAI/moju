"""
Moju Studio — interactive audit explorer.

Run from the repository root::

    pip install -e ".[studio,viz]"
    streamlit run apps/moju_studio/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``streamlit run apps/moju_studio/Home.py`` with repo root on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from apps.moju_studio.studio_streamlit_extras import studio_sidebar_branding_and_nav

st.set_page_config(
    page_title="Moju Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    studio_sidebar_branding_and_nav()

st.markdown("# Moju Studio")
st.markdown(
    "Explore governing-law and constitutive residuals on your **state_pred** (Path B) or builder-based "
    "state (Path A), with an interactive **Plotly** dashboard and session log for multi-step runs."
)
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
**Setup**

- `pip install -e ".[studio,viz]"` from the **repository root**
- Streamlit **≥ 1.33**
- Run: `streamlit run apps/moju_studio/Home.py`
"""
    )
with c2:
    st.markdown(
        """
**Start here**

- Open **Audit** in the sidebar: upload data, configure laws/models, run Path A or B
- **Quick start** — minimal Path B workflow
- **Help** — Streamlit patterns used in this app
"""
    )
st.caption("Theme and server options: `.streamlit/config.toml` at the repo root when you start Streamlit there.")
