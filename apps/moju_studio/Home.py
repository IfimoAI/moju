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

st.set_page_config(
    page_title="Moju Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Moju Studio")
st.markdown(
    """
Welcome. Use the **sidebar** (or the links below) to open **Audit**, **Quick start**, or **Help**.

**Requirements:** `pip install "moju[studio,viz]"` (Streamlit **>= 1.33**). Run from the **repository root** so `.streamlit/config.toml` applies.

**Docs:** repository README and `moju.monitor` API.
"""
)

try:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("pages/1_Audit.py", label="Open Audit", icon="🔬")
    with c2:
        st.page_link("pages/2_Quick_start.py", label="Quick start", icon="📘")
    with c3:
        st.page_link("pages/3_Help.py", label="Help and UX", icon="❓")
except Exception:  # noqa: BLE001
    st.info("Select a page under **pages** in the sidebar to continue.")
