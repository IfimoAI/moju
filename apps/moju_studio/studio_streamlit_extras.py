"""
Streamlit UX helpers: caching registries, safe use of toast / status / fragment / dialog.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Tuple

import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def cached_registry_names() -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Laws, constitutive models, scaling groups — stable strings for selectboxes."""
    from apps.moju_studio.studio_core import list_registered_law_names
    from moju.monitor import list_constitutive_models, list_scaling_closure_ids

    return (
        tuple(list_registered_law_names()),
        tuple(list_constitutive_models()),
        tuple(list_scaling_closure_ids()),
    )


def toast(msg: str, *, icon: Optional[str] = None) -> None:
    if hasattr(st, "toast"):
        st.toast(msg, icon=icon)
    else:
        st.caption(msg)


@contextmanager
def pipeline_status(message: str) -> Iterator[Any]:
    """``st.status`` with spinner fallback for older Streamlit."""
    if hasattr(st, "status"):
        with st.status(message, expanded=True) as status:
            yield status
    else:
        with st.spinner(message):
            yield None


def status_update(status: Any, label: str) -> None:
    if status is None:
        return
    try:
        status.update(label=label, state="running")
    except (TypeError, AttributeError):
        pass


def status_complete(status: Any, label: str) -> None:
    if status is None:
        return
    try:
        status.update(label=label, state="complete")
    except (TypeError, AttributeError):
        pass


def as_fragment(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Apply ``@st.fragment`` when available."""
    frag = getattr(st, "fragment", None)
    if frag is not None:
        return frag(fn)
    return fn


def run_dialog_if_available(dialog_fn: Callable[[], None], trigger: bool) -> None:
    """Call ``dialog_fn`` when ``trigger`` and ``st.dialog`` exists."""
    if not trigger:
        return
    d = getattr(st, "dialog", None)
    if d is not None:
        dialog_fn()
