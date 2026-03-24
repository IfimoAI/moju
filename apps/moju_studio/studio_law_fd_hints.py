"""
Human-readable FD prerequisite lines for Studio (selected laws).

Uses the same recipes as :mod:`moju.monitor.law_fd_recipes`.
"""

from __future__ import annotations

from typing import List

from apps.moju_studio.config_forms import law_parameter_names

from moju.monitor.law_fd_recipes import (
    LAW_FD_RECIPES,
    LawFDArgRecipe,
    _resolve_source_state_key,
)


def _recipe_line(law_name: str, arg_name: str, recipe: LawFDArgRecipe) -> str:
    sm_identity = {arg_name: arg_name}
    target_sk = arg_name
    src = _resolve_source_state_key(recipe, arg_name, target_sk, sm_identity)
    prim = src or "(could not infer — use Expert JSON state_map)"
    return f"- **`{arg_name}`** — FD kind `{recipe.kind}` from primitive **`{prim}`** when that key is missing."


def law_fd_help_markdown(law_name: str) -> str:
    """Single law: bullet list of FD recipes + grid note."""
    recipes = LAW_FD_RECIPES.get(law_name) or {}
    args = []
    try:
        args = law_parameter_names(law_name)
    except Exception:  # noqa: BLE001
        pass

    lines: List[str] = [f"**{law_name}**"]
    if not recipes:
        if args:
            lines.append(
                "No registered `LAW_FD_RECIPES` entries — provide every argument in `state_pred` or constants: "
                + ", ".join(f"`{a}`" for a in args)
                + "."
            )
        else:
            lines.append("No FD recipes registered for this law.")
        return "\n".join(lines)

    for arg_name, recipe in recipes.items():
        lines.append(_recipe_line(law_name, arg_name, recipe))

    lines.append(
        "- **Grid:** include **`x`** in `state_pred` (and **`y`**, **`z`** for multi-D meshgrid layouts). "
        "Temporal FD needs **`t`** when the law uses time derivatives (`T_t`, etc.)."
    )
    return "\n".join(lines)


def format_laws_fd_help(law_names: List[str]) -> str:
    """Markdown for multiple laws (e.g. Studio multiselect)."""
    if not law_names:
        return ""
    blocks = [law_fd_help_markdown(n) for n in law_names]
    return "\n\n".join(blocks)
