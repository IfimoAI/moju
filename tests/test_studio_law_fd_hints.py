"""Studio law FD prerequisite text (no Streamlit)."""

from apps.moju_studio.studio_law_fd_hints import format_laws_fd_help, law_fd_help_markdown


def test_fourier_conduction_mentions_t_and_laplacian():
    md = law_fd_help_markdown("fourier_conduction")
    assert "T_laplacian" in md
    assert "laplacian" in md
    assert "T" in md
    assert "x" in md


def test_format_laws_fd_help_multiple():
    md = format_laws_fd_help(["laplace_equation", "fourier_conduction"])
    assert "laplace_equation" in md
    assert "fourier_conduction" in md


def test_law_without_recipes_lists_args():
    md = law_fd_help_markdown("laplace_beltrami")
    assert "laplace_beltrami" in md
    assert "No registered" in md or "LAW_FD_RECIPES" in md
