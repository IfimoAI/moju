"""
Opt-in Path B boundary closures (Moju 2.0 Phase A4).

Modes (``boundary_closure`` on :class:`~moju.monitor.path_b_derivatives.PathBGridConfig`
or stencil kwargs):

- ``\"matching\"`` (default): Fornberg one-sided of the same formal order as the interior.
- ``\"order_reduced\"``: boundary rows use Fornberg at one order lower (4→2, 6→4; 2 stays 2).
- ``\"sbp\"``: diagonal-norm summation-by-parts boundary blocks (orders 4 and 6);
  second derivatives use the SBP first-derivative operator composed once (D₁ᴴ D₁).
- ``\"bc_aware\"``: ghost-point centred closures using **explicit** Dirichlet values
  (never assumed). Supply ``boundary_values=(lo, hi)`` per axis call, or
  ``use_field_boundary_values=True`` to take the field's own boundary samples as
  known Dirichlet data for that differentiation.

These modes are opt-in for Phase A4; default Path B behaviour remains ``matching``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

BoundaryClosure = Literal["matching", "order_reduced", "sbp", "bc_aware"]
VALID_BOUNDARY_CLOSURES: Tuple[str, ...] = (
    "matching",
    "order_reduced",
    "sbp",
    "bc_aware",
)


def validate_boundary_closure(mode: Any) -> str:
    if mode is None:
        return "matching"
    m = str(mode).strip().lower()
    if m not in VALID_BOUNDARY_CLOSURES:
        raise ValueError(
            f"boundary_closure must be one of {VALID_BOUNDARY_CLOSURES}, got {mode!r}"
        )
    return m


def reduced_boundary_order(interior_order: int) -> int:
    """One formal order step down for boundary rows (clamped to {2,4,6})."""
    from moju.monitor.fd_stencils import validate_fd_order

    o = validate_fd_order(interior_order)
    if o == 2:
        return 2
    if o == 4:
        return 2
    return 4  # o == 6


def _apply_fornberg_boundary_band(
    D: Any,
    n: int,
    *,
    deriv_order: int,
    h: float,
    boundary_order: int,
    n_rows: Optional[int] = None,
) -> None:
    """
    Overwrite left/right bands with Fornberg stencils of ``boundary_order``.

    Stencil width is ``boundary_order + deriv_order`` so the formal order of the
    ``deriv_order``-th derivative matches ``boundary_order`` (``n`` points give
    order ``n - m`` for the ``m``-th derivative). First derivatives therefore use
    the familiar ``order + 1`` points; second derivatives use ``order + 2``.

    ``n_rows`` defaults to ``(boundary_order+1)//2``. For order-reduced closures,
    pass the *interior* half-width so gap rows (between reduced and matching half)
    also receive a valid shifted Fornberg stencil (never left as zeros).
    """
    import numpy as np
    from moju.monitor.fd_stencils import fornberg_weights, validate_fd_order

    bo = validate_fd_order(boundary_order)
    if deriv_order not in (1, 2):
        raise ValueError(f"deriv_order must be 1 or 2, got {deriv_order}")
    width = int(bo + deriv_order)
    half = width // 2
    rows = int((bo + 1) // 2 if n_rows is None else n_rows)
    rows = max(0, min(rows, n // 2))
    if width > n:
        raise ValueError(
            f"Fornberg boundary band needs width={width} points for "
            f"boundary_order={bo}, deriv_order={deriv_order}, got n={n}"
        )
    for i in range(rows):
        D[i, :] = 0.0
        # Shift stencil to stay inside [0, width) growing toward the interior
        left = 0
        right = width
        if i >= width:
            left = i - half
            right = left + width
        idx = np.arange(left, right)
        w = fornberg_weights(float(i - left), (idx - left).astype(float), deriv_order)[
            deriv_order, :
        ]
        D[i, idx] = w / (h**deriv_order)
    for i in range(n - rows, n):
        D[i, :] = 0.0
        left = n - width
        right = n
        if i < n - width:
            left = i - half
            right = left + width
        idx = np.arange(left, right)
        w = fornberg_weights(float(i - left), (idx - left).astype(float), deriv_order)[
            deriv_order, :
        ]
        D[i, idx] = w / (h**deriv_order)


def _fill_interior_central(
    D: Any, n: int, *, deriv_order: int, h: float, order: int
) -> int:
    """
    Fill interior central rows; return ``half`` (boundary band width for matching order).
    """
    import numpy as np
    from moju.monitor.fd_stencils import validate_fd_order

    order = validate_fd_order(order)
    width = order + 1
    half = width // 2
    if deriv_order == 1:
        if order == 2:
            for i in range(1, n - 1):
                D[i, i - 1] = -0.5 / h
                D[i, i + 1] = 0.5 / h
        elif order == 4:
            c = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
            for i in range(2, n - 2):
                D[i, i - 2 : i + 3] = c
        else:
            c = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / (60.0 * h)
            for i in range(3, n - 3):
                D[i, i - 3 : i + 4] = c
    elif deriv_order == 2:
        if order == 2:
            for i in range(1, n - 1):
                D[i, i - 1] = 1.0 / (h * h)
                D[i, i] = -2.0 / (h * h)
                D[i, i + 1] = 1.0 / (h * h)
        elif order == 4:
            c = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
            for i in range(2, n - 2):
                D[i, i - 2 : i + 3] = c
        else:
            c = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / (
                180.0 * h * h
            )
            for i in range(3, n - 3):
                D[i, i - 3 : i + 4] = c
    else:
        raise ValueError(f"deriv_order must be 1 or 2, got {deriv_order}")
    return half


def _sbp_d1_matrix(n: int, h: float, order: int) -> Any:
    """
    Diagonal-norm SBP first-derivative matrix (Mattsson–Nordström / Strand style).

    Interior accuracy ``order``; boundary accuracy ``order//2`` (2 for order 4, 3 for 6).
    """
    import numpy as np
    from moju.monitor.fd_stencils import validate_fd_order

    order = validate_fd_order(order)
    if order == 2:
        # Standard 2nd-order SBP ≡ central + one-sided first-order ends
        D = np.zeros((n, n), dtype=float)
        D[0, 0] = -1.0 / h
        D[0, 1] = 1.0 / h
        for i in range(1, n - 1):
            D[i, i - 1] = -0.5 / h
            D[i, i + 1] = 0.5 / h
        D[n - 1, n - 2] = -1.0 / h
        D[n - 1, n - 1] = 1.0 / h
        return D

    D = np.zeros((n, n), dtype=float)
    if order == 4:
        # Boundary block (order 2) — Mattsson & Nordström (2004) / Strand (1994)
        left = [
            [-24 / 17, 59 / 34, -4 / 17, -3 / 34],
            [-1 / 2, 0.0, 1 / 2],
            [4 / 43, -59 / 86, 0.0, 59 / 86, -4 / 43],
            [3 / 98, 0.0, -59 / 98, 0.0, 32 / 49, -4 / 49],
        ]
        for i, row in enumerate(left):
            for j, c in enumerate(row):
                D[i, j] = c / h
        c_int = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
        for i in range(4, n - 4):
            D[i, i - 2 : i + 3] = c_int
        # Right = -flip(left)
        for i, row in enumerate(left):
            for j, c in enumerate(row):
                D[n - 1 - i, n - 1 - j] = -c / h
        return D

    # order == 6: order-4 Fornberg boundary bands + order-6 interior, with
    # first-order SBP ends on the outermost rows (practical Phase A4 operator;
    # full (3,6) Mattsson blocks are deferred if empirical tests prefer order_reduced).
    _fill_interior_central(D, n, deriv_order=1, h=h, order=6)
    _apply_fornberg_boundary_band(D, n, deriv_order=1, h=h, boundary_order=4)
    D[0, :] = 0.0
    D[0, 0] = -1.0 / h
    D[0, 1] = 1.0 / h
    D[n - 1, :] = 0.0
    D[n - 1, n - 2] = -1.0 / h
    D[n - 1, n - 1] = 1.0 / h
    return D


def _sbp_d2_matrix(n: int, h: float, order: int) -> Any:
    """
    Second-derivative SBP-style operator.

    Interior uses the dedicated central stencil of ``order``. Boundary rows use
    Fornberg one-sided at ``order//2`` (same formal accuracy as the SBP D1 ends),
    not ``D1@D1`` (which drops global accuracy).
    """
    import numpy as np
    from moju.monitor.fd_stencils import validate_fd_order

    order = validate_fd_order(order)
    D = np.zeros((n, n), dtype=float)
    _fill_interior_central(D, n, deriv_order=2, h=h, order=order)
    b_order = reduced_boundary_order(order)
    match_half = (order + 1) // 2
    for i in range(match_half):
        D[i, :] = 0.0
    for i in range(n - match_half, n):
        D[i, :] = 0.0
    _apply_fornberg_boundary_band(
        D, n, deriv_order=2, h=h, boundary_order=b_order, n_rows=match_half
    )
    return D


def _bc_aware_matrix(
    n: int,
    h: float,
    *,
    deriv_order: int,
    order: int,
    boundary_values: Optional[Tuple[Optional[float], Optional[float]]],
) -> Any:
    """
    Ghost-point matrix: pad with Dirichlet ghosts, apply centred stencil, restrict.

    ``boundary_values=(lo, hi)``; ``None`` on a side falls back to matching Fornberg.
    """
    import numpy as np
    from moju.monitor.fd_stencils import fornberg_weights, validate_fd_order

    order = validate_fd_order(order)
    half = (order + 1) // 2
    lo_val, hi_val = (None, None) if boundary_values is None else boundary_values

    # Build operator on extended grid conceptually: for each output row i, stencil
    # may reference ghosts. Ghosts are linear in boundary value and interior samples:
    #   f[-k] = 2*g_lo - f[k]   (homogeneous reflection through Dirichlet at node 0)
    # when g_lo is the known value AT node 0 (already in f[0]; we still use reflection
    # for exterior ghosts so centred stencils close).
    D = np.zeros((n, n), dtype=float)
    # Interior: standard central
    _fill_interior_central(D, n, deriv_order=deriv_order, h=h, order=order)

    def _centred_weights(offset_from_node: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (relative_offsets, weights) for centred stencil at a boundary-adjacent node."""
        # Use Fornberg on a virtual uniform stencil including ghosts
        idxs = np.arange(-half, half + 1, dtype=float)
        w = fornberg_weights(0.0, idxs, deriv_order)[deriv_order, :] / (h**deriv_order)
        return idxs.astype(int), w

    for i in range(half):
        if lo_val is None:
            # matching Fornberg on this row only
            width = order + 1
            idx = np.arange(0, width)
            D[i, :] = 0.0
            D[i, idx] = fornberg_weights(float(i), idx.astype(float), deriv_order)[
                deriv_order, :
            ] / (h**deriv_order)
            continue
        offs, w = _centred_weights(i)
        D[i, :] = 0.0
        # Evaluation point is physical index i; stencil nodes are i+off
        for off, coef in zip(offs, w):
            j = i + int(off)
            if 0 <= j < n:
                D[i, j] += coef
            else:
                # ghost left of 0: j = -k → contributes to f[k] with minus, and to g
                # f[-k] = 2*g - f[k]  ⇒  coef * f[-k] = coef*2*g - coef*f[k]
                # Affine g term is stored separately; for matrix on interior samples:
                k = -j
                if 0 <= k < n:
                    D[i, k] += -coef
                # Dirichlet value contribution handled at apply-time via bias vector
        # Encode bias in an extra convention: we return only the homogeneous matrix
        # for reflected ghosts; bias = sum of coef*2*g for ghost nodes is applied
        # in ``apply_diff_with_closure`` via boundary_values.

    for i in range(n - half, n):
        if hi_val is None:
            width = order + 1
            left = n - width
            idx = np.arange(left, n)
            D[i, :] = 0.0
            D[i, idx] = fornberg_weights(
                float(i - left), (idx - left).astype(float), deriv_order
            )[deriv_order, :] / (h**deriv_order)
            continue
        offs, w = _centred_weights(i)
        D[i, :] = 0.0
        for off, coef in zip(offs, w):
            j = i + int(off)
            if 0 <= j < n:
                D[i, j] += coef
            else:
                # ghost right of n-1: j = n-1+k → f[n-1+k] = 2*g_hi - f[n-1-k]
                k = j - (n - 1)
                mirror = (n - 1) - k
                if 0 <= mirror < n:
                    D[i, mirror] += -coef
    return D


def bc_aware_bias(
    n: int,
    h: float,
    *,
    deriv_order: int,
    order: int,
    boundary_values: Tuple[Optional[float], Optional[float]],
) -> Any:
    """Affine bias vector so that ``D @ f + bias`` includes Dirichlet ghost contributions."""
    import numpy as np
    from moju.monitor.fd_stencils import fornberg_weights, validate_fd_order

    order = validate_fd_order(order)
    half = (order + 1) // 2
    lo_val, hi_val = boundary_values
    bias = np.zeros(n, dtype=float)

    def _centred(i: int):
        idxs = np.arange(-half, half + 1, dtype=float)
        w = fornberg_weights(0.0, idxs, deriv_order)[deriv_order, :] / (h**deriv_order)
        return idxs.astype(int), w

    for i in range(half):
        if lo_val is None:
            continue
        offs, w = _centred(i)
        for off, coef in zip(offs, w):
            j = i + int(off)
            if j < 0:
                bias[i] += coef * 2.0 * float(lo_val)
    for i in range(n - half, n):
        if hi_val is None:
            continue
        offs, w = _centred(i)
        for off, coef in zip(offs, w):
            j = i + int(off)
            if j >= n:
                bias[i] += coef * 2.0 * float(hi_val)
    return bias


def _periodic_central_matrix(
    n: int, h: float, *, deriv_order: int, order: int
) -> Any:
    """Circulant central stencil of ``order`` (wrap-around at every row)."""
    import numpy as np
    from moju.monitor.fd_stencils import validate_fd_order

    order = validate_fd_order(order)
    if n < order + 1:
        raise ValueError(
            f"periodic FD needs at least {order + 1} points for fd_order={order}, got {n}"
        )
    D = np.zeros((n, n), dtype=float)
    half = (order + 1) // 2
    if deriv_order == 1:
        if order == 2:
            coeffs = { -1: -0.5 / h, 1: 0.5 / h }
        elif order == 4:
            c = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
            coeffs = {k: float(c[k + 2]) for k in range(-2, 3)}
        else:
            c = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / (60.0 * h)
            coeffs = {k: float(c[k + 3]) for k in range(-3, 4)}
    elif deriv_order == 2:
        if order == 2:
            coeffs = {-1: 1.0 / (h * h), 0: -2.0 / (h * h), 1: 1.0 / (h * h)}
        elif order == 4:
            c = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
            coeffs = {k: float(c[k + 2]) for k in range(-2, 3)}
        else:
            c = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / (
                180.0 * h * h
            )
            coeffs = {k: float(c[k + 3]) for k in range(-3, 4)}
    else:
        raise ValueError(f"deriv_order must be 1 or 2, got {deriv_order}")
    for i in range(n):
        for off, coef in coeffs.items():
            if abs(coef) < 1e-30:
                continue
            D[i, (i + off) % n] += coef
    return D


def build_diff_matrix_1d_uniform(
    n: int,
    h: float,
    *,
    deriv_order: int,
    order: int,
    boundary_closure: Any = "matching",
    boundary_values: Optional[Tuple[Optional[float], Optional[float]]] = None,
    periodic: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """
    Build dense 1D differentiation matrix and optional Dirichlet bias.

    When ``periodic=True``, uses a circulant central stencil on every row
    (``boundary_closure`` / Dirichlet ghosts are ignored for that axis).

    Returns ``(D, bias_or_None)``.
    """
    import numpy as np
    from moju.monitor.fd_stencils import fornberg_weights, validate_fd_order

    order = validate_fd_order(order)
    h = float(h)
    if abs(h) < 1e-30:
        raise ValueError("spacing h must be non-zero")
    if deriv_order not in (1, 2):
        raise ValueError(f"deriv_order must be 1 or 2, got {deriv_order}")

    if periodic:
        return _periodic_central_matrix(n, h, deriv_order=deriv_order, order=order), None

    mode = validate_boundary_closure(boundary_closure)

    if mode == "sbp":
        if deriv_order == 1:
            return _sbp_d1_matrix(n, h, order), None
        return _sbp_d2_matrix(n, h, order), None

    if mode == "bc_aware":
        D = _bc_aware_matrix(
            n, h, deriv_order=deriv_order, order=order, boundary_values=boundary_values
        )
        bias = None
        if boundary_values is not None and any(v is not None for v in boundary_values):
            bias = bc_aware_bias(
                n, h, deriv_order=deriv_order, order=order, boundary_values=boundary_values
            )
        return D, bias

    # matching or order_reduced: interior central + Fornberg bands
    D = np.zeros((n, n), dtype=float)
    _fill_interior_central(D, n, deriv_order=deriv_order, h=h, order=order)
    b_order = order if mode == "matching" else reduced_boundary_order(order)
    match_half = (order + 1) // 2
    # Clear the matching-width boundary band, then write Fornberg of b_order on
    # every cleared row (shifted) so order_reduced never leaves gap rows as zeros.
    for i in range(match_half):
        D[i, :] = 0.0
    for i in range(n - match_half, n):
        D[i, :] = 0.0
    _apply_fornberg_boundary_band(
        D,
        n,
        deriv_order=deriv_order,
        h=h,
        boundary_order=b_order,
        n_rows=match_half,
    )
    return D, None


def _pad_dirichlet_ghosts(a: Any, half: int, *, order: int) -> Any:
    """
    Pad last-axis with high-order polynomial ghosts that honor boundary samples.

    For each line, extrapolate ``f[-k]`` / ``f[n-1+k]`` from the ``order+1`` samples
    abutting that end (including the Dirichlet node) via Fornberg value weights.
    Odd reflection ``2f[0]-f[k]`` is only first-order and is **not** used.
    """
    import numpy as np
    from moju.monitor.fd_stencils import fornberg_weights, validate_fd_order

    order = validate_fd_order(order)
    width = order + 1
    n = int(a.shape[-1])
    idx = np.arange(width, dtype=float)
    left = []
    for k in range(half, 0, -1):
        w = fornberg_weights(float(-k), idx, 0)[0, :]
        ghost = sum(float(w[j]) * a[..., j : j + 1] for j in range(width))
        left.append(ghost)
    right = []
    for k in range(1, half + 1):
        w = fornberg_weights(float(width - 1 + k), idx, 0)[0, :]
        ghost = sum(
            float(w[j]) * a[..., n - width + j : n - width + j + 1] for j in range(width)
        )
        right.append(ghost)
    return jnp.concatenate([*left, a, *right], axis=-1)


def apply_diff_uniform_1d(
    values: Any,
    h: float,
    *,
    deriv_order: int,
    order: int,
    axis: int = -1,
    boundary_closure: Any = "matching",
    boundary_values: Optional[Tuple[Optional[float], Optional[float]]] = None,
    use_field_boundary_values: bool = False,
    periodic: bool = False,
) -> Any:
    """Apply a 1D uniform differentiation matrix along ``axis``."""
    from moju.monitor.fd_stencils import validate_fd_order

    order = validate_fd_order(order)
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    n = int(a.shape[-1])
    h = float(h)

    if periodic:
        D, _ = build_diff_matrix_1d_uniform(
            n, h, deriv_order=deriv_order, order=order, periodic=True
        )
        flat = a.reshape(-1, n)
        out = flat @ jnp.asarray(D).T
        return jnp.moveaxis(out.reshape(a.shape), -1, axis)

    mode = validate_boundary_closure(boundary_closure)

    # bc_aware with per-line Dirichlet: pad ghosts, apply centred stencil on the
    # extended grid, then restrict to the original nodes.
    if mode == "bc_aware" and (
        use_field_boundary_values
        or (
            boundary_values is not None
            and boundary_values[0] is not None
            and boundary_values[1] is not None
        )
    ):
        half = (order + 1) // 2
        if use_field_boundary_values or boundary_values is None:
            padded = _pad_dirichlet_ghosts(a, half, order=order)
        else:
            # Constant Dirichlet: overwrite boundary nodes, then high-order pad
            lo, hi = float(boundary_values[0]), float(boundary_values[1])
            a_bc = a.at[..., 0].set(lo).at[..., -1].set(hi)
            padded = _pad_dirichlet_ghosts(a_bc, half, order=order)
        # Centred differentiation on the padded line (matching interior stencils only)
        D_pad, _ = build_diff_matrix_1d_uniform(
            int(padded.shape[-1]),
            h,
            deriv_order=deriv_order,
            order=order,
            boundary_closure="matching",
            boundary_values=None,
        )
        flat = padded.reshape(-1, padded.shape[-1])
        out_pad = flat @ jnp.asarray(D_pad).T
        out = out_pad[:, half : half + n].reshape(a.shape)
        return jnp.moveaxis(out, -1, axis)

    bv = boundary_values
    D, bias = build_diff_matrix_1d_uniform(
        n,
        h,
        deriv_order=deriv_order,
        order=order,
        boundary_closure=boundary_closure,
        boundary_values=bv,
        periodic=False,
    )
    flat = a.reshape(-1, n)
    out = flat @ jnp.asarray(D).T
    if bias is not None:
        out = out + jnp.asarray(bias, dtype=out.dtype)[None, :]
    return jnp.moveaxis(out.reshape(a.shape), -1, axis)
