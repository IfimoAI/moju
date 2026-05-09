# Path B finite differences (law FD)

When **`auto_path_b_derivatives`** and **`fill_law_fd`** (or **`fill_law_recipes`** inside `fill_path_b_derivatives`) run, Moju can fill missing law inputs (e.g. `phi_laplacian`, `T_t`, `u_grad`) from field primitives on a structured grid.

## Default accuracy: `PathBGridConfig.fd_order`

- **`fd_order=4`** (default): explicit **4th-order** centered and one-sided stencils on **uniform** spacing along each axis where the layout is **separable** (1D meshgrid, rectilinear meshgrid with 1D axis arrays, separable layouts). Laplacians use the **sum of per-axis second derivatives**, not a composition of two first-derivative passes.
- **`fd_order=2`**: legacy behavior via **`jnp.gradient`** (and non-uniform 1D spacing where applicable), for comparisons or very coarse grids.

Studio / JSON: **`path_b_grid_from_options`** accepts **`fd_order`** in **`{2, 4}`** (default **4**).

## Minimum grid size

4th-order stencils need **at least 5 points** per axis where they apply (`MIN_POINTS_FD_ORDER_4`). With fewer points, the implementation **falls back** to 2nd-order **`jnp.gradient`**-style differencing along that axis.

## Non-uniform spacing and curvilinear meshes

- **Non-uniform 1D** coordinates: when **`fd_order=4`** but spacing is not uniform, the code **falls back to 2nd order** and records a **warning** (per axis / context).
- **Full curvilinear meshgrid** (coordinates full **`K`-shaped**, not rectilinear 1D axes): **`fd_order=4`** is **not** applied; **`jnp.gradient`** (**2nd order**) is used with a **warning**.
- **Non-uniform time** **`t`**: same policy for **`d/dt`** and **`d²/dt²`**.

These paths are intentional: variable-mesh or metric-aware 4th-order operators are not implemented.

## Float precision

**Float32** with small **`h`** and **high-order** differences amplifies round-off; error vs refinement can **stall or worsen** in float32. For FD-heavy Path B audits or refinement studies, prefer **`jax.config.update("jax_enable_x64", True)`** and **`float64`** arrays.

## Path A

Workflows that supply all law inputs from a **state builder / NN** do not use this stack unless a key is missing and FD fill runs.
