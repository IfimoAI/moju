# Before / After Moju — Corrected code snippets

Reference for the "Hardwiring reality into every iteration" graphic. All syntax errors and typos fixed with minimal format changes; APIs match the moju codebase.

---

## BEFORE MOJU: Pure JAX (PINN)

```python
import jax
# Definitive derivatives
dx = jax.grad(u)(x)
dy = jax.grad(jax.grad(u))(x)
# Hard-coded properties (μ, ρ)
mu = get_mu(T)
rho = get_rho()

def momentum_residual(u, me):
    residual = jax.grad(u)(x)
    residual = Laws.momentum_navier_stokes(
        u_t, u, u_grad, p_grad, u_laplacian, re
    )
    residual = residual - jax.grad(jax.grad(u))(x)
    residual = residual - viscous_term(jax.grad(u))(x)
    residual = residual + jax.grad(jax.grad(u))(x)
    return residual
```

- Manual autodiff complexity.
- Hard-coded properties (μ, ρ).
- Non-standard residual handling.

---

## AFTER MOJU: Physics-native AI

```python
from moju.piratio import Models, Laws, Operators

# Compact derivative
u_laplacian = Operators.laplacian(u_net, params, x_grid)

# Physical property call
mu = Models.sutherland_mu(T, mu0, T0, S)

# Unified residual
residual = Laws.momentum_navier_stokes(u_t, u, u_grad, p_grad, u_laplacian, re)
```

- **Operators.laplacian**: signature `(func, params, x)`; here `u_net` is the network callable.
- **Models.sutherland_mu**: signature `(T, mu0, T0, S)`.
- **Laws.momentum_navier_stokes**: signature `(u_t, u, u_grad, p_grad, u_laplacian, re)`.

Use this file when redrawing the graphic or copying snippets.
