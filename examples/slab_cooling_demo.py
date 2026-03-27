#!/usr/bin/env python3
"""
Transient cooling of an aluminum slab (1D).

  - Training PDE (interior): T_t - alpha(x,t) T_xx = 0 with alpha = k(T)/(rho(T)*cp) [m^2/s].
  - Moju law: ``Laws.fourier_conduction`` is the same balance written as
    T_t - (Fo*L^2/t) T_xx = 0 with Fo = alpha*t/L^2 (``Groups.fo``). Use **physical** T_t, T_xx,
    t [s], L [m], and alpha [m^2/s] so this matches the training residual.

  - Law-linked implied audit (default ``law_implied_audits=True``): compares
    ``Models.thermal_diffusivity(k, rho, cp)`` to alpha_implied = T_t/T_xx from the same fields.
    If those numbers disagree, check that T_t and T_xx are the same derivatives you use in the
    PDE (SI-consistent), not a different nondimensionalization. See ``docs/law_implied_audits.md``.

  - Scaling audits (Fo ``ref_delta``) only run when ``state_ref`` is passed; this demo omits
    them and relies on laws + groups + constitutive implied rows only.

  - **Training loss:** Interior collocation only (no explicit initial/boundary penalties in the
    objective). **Normalization:** inputs :math:`\\tau=(t-t_{\\min})/(t_{\\max}-t_{\\min})`,
    :math:`\\xi=x/L`; network predicts :math:`\\theta\\in(0,1)` with ``sigmoid``, and
    :math:`T=T_\\infty+(T_i-T_\\infty)\\theta`. The interior PDE residual is scaled so the loss is
    :math:`O(1)` at initialization.

Run: pip install moju[report] && python examples/slab_cooling_demo.py
"""

import jax
import jax.numpy as jnp
import optax

from moju.piratio import Operators
from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor import ResidualEngine, build_loss, audit, visualize

L = 0.02
k_solid = 200.0
rho_ref = 2700.0
cp = 900.0
T_inf = 300.0
T_i = 500.0
t_min = 1.0
t_max = 60.0


def init_mlp(key, widths):
    params = []
    for m, n in zip(widths[:-1], widths[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (n, m)) * jnp.sqrt(2.0 / m)
        b = jnp.zeros((n,))
        params.append({"W": W, "b": b})
    return params


def mlp(params, tx):
    h = tx
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"].T + layer["b"])
    out = params[-1]
    return h @ out["W"].T + out["b"]


def k_model(T):
    T_ref = (T_i + T_inf) / 2.0
    return k_solid * (1.0 + 0.001 * (T - T_ref))


def rho_model(T):
    T_ref = (T_i + T_inf) / 2.0
    return rho_ref * (1.0 - 0.0001 * (T - T_ref))


_delta_T = T_i - T_inf
_T_mid = (T_i + T_inf) / 2.0
_alpha_mid = k_model(_T_mid) / (rho_model(_T_mid) * cp)
_pde_residual_scale = _delta_T * max(1.0 / (t_max - t_min), float(_alpha_mid / (L**2)))


def _coords_norm(t, x):
    """Map physical (t, x) to normalized inputs ``(tau, xi)`` for the MLP."""
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    dt = t_max - t_min
    if t.ndim == 0 and x.ndim == 1:
        tau = jnp.broadcast_to((t - t_min) / dt, x.shape[:-1] + (1,))
        xi = x / L
    elif t.ndim == 1 and x.ndim == 2:
        tau = ((t - t_min) / dt)[:, None]
        xi = x / L
    else:
        tau = jnp.broadcast_to((t - t_min) / dt, x.shape[:-1] + (1,))
        xi = x / L
    return jnp.concatenate([tau, xi], axis=-1)


def theta_field(params, t, x):
    """Dimensionless temperature :math:`\\theta=(T-T_\\infty)/(T_i-T_\\infty)` in (0, 1)."""
    tx = _coords_norm(t, x)
    raw = mlp(params, tx)[..., 0]
    out = jax.nn.sigmoid(raw)
    return jnp.squeeze(out) if out.ndim > 0 and out.size == 1 else out


def scalar_field(params, t, x):
    theta = theta_field(params, t, x)
    T = T_inf + _delta_T * theta
    return jnp.squeeze(T) if T.ndim > 0 and T.size == 1 else T


def T_t_batch(params, t, x):
    return Operators.time_derivative(scalar_field, params, t, x)


def T_xx_batch(params, t, x):
    def body(ti, xi):
        return Operators.laplacian(
            lambda p, x_in: scalar_field(p, ti, x_in), params, xi
        )

    return jax.vmap(body)(t, x)


def T_x_batch(params, t, x):
    """Spatial derivative dT/dx at (t, x). t (N,), x (N, 1)."""

    def body(ti, xi):
        grad = Operators.gradient(lambda p, x_in: scalar_field(p, ti, x_in), params, xi)
        return grad[0] if grad.shape == (1,) else grad

    return jax.vmap(body)(t, x)


def physics_loss_interior(params, t, x):
    T = scalar_field(params, t, x)
    kappa = k_model(T)
    rho_val = rho_model(T)
    alpha_loc = kappa / (rho_val * cp)
    return T_t_batch(params, t, x) - alpha_loc * T_xx_batch(params, t, x)


def make_loss_fn(t_int, x_int):
    """Mean squared scaled interior PDE residual only (no IC/BC terms)."""

    def loss_fn(params):
        r_int = physics_loss_interior(params, t_int, x_int) / _pde_residual_scale
        return jnp.mean(r_int**2)

    return loss_fn


def make_train_step(loss_fn):
    """Return a jitted step that closes over ``loss_fn`` (uses module-level ``optimizer``)."""

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


engine = ResidualEngine(
    constants={"cp": cp},
    laws=[
        {
            "name": "fourier_conduction",
            "state_map": {
                "T_t": "T_t",
                "T_laplacian": "T_xx",
                "fo": "fo",
                "t": "t",
                "L": "L",
            },
            "fn": Laws.fourier_conduction,
        },
    ],
    groups=[
        {
            "name": "fo",
            "state_map": {"alpha": "alpha", "t": "t", "L": "L"},
            "output_key": "fo",
            "fn": Groups.fo,
        },
    ],
)


def build_state_for_engine(params, t, x):
    T = scalar_field(params, t, x)
    T_t = T_t_batch(params, t, x)
    T_x = T_x_batch(params, t, x)
    T_xx = T_xx_batch(params, t, x)
    kappa = k_model(T)
    rho = rho_model(T)
    alpha = kappa / (rho * cp)
    Lb = jnp.broadcast_to(L, t.shape)
    cp_b = jnp.broadcast_to(cp, t.shape)
    # fo = alpha*t/L^2 (dimensionless; same alpha as in PDE and Fourier law).
    fo = alpha * t / (Lb**2)
    return {
        "T": T,
        "T_t": T_t,
        "T_x": T_x,
        "T_xx": T_xx,
        "t": t,
        "L": Lb,
        "kappa": kappa,
        "rho": rho,
        "cp": cp_b,
        "alpha": alpha,
        "k": kappa,
        "k_solid": kappa,
        "fo": fo,
    }


def monitor_with_engine(params, t, x):
    state_pred = build_state_for_engine(params, t, x)
    residuals = engine.compute_residuals(state_pred, log_to_python=True)
    return build_loss(residuals)


optimizer = optax.adam(1e-3)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_t, n_x = 32, 24
    t_flat = jnp.linspace(t_min, t_max, n_t)
    x_flat = jnp.linspace(0.0, L, n_x)
    t_col, x_col = jnp.meshgrid(t_flat, x_flat, indexing="ij")
    t_col = t_col.reshape(-1)
    x_col = x_col.reshape(-1, 1)

    loss_fn = make_loss_fn(t_col, x_col)
    train_step = make_train_step(loss_fn)

    params = init_mlp(key, [2, 48, 48, 1])
    opt_state = optimizer.init(params)

    for step in range(1500):
        params, opt_state, loss = train_step(params, opt_state)
        if step % 150 == 0:
            law_loss = monitor_with_engine(params, t_col, x_col)
            print(f"step {step:4d}  loss={float(loss):.3e}  law_loss(engine)={float(law_loss):.3e}")

    state_final = build_state_for_engine(params, t_col, x_col)
    residuals_final = engine.compute_residuals(state_final, log_to_python=True)

    report = audit(
        engine.log,
        export_dir=".",
        save_residuals=True,
        last_residual_dict=residuals_final,
        model_name="SlabCooling-1D",
        model_id="demo-slab",
    )
    print("Overall admissibility score:", report["overall_admissibility_score"])
    print("Overall admissibility level:", report["overall_admissibility_level"])

    fig = visualize(engine.log, engine=engine)
    if fig is not None:
        if hasattr(fig, "write_html"):
            fig.write_html("slab_cooling_diagnostics.html")
            print("Saved slab_cooling_diagnostics.html")
        else:
            fig.savefig("slab_cooling_diagnostics.png", dpi=150, bbox_inches="tight")
            print("Saved slab_cooling_diagnostics.png")
