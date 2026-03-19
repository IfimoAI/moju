#!/usr/bin/env python3
"""
Transient cooling of an aluminum slab with convection (1D).

  - PDE: rho(T)*cp*T_t = k(T)*T_xx; audit uses Laws.fourier_conduction.
  - Monitor: this demo focuses on governing-law residuals (Laws.fourier_conduction).
    See `examples/monitor_chain_spatial_demo.py` and `examples/monitor_chain_temporal_demo.py`
    for the new Model/Group chain-closure audits.

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
h = 500.0
T_inf = 300.0
T_i = 500.0
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


def scalar_field(params, t, x):
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    if t.ndim == 0 and x.ndim == 1:
        tx = jnp.concatenate([jnp.broadcast_to(t, x.shape[:-1] + (1,)), x], axis=-1)
    elif t.ndim == 1 and x.ndim == 2:
        tx = jnp.concatenate([t[:, None], x], axis=-1)
    else:
        tx = jnp.concatenate([jnp.broadcast_to(t, x.shape[:-1] + (1,)), x], axis=-1)
    out = mlp(params, tx)[..., 0]
    return jnp.squeeze(out) if out.ndim > 0 and out.size == 1 else out


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


def k_model(T):
    T_ref = (T_i + T_inf) / 2.0
    return k_solid * (1.0 + 0.001 * (T - T_ref))


def rho_model(T):
    T_ref = (T_i + T_inf) / 2.0
    return rho_ref * (1.0 - 0.0001 * (T - T_ref))


def physics_loss_interior(params, t, x):
    T = scalar_field(params, t, x)
    kappa = k_model(T)
    rho_val = rho_model(T)
    alpha_loc = kappa / (rho_val * cp)
    return T_t_batch(params, t, x) - alpha_loc * T_xx_batch(params, t, x)


def loss_fn(params, t, x):
    return jnp.mean(physics_loss_interior(params, t, x) ** 2)


@jax.jit
def train_step(params, opt_state, t, x):
    loss, grads = jax.value_and_grad(loss_fn)(params, t, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


engine = ResidualEngine(
    constants={"cp": cp, "h": h},
    laws=[
        {
            "name": "fourier_conduction",
            "state_map": {
                "T_t": "T_t",
                "T_laplacian": "T_xx",
                "fo": "Fo",
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
            "output_key": "Fo",
            "fn": Groups.fo,
        },
        {
            "name": "bi",
            "state_map": {"h": "h", "L": "L", "k_solid": "kappa"},
            "output_key": "Bi",
            "fn": Groups.bi,
        },
    ],
    constitutive_audit=[
        {
            "name": "thermal_diffusivity",
            "output_key": "alpha",
            "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
            "predicted_spatial": ["k", "rho"],
            "predicted_temporal": ["k", "rho"],
        }
    ],
    scaling_audit=[
        {
            "name": "fo",
            "output_key": "Fo",
            "state_map": {"alpha": "alpha", "t": "t", "L": "L"},
            "predicted_spatial": ["alpha"],
            "predicted_temporal": ["alpha", "t"],
        },
        {
            "name": "bi",
            "output_key": "Bi",
            "state_map": {"h": "h", "L": "L", "k_solid": "kappa"},
            "predicted_spatial": ["kappa"],
            "predicted_temporal": ["kappa"],
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
    hb = jnp.broadcast_to(h, t.shape)
    # Derivatives for Model/Group chain audits.
    dk_dT = 0.001 * k_solid
    drho_dT = -0.0001 * rho_ref
    d_k_dx = dk_dT * T_x
    d_k_dt = dk_dT * T_t
    d_rho_dx = drho_dT * T_x
    d_rho_dt = drho_dT * T_t
    d_alpha_dx = (1.0 / (rho * cp)) * d_k_dx - (kappa / (rho**2 * cp)) * d_rho_dx
    d_alpha_dt = (1.0 / (rho * cp)) * d_k_dt - (kappa / (rho**2 * cp)) * d_rho_dt
    d_t_dt = jnp.ones_like(t)
    # Fo = alpha*t/L^2, Bi = h*L/kappa
    Fo = alpha * t / (Lb**2)
    Bi = hb * Lb / kappa
    d_Fo_dx = (t / (Lb**2)) * d_alpha_dx
    d_Fo_dt = (t / (Lb**2)) * d_alpha_dt + (alpha / (Lb**2)) * d_t_dt
    d_Bi_dx = -(hb * Lb / (kappa**2)) * d_k_dx
    d_Bi_dt = -(hb * Lb / (kappa**2)) * d_k_dt
    return {
        "T": T,
        "T_t": T_t,
        "T_x": T_x,
        "T_xx": T_xx,
        "t": t,
        "L": Lb,
        "kappa": kappa,
        "rho": rho,
        "alpha": alpha,
        "k": kappa,
        "k_solid": kappa,
        "h": hb,
        # Derivative convention for chain closures
        "d_k_dx": d_k_dx,
        "d_k_dt": d_k_dt,
        "d_rho_dx": d_rho_dx,
        "d_rho_dt": d_rho_dt,
        "d_alpha_dx": d_alpha_dx,
        "d_alpha_dt": d_alpha_dt,
        "d_t_dt": d_t_dt,
        "Fo": Fo,
        "Bi": Bi,
        "d_Fo_dx": d_Fo_dx,
        "d_Fo_dt": d_Fo_dt,
        "d_Bi_dx": d_Bi_dx,
        "d_Bi_dt": d_Bi_dt,
    }


def monitor_with_engine(params, t, x):
    state_pred = build_state_for_engine(params, t, x)
    residuals = engine.compute_residuals(state_pred, log_to_python=True)
    return build_loss(residuals)


optimizer = optax.adam(1e-3)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n_t, n_x = 32, 24
    t_flat = jnp.linspace(1.0, t_max, n_t)
    x_flat = jnp.linspace(0.0, L, n_x)
    t_col, x_col = jnp.meshgrid(t_flat, x_flat, indexing="ij")
    t_col = t_col.reshape(-1)
    x_col = x_col.reshape(-1, 1)

    params = init_mlp(key, [2, 48, 48, 1])
    opt_state = optimizer.init(params)

    for step in range(1500):
        params, opt_state, loss = train_step(params, opt_state, t_col, x_col)
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

    fig = visualize(engine.log, backend="matplotlib")
    if fig is not None:
        fig.savefig("slab_cooling_diagnostics.png", dpi=150, bbox_inches="tight")
        print("Saved slab_cooling_diagnostics.png")
