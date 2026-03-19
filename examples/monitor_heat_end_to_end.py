#!/usr/bin/env python3
"""
End-to-end monitor example: 1D heat diffusion with NN T(t,x).

Shows:
  - Path A style state building (compute T, T_t, T_xx from the NN)
  - Laws.fourier_conduction residual + build_loss
  - Model/Group chain audits (fo, bi, thermal_diffusivity) with both chain_dx and chain_dt
  - Required keys introspection
  - audit() PDF export when moju[report] installed
"""

import jax
import jax.numpy as jnp
import optax

from moju.piratio import Operators
from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor import ResidualEngine, audit, build_loss


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


def T_field(params, t, x):
    # tx: (N,2) with columns [t,x]
    if t.ndim == 1 and x.ndim == 2:
        tx = jnp.concatenate([t[:, None], x], axis=-1)
    else:
        tx = jnp.concatenate([jnp.asarray(t)[..., None], jnp.asarray(x)], axis=-1)
    return mlp(params, tx)[..., 0]


def build_state(params, t, x, *, L, h, cp, k_solid, rho_ref):
    T = T_field(params, t, x)
    T_t = Operators.time_derivative(T_field, params, t, x)

    def lap_body(ti, xi):
        return Operators.laplacian(lambda p, x_in: T_field(p, ti, x_in), params, xi)

    T_xx = jax.vmap(lap_body)(t, x)

    def grad_body(ti, xi):
        g = Operators.gradient(lambda p, x_in: T_field(p, ti, x_in), params, xi)
        return g[0] if g.shape == (1,) else g

    T_x = jax.vmap(grad_body)(t, x)

    # Simple constitutive k(T), rho(T)
    kappa = k_solid * (1.0 + 0.001 * (T - 300.0))
    rho = rho_ref * (1.0 - 0.0001 * (T - 300.0))
    alpha = kappa / (rho * cp)

    # Derivatives needed for chain audits (same math as slab demo)
    dk_dT = 0.001 * k_solid
    drho_dT = -0.0001 * rho_ref
    d_k_dx = dk_dT * T_x
    d_k_dt = dk_dT * T_t
    d_rho_dx = drho_dT * T_x
    d_rho_dt = drho_dT * T_t
    d_alpha_dx = (1.0 / (rho * cp)) * d_k_dx - (kappa / (rho**2 * cp)) * d_rho_dx
    d_alpha_dt = (1.0 / (rho * cp)) * d_k_dt - (kappa / (rho**2 * cp)) * d_rho_dt

    Lb = jnp.broadcast_to(L, t.shape)
    hb = jnp.broadcast_to(h, t.shape)
    Fo = alpha * t / (Lb**2)
    Bi = hb * Lb / kappa
    d_t_dt = jnp.ones_like(t)
    d_Fo_dx = (t / (Lb**2)) * d_alpha_dx
    d_Fo_dt = (t / (Lb**2)) * d_alpha_dt + (alpha / (Lb**2)) * d_t_dt
    d_Bi_dx = -(hb * Lb / (kappa**2)) * d_k_dx
    d_Bi_dt = -(hb * Lb / (kappa**2)) * d_k_dt

    return {
        "T": T,
        "T_t": T_t,
        "T_xx": T_xx,
        "t": t,
        "L": Lb,
        "h": hb,
        "cp": jnp.broadcast_to(cp, t.shape),
        "kappa": kappa,
        "k": kappa,
        "rho": rho,
        "alpha": alpha,
        "k_solid": kappa,
        "Fo": Fo,
        "Bi": Bi,
        # derivatives
        "d_k_dx": d_k_dx,
        "d_k_dt": d_k_dt,
        "d_rho_dx": d_rho_dx,
        "d_rho_dt": d_rho_dt,
        "d_alpha_dx": d_alpha_dx,
        "d_alpha_dt": d_alpha_dt,
        "d_t_dt": d_t_dt,
        "d_Fo_dx": d_Fo_dx,
        "d_Fo_dt": d_Fo_dt,
        "d_Bi_dx": d_Bi_dx,
        "d_Bi_dt": d_Bi_dt,
    }


def main():
    L = 0.02
    h = 500.0
    cp = 900.0
    k_solid = 200.0
    rho_ref = 2700.0

    engine = ResidualEngine(
        constants={"cp": cp, "h": h},
        laws=[
            {
                "name": "fourier_conduction",
                "state_map": {"T_t": "T_t", "T_laplacian": "T_xx", "fo": "Fo", "t": "t", "L": "L"},
                "fn": Laws.fourier_conduction,
            }
        ],
        groups=[
            {"name": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}, "output_key": "Fo", "fn": Groups.fo},
            {"name": "bi", "state_map": {"h": "h", "L": "L", "k_solid": "kappa"}, "output_key": "Bi", "fn": Groups.bi},
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
            {"name": "fo", "output_key": "Fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}, "predicted_spatial": ["alpha"], "predicted_temporal": ["alpha", "t"]},
            {"name": "bi", "output_key": "Bi", "state_map": {"h": "h", "L": "L", "k_solid": "kappa"}, "predicted_spatial": ["kappa"], "predicted_temporal": ["kappa"]},
        ],
    )

    print("Required state keys:", sorted(engine.required_state_keys()))
    print("Required derivative keys:", sorted(engine.required_derivative_keys()))

    key = jax.random.PRNGKey(0)
    params = init_mlp(key, [2, 32, 32, 1])
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    # Collocation
    t = jnp.linspace(1.0, 10.0, 64)
    x = jnp.linspace(0.0, L, 64)[:, None]
    t_col, x_col = jnp.meshgrid(t, x[:, 0], indexing="ij")
    t_col = t_col.reshape(-1)
    x_col = x_col.reshape(-1, 1)

    def loss_fn(p):
        state = build_state(p, t_col, x_col, L=L, h=h, cp=cp, k_solid=k_solid, rho_ref=rho_ref)
        residuals = engine.compute_residuals(state, log_to_python=False)
        return build_loss(residuals)

    for _ in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    state = build_state(params, t_col, x_col, L=L, h=h, cp=cp, k_solid=k_solid, rho_ref=rho_ref)
    residuals = engine.compute_residuals(state, log_to_python=True)
    print("Residual categories:", residuals.keys())

    report = audit(engine.log, export_dir=".", last_residual_dict=residuals, save_residuals=True, model_name="Heat-1D", model_id="heat-demo")
    print("Overall score:", report["overall_admissibility_score"])
    print("Per-category:", report.get("per_category", {}))


if __name__ == "__main__":
    main()

