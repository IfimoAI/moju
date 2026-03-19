#!/usr/bin/env python3
"""
End-to-end monitor example: 1D viscous Burgers equation with NN u(t,x).

Shows:
  - NN u(params,t,x)
  - Operators: u_t, u_x, u_xx
  - Laws.burgers_equation residual
  - Required keys introspection
  - audit() PDF export when moju[report] installed
"""

import jax
import jax.numpy as jnp
import optax

from moju.piratio import Operators
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


def u_field(params, t, x):
    tx = jnp.concatenate([t[:, None], x], axis=-1)
    return mlp(params, tx)[..., 0]


def build_state(params, t, x, *, Re, U, L):
    u = u_field(params, t, x)
    u_t = Operators.time_derivative(u_field, params, t, x)

    def grad_body(ti, xi):
        g = Operators.gradient(lambda p, x_in: u_field(p, jnp.asarray([ti]), x_in[None, :])[0], params, xi)
        return g[0] if g.shape == (1,) else g

    u_x = jax.vmap(grad_body)(t, x)

    def lap_body(ti, xi):
        return Operators.laplacian(
            lambda p, x_in: u_field(p, jnp.asarray([ti]), x_in[None, :])[0], params, xi
        )

    u_xx = jax.vmap(lap_body)(t, x)

    # burgers_equation expects vector forms (..., d)
    u_vec = u[:, None]
    u_t_vec = u_t[:, None]
    u_grad = u_x[:, None, None]
    u_lap = u_xx[:, None]

    return {
        "u": u_vec,
        "u_t": u_t_vec,
        "u_grad": u_grad,
        "u_laplacian": u_lap,
        "Re": jnp.broadcast_to(Re, (u.shape[0],)),
        "U": jnp.broadcast_to(U, (u.shape[0],)),
        "L": jnp.broadcast_to(L, (u.shape[0],)),
    }


def main():
    Re = 100.0
    U = 1.0
    L = 1.0

    engine = ResidualEngine(
        laws=[
            {
                "name": "burgers_equation",
                "state_map": {
                    "u_t": "u_t",
                    "u": "u",
                    "u_grad": "u_grad",
                    "u_laplacian": "u_laplacian",
                    "re": "Re",
                    "U": "U",
                    "L": "L",
                },
                "fn": Laws.burgers_equation,
            }
        ],
    )

    print("Required state keys:", sorted(engine.required_state_keys()))

    key = jax.random.PRNGKey(0)
    params = init_mlp(key, [2, 32, 32, 1])
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    t = jnp.linspace(0.1, 1.0, 64)
    x = jnp.linspace(0.0, L, 64)[:, None]
    t_col, x_col = jnp.meshgrid(t, x[:, 0], indexing="ij")
    t_col = t_col.reshape(-1)
    x_col = x_col.reshape(-1, 1)

    def loss_fn(p):
        state = build_state(p, t_col, x_col, Re=Re, U=U, L=L)
        residuals = engine.compute_residuals(state, log_to_python=False)
        return build_loss(residuals)

    for _ in range(5):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    state = build_state(params, t_col, x_col, Re=Re, U=U, L=L)
    residuals = engine.compute_residuals(state, log_to_python=True)
    print("Loss:", float(build_loss(residuals)))

    report = audit(engine.log, export_dir=".", last_residual_dict=residuals, save_residuals=True, model_name="Burgers-1D", model_id="burgers-demo")
    print("Overall score:", report["overall_admissibility_score"])


if __name__ == "__main__":
    main()

