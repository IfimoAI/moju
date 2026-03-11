import functools
import jax
import jax.numpy as jnp


class Operators:
    """
    JIT-compatible differential operators for SciML.
    Optimized for use with neural-network functions f(params, x) or f(params, t, x).
    Inputs may have a leading batch dimension (N, d); single-point (d,) and batched (N, d) both supported.
    """

    # --- Single-point implementations (JIT'd, used by public methods and by vmap) ---

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _gradient_single(func, params, x):
        return jax.grad(func, argnums=1)(params, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _jacobian_single(func, params, x):
        return jax.jacfwd(func, argnums=1)(params, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _divergence_single(func, params, x):
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return jnp.trace(jac, axis1=-2, axis2=-1)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _laplacian_single(func, params, x):
        dim = x.shape[-1]
        eye = jnp.eye(dim)

        def grad_fn(x_inner):
            return jax.jacfwd(func, argnums=1)(params, x_inner)

        def jvp_ith(ei):
            _, tan = jax.jvp(grad_fn, (x,), (ei,))
            return tan

        tangents = jax.vmap(jvp_ith)(eye)
        if tangents.ndim == 2:
            return jnp.trace(tangents)
        return jnp.einsum("ijj->i", tangents)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _curl_2d_single(func, params, x):
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return jac[..., 1, 0] - jac[..., 0, 1]

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _curl_3d_single(func, params, x):
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return jnp.stack(
            [
                jac[..., 2, 1] - jac[..., 1, 2],
                jac[..., 0, 2] - jac[..., 2, 0],
                jac[..., 1, 0] - jac[..., 0, 1],
            ],
            axis=-1,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _time_derivative_single(func, params, t, x):
        return jax.jacfwd(func, argnums=1)(params, t, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _time_derivative_second_single(func, params, t, x):
        first_dt = jax.jacfwd(func, argnums=1)
        return jax.jacfwd(first_dt, argnums=1)(params, t, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def _symmetric_gradient_single(func, params, x):
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return 0.5 * (jac + jnp.swapaxes(jac, -2, -1))

    # --- Public API: single-point or batched via dispatch ---

    @staticmethod
    def gradient(func, params, x):
        """
        Gradient of a scalar field with respect to input x.

        :param func: Callable (params, x) -> scalar.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (N, d) for batch.
        :return: Gradient vector. Same shape as x.

        Use case: Pressure or temperature gradients.
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._gradient_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._gradient_single(func, params, x)

    @staticmethod
    def jacobian(func, params, x):
        """
        Jacobian of a vector field with respect to input x.

        :param func: Callable (params, x) -> vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (N, d) for batch.
        :return: Jacobian matrix. Shape (out_dim, in_dim) or (N, out_dim, in_dim).

        Use case: Velocity gradients (du/dx).
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._jacobian_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._jacobian_single(func, params, x)

    @staticmethod
    def divergence(func, params, x):
        """
        Divergence of a vector field (trace of the Jacobian).

        :param func: Callable (params, x) -> vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (N, d) for batch.
        :return: Scalar divergence. Shape () or (N,).

        Use case: Mass continuity (div u).
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._divergence_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._divergence_single(func, params, x)

    @staticmethod
    def laplacian(func, params, x):
        """
        Laplacian (trace of Hessian) using JVP; faster than full Hessian for high-dimensional x.

        :param func: Callable (params, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (N, d) for batch.
        :return: Laplacian. Scalar or (d_out,) per point; shape (N,) or (N, d_out) when batched.

        Use case: Viscous terms or heat diffusion.
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._laplacian_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._laplacian_single(func, params, x)

    @staticmethod
    def curl_2d(func, params, x):
        """
        Scalar curl (vorticity) in 2D: dv/dx - du/dy.

        :param func: Callable (params, x) -> [u, v]. Shape (..., 2).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (2,) or (N, 2) for batch.
        :return: Scalar vorticity. Shape () or (N,).

        Use case: 2D incompressible flow vorticity.
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._curl_2d_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._curl_2d_single(func, params, x)

    @staticmethod
    def curl_3d(func, params, x):
        """
        Vector curl in 3D: (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy).

        :param func: Callable (params, x) -> [u, v, w]. Shape (..., 3).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (3,) or (N, 3) for batch.
        :return: Vorticity vector. Shape (3,) or (N, 3).

        Use case: 3D flow vorticity; Faraday's law (curl E).
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._curl_3d_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._curl_3d_single(func, params, x)

    @staticmethod
    @jax.jit
    def advection(u, u_grad):
        """
        Convective acceleration (u·grad)u.

        :param u: Velocity vector [m/s]. Shape (..., d).
        :param u_grad: Velocity Jacobian from Operators.jacobian [1/s]. Shape (..., d, d).
        :return: Advection vector [m/s^2]. Shape (..., d).

        Use case: Momentum equations (Navier-Stokes, Euler, Burgers).
        """
        return jnp.einsum("...ij,...j->...i", u_grad, u)

    @staticmethod
    def time_derivative(func, params, t, x):
        """
        First time derivative d(phi)/dt for unsteady PDEs.

        :param func: Callable (params, t, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param t: Time [s]. Scalar or (N,) when x is batched.
        :param x: Spatial coordinate vector. Shape (..., d) or (N, d) for batch.
        :return: Time derivative of func output. Same shape as func output; (N, ...) when batched.

        Use case: Unsteady continuity, momentum, heat, diffusion, Faraday, Burgers.
        """
        if x.ndim > 1:
            if jnp.ndim(t) == 0:
                return jax.vmap(
                    lambda xi: Operators._time_derivative_single(func, params, t, xi),
                    in_axes=(0,),
                )(x)
            return jax.vmap(
                Operators._time_derivative_single,
                in_axes=(None, None, 0, 0),
            )(func, params, t, x)
        return Operators._time_derivative_single(func, params, t, x)

    @staticmethod
    def time_derivative_second(func, params, t, x):
        """
        Second time derivative d²(phi)/dt².

        :param func: Callable (params, t, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param t: Time [s]. Scalar or (N,) when x is batched.
        :param x: Spatial coordinate vector. Shape (..., d) or (N, d) for batch.
        :return: Second time derivative. Same shape as func output; (N, ...) when batched.

        Use case: Wave equation, acoustics, second-order-in-time dynamics.
        """
        if x.ndim > 1:
            if jnp.ndim(t) == 0:
                return jax.vmap(
                    lambda xi: Operators._time_derivative_second_single(
                        func, params, t, xi
                    ),
                    in_axes=(0,),
                )(x)
            return jax.vmap(
                Operators._time_derivative_second_single,
                in_axes=(None, None, 0, 0),
            )(func, params, t, x)
        return Operators._time_derivative_second_single(func, params, t, x)

    @staticmethod
    def symmetric_gradient(func, params, x):
        """
        Symmetric part of the Jacobian (strain tensor from displacement).

        :param func: Callable (params, x) -> displacement vector. Shape (..., d).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (N, d) for batch.
        :return: Symmetric gradient (strain tensor). Shape (d, d) or (N, d, d).

        Use case: Linear elasticity (Hooke), strain from displacement field.
        """
        if x.ndim > 1:
            return jax.vmap(
                functools.partial(Operators._symmetric_gradient_single, func, params),
                in_axes=(0,),
            )(x)
        return Operators._symmetric_gradient_single(func, params, x)
