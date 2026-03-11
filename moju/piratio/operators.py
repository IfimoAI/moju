import functools
import jax
import jax.numpy as jnp


class Operators:
    """
    JIT-compatible differential operators for SciML.
    Optimized for use with neural-network functions f(params, x) or f(params, t, x).
    """

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def gradient(func, params, x):
        """
        Gradient of a scalar field with respect to input x.

        :param func: Callable (params, x) -> scalar.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (..., d).
        :return: Gradient vector. Same shape as x.

        Use case: Pressure or temperature gradients. For batched x use jax.vmap(partial(Operators.gradient, func, params), in_axes=(0,)).
        """
        return jax.grad(func, argnums=1)(params, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def jacobian(func, params, x):
        """
        Jacobian of a vector field with respect to input x.

        :param func: Callable (params, x) -> vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (..., d).
        :return: Jacobian matrix. Shape (..., out_dim, in_dim).

        Use case: Velocity gradients (du/dx). For batched x use jax.vmap(partial(Operators.jacobian, func, params), in_axes=(0,)).
        """
        return jax.jacfwd(func, argnums=1)(params, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def divergence(func, params, x):
        """
        Divergence of a vector field (trace of the Jacobian).

        :param func: Callable (params, x) -> vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (..., d).
        :return: Scalar divergence. Shape (...).

        Use case: Mass continuity (div u).
        """
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return jnp.trace(jac, axis1=-2, axis2=-1)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def laplacian(func, params, x):
        """
        Laplacian (trace of Hessian) using JVP; faster than full Hessian for high-dimensional x.

        :param func: Callable (params, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,).
        :return: Laplacian. Scalar if func returns scalar; shape (d_out,) if func returns vector.

        Use case: Viscous terms or heat diffusion. For batched x use jax.vmap(partial(Operators.laplacian, func, params), in_axes=(0,)).
        """
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
    def curl_2d(func, params, x):
        """
        Scalar curl (vorticity) in 2D: dv/dx - du/dy.

        :param func: Callable (params, x) -> [u, v]. Shape (..., 2).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (2,) or (..., 2).
        :return: Scalar vorticity. Shape (...).

        Use case: 2D incompressible flow vorticity.
        """
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return jac[..., 1, 0] - jac[..., 0, 1]

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def curl_3d(func, params, x):
        """
        Vector curl in 3D: (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy).

        :param func: Callable (params, x) -> [u, v, w]. Shape (..., 3).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (3,) or (..., 3).
        :return: Vorticity vector. Shape (..., 3).

        Use case: 3D flow vorticity; Faraday's law (curl E).
        """
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
    @functools.partial(jax.jit, static_argnums=0)
    def time_derivative(func, params, t, x):
        """
        First time derivative d(phi)/dt for unsteady PDEs.

        :param func: Callable (params, t, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param t: Time [s].
        :param x: Spatial coordinate vector. Shape (..., d).
        :return: Time derivative of func output. Same shape as func(params, t, x).

        Use case: Unsteady continuity, momentum, heat, diffusion, Faraday, Burgers.
        """
        return jax.jacfwd(func, argnums=1)(params, t, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def time_derivative_second(func, params, t, x):
        """
        Second time derivative d²(phi)/dt².

        :param func: Callable (params, t, x) -> scalar or vector.
        :param params: Pytree of NN weights.
        :param t: Time [s].
        :param x: Spatial coordinate vector. Shape (..., d).
        :return: Second time derivative. Same shape as func(params, t, x).

        Use case: Wave equation, acoustics, second-order-in-time dynamics.
        """
        first_dt = jax.jacfwd(func, argnums=1)
        return jax.jacfwd(first_dt, argnums=1)(params, t, x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=0)
    def symmetric_gradient(func, params, x):
        """
        Symmetric part of the Jacobian (strain tensor from displacement).

        :param func: Callable (params, x) -> displacement vector. Shape (..., d).
        :param params: Pytree of NN weights.
        :param x: Input coordinate vector. Shape (d,) or (..., d).
        :return: Symmetric gradient (strain tensor). Shape (..., d, d).

        Use case: Linear elasticity (Hooke), strain from displacement field.
        """
        jac = jax.jacfwd(func, argnums=1)(params, x)
        return 0.5 * (jac + jnp.swapaxes(jac, -2, -1))
