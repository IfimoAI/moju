"""Tests for moju.piratio.Operators (differential operators on fields)."""

import pytest
import jax.numpy as jnp
from moju.piratio import Operators


def _scalar_squared(params, x):
    """f(x) = sum(x^2). Gradient 2*x, Laplacian 2*dim."""
    return jnp.sum(x ** 2)


def _linear_scalar(params, x):
    """f(x) = sum(x). Gradient ones, Laplacian 0."""
    return jnp.sum(x)


def _vector_identity(params, x):
    """Vector field F(x)=x. Jacobian I, divergence = dim."""
    return x


class TestOperatorsGradient:
    """Gradient of scalar field."""

    def test_gradient_of_sum_squared(self, rtol, atol):
        """grad(sum(x^2)) = 2*x."""
        params = {}
        x = jnp.array([1.0, 2.0, 3.0])
        grad = Operators.gradient(_scalar_squared, params, x)
        expected = 2.0 * x
        assert jnp.allclose(grad, expected, rtol=rtol, atol=atol)

    def test_gradient_of_linear(self, rtol, atol):
        """grad(sum(x)) = ones."""
        params = {}
        x = jnp.array([1.0, -1.0, 0.0])
        grad = Operators.gradient(_linear_scalar, params, x)
        assert jnp.allclose(grad, jnp.ones(3), rtol=rtol, atol=atol)


class TestOperatorsLaplacian:
    """Laplacian of scalar field."""

    def test_laplacian_of_sum_squared(self, rtol, atol):
        """Laplacian(sum(x^2)) = 2*dim."""
        params = {}
        x = jnp.array([1.0, 2.0])
        lap = Operators.laplacian(_scalar_squared, params, x)
        assert jnp.allclose(lap, 4.0, rtol=rtol, atol=atol)  # 2*2

    def test_laplacian_of_linear_is_zero(self, rtol, atol):
        """Laplacian(sum(x)) = 0."""
        params = {}
        x = jnp.array([1.0, 2.0, 3.0])
        lap = Operators.laplacian(_linear_scalar, params, x)
        assert jnp.allclose(lap, 0.0, rtol=rtol, atol=atol)


class TestOperatorsJacobianAndDivergence:
    """Jacobian and divergence of vector field."""

    def test_jacobian_of_identity(self, rtol, atol):
        """For F(x)=x, Jacobian is I."""
        params = {}
        x = jnp.array([1.0, 2.0])
        jac = Operators.jacobian(_vector_identity, params, x)
        assert jnp.allclose(jac, jnp.eye(2), rtol=rtol, atol=atol)

    def test_divergence_of_identity(self, rtol, atol):
        """For F(x)=x, div F = dim."""
        params = {}
        x = jnp.array([1.0, 2.0, 3.0])
        div = Operators.divergence(_vector_identity, params, x)
        assert jnp.allclose(div, 3.0, rtol=rtol, atol=atol)


class TestOperatorsAdvection:
    """Advection (u·grad)u."""

    def test_advection_formula(self, rtol, atol):
        """advection(u, u_grad) = (u_grad)^T @ u."""
        u = jnp.array([1.0, 2.0])
        u_grad = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        adv = Operators.advection(u, u_grad)
        expected = jnp.einsum("ij,j->i", u_grad, u)
        assert jnp.allclose(adv, expected, rtol=rtol, atol=atol)


class TestOperatorsBatch:
    """Batched evaluation."""

    def test_gradient_batch_same_as_scalar(self, rtol, atol):
        """Batched gradient matches scalar at each point."""
        params = {}
        x_batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        grad_batch = Operators.gradient(_scalar_squared, params, x_batch)
        assert grad_batch.shape == (3, 2)
        for i in range(3):
            gi = Operators.gradient(_scalar_squared, params, x_batch[i])
            assert jnp.allclose(grad_batch[i], gi, rtol=rtol, atol=atol)

    def test_curl_2d_scalar_output(self, rtol, atol):
        """2D curl of (y, -x) is scalar: d(-x)/dx - dy/dy = -1 - 1 = -2."""
        def field(params, x):
            return jnp.array([x[1], -x[0]])
        params = {}
        x = jnp.array([1.0, 2.0])
        curl = Operators.curl_2d(field, params, x)
        # Jacobian of (y, -x) is [[0, 1], [-1, 0]]; curl = J[1,0] - J[0,1] = -1 - 1 = -2
        assert jnp.allclose(curl, -2.0, rtol=rtol, atol=atol)
