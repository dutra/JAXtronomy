"""NIE in Cartesian ellipticity, including its differentiable circular limit."""
from jax import jit
import jax.numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["NIE_E"]


def _atan_sqrt_ratio(u):
    """atan(sqrt(u))/sqrt(u), evaluated without a singular intermediate at zero."""
    small = jnp.abs(u) < 0.01
    root = jnp.sqrt(jnp.where(small, 1.0 + 0j, u))
    series = jnp.full_like(u, -1.0 / 19.0)
    for n in range(8, -1, -1):
        series = (-1.0) ** n / (2 * n + 1) + u * series
    return jnp.where(small, series, jnp.arctan(root) / root)


def _log1p_ratio(u):
    """log1p(u)/u with regular radius derivatives when two cores coincide."""
    small = jnp.abs(u) < 0.01
    safe = jnp.where(small, 1.0, u)
    series = jnp.full_like(u, -1.0 / 10.0)
    for n in range(8, -1, -1):
        series = (-1.0) ** n / (n + 1) + u * series
    return jnp.where(small, series, jnp.log1p(safe) / safe)


def _geometry(x, y, e1, e2, center_x, center_y):
    x, y = jnp.asarray(x) - center_x, jnp.asarray(y) - center_y
    e_sq = e1 * e1 + e2 * e2
    d = 1.0 - e_sq
    qxx = (1 + e_sq - 2 * e1) / d
    qxy = -2 * e2 / d
    qyy = (1 + e_sq + 2 * e1) / d
    rx, ry = qxx * x + qxy * y, qxy * x + qyy * y
    radius_sq = x * rx + y * ry
    e = e1 + 1j * e2
    v = (x + 1j * y) - e * (x - 1j * y)
    return x, y, d, e, v, radius_sq, rx, ry


def _nie_terms(x, y, r, e1, e2, center_x, center_y):
    x, y, d, e, v, radius_sq, rx, ry = _geometry(x, y, e1, e2, center_x, center_y)
    radius = jnp.sqrt(r * r + radius_sq)
    denom = radius + r
    h = v / denom
    u = jnp.conj(e) * h * h / d
    hx = ((1 - e) - h * rx / radius) / denom
    hy = (1j * (1 + e) - h * ry / radius) / denom
    return h, u, hx, hy, radius


def _hessian_components(ax, ay):
    cross = 0.5 * (jnp.imag(ax) + jnp.real(ay))
    return jnp.real(ax), cross, cross, jnp.imag(ay)


class NIE_E(LensProfileBase):
    """The NIE mass ellipse, with no angle conversion.

    Uses the usual product-average theta_E and s_scale conventions. The regular
    domain is e1**2 + e2**2 < 1 and s_scale > 0. With zero core the origin is
    physically singular. Invalid shapes are not clipped into the valid domain.
    """

    param_names = ["theta_E", "e1", "e2", "s_scale", "center_x", "center_y"]
    lower_limit_default = dict(
        theta_E=0, e1=-0.5, e2=-0.5, s_scale=0, center_x=-100, center_y=-100
    )
    upper_limit_default = dict(
        theta_E=10, e1=0.5, e2=0.5, s_scale=100, center_x=100, center_y=100
    )

    @staticmethod
    @jit
    def derivatives(x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        h, u, _, _, _ = _nie_terms(x, y, s_scale, e1, e2, center_x, center_y)
        alpha = theta_E * h * _atan_sqrt_ratio(u)
        return jnp.real(alpha), jnp.imag(alpha)

    @staticmethod
    @jit
    def hessian(x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        _, u, hx, hy, _ = _nie_terms(x, y, s_scale, e1, e2, center_x, center_y)
        # d[h F(c h**2)] / dh = 1 / (1 + c h**2).
        return _hessian_components(theta_E * hx / (1 + u), theta_E * hy / (1 + u))

    @staticmethod
    @jit
    def function(x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        x, y, d, _, _, _, _, _ = _geometry(x, y, e1, e2, center_x, center_y)
        h, u, _, _, radius = _nie_terms(x, y, s_scale, e1, e2, 0, 0)
        alpha = theta_E * h * _atan_sqrt_ratio(u)
        log_arg = x * x + y * y + 2 * s_scale * radius + 2 * s_scale**2 * (2 - d) / d
        return x * jnp.real(alpha) + y * jnp.imag(alpha) - theta_E * s_scale / 2 * jnp.log(log_arg)

    @staticmethod
    @jit
    def density_lens(r, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        return theta_E / (2 * jnp.pi * (jnp.asarray(r)**2 + s_scale**2))

    @staticmethod
    @jit
    def mass_3d_lens(r, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        r = jnp.asarray(r)
        return 2 * theta_E * (r - s_scale * jnp.arctan2(r, s_scale))
