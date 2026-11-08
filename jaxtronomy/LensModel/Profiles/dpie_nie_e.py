"""Difference of two NIE mass ellipses in Cartesian ellipticity."""
from jax import jit
import jax.numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from jaxtronomy.LensModel.Profiles.nie_e import _atan_sqrt_ratio, _geometry, _hessian_components, _log1p_ratio

__all__ = ["DPIE_NIE_E"]


def _dpie_terms(x, y, a, b, e1, e2, center_x, center_y):
    x, y, d, e, v, radius_sq, rx, ry = _geometry(x, y, e1, e2, center_x, center_y)
    ua, ub = jnp.sqrt(a * a + radius_sq), jnp.sqrt(b * b + radius_sq)
    da, db = ua + a, ub + b
    c = jnp.conj(e) / d
    total = ua + ub
    m = 1 + (a + b) / total
    denom = da * db + c * v * v
    g = v * m / denom
    u = c * (b - a)**2 * g * g
    gradients = []
    for vj, rj in ((1 - e, rx), (1j * (1 + e), ry)):
        uaj, ubj = rj / ua, rj / ub
        mj = -(a + b) * (uaj + ubj) / total**2
        dj = uaj * db + da * ubj + 2 * c * v * vj
        gradients.append((vj * m + v * mj - g * dj) / denom)
    return g, u, gradients[0], gradients[1], ua, ub


class DPIE_NIE_E(LensProfileBase):
    """NIE(Ra) - NIE(Rs), with the original sigma0 normalization.

    The physical domain is 0 < Ra < Rs and e1**2 + e2**2 < 1. Radii are not
    swapped or clipped. The factored difference avoids subtracting nearly equal
    deflections; it also removes the explicit 1/(Rs - Ra) normalization.
    """

    param_names = ["sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = dict(
        sigma0=0, Ra=0, Rs=0, e1=-0.5, e2=-0.5, center_x=-100, center_y=-100
    )
    upper_limit_default = dict(
        sigma0=10, Ra=100, Rs=100, e1=0.5, e2=0.5, center_x=100, center_y=100
    )

    @staticmethod
    @jit
    def derivatives(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        g, u, _, _, _, _ = _dpie_terms(x, y, Ra, Rs, e1, e2, center_x, center_y)
        alpha = 2 * sigma0 * Ra * Rs * g * _atan_sqrt_ratio(u)
        return jnp.real(alpha), jnp.imag(alpha)

    @staticmethod
    @jit
    def hessian(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        _, u, gx, gy, _, _ = _dpie_terms(x, y, Ra, Rs, e1, e2, center_x, center_y)
        scale = 2 * sigma0 * Ra * Rs / (1 + u)
        return _hessian_components(scale * gx, scale * gy)

    @staticmethod
    @jit
    def function(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        x, y, d, _, _, _, _, _ = _geometry(x, y, e1, e2, center_x, center_y)
        g, u, _, _, ua, ub = _dpie_terms(x, y, Ra, Rs, e1, e2, 0, 0)
        alpha = 2 * sigma0 * Ra * Rs * g * _atan_sqrt_ratio(u)
        shape = (2 - d) / d
        ha = x * x + y * y + 2 * Ra * ua + 2 * Ra**2 * shape
        gap = Rs - Ra
        # (H_b - H_a)/(b - a), factored before taking the logarithm.
        dh = 2 * (ub + Ra * (Ra + Rs) / (ua + ub) + (Ra + Rs) * shape)
        log_term = jnp.log(ha) + Rs * (dh / ha) * _log1p_ratio(gap * dh / ha)
        return x * jnp.real(alpha) + y * jnp.imag(alpha) + sigma0 * Ra * Rs * log_term

    @staticmethod
    @jit
    def density_lens(r, sigma0, Ra, Rs, e1=0, e2=0, center_x=0, center_y=0):
        r2 = jnp.asarray(r)**2
        return sigma0 * Ra * Rs * (Ra + Rs) / (jnp.pi * (r2 + Ra**2) * (r2 + Rs**2))

    @staticmethod
    @jit
    def mass_3d_lens(r, sigma0, Ra, Rs, e1=0, e2=0, center_x=0, center_y=0):
        r = jnp.asarray(r)
        return 4 * sigma0 * Ra * Rs / (Rs - Ra) * (Rs * jnp.arctan2(r, Rs) - Ra * jnp.arctan2(r, Ra))
