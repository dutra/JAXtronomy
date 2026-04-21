from jaxtronomy.LensModel.Profiles.pseudo_jaffe_compact import PseudoJaffeCompact
import jaxtronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from jax import jit, numpy as jnp

__all__ = ["PseudoJaffeEllipsePotentialCompact"]


class PseudoJaffeEllipsePotentialCompact(LensProfileBase):
    param_names = ["sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    spherical = PseudoJaffeCompact()
    _diff = 0.000001

    @staticmethod
    @jit
    def function(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )
        return PseudoJaffeEllipsePotentialCompact.spherical.function(x_, y_, sigma0, Ra, Rs)

    @staticmethod
    @jit
    def derivatives(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )
        e = param_util.q2e(q)
        cos_phi = jnp.cos(phi_G)
        sin_phi = jnp.sin(phi_G)
        f_x_prim, f_y_prim = PseudoJaffeEllipsePotentialCompact.spherical.derivatives(
            x_, y_, sigma0, Ra, Rs, center_x=0, center_y=0
        )
        f_x_prim *= jnp.sqrt(1 - e)
        f_y_prim *= jnp.sqrt(1 + e)
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y

    @staticmethod
    @jit
    def hessian(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        alpha_ra, alpha_dec = PseudoJaffeEllipsePotentialCompact.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )
        diff = PseudoJaffeEllipsePotentialCompact._diff
        alpha_ra_dx, alpha_dec_dx = PseudoJaffeEllipsePotentialCompact.derivatives(
            x + diff, y, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )
        alpha_ra_dy, alpha_dec_dy = PseudoJaffeEllipsePotentialCompact.derivatives(
            x, y + diff, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_xy, f_yx, f_yy

    @staticmethod
    @jit
    def mass_3d_lens(r, sigma0, Ra, Rs, e1=0, e2=0):
        return PseudoJaffeEllipsePotentialCompact.spherical.mass_3d_lens(r, sigma0, Ra, Rs)
