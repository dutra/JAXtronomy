from jax import jit, vmap
import jax.numpy as jnp
import jaxtronomy.Util.param_util as param_util
import jaxtronomy.Util.util as util

from jaxtronomy.LensModel.Profiles.nie import NIEMajorAxis
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["DPIENIE"]


class DPIENIE(LensProfileBase):
    """dPIE profile represented as the difference of two NIE profiles."""

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

    @staticmethod
    @jit
    def function(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = DPIENIE._param_convert(sigma0, Ra, Rs)
        b, q, phi_G, s_pair = DPIENIE._major_axis_params(theta_E, e1, e2, Ra, Rs)
        x_rot, y_rot = DPIENIE._shift_rotate(x, y, center_x, center_y, phi_G)
        f_pair = DPIENIE._axis_function_pair(x_rot, y_rot, b, s_pair, q)
        return f_pair[0] - f_pair[1]

    @staticmethod
    @jit
    def derivatives(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = DPIENIE._param_convert(sigma0, Ra, Rs)
        b, q, phi_G, s_pair = DPIENIE._major_axis_params(theta_E, e1, e2, Ra, Rs)
        x_rot, y_rot = DPIENIE._shift_rotate(x, y, center_x, center_y, phi_G)
        f_x_pair, f_y_pair = DPIENIE._axis_derivatives_pair(x_rot, y_rot, b, s_pair, q)
        f_x_rot = f_x_pair[0] - f_x_pair[1]
        f_y_rot = f_y_pair[0] - f_y_pair[1]
        return util.rotate(f_x_rot, f_y_rot, -phi_G)

    @staticmethod
    @jit
    def hessian(x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        theta_E, Ra, Rs = DPIENIE._param_convert(sigma0, Ra, Rs)
        b, q, phi_G, s_pair = DPIENIE._major_axis_params(theta_E, e1, e2, Ra, Rs)
        x_rot, y_rot = DPIENIE._shift_rotate(x, y, center_x, center_y, phi_G)
        f_xx_pair, f_xy_pair, _, f_yy_pair = DPIENIE._axis_hessian_pair(
            x_rot, y_rot, b, s_pair, q
        )
        f_xx_rot = f_xx_pair[0] - f_xx_pair[1]
        f_xy_rot = f_xy_pair[0] - f_xy_pair[1]
        f_yy_rot = f_yy_pair[0] - f_yy_pair[1]

        kappa = 0.5 * (f_xx_rot + f_yy_rot)
        gamma1_rot = 0.5 * (f_xx_rot - f_yy_rot)
        gamma2_rot = f_xy_rot
        gamma1 = jnp.cos(2 * phi_G) * gamma1_rot - jnp.sin(2 * phi_G) * gamma2_rot
        gamma2 = jnp.sin(2 * phi_G) * gamma1_rot + jnp.cos(2 * phi_G) * gamma2_rot
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    @jit
    def _major_axis_params(theta_E, e1, e2, Ra, Rs):
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E_major_axis = DPIENIE._theta_E_prod_average2major_axis(theta_E, q)
        b = theta_E_major_axis * jnp.sqrt((1 + q**2) / 2)
        s_pair = jnp.asarray([Ra, Rs]) / jnp.sqrt(q)
        return b, q, phi_G, s_pair

    @staticmethod
    @jit
    def _shift_rotate(x, y, center_x, center_y, phi_G):
        x_shift = x - center_x
        y_shift = y - center_y
        return util.rotate(x_shift, y_shift, phi_G)

    @staticmethod
    @jit
    def _axis_function_pair(x_rot, y_rot, b, s_pair, q):
        return vmap(NIEMajorAxis.function, in_axes=(None, None, None, 0, None))(
            x_rot, y_rot, b, s_pair, q
        )

    @staticmethod
    @jit
    def _axis_derivatives_pair(x_rot, y_rot, b, s_pair, q):
        return vmap(NIEMajorAxis.derivatives, in_axes=(None, None, None, 0, None))(
            x_rot, y_rot, b, s_pair, q
        )

    @staticmethod
    @jit
    def _axis_hessian_pair(x_rot, y_rot, b, s_pair, q):
        return vmap(NIEMajorAxis.hessian, in_axes=(None, None, None, 0, None))(
            x_rot, y_rot, b, s_pair, q
        )

    @staticmethod
    @jit
    def _theta_E_prod_average2major_axis(theta_E, q):
        theta_E_major_axis = theta_E / (jnp.sqrt((1.0 + q**2) / (2.0 * q)))
        return theta_E_major_axis

    @staticmethod
    @jit
    def _param_convert(sigma0, Ra, Rs):
        Ra, Rs = DPIENIE._sort_ra_rs(Ra, Rs)
        theta_E = 2.0 * sigma0 * Ra * Rs / (Rs - Ra)
        return theta_E, Ra, Rs

    @staticmethod
    @jit
    def _sort_ra_rs(Ra, Rs):
        Ra = jnp.where(Ra < 1.0e-8, 1.0e-8, Ra)
        Rs = jnp.where(Rs < 1.0e-8, 1.0e-8, Rs)
        ra_sorted = jnp.where(Rs < Ra, Rs, Ra)
        rs_sorted = jnp.where(Rs < Ra, Ra, Rs)
        rs_sorted = jnp.where(rs_sorted < ra_sorted + 1.0e-8, ra_sorted + 2.0e-8, rs_sorted)
        return ra_sorted, rs_sorted
