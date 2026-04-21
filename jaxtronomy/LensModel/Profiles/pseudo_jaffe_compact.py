import numpy as np
from jax import jit, numpy as jnp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["PseudoJaffeCompact"]


class PseudoJaffeCompact(LensProfileBase):
    param_names = ["sigma0", "Ra", "Rs", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "center_x": 100,
        "center_y": 100,
    }

    _s = 0.0001
    _match_frac = 0.8

    @staticmethod
    @jit
    def _sort_ra_rs(Ra, Rs):
        Ra_new = jnp.where(Ra >= Rs, Rs, Ra)
        Rs_new = jnp.where(Ra >= Rs, Ra, Rs)
        Ra_new = jnp.where(Ra_new < 0.00000001, 0.00000001, Ra_new)
        Rs_new = jnp.where(Rs_new < Ra_new + 0.00000001, Ra_new + 0.00000002, Rs_new)
        return Ra_new, Rs_new

    @staticmethod
    @jit
    def _g(u):
        return u / (1 + jnp.sqrt(1 + u**2))

    @staticmethod
    @jit
    def _g_prime(u):
        sqrt_term = jnp.sqrt(1 + u**2)
        return 1.0 / (sqrt_term * (1.0 + sqrt_term))

    @staticmethod
    @jit
    def _g_second(u):
        sqrt_term = jnp.sqrt(1 + u**2)
        return -u * (1.0 + 2.0 * sqrt_term) / (sqrt_term**3 * (1.0 + sqrt_term) ** 2)

    @staticmethod
    @jit
    def alpha_r_standard(r, sigma0, Ra, Rs):
        Ra, Rs = PseudoJaffeCompact._sort_ra_rs(Ra, Rs)
        return 2 * sigma0 * Ra * Rs / (Rs - Ra) * (
            PseudoJaffeCompact._g(r / Ra) - PseudoJaffeCompact._g(r / Rs)
        )

    @staticmethod
    @jit
    def d_alpha_dr_standard(r, sigma0, Ra, Rs):
        Ra, Rs = PseudoJaffeCompact._sort_ra_rs(Ra, Rs)
        prefactor = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        return prefactor * (
            PseudoJaffeCompact._g_prime(r / Ra) / Ra - PseudoJaffeCompact._g_prime(r / Rs) / Rs
        )

    @staticmethod
    @jit
    def d2_alpha_dr2_standard(r, sigma0, Ra, Rs):
        Ra, Rs = PseudoJaffeCompact._sort_ra_rs(Ra, Rs)
        prefactor = 2 * sigma0 * Ra * Rs / (Rs - Ra)
        return prefactor * (
            PseudoJaffeCompact._g_second(r / Ra) / Ra**2 - PseudoJaffeCompact._g_second(r / Rs) / Rs**2
        )

    @staticmethod
    @jit
    def potential_standard(r, sigma0, Ra, Rs):
        Ra, Rs = PseudoJaffeCompact._sort_ra_rs(Ra, Rs)
        return (
            -2
            * sigma0
            * Ra
            * Rs
            / (Rs - Ra)
            * (
                jnp.sqrt(Rs**2 + r**2)
                - jnp.sqrt(Ra**2 + r**2)
                + Ra * jnp.log(Ra + jnp.sqrt(Ra**2 + r**2))
                - Rs * jnp.log(Rs + jnp.sqrt(Rs**2 + r**2))
            )
        )

    @staticmethod
    @jit
    def _quintic_coefficients(alpha_match, d_alpha_match, d2_alpha_match, delta):
        m0 = delta * d_alpha_match
        n0 = delta**2 * d2_alpha_match
        c0 = alpha_match
        c1 = m0
        c2 = 0.5 * n0
        c3 = -10.0 * alpha_match - 6.0 * m0 - 1.5 * n0
        c4 = 15.0 * alpha_match + 8.0 * m0 + 1.5 * n0
        c5 = -6.0 * alpha_match - 3.0 * m0 - 0.5 * n0
        return c0, c1, c2, c3, c4, c5

    @staticmethod
    @jit
    def _shell_values(t, coeffs, delta):
        c0, c1, c2, c3, c4, c5 = coeffs
        alpha = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
        d_alpha_dt = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
        d2_alpha_dt2 = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
        potential_offset = delta * (
            c0 * t
            + c1 * t**2 / 2.0
            + c2 * t**3 / 3.0
            + c3 * t**4 / 4.0
            + c4 * t**5 / 5.0
            + c5 * t**6 / 6.0
        )
        return alpha, d_alpha_dt / delta, d2_alpha_dt2 / delta**2, potential_offset

    @staticmethod
    @jit
    def _radial_quantities(r, sigma0, Ra, Rs):
        Ra, Rs = PseudoJaffeCompact._sort_ra_rs(Ra, Rs)
        r_match = PseudoJaffeCompact._match_frac * Rs
        delta = Rs - r_match
        alpha_standard = PseudoJaffeCompact.alpha_r_standard(r, sigma0, Ra, Rs)
        d_alpha_standard = PseudoJaffeCompact.d_alpha_dr_standard(r, sigma0, Ra, Rs)
        d2_alpha_standard = PseudoJaffeCompact.d2_alpha_dr2_standard(r, sigma0, Ra, Rs)
        phi_standard = PseudoJaffeCompact.potential_standard(r, sigma0, Ra, Rs)

        alpha_match = PseudoJaffeCompact.alpha_r_standard(r_match, sigma0, Ra, Rs)
        d_alpha_match = PseudoJaffeCompact.d_alpha_dr_standard(r_match, sigma0, Ra, Rs)
        d2_alpha_match = PseudoJaffeCompact.d2_alpha_dr2_standard(r_match, sigma0, Ra, Rs)
        phi_match = PseudoJaffeCompact.potential_standard(r_match, sigma0, Ra, Rs)
        coeffs = PseudoJaffeCompact._quintic_coefficients(alpha_match, d_alpha_match, d2_alpha_match, delta)

        t = (r - r_match) / delta
        alpha_shell, d_alpha_shell, d2_alpha_shell, phi_offset = PseudoJaffeCompact._shell_values(t, coeffs, delta)
        phi_shell = phi_match + phi_offset
        _, _, _, phi_cut_offset = PseudoJaffeCompact._shell_values(jnp.ones_like(r), coeffs, delta)
        phi_cut = phi_match + phi_cut_offset

        shell_mask = jnp.logical_and(r > r_match, r < Rs)
        outer_mask = r >= Rs
        alpha = jnp.where(shell_mask, alpha_shell, alpha_standard)
        alpha = jnp.where(outer_mask, 0.0, alpha)
        d_alpha = jnp.where(shell_mask, d_alpha_shell, d_alpha_standard)
        d_alpha = jnp.where(outer_mask, 0.0, d_alpha)
        d2_alpha = jnp.where(shell_mask, d2_alpha_shell, d2_alpha_standard)
        d2_alpha = jnp.where(outer_mask, 0.0, d2_alpha)
        phi = jnp.where(shell_mask, phi_shell, phi_standard)
        phi = jnp.where(outer_mask, phi_cut, phi)
        return alpha, d_alpha, d2_alpha, phi

    @staticmethod
    @jit
    def alpha_r(r, sigma0, Ra, Rs):
        alpha, _, _, _ = PseudoJaffeCompact._radial_quantities(r, sigma0, Ra, Rs)
        return alpha

    @staticmethod
    @jit
    def d_alpha_dr(r, sigma0, Ra, Rs):
        _, d_alpha, _, _ = PseudoJaffeCompact._radial_quantities(r, sigma0, Ra, Rs)
        return d_alpha

    @staticmethod
    @jit
    def d2_alpha_dr2(r, sigma0, Ra, Rs):
        _, _, d2_alpha, _ = PseudoJaffeCompact._radial_quantities(r, sigma0, Ra, Rs)
        return d2_alpha

    @staticmethod
    @jit
    def function(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        _, _, _, phi = PseudoJaffeCompact._radial_quantities(r, sigma0, Ra, Rs)
        return phi

    @staticmethod
    @jit
    def derivatives(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        r_safe = jnp.where(r < PseudoJaffeCompact._s, PseudoJaffeCompact._s, r)
        alpha_r = PseudoJaffeCompact.alpha_r(r_safe, sigma0, Ra, Rs)
        f_x = alpha_r * x_ / r_safe
        f_y = alpha_r * y_ / r_safe
        zeros = jnp.zeros_like(f_x)
        return jnp.where(r == 0, zeros, f_x), jnp.where(r == 0, zeros, f_y)

    @staticmethod
    @jit
    def hessian(x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        x_ = jnp.array(x - center_x, dtype=float)
        y_ = jnp.array(y - center_y, dtype=float)
        r = jnp.sqrt(x_**2 + y_**2)
        r_safe = jnp.where(r < PseudoJaffeCompact._s, PseudoJaffeCompact._s, r)
        alpha = PseudoJaffeCompact.alpha_r(r_safe, sigma0, Ra, Rs)
        d_alpha = PseudoJaffeCompact.d_alpha_dr(r_safe, sigma0, Ra, Rs)
        f_xx = d_alpha * x_**2 / r_safe**2 + alpha * y_**2 / r_safe**3
        f_yy = d_alpha * y_**2 / r_safe**2 + alpha * x_**2 / r_safe**3
        f_xy = x_ * y_ * (d_alpha / r_safe**2 - alpha / r_safe**3)
        zeros = jnp.zeros_like(f_xx)
        return (
            jnp.where(r == 0, zeros, f_xx),
            jnp.where(r == 0, zeros, f_xy),
            jnp.where(r == 0, zeros, f_xy),
            jnp.where(r == 0, zeros, f_yy),
        )

    @staticmethod
    @jit
    def mass_3d_lens(r, sigma0, Ra, Rs):
        rho0 = PseudoJaffeCompact.sigma2rho(sigma0, Ra, Rs)
        return (
            4
            * np.pi
            * rho0
            * Ra**2
            * Rs**2
            / (Rs**2 - Ra**2)
            * (Rs * jnp.arctan(r / Rs) - Ra * jnp.arctan(r / Ra))
        )

    @staticmethod
    @jit
    def rho2sigma(rho0, Ra, Rs):
        return np.pi * rho0 * Ra * Rs / (Rs + Ra)

    @staticmethod
    @jit
    def sigma2rho(sigma0, Ra, Rs):
        return (Rs + Ra) / Ra / Rs / np.pi * sigma0
