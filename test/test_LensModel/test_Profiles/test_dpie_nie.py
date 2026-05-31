import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.LensModel.lens_model_bulk import LensModelBulk
from jaxtronomy.LensModel.Profiles.dpie_nie import DPIENIE
from lenstronomy.LensModel.Profiles.dpie_nie import DPIENIE as DPIENIE_ref


class TestDPIENIE(object):
    def setup_method(self):
        self.profile = DPIENIE()
        self.profile_ref = DPIENIE_ref()

    def test_matches_lenstronomy_profile(self):
        x = np.array([0.2, 1.0, 3.0])
        y = np.array([0.4, 2.0, -1.0])
        kwargs = {
            "sigma0": 1.2,
            "Ra": 0.15,
            "Rs": 3.0,
            "e1": 0.05,
            "e2": -0.02,
            "center_x": 0.1,
            "center_y": -0.1,
        }

        npt.assert_allclose(
            np.asarray(self.profile.function(jnp.asarray(x), jnp.asarray(y), **kwargs)),
            self.profile_ref.function(x, y, **kwargs),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        npt.assert_allclose(
            np.asarray(self.profile.derivatives(jnp.asarray(x), jnp.asarray(y), **kwargs)),
            self.profile_ref.derivatives(x, y, **kwargs),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        npt.assert_allclose(
            np.asarray(self.profile.hessian(jnp.asarray(x), jnp.asarray(y), **kwargs)),
            self.profile_ref.hessian(x, y, **kwargs),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_lens_model_bulk_ray_shooting(self):
        model = LensModelBulk(unique_lens_model_list=["DPIE_NIE"], multi_plane=False)
        x = np.array([0.2, 1.0, 3.0])
        y = np.array([0.4, 2.0, -1.0])
        kwargs_lens = [
            {
                "sigma0": 1.2,
                "Ra": 0.15,
                "Rs": 3.0,
                "e1": 0.05,
                "e2": -0.02,
                "center_x": 0.1,
                "center_y": -0.1,
            }
        ]
        ray_shooting_kwargs = model.prepare_ray_shooting_kwargs(
            lens_model_list=["DPIE_NIE"],
            kwargs_lens=kwargs_lens,
        )

        beta_x, beta_y = model.ray_shooting(x, y, ray_shooting_kwargs)

        assert np.isfinite(np.asarray(beta_x)).all()
        assert np.isfinite(np.asarray(beta_y)).all()


if __name__ == "__main__":
    pytest.main()
