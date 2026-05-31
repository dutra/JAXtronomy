import jax

jax.config.update("jax_enable_x64", True)

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.LensModel.single_plane import SinglePlane as SinglePlane_ref
from jaxtronomy.LensModel.single_plane import SinglePlane
from jaxtronomy.LensModel.single_plane_bulk import SinglePlaneBulk


class TestSinglePlaneBulk(object):
    """Tests the SinglePlaneBulk routines."""

    def setup_method(self):
        self.lens_model_list = ["NFW", "NFW"]
        self.singleplane = SinglePlaneBulk(
            unique_lens_model_list=self.lens_model_list,
        )

        self.singleplane_ref = SinglePlane_ref(
            lens_model_list=self.lens_model_list,
        )

        kwargs_nfw1 = {"Rs": 1.3, "alpha_Rs": 2.18, "center_x": 0.1, "center_y": -2.1}
        self.kwargs_lens = [
            kwargs_nfw1,
            kwargs_nfw1,
        ]

    def test_ray_shooting(self):
        x = np.tile(np.linspace(-5, 5, 20), 20)
        y = np.repeat(np.linspace(-5, 5, 20), 20)

        ray_shooting_kwargs = self.singleplane.prepare_ray_shooting_kwargs(
            lens_model_list=self.lens_model_list,
            kwargs_lens=self.kwargs_lens,
            num_deflectors=10,
        )
        f_x, f_y = self.singleplane.ray_shooting(x, y, **ray_shooting_kwargs)
        f_x_ref, f_y_ref = self.singleplane_ref.ray_shooting(x, y, self.kwargs_lens)
        npt.assert_allclose(f_x, f_x_ref, atol=1e-12, rtol=1e-12)
        npt.assert_allclose(f_y, f_y_ref, atol=1e-12, rtol=1e-12)

    def test_hessian_dpie_nie_shear(self):
        lens_model_list = ["DPIE_NIE", "SHEAR", "DPIE_NIE"]
        unique_lens_model_list = ["DPIE_NIE", "SHEAR"]
        singleplane = SinglePlaneBulk(unique_lens_model_list=unique_lens_model_list)
        singleplane_ref = SinglePlane(lens_model_list=lens_model_list)
        kwargs_lens = [
            {
                "sigma0": 1.2,
                "Ra": 0.15,
                "Rs": 3.0,
                "e1": 0.05,
                "e2": -0.02,
                "center_x": 0.1,
                "center_y": -0.1,
            },
            {"gamma1": 0.08, "gamma2": -0.03},
            {
                "sigma0": 0.4,
                "Ra": 0.08,
                "Rs": 1.7,
                "e1": -0.03,
                "e2": 0.04,
                "center_x": -0.2,
                "center_y": 0.3,
            },
        ]
        x = np.asarray([0.2, 1.0, 3.0], dtype=float)
        y = np.asarray([0.4, 2.0, -1.0], dtype=float)
        ray_shooting_kwargs = singleplane.prepare_ray_shooting_kwargs(
            lens_model_list=lens_model_list,
            kwargs_lens=kwargs_lens,
            num_deflectors=5,
        )

        hessian = singleplane.hessian(x, y, **ray_shooting_kwargs)
        hessian_ref = singleplane_ref.hessian(x, y, kwargs_lens)

        for value, reference in zip(hessian, hessian_ref):
            npt.assert_allclose(value, reference, atol=1e-12, rtol=1e-12)

    def test_hessian_dpie_nie(self):
        lens_model_list = ["DPIE_NIE"]
        singleplane = SinglePlaneBulk(unique_lens_model_list=["DPIE_NIE"])
        singleplane_ref = SinglePlane(lens_model_list=lens_model_list)
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
        x = np.asarray([0.2, 1.0, 3.0], dtype=float)
        y = np.asarray([0.4, 2.0, -1.0], dtype=float)
        ray_shooting_kwargs = singleplane.prepare_ray_shooting_kwargs(
            lens_model_list=lens_model_list,
            kwargs_lens=kwargs_lens,
            num_deflectors=3,
        )

        hessian = singleplane.hessian(x, y, **ray_shooting_kwargs)
        hessian_ref = singleplane_ref.hessian(x, y, kwargs_lens)

        for value, reference in zip(hessian, hessian_ref):
            npt.assert_allclose(value, reference, atol=1e-12, rtol=1e-12)

    def test_hessian_shear(self):
        lens_model_list = ["SHEAR"]
        singleplane = SinglePlaneBulk(unique_lens_model_list=["SHEAR"])
        singleplane_ref = SinglePlane(lens_model_list=lens_model_list)
        kwargs_lens = [{"gamma1": 0.08, "gamma2": -0.03}]
        x = np.asarray([0.2, 1.0, 3.0], dtype=float)
        y = np.asarray([0.4, 2.0, -1.0], dtype=float)
        ray_shooting_kwargs = singleplane.prepare_ray_shooting_kwargs(
            lens_model_list=lens_model_list,
            kwargs_lens=kwargs_lens,
            num_deflectors=3,
        )

        hessian = singleplane.hessian(x, y, **ray_shooting_kwargs)
        hessian_ref = singleplane_ref.hessian(x, y, kwargs_lens)

        for value, reference in zip(hessian, hessian_ref):
            npt.assert_allclose(value, reference, atol=1e-12, rtol=1e-12)

    def test_raises(self):
        # too many lens models, doesnt match with kwargs_lens
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.singleplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list + ["NFW"],
                kwargs_lens=self.kwargs_lens,
            )
        # provided num_deflectors is smaller than the number of deflectors in lens model list
        with pytest.raises(ValueError):
            ray_shooting_kwargs = self.singleplane.prepare_ray_shooting_kwargs(
                lens_model_list=self.lens_model_list,
                kwargs_lens=self.kwargs_lens,
                num_deflectors=1,
            )
