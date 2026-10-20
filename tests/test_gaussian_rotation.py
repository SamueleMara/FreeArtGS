import unittest
from types import SimpleNamespace

import torch
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.data.utils.dataparsers_utils import apply_left_se3_to_gaussians
from nerfstudio.models.splatfacto import SplatfactoModel


class GaussianRotationTest(unittest.TestCase):
    def test_covariance_follows_world_rotation_and_preserves_legacy_order(self):
        q = torch.tensor([[.2, .3, .4, .8]])
        q /= q.norm(dim=-1, keepdim=True)
        scales = torch.tensor([[.01, .1, 1.]])
        gaussians = dict(means=torch.tensor([[1., 2., 3.]]), quats=q, scales=scales.log(),
                         features_dc=torch.zeros(1, 3), features_rest=torch.zeros(1, 0, 3),
                         opacities=torch.zeros(1, 1))
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transform = torch.eye(4)
        transform[:3, :3] = rotation
        rotated = apply_left_se3_to_gaussians(gaussians, transform, torch.device("cpu"))
        model = SimpleNamespace(config=SimpleNamespace(quaternion_order="xyzw"))

        def covariance(quats):
            r = quat_to_rotmat(SplatfactoModel.quats_for_gsplat(model, quats))
            return (r @ torch.diag_embed(scales.square()) @ r.transpose(-1, -2))[0]

        torch.testing.assert_close(covariance(rotated["quats"]), rotation @ covariance(q) @ rotation.T)
        torch.testing.assert_close(rotated["means"], gaussians["means"] @ rotation.T)
        model.config.quaternion_order = "legacy"
        torch.testing.assert_close(SplatfactoModel.quats_for_gsplat(model, q), q)


if __name__ == "__main__":
    unittest.main()
