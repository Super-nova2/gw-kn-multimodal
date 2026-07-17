import importlib.util
import unittest
from pathlib import Path

import torch


MODEL_PATH = Path(__file__).resolve().parents[2] / "Model" / "model.py"
SPEC = importlib.util.spec_from_file_location("magiks_model", MODEL_PATH)
magiks_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(magiks_model)


def trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_physical_dual_model(**overrides):
    kwargs = {
        "fusion_mode": "physical_dual_hgw",
        "dual_fusion": True,
        "use_lightweight_gw": True,
        "enc_dim": 128,
        "proj_dim": 256,
        "ref_time_dim": 64,
        "fusion_attn_dim": None,
        "fusion_hidden_dim": None,
        "gw_dropout": 0.1,
        "opt_dropout": 0.1,
        "proj_dropout": 0.1,
        "feature_dropout": 0.1,
        "fusion_dropout": 0.2,
        "temp_init": 0.07,
        "temp_min": 0.01,
        "temp_max": 0.5,
        "time_compat_weight": 0.0,
        "time_compat_max_penalty": 4.0,
        "use_similarity_as_cls_input": False,
        "use_cred_level_feature": False,
        "use_time_delta_cls_feature": False,
        "fusion_physical_weight": 2.0,
        "fusion_spatial_weight": 1.0,
    }
    kwargs.update(overrides)
    return magiks_model.MAGIKSModel(**kwargs)


def make_inputs(batch_size=2, n_obs=5, n_ref=4, skymap_len=192):
    torch.manual_seed(7)
    gw_s = torch.randn(batch_size, 7)
    gw_m = torch.randn(batch_size, 7, skymap_len)
    opt_coords = torch.tensor([[10.0, -20.0], [120.0, 35.0]], dtype=torch.float32)
    opt_t = torch.linspace(-0.1, 0.2, n_obs).repeat(batch_size, 1)
    opt_ref_t = torch.linspace(-0.1, 0.2, n_ref).repeat(batch_size, 1)
    opt_v = torch.randn(batch_size, n_obs, 6)
    opt_mask = torch.ones(batch_size, n_obs, 6)
    opt_err = torch.full((batch_size, n_obs, 6), 0.05)
    return gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err


class PhysicalDualArchitectureTest(unittest.TestCase):
    def test_v10_physical_dual_parameter_count_is_preserved(self):
        model = build_physical_dual_model()

        self.assertEqual(trainable_params(model), 547_908)
        self.assertEqual(trainable_params(model.optical_encoder.curve_encoder), 12_161)
        self.assertEqual(trainable_params(model.optical_encoder.coord_encoder), 17_280)
        self.assertEqual(trainable_params(model.optical_encoder.contrastive_head), 132_096)
        self.assertEqual(trainable_params(model.gw_encoder.contrastive_head), 132_096)

    def test_v11_rebalanced_architecture_shapes_and_parameter_budget(self):
        model = build_physical_dual_model(
            proj_dim=128,
            fusion_attn_dim=128,
            optical_curve_dim=192,
            optical_coord_dim=64,
            optical_curve_hidden_dim=256,
            contrastive_hidden_dim=128,
        )
        model.eval()

        gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err = make_inputs()

        with torch.no_grad():
            feat_g, feat_o, h_l, h_gw = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
            )
            coord_feat = model.encode_coord_query(opt_coords)
            logits = model.fusion_logits(
                feat_g,
                h_l,
                z_l=feat_o,
                H_gw=h_gw,
                gw_s=gw_s,
                gw_m=gw_m,
                opt_coords=opt_coords,
            )

        self.assertEqual(feat_g.shape, (2, 128))
        self.assertEqual(feat_o.shape, (2, 128))
        self.assertEqual(h_l.shape, (2, 4, 192))
        self.assertEqual(coord_feat.shape, (2, 64))
        self.assertEqual(logits.shape, (2, 2))

        curve_params = trainable_params(model.optical_encoder.curve_encoder)
        coord_params = trainable_params(model.optical_encoder.coord_encoder)
        contrastive_params = (
            trainable_params(model.gw_encoder.contrastive_head)
            + trainable_params(model.optical_encoder.contrastive_head)
            + model.log_temp.numel()
        )

        self.assertGreater(curve_params, coord_params)
        self.assertLess(contrastive_params, 120_000)
        self.assertLess(trainable_params(model), 547_908)

    def test_optical_only_classifier_can_match_v11_curve_encoder_shape(self):
        model = magiks_model.OpticalKNClassifier(
            optical_input_dim=6,
            ref_time_dim=8,
            enc_dim=16,
            optical_curve_dim=24,
            optical_curve_hidden_dim=32,
            num_heads=2,
            k_dim=8,
            opt_dropout=0.1,
            feature_dropout=0.1,
            head_hidden_dim=24,
            head_dropout=0.2,
        )
        model.eval()

        opt_t = torch.linspace(-0.1, 0.2, 5).repeat(3, 1)
        opt_ref_t = torch.linspace(-0.1, 0.2, 4).repeat(3, 1)
        opt_v = torch.randn(3, 5, 6)
        opt_mask = torch.ones(3, 5, 6)
        opt_err = torch.full((3, 5, 6), 0.05)

        with torch.no_grad():
            logits = model(opt_t, opt_v, opt_ref_t, opt_mask, opt_err)

        self.assertEqual(model.optical_encoder.output_dim, 24)
        self.assertEqual(model.feature_dim, 48)
        self.assertEqual(model.classifier[0].in_features, 48)
        self.assertEqual(logits.shape, (3, 1))

    def test_optical_only_classifier_loads_albef_v11_curve_encoder_state(self):
        albef = build_physical_dual_model(
            enc_dim=16,
            proj_dim=8,
            ref_time_dim=8,
            optical_curve_dim=24,
            optical_coord_dim=12,
            optical_curve_hidden_dim=32,
            contrastive_hidden_dim=16,
            fusion_attn_dim=8,
            fusion_hidden_dim=16,
        )
        optical_only = magiks_model.OpticalKNClassifier(
            optical_input_dim=6,
            ref_time_dim=8,
            enc_dim=16,
            optical_curve_dim=24,
            optical_curve_hidden_dim=32,
            num_heads=4,
            k_dim=64,
        )

        missing, unexpected = optical_only.load_optical_encoder_from_magiks_state_dict(
            albef.state_dict(),
            strict=True,
        )

        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])
        self.assertEqual(
            optical_only.optical_encoder_init_info["source_mode"],
            "curve_encoder_only",
        )


if __name__ == "__main__":
    unittest.main()
