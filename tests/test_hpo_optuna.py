import importlib.util
import math
from pathlib import Path
import unittest


def load_hpo_module():
    module_path = Path(__file__).resolve().parents[1] / "Model" / "scripts" / "hpo" / "hpo_optuna.py"
    spec = importlib.util.spec_from_file_location("hpo_optuna", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HpoObjectiveTests(unittest.TestCase):
    def test_fusion_gallery_priority_objective_uses_saved_metrics(self):
        hpo = load_hpo_module()
        results = {
            "val_fusion_gallery_mrr": 0.80,
            "val_fusion_gallery_recall_at_1": 0.75,
            "val_auprc": 0.90,
            "val_auroc": 0.95,
        }

        score = hpo.compute_objective_score(results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"])

        expected = (0.70 * 0.80) + (0.15 * 0.75) + (0.10 * 0.90) + (0.05 * 0.95)
        self.assertTrue(math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-12))
        summary = hpo.format_metric_summary(
            results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"].keys()
        )
        self.assertIn("FG_MRR=0.800000", summary)
        self.assertIn("FG_R@1=0.750000", summary)

    def test_fusion_gallery_priority_objective_nan_when_required_metric_missing(self):
        hpo = load_hpo_module()
        results = {
            "best_ckpt_metric": "fusion_gallery_mrr",
            "best_ckpt_score": 0.80,
            "val_auprc": 0.90,
            "val_auroc": 0.95,
        }

        score = hpo.compute_objective_score(results, hpo.OBJECTIVE_PRESETS["fusion_gallery_priority"])

        self.assertTrue(math.isnan(score))


if __name__ == "__main__":
    unittest.main()
