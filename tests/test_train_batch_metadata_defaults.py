import inspect
import os
import sys
import unittest


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from scripts.train import train  # noqa: E402


class TrainBatchMetadataDefaultsTests(unittest.TestCase):
    def test_train_initializes_optional_optical_time_metadata_before_unpack(self):
        source = inspect.getsource(train.train)
        marker = "for batch_idx, batch_data in enumerate(pbar):"
        self.assertIn(marker, source)

        loop_preamble = source.split(marker, 1)[1].split("if has_negatives:", 1)[0]

        self.assertIn("_opt_zero_time_mjd_base = None", loop_preamble)
        self.assertIn("_opt_first_detection_mjd = None", loop_preamble)


if __name__ == "__main__":
    unittest.main()
