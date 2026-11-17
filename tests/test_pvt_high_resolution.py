import copy
from pathlib import Path
import unittest
import yaml
from scripts import train_pvt_high_resolution as p

META = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/LOCAL-PROP-009-PVT448.yaml").read_text())


class HighResolutionTests(unittest.TestCase):
    def test_continuation_preserves_source_fold_and_validation(self):
        base, cfg = p.resolve(META)
        for key in ("fold", "seed", "dataset_root", "team_root", "reference_root"):
            self.assertEqual(base[key], cfg[key])
        self.assertEqual(cfg["training"]["source_batch"], 256)
        self.assertEqual(cfg["training"]["microbatch"], 16)
        self.assertEqual(cfg["validation"]["student_inference_size"], 352)
        self.assertEqual(META["source_epoch_offset"]+1, 801)

    def test_external_or_unmatched_validation_configuration_rejected(self):
        for where, key, value in (("training_overrides", "pretrained", True),
                                  ("training_overrides", "epochs", 25),
                                  ("validation_overrides", "student_inference_size", 448),
                                  ("execution", "target_images_used", True)):
            bad = copy.deepcopy(META); bad[where][key] = value
            with self.assertRaises(ValueError): p.resolve(bad)

    def test_final_gate_requires_primary_gain_and_both_guards(self):
        baseline = dict(primary=.8655, clean=.8904, mean_blur=.7274)
        final = dict(primary=.8700, clean=.891, mean_blur=.730)
        self.assertTrue(p.screening(final, baseline, META["gate"])["passed"])
        for key, value in (("primary", .866), ("clean", .879), ("mean_blur", .715)):
            bad = dict(final); bad[key] = value
            self.assertFalse(p.screening(bad, baseline, META["gate"])["passed"])


if __name__ == "__main__": unittest.main()
