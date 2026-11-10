"""Synthetic directory test; no organizer images used."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.inspect_dataset import inspect


class InspectTests(unittest.TestCase):
    def test_synthetic_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_train = root / "train" / "0_Synthetic"
            sample_train.mkdir(parents=True)
            (sample_train / "sample.jpg").write_bytes(b"not-a-real-image")
            sample_test = root / "test"
            sample_test.mkdir()
            (sample_test / "test_0001.jpg").write_bytes(b"not-a-real-image")
            with (root / "train.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "label"])
                writer.writerow(["sample.jpg", 0])
            stats = inspect(root)
            self.assertEqual(1, stats["train_class_dirs"])
            self.assertEqual(1, stats["train_images"])
            self.assertEqual(1, stats["test_images"])
            self.assertEqual(1, stats["csv"]["train.csv"]["rows"])


if __name__ == "__main__":
    unittest.main()
