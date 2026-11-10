"""Synthetic submission validator tests."""
import csv
import tempfile
import unittest
from pathlib import Path
from scripts.validate_submission import EXPECTED_ROWS, validate


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.sample = root / "sample_submission.csv"
        self.file = root / "submission.csv"
        self.ids = [f"test_{n:04d}.jpg" for n in range(1, EXPECTED_ROWS + 1)]
        self.write(self.sample, self.ids, [0] * EXPECTED_ROWS)
        self.write(self.file, self.ids, [0] * EXPECTED_ROWS)

    @staticmethod
    def write(path, ids, labels):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "label"])
            writer.writerows(zip(ids, labels))

    def test_valid(self):
        self.assertEqual([], validate(self.file, self.sample))

    def test_bad_label(self):
        labels = [0] * EXPECTED_ROWS
        labels[3] = 30
        self.write(self.file, self.ids, labels)
        self.assertTrue(any("Invalid labels" in s for s in validate(self.file, self.sample)))

    def test_wrong_id_order(self):
        ids = list(self.ids)
        ids[0], ids[1] = ids[1], ids[0]
        self.write(self.file, ids, [0] * EXPECTED_ROWS)
        self.assertTrue(any("ID/order mismatch" in s for s in validate(self.file, self.sample)))

    def test_missing_row(self):
        self.write(self.file, self.ids[:-1], [0] * (EXPECTED_ROWS - 1))
        self.assertTrue(any("Expected 2010" in s for s in validate(self.file, self.sample)))

    def test_decimal_label_invalid(self):
        labels = [0] * EXPECTED_ROWS
        labels[2] = "1.0"
        self.write(self.file, self.ids, labels)
        self.assertTrue(any("Invalid labels" in s for s in validate(self.file, self.sample)))


if __name__ == "__main__":
    unittest.main()
