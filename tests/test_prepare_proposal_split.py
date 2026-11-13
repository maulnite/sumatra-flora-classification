"""Synthetic checks for transitive scene purging and metadata alignment."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_proposal_split import prepare, read_rows


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ProposalSplitTests(unittest.TestCase):
    def inputs(self, root):
        train, groups, matches = [], [], []
        for label in range(30):
            for member in range(4):
                index = len(train)
                row = {"id": f"{label}_synthetic/{member}.jpg", "label": label}
                train.append(row)
                groups.append({**row, "strict_group": index, "broad_group": index,
                               "provisional_broad_fold": 2 if member == 0 else 0})
                matches.append({"idx": -1, "inlier": 0, "label": -1})
        # Validation image 0 -> train image 1 -> train image 2.
        matches[0] = {"idx": 1, "inlier": 16, "label": 0}
        matches[1] = {"idx": 2, "inlier": 20, "label": 0}
        paths = [root / name for name in ("train.csv", "groups.csv", "sift.csv")]
        for path, rows in zip(paths, (train, groups, matches)):
            write_rows(path, rows)
        return paths

    def test_purge_includes_transitive_neighbors_and_freeze_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.inputs(root)
            output = root / "plan"
            result = prepare(*paths, output)
            rows = read_rows(output / "split_manifest.csv")
            self.assertEqual((88, 30, 2), (result["train_rows"], result["validation_rows"], result["purged_rows"]))
            self.assertEqual("validation", rows[0]["role"])
            self.assertEqual("purged_scene_neighbor", rows[1]["role"])
            self.assertEqual("purged_scene_neighbor", rows[2]["role"])
            self.assertEqual("train", rows[3]["role"])
            before = (output / "split_manifest.csv").read_bytes()
            repeated = prepare(*paths, output)
            self.assertEqual(result["manifest_sha256"], repeated["manifest_sha256"])
            self.assertEqual(before, (output / "split_manifest.csv").read_bytes())

    def test_reordered_metadata_is_rejected_before_writing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.inputs(root)
            groups = read_rows(paths[1])
            groups[0], groups[1] = groups[1], groups[0]
            write_rows(paths[1], groups)
            with self.assertRaisesRegex(ValueError, "row order"):
                prepare(*paths, root / "plan")
            self.assertFalse((root / "plan/split_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
