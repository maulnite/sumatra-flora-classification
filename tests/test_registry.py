"""Synthetic tests for the experiment registry (no data, no GPU, no network)."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.registry import best_oof, load_protocols, render_table, validate

PROTOCOLS = {"scene5_v4rot": {"description": "test"}, "leaderboard_only": {"description": "lb"}}


def make_exp(**overrides):
    exp = {
        "id": "EXP-900",
        "title": "synthetic",
        "owner": "tester",
        "branch": "kaka",
        "status": "completed",
        "date": "2026-11-12",
        "protocol": "scene5_v4rot",
        "summary": "x",
        "results": [{"name": "OOF @224", "scope": "oof", "resolution": 224, "macro_f1": 0.75}],
        "leaderboard": {"public": None, "private": None},
        "evidence": {"report": "experiments/EXP-900.md"},
    }
    exp.update(overrides)
    return exp


class RegistryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        (self.base / "experiments").mkdir()
        (self.base / "experiments" / "EXP-900.md").write_text("report", encoding="utf-8")

    def test_valid_experiment_passes(self):
        self.assertEqual(validate(make_exp(), PROTOCOLS, self.base), [])

    def test_unknown_protocol_is_rejected(self):
        errs = validate(make_exp(protocol="made_up"), PROTOCOLS, self.base)
        self.assertTrue(any("protocol" in e for e in errs))

    def test_score_out_of_range_is_rejected(self):
        exp = make_exp(results=[{"name": "bad", "scope": "oof", "macro_f1": 1.5}])
        self.assertTrue(any("not in [0, 1]" in e for e in validate(exp, PROTOCOLS, self.base)))

    def test_leaderboard_score_out_of_range_is_rejected(self):
        exp = make_exp(leaderboard={"public": 87.0, "private": None})
        self.assertTrue(any("leaderboard.public" in e for e in validate(exp, PROTOCOLS, self.base)))

    def test_completed_needs_a_score_and_existing_report(self):
        no_score = make_exp(results=[{"name": "none", "scope": "oof", "macro_f1": None}])
        self.assertTrue(validate(no_score, PROTOCOLS, self.base))
        no_report = make_exp(evidence={"report": "experiments/missing.md"})
        self.assertTrue(any("does not exist" in e for e in validate(no_report, PROTOCOLS, self.base)))

    def test_planned_experiment_may_have_no_score(self):
        exp = make_exp(status="planned", results=[{"name": "tbd", "scope": "oof", "macro_f1": None}])
        self.assertEqual(validate(exp, PROTOCOLS, self.base), [])

    def test_best_oof_ignores_non_oof_scopes(self):
        exp = make_exp(results=[
            {"name": "fold0", "scope": "fold0", "macro_f1": 0.99},
            {"name": "oof", "scope": "oof", "macro_f1": 0.7},
        ])
        self.assertEqual(best_oof(exp), 0.7)

    def test_render_contains_id_and_protocol_definitions(self):
        table = render_table([make_exp()], PROTOCOLS)
        self.assertIn("EXP-900", table)
        self.assertIn("| `scene5_v4rot` | test |", table)

    def test_shipped_protocols_file_is_valid_json(self):
        protocols = load_protocols(Path(__file__).resolve().parents[1] / "docs" / "protocols.json")
        self.assertIn("scene5_v4rot", protocols)
        json.dumps(protocols)


if __name__ == "__main__":
    unittest.main()
