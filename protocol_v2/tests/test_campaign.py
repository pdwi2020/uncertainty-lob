from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from protocol_v2.campaign import attempt_succeeded, build_runs, run_id
from protocol_v2.io_utils import atomic_write_json


class CampaignTests(unittest.TestCase):
    def test_matrix_expansion_is_cartesian(self) -> None:
        campaign = {
            "defaults": {"horizon": 10, "micro_mode": "none"},
            "phases": [{
                "name": "test",
                "matrix": {
                    "model": ["deeplob", "evlob"],
                    "fold": [1, 2],
                    "seed": [0, 1],
                },
            }],
        }
        runs = build_runs(campaign)
        self.assertEqual(len(runs), 8)
        self.assertEqual({run["fold"] for run in runs}, {1, 2})

    def test_run_id_includes_sensitivity_parameters(self) -> None:
        identifier = run_id({
            "phase": "sde",
            "model": "lob_sde",
            "fold": 9,
            "horizon": 10,
            "seed": 2,
            "micro_mode": "corrected",
            "dt": 0.025,
            "n_rollouts": 50,
        })
        self.assertIn("dt0p025", identifier)
        self.assertIn("r50", identifier)

    def test_completed_status_overrides_native_nonzero_exit(self) -> None:
        with TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            atomic_write_json(status_path, {"state": "completed"})
            self.assertTrue(attempt_succeeded(3221225477, status_path))

    def test_nonzero_exit_without_completion_is_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            atomic_write_json(status_path, {"state": "initializing"})
            self.assertFalse(attempt_succeeded(3221225477, status_path))


if __name__ == "__main__":
    unittest.main()
