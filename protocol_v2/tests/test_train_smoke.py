from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from protocol_v2.train import train_one


def matrix(samples: int) -> np.ndarray:
    rng = np.random.default_rng(42 + samples)
    value = rng.normal(size=(149, samples)).astype(np.float32)
    for horizon in range(5):
        value[144 + horizon] = np.arange(samples) % 3 + 1
    return value


class TrainSmokeTests(unittest.TestCase):
    def test_cpu_training_run_completes_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "data" / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore" / "NoAuction_Zscore_Training"
            test_dir = root / "data" / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore" / "NoAuction_Zscore_Testing"
            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)
            np.save(train_dir / "Train_Dst_NoAuction_ZScore_CF_1.npy", matrix(180))
            np.save(test_dir / "Test_Dst_NoAuction_ZScore_CF_1.npy", matrix(90))
            run_dir = root / "run"
            config = {
                "run_id": "cpu-smoke",
                "run_dir": str(run_dir),
                "data_root": str(root / "data"),
                "cache_root": str(root / "cache"),
                "model": "cnn_lstm_no_attention",
                "fold": 1,
                "horizon": 10,
                "seed": 0,
                "micro_mode": "none",
                "require_cuda": False,
                "resume": True,
                "amp": False,
                "sequence_length": 10,
                "validation_fraction": 0.20,
                "batch_size": 32,
                "eval_batch_size": 64,
                "num_workers": 0,
                "epochs": 1,
                "early_stopping_patience": 1,
                "scheduler_patience": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "max_grad_norm": 1.0,
                "hidden": 8,
            }
            self.assertEqual(train_one(config), 0)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertTrue((run_dir / "best.pt").exists())
            self.assertTrue((run_dir / "test_logits.pt").exists())


if __name__ == "__main__":
    unittest.main()
