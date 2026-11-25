from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from protocol_v2.fi2010 import (
    FI2010SequenceDataset,
    audit_fold_sizes,
    corrected_micro_features,
    discover_fold_pairs,
    legacy_micro_features,
    load_fold,
)
from protocol_v2.metrics import classification_metrics


def make_matrix(samples: int, offset: float = 0.0) -> np.ndarray:
    matrix = np.zeros((149, samples), dtype=np.float32)
    timeline = np.arange(samples, dtype=np.float32) + offset
    for level in range(10):
        base = level * 4
        matrix[base] = 100.0 + timeline * 0.01 + level
        matrix[base + 1] = 20.0 + (timeline % 7) + level
        matrix[base + 2] = 99.5 + timeline * 0.01 - level
        matrix[base + 3] = 22.0 + (timeline % 5) + level
    for horizon in range(5):
        matrix[144 + horizon] = (np.arange(samples) + horizon) % 3 + 1
    return matrix


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_dir = self.root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore" / "NoAuction_Zscore_Training"
        self.test_dir = self.root / "BenchmarkDatasets" / "NoAuction" / "1.NoAuction_Zscore" / "NoAuction_Zscore_Testing"
        self.train_dir.mkdir(parents=True)
        self.test_dir.mkdir(parents=True)

        train_sizes = [130, 190, 250]
        test_sizes = [60, 60, 40]
        for fold, (train_size, test_size) in enumerate(zip(train_sizes, test_sizes), start=1):
            np.save(self.train_dir / f"Train_Dst_NoAuction_ZScore_CF_{fold}.npy", make_matrix(train_size, fold))
            np.save(self.test_dir / f"Test_Dst_NoAuction_ZScore_CF_{fold}.npy", make_matrix(test_size, 100 + fold))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_discovers_paired_anchored_folds(self) -> None:
        pairs = discover_fold_pairs(self.root)
        self.assertEqual([pair.fold for pair in pairs], [1, 2, 3])
        rows = audit_fold_sizes(pairs)
        self.assertTrue(rows[1]["increment_matches_previous_test"])
        self.assertTrue(rows[2]["increment_matches_previous_test"])

    def test_fold_loader_does_not_concatenate(self) -> None:
        pair = discover_fold_pairs(self.root)[-1]
        loaded = load_fold(pair, micro_mode="corrected")
        self.assertEqual(loaded.train_features.shape, (250, 40))
        self.assertEqual(loaded.test_features.shape, (40, 40))
        self.assertEqual(loaded.train_micro.shape, (250, 7))
        self.assertTrue(np.isfinite(loaded.train_micro).all())

    def test_sequence_label_is_last_event(self) -> None:
        pair = discover_fold_pairs(self.root)[0]
        loaded = load_fold(pair, micro_mode="none")
        dataset = FI2010SequenceDataset(
            loaded.train_features,
            loaded.train_labels,
            horizon=10,
            sequence_length=10,
        )
        x, y, micro = dataset[0]
        self.assertEqual(tuple(x.shape), (10, 40))
        self.assertEqual(int(y), int(loaded.train_labels[9, 4] - 1))
        self.assertEqual(micro.numel(), 0)
        self.assertEqual(len(dataset), 121)

    def test_legacy_and_corrected_features_are_explicitly_different(self) -> None:
        features = make_matrix(20)[:40].T
        legacy = legacy_micro_features(features)
        corrected = corrected_micro_features(features)
        self.assertEqual(legacy.shape, (20, 7))
        self.assertEqual(corrected.shape, (20, 7))
        self.assertTrue(np.isfinite(legacy).all())
        self.assertTrue(np.isfinite(corrected).all())
        self.assertFalse(np.allclose(legacy, corrected))

    def test_metrics_are_perfect_for_perfect_logits(self) -> None:
        logits = torch.tensor([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
        labels = torch.tensor([0, 1, 2])
        result = classification_metrics(logits, labels)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        self.assertLess(result["ece"], 0.001)


if __name__ == "__main__":
    unittest.main()
