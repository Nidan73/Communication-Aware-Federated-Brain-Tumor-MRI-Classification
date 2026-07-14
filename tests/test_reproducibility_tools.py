from __future__ import annotations

import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from paired_model_statistics import (  # noqa: E402
    align_predictions,
    analyze_seed,
    draw_paired_bootstrap_indices,
    exact_mcnemar_counts,
    holm_adjust_pvalues,
    paired_bootstrap,
)
from reproducibility_common import CLASSES, SEEDS, stable_sample_id  # noqa: E402
from run_federated_benchmark import (  # noqa: E402
    build_noniid_split,
    create_new_output_directory,
    make_loader,
    seed_dataloader_worker,
    validate_client_partition,
)
from summarize_multiseed_results import build_summary_rows  # noqa: E402


class PartitionTests(unittest.TestCase):
    def synthetic_manifest(self) -> pd.DataFrame:
        rows = []
        for class_name in CLASSES:
            for index in range(10):
                path = f"images/{class_name}/{index}.jpg"
                rows.append(
                    {
                        "sample_id": stable_sample_id(path),
                        "file_path": path,
                        "class_name": class_name,
                        "cluster_id": f"{class_name}-{index}",
                        "original_split": "synthetic",
                        "final_split": "train",
                    }
                )
        return pd.DataFrame(rows)

    def test_noniid_partition_reproducible_and_valid(self) -> None:
        manifest = self.synthetic_manifest()
        first = build_noniid_split(manifest, clients=4, seed=123, dominant_fraction=0.70)
        second = build_noniid_split(manifest, clients=4, seed=123, dominant_fraction=0.70)
        self.assertTrue(
            first.set_index("file_path")["client_id"].sort_index().equals(
                second.set_index("file_path")["client_id"].sort_index()
            )
        )
        validated = validate_client_partition(
            first,
            manifest,
            clients=4,
            seed=123,
            dominant_fraction=0.70,
            source="synthetic",
        )
        counts = validated.groupby(["client_id", "class_name"]).size().unstack(fill_value=0)
        for _, row in counts.iterrows():
            self.assertEqual(sorted(row.tolist()), [1, 1, 1, 7])

    def test_stable_sample_id(self) -> None:
        self.assertEqual(stable_sample_id("a/b.jpg"), stable_sample_id("a/b.jpg"))
        self.assertNotEqual(stable_sample_id("a/b.jpg"), stable_sample_id("a/c.jpg"))

    def test_dataloader_generator_and_worker_seed_are_deterministic(self) -> None:
        manifest = self.synthetic_manifest().head(1)
        loader = make_loader(
            manifest,
            transform=None,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            seed=2025,
            device=torch.device("cpu"),
        )
        self.assertEqual(loader.generator.initial_seed(), 2025)
        with patch("torch.initial_seed", return_value=987654321):
            seed_dataloader_worker(0)
            first = (random.random(), float(np.random.random()))
            seed_dataloader_worker(1)
            second = (random.random(), float(np.random.random()))
        self.assertEqual(first, second)

    def test_output_directory_guard_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-result"
            create_new_output_directory(output)
            self.assertTrue((output / "checkpoints").is_dir())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                create_new_output_directory(output)


class SummaryTests(unittest.TestCase):
    def test_summary_uses_sample_sd(self) -> None:
        results = {}
        values = {42: 0.80, 123: 0.82, 2025: 0.84}
        for model_name in ("efficientnet_b0", "resnet50"):
            for seed in SEEDS:
                results[(model_name, seed)] = {
                    "test_metrics": {
                        "accuracy": values[seed],
                        "macro_f1": values[seed] - 0.01,
                    },
                    "selection": {"best_round": 10},
                    "communication": {
                        "state_dict_mib": 5.0,
                        "total_communication_mib": 800.0,
                    },
                }
        rows = build_summary_rows(results)
        self.assertEqual(len(rows), 2)
        self.assertTrue(math.isclose(rows[0]["mean_accuracy"], 0.82))
        self.assertTrue(math.isclose(rows[0]["sample_sd_accuracy"], 0.02))


class PairedStatisticsTests(unittest.TestCase):
    def prediction_frame(self, predictions: list[str], model: str) -> pd.DataFrame:
        true = ["glioma", "glioma", "meningioma", "meningioma", "notumor", "notumor", "pituitary", "pituitary"]
        paths = [f"test/{index}.jpg" for index in range(len(true))]
        return pd.DataFrame(
            {
                "sample_id": [stable_sample_id(path) for path in paths],
                "image_path": paths,
                "true_label": true,
                "predicted_label": predictions,
                "correct": [int(a == b) for a, b in zip(true, predictions)],
                "model": model,
                "seed": 42,
            }
        )

    def test_exact_mcnemar_and_deterministic_bootstrap(self) -> None:
        efficientnet = self.prediction_frame(
            ["glioma", "glioma", "meningioma", "meningioma", "notumor", "notumor", "pituitary", "pituitary"],
            "efficientnet_b0",
        )
        resnet = self.prediction_frame(
            ["meningioma", "glioma", "glioma", "meningioma", "notumor", "notumor", "pituitary", "pituitary"],
            "resnet50",
        )
        efficientnet, resnet = align_predictions(efficientnet, resnet, expected_samples=8)
        counts = exact_mcnemar_counts(efficientnet, resnet)
        self.assertEqual(counts["only_efficientnet_correct"], 2)
        self.assertEqual(counts["only_resnet_correct"], 0)
        self.assertTrue(math.isclose(counts["mcnemar_exact_two_sided_p"], 0.5))
        first = paired_bootstrap(
            efficientnet,
            resnet,
            resamples=10_000,
            bootstrap_seed=1234,
            chunk_size=250,
        )
        second = paired_bootstrap(
            efficientnet,
            resnet,
            resamples=10_000,
            bootstrap_seed=1234,
            chunk_size=250,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["bootstrap_stratified"])
        analyzed = analyze_seed(
            42,
            efficientnet,
            resnet,
            bootstrap_resamples=10_000,
            bootstrap_seed=1234,
            expected_samples=8,
        )
        self.assertTrue(analyzed["accuracy_difference_efficientnet_minus_resnet"] > 0)

    def test_stratified_bootstrap_preserves_class_counts_and_pairing(self) -> None:
        true_labels = np.repeat(np.arange(len(CLASSES), dtype=np.int64), 360)
        efficientnet_indices, resnet_indices = draw_paired_bootstrap_indices(
            true_labels,
            resamples=64,
            rng=np.random.default_rng(2025),
            stratified=True,
        )
        self.assertIs(efficientnet_indices, resnet_indices)
        self.assertTrue(np.array_equal(efficientnet_indices, resnet_indices))
        sampled_labels = true_labels[efficientnet_indices]
        for class_index in range(len(CLASSES)):
            counts = np.sum(sampled_labels == class_index, axis=1)
            self.assertTrue(np.all(counts == 360))

    def test_holm_adjustment(self) -> None:
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_adjust_pvalues(raw)
        self.assertTrue(np.allclose(adjusted, [0.03, 0.06, 0.06]))
        self.assertEqual(raw, [0.01, 0.04, 0.03])

    def test_alignment_rejects_mismatched_ids(self) -> None:
        predictions = list(CLASSES) * 2
        efficientnet = self.prediction_frame(predictions, "efficientnet_b0")
        resnet = self.prediction_frame(predictions, "resnet50")
        resnet.loc[0, "sample_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "sample ID sets differ"):
            align_predictions(efficientnet, resnet, expected_samples=8)


if __name__ == "__main__":
    unittest.main()
