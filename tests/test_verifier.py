"""Fabricated evidence used ONLY to test verification; never benchmark results.

No model is downloaded, trained, or used to produce this fixture. All responses
come from the deterministic rule engine, including those labeled as models.
"""
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.verify_results import VerificationError, verify
from support_data import CATEGORIES, build_dataset, rule_response, score_response, summarize, validate_dataset

METHODS = ("base", "sft", "dpo", "rule_engine")
SPLITS = ("development", "test")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rehash(root):
    write_json(root / "manifest_sha256.json", {
        p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest_sha256.json" and p.relative_to(root).parts[0] not in ("adapters", "checkpoints")})


def fixture(root):
    """Create internally consistent, explicitly fabricated evidence in temp space."""
    config = read_json(PROJECT / "config.json")
    config["bootstrap_resamples"] = 101
    data = build_dataset(config["seed"])
    write_json(root / "config.json", config)
    write_json(root / "SYNTHETIC_TEST_FIXTURE.json", {"fabricated": True, "benchmark_result": False})
    write_json(root / "run_status.json", {"status": "complete", "benchmark_completed": True,
        "completed_stages": ["sft", "dpo"], "human_review_completed": False})
    write_json(root / "protocol.json", {"methods": ["base", "sft", "dpo"], "reference_baseline": "rule_engine",
        "selection": "None: fixed steps and unchanged decoding; no development or test tuning",
        "primary_metric": "structured_pass", "test_evaluation_after_both_training_stages": True,
        "max_new_tokens": config["max_new_tokens"], "do_sample": False, "adapter_merging": False,
        "reply_semantic_quality_measured_by_primary_metric": False})
    for name in ("environment.json", "parameters.json", "adapter_backups.json"):
        write_json(root / name, {"SYNTHETIC_TEST_FIXTURE": True})
    for name in ("train_evaluate.py", "support_data.py"):
        path = root / "code" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Synthetic fixture only; no model execution.\n")
    for split, rows in data.items():
        write_jsonl(root / "data" / (split + ".jsonl"), rows)
    write_json(root / "data_audit.json", validate_dataset(data))
    write_json(root / "data_manifest.json", {split: {"n": len(rows), "sha256": sha(root / "data" / (split + ".jsonl"))}
        for split, rows in data.items()})
    predictions = []
    metrics = {}
    for method in METHODS:
        metrics[method] = {}
        for split in SPLITS:
            outputs = []
            for row in data[split]:
                response = rule_response(row)
                outputs.append(response)
                predictions.append({"method": method, "split": split, "id": row["id"], "category": row["category"],
                    "response": response, "scores": score_response(row, response), "hit_token_limit": False,
                    "generation_seconds": None if method == "rule_engine" else 0.01,
                    "generated_tokens_including_eos": None if method == "rule_engine" else 10})
            metrics[method][split] = summarize(data[split], outputs)
    write_jsonl(root / "predictions.jsonl", predictions)
    write_json(root / "metrics.json", metrics)
    fields = ["method", "split"] + [key for key, value in metrics["base"]["test"].items() if not isinstance(value, dict)]
    with (root / "metrics_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for method in METHODS:
            for split in SPLITS:
                writer.writerow({"method": method, "split": split, **{key: metrics[method][split][key] for key in fields[2:]}})
    write_json(root / "paired_comparisons.json", {name: {"mean_delta": 0.0, "ci95": [0.0, 0.0],
        "b_better": 0, "a_better": 0, "ties": len(data["test"]), "resamples": config["bootstrap_resamples"],
        "seed": config["seed"], "interpretation": "Descriptive paired case bootstrap; no multiplicity or domain-shift adjustment"}
        for name in ("sft_minus_base", "dpo_minus_sft", "dpo_minus_base")})
    train_ids = [row["id"] for row in data["train"]]
    indices = np.random.default_rng(config["seed"] + 2).permutation(len(train_ids))[:config["dpo_examples"]]
    dpo_rows = [data["train"][i] for i in indices]
    dpo_ids = [row["id"] for row in dpo_rows]
    write_json(root / "dpo_training_ids.json", dpo_ids)
    write_json(root / "training_mixture.json", {
        "sft_categories": dict(Counter(row["category"] for row in data["train"])),
        "dpo_categories": dict(Counter(row["category"] for row in dpo_rows)),
        "dpo_rejection_types": dict(Counter(row["rejection_type"] for row in dpo_rows)),
        "subset_selection": "Fixed seeded random subset of training rows; no held-out information used"})
    write_jsonl(root / "dpo_reference_logps.jsonl", [{"id": ticket, "chosen_logp": -10.0, "rejected_logp": -12.0} for ticket in dpo_ids])
    write_json(root / "dpo_reference.json", {"sha256": sha(root / "dpo_reference_logps.jsonl"), "seconds": 0.1,
        "reference": "Frozen SFT adapter on the same NF4 base", "reduction": "completion token sum including EOS"})
    timings = {}
    for stage, ids, seed in (("sft", train_ids, config["seed"]), ("dpo", dpo_ids, config["seed"] + 1)):
        steps = config[stage + "_steps"]
        count = steps * config["gradient_accumulation"]
        write_jsonl(root / (stage + "_training_log.jsonl"), [{"step": step, "loss": math.log(2), "gradient_norm_before_clip": 1.0}
            for step in range(1, steps + 1)])
        rng = np.random.default_rng(seed)
        order = [ids[i] for _ in range(math.ceil(count / len(ids))) for i in rng.permutation(len(ids))][:count]
        write_json(root / (stage + "_training_order.json"), order)
        timings[stage] = {"seconds": 0.1, "optimizer_steps": steps, "training_example_presentations": count,
            "peak_allocated_gpu_bytes": 0, "peak_reserved_gpu_bytes": 0}
    write_json(root / "timing.json", timings)
    rng = random.Random(config["seed"])
    selected = []
    for category in sorted(CATEGORIES):
        selected.extend(rng.sample(sorted(row["id"] for row in data["test"] if row["category"] == category), 2))
    mapping = [{"review_id": f"R{100000000 + i}", "method": method, "id": ticket}
        for i, (method, ticket) in enumerate((method, ticket) for method in METHODS[:3] for ticket in selected)]
    write_json(root / "review_key.json", {"mapping": mapping})
    row_map = {row["id"]: row for row in data["test"]}
    with (root / "manual_review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["review_id", "prompt", "response", "reply_factual", "instruction_following", "unsupported_advice", "notes"])
        writer.writeheader()
        for entry in mapping:
            row = row_map[entry["id"]]
            writer.writerow({"review_id": entry["review_id"], "prompt": row["prompt"], "response": rule_response(row)})
    rehash(root)


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_temp = tempfile.TemporaryDirectory()
        cls.template = Path(cls.template_temp.name)
        fixture(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.template_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic_fixture"
        shutil.copytree(self.template, self.root)

    def edit_json(self, name, edit):
        path = self.root / name
        value = read_json(path); edit(value); write_json(path, value)

    def rejected_after_rehash(self, message):
        rehash(self.root)
        with self.assertRaisesRegex(VerificationError, message):
            verify(self.root)

    def test_consistent_synthetic_fixture_and_blank_ratings_pass(self):
        report = verify(self.root)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["predictions"], 720)
        self.assertEqual(report["review_rows_rated"], 0)

    def test_cli_works_without_importing_torch(self):
        command = '''import builtins, runpy, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Verifier imported torch")
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-c", command, str(PROJECT / "scripts" / "verify_results.py"), str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(json.loads(result.stdout)["status"], "verified")
        source = (PROJECT / "scripts" / "verify_results.py").read_text()
        self.assertNotIn("import torch", source)
        self.assertNotIn("from train_evaluate", source)

    def test_manifest_detects_changed_file_without_rehash(self):
        self.edit_json("metrics.json", lambda value: value["base"]["test"].update(structured_pass=0.0))
        with self.assertRaisesRegex(VerificationError, "Manifest hash"):
            verify(self.root)

    def test_manifest_requires_complete_coverage(self):
        self.edit_json("manifest_sha256.json", lambda value: value.pop("metrics.json"))
        with self.assertRaisesRegex(VerificationError, "Manifest file coverage"):
            verify(self.root)

    def test_rehashed_flags_are_recomputed(self):
        path = self.root / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["scores"]["structured_pass"] = False
        write_jsonl(path, rows)
        self.rejected_after_rehash("prediction scores")

    def test_rehashed_metrics_are_recomputed(self):
        self.edit_json("metrics.json", lambda value: value["base"]["test"].update(structured_pass=0.0))
        self.rejected_after_rehash("metrics.base.test.structured_pass")

    def test_rehashed_csv_is_recomputed(self):
        path = self.root / "metrics_summary.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
        rows[0]["structured_pass"] = "0"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        self.rejected_after_rehash("metric CSV")

    def test_rehashed_bootstrap_is_recomputed(self):
        self.edit_json("paired_comparisons.json", lambda value: value["sft_minus_base"].update(ci95=[-0.5, 0.5]))
        self.rejected_after_rehash("paired_comparisons")

    def test_duplicate_prediction_cannot_replace_missing_case(self):
        path = self.root / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1] = rows[0].copy(); write_jsonl(path, rows)
        self.rejected_after_rehash("duplicate prediction")

    def test_training_order_cannot_use_heldout_id(self):
        test_id = json.loads((self.root / "data/test.jsonl").read_text().splitlines()[0])["id"]
        self.edit_json("sft_training_order.json", lambda value: value.__setitem__(0, test_id))
        self.rejected_after_rehash("sft_training_order")

    def test_reference_hash_is_checked_after_manifest_rehash(self):
        self.edit_json("dpo_reference.json", lambda value: value.update(sha256="0" * 64))
        self.rejected_after_rehash("DPO reference hash")

    def test_reference_cannot_contain_nonfinite_scores(self):
        path = self.root / "dpo_reference_logps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["chosen_logp"] = float("nan"); write_jsonl(path, rows)
        self.rejected_after_rehash("Nonfinite JSON")

    def test_review_key_cannot_duplicate_one_model_case(self):
        self.edit_json("review_key.json", lambda value: value["mapping"].__setitem__(1, value["mapping"][0].copy()))
        self.rejected_after_rehash("fixed balanced 12 tickets")

    def test_interrupted_run_is_not_completed(self):
        self.edit_json("run_status.json", lambda value: value.update(status="interrupted", benchmark_completed=False))
        self.rejected_after_rehash("not a completed benchmark")


if __name__ == "__main__":
    unittest.main()
