"""Verify exported evidence without importing torch or loading any model weights.

This checks internal consistency, not that a GPU run really occurred. Keep the
original exported folder unchanged; enter human ratings in a separate copy.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support_data import CATEGORIES, build_dataset, rule_response, score_response, summarize, validate_dataset

METHODS = ("base", "sft", "dpo", "rule_engine")
SPLITS = ("development", "test")
EXCLUDED = {"adapters", "checkpoints"}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=strict_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(VerificationError(f"Nonfinite JSON value: {x}")))


def read_json(path):
    return loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f"Duplicate CSV columns: {path.name}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"Malformed CSV row: {path.name}")
    return fields, rows


def finite_number(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")
    require(not nonnegative or value >= 0, f"{label} must be nonnegative")


def same(actual, expected, label):
    """Compare nested metrics while keeping boolean flags distinct from 0/1."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{label}: keys differ")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{label}: list length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{label}[{i}]")
    elif type(expected) is bool:
        require(type(actual) is bool and actual == expected, f"{label}: boolean differs")
    elif type(expected) in (int, float):
        finite_number(actual, label)
        require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: number differs")
    else:
        require(actual == expected, f"{label}: value differs")


def expected_order(ids, steps, accumulation, seed):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < steps * accumulation:
        result.extend(ids[i] for i in rng.permutation(len(ids)))
    return result[:steps * accumulation]


def recompute_comparisons(predictions, seed, resamples):
    indexed = {(p["method"], p["id"]): int(p["scores"]["structured_pass"])
               for p in predictions if p["split"] == "test"}
    ids = sorted(p["id"] for p in predictions if p["split"] == "test" and p["method"] == "base")
    result = {}
    for a, b in (("base", "sft"), ("sft", "dpo"), ("base", "dpo")):
        differences = [indexed[b, i] - indexed[a, i] for i in ids]
        frequencies = [differences.count(value) for value in (-1, 0, 1)]
        bootstrap = np.random.default_rng(seed).multinomial(len(ids), np.asarray(frequencies) / len(ids), size=resamples)
        bootstrap_means = (bootstrap[:, 2] - bootstrap[:, 0]) / len(ids)
        result[b + "_minus_" + a] = {
            "mean_delta": sum(differences) / len(ids),
            "ci95": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
            "b_better": frequencies[2], "a_better": frequencies[0], "ties": frequencies[1],
            "resamples": resamples, "seed": seed,
            "interpretation": "Descriptive paired case bootstrap; no multiplicity or domain-shift adjustment",
        }
    return result


def verify(run_dir):
    root = Path(run_dir).resolve()
    require(root.is_dir(), "Run folder does not exist")
    paths = [p for p in root.rglob("*") if p.relative_to(root).parts[0] not in EXCLUDED]
    require(not any(p.is_symlink() for p in paths), "Evidence must not contain symlinks")
    files = {p.relative_to(root).as_posix(): p for p in paths if p.is_file() and p.name != "manifest_sha256.json"}
    manifest = read_json(root / "manifest_sha256.json")
    require(isinstance(manifest, dict) and set(manifest) == set(files), "Manifest file coverage differs")
    for name, path in files.items():
        require(manifest[name] == digest(path), f"Manifest hash mismatch: {name}")
    required = {"config.json", "run_status.json", "protocol.json", "environment.json", "parameters.json",
        "data_audit.json", "data_manifest.json", "predictions.jsonl", "metrics.json", "metrics_summary.csv",
        "paired_comparisons.json", "dpo_training_ids.json", "dpo_reference.json", "dpo_reference_logps.jsonl",
        "sft_training_log.jsonl", "dpo_training_log.jsonl", "sft_training_order.json", "dpo_training_order.json",
        "timing.json", "adapter_backups.json", "manual_review.csv", "review_key.json", "training_mixture.json",
        "code/train_evaluate.py", "code/support_data.py"}
    require(required <= files.keys(), f"Required evidence missing: {sorted(required - files.keys())}")
    config = read_json(root / "config.json")
    for key in ("seed", "sft_steps", "dpo_steps", "gradient_accumulation", "dpo_examples", "bootstrap_resamples", "max_new_tokens"):
        require(type(config.get(key)) is int and config[key] >= (0 if key == "seed" else 1), f"Invalid config: {key}")
    require(config.get("eval_limit") is None, "Limited evaluation is not a completed benchmark")
    status = read_json(root / "run_status.json")
    require(status.get("status") == "complete" and status.get("benchmark_completed") is True,
            "Run is not a completed benchmark")
    require(status.get("completed_stages") == ["sft", "dpo"], "Training stages are incomplete or reordered")
    protocol = read_json(root / "protocol.json")
    same(protocol, {"methods": ["base", "sft", "dpo"], "reference_baseline": "rule_engine",
        "selection": "None: fixed steps and unchanged decoding; no development or test tuning",
        "primary_metric": "structured_pass", "test_evaluation_after_both_training_stages": True,
        "max_new_tokens": config["max_new_tokens"], "do_sample": False, "adapter_merging": False,
        "reply_semantic_quality_measured_by_primary_metric": False}, "protocol")

    data = {split: read_jsonl(root / "data" / (split + ".jsonl")) for split in ("train", *SPLITS)}
    audit = validate_dataset(data)
    require(data == build_dataset(config["seed"]), "Dataset differs from deterministic rebuild")
    same(read_json(root / "data_audit.json"), audit, "data_audit")
    same(read_json(root / "data_manifest.json"), {
        split: {"n": len(rows), "sha256": digest(root / "data" / (split + ".jsonl"))} for split, rows in data.items()
    }, "data_manifest")
    rows_by_key = {(split, row["id"]): row for split, rows in data.items() for row in rows}

    predictions = read_jsonl(root / "predictions.jsonl")
    expected_keys = {(method, split, row["id"]) for method in METHODS for split in SPLITS for row in data[split]}
    indexed = {}
    for record in predictions:
        key = (record.get("method"), record.get("split"), record.get("id"))
        require(key in expected_keys and key not in indexed, f"Unknown or duplicate prediction: {key}")
        indexed[key] = record
        row = rows_by_key[key[1:]]
        require(record.get("category") == row["category"], f"Prediction category differs: {key}")
        require(isinstance(record.get("response"), str), f"Response must be text: {key}")
        same(record.get("scores"), score_response(row, record["response"]), f"prediction scores {key}")
        require(type(record.get("hit_token_limit")) is bool, f"Invalid token-limit flag: {key}")
        if key[0] == "rule_engine":
            require(record["response"] == rule_response(row), f"Rule baseline differs: {key}")
            require(record.get("generation_seconds") is None and record.get("generated_tokens_including_eos") is None
                    and record["hit_token_limit"] is False, f"Rule timing differs: {key}")
        else:
            finite_number(record.get("generation_seconds"), f"generation time {key}", True)
            tokens = record.get("generated_tokens_including_eos")
            require(type(tokens) is int and 1 <= tokens <= config["max_new_tokens"], f"Invalid token count: {key}")
            require(not record["hit_token_limit"] or tokens == config["max_new_tokens"], f"Token-limit count differs: {key}")
    require(indexed.keys() == expected_keys, "Prediction coverage is incomplete")
    recomputed = {method: {split: summarize(data[split], [indexed[method, split, row["id"]]["response"]
        for row in data[split]]) for split in SPLITS} for method in METHODS}
    same(read_json(root / "metrics.json"), recomputed, "metrics")
    scalar_fields = [k for k, value in recomputed["base"]["test"].items() if not isinstance(value, (dict, list))]
    fields, csv_rows = read_csv(root / "metrics_summary.csv")
    require(fields == ["method", "split", *scalar_fields], "Metric CSV columns differ")
    seen = set()
    for row in csv_rows:
        key = (row["method"], row["split"])
        require(key[0] in METHODS and key[1] in SPLITS and key not in seen, "Unknown/duplicate metric CSV row")
        seen.add(key)
        for field in scalar_fields:
            same(float(row[field]), recomputed[key[0]][key[1]][field], f"metric CSV {key}.{field}")
    require(seen == {(method, split) for method in METHODS for split in SPLITS}, "Metric CSV coverage differs")
    same(read_json(root / "paired_comparisons.json"), recompute_comparisons(predictions, config["seed"],
         config["bootstrap_resamples"]), "paired_comparisons")

    train_ids = [r["id"] for r in data["train"]]
    require(config["dpo_examples"] <= len(train_ids), "DPO subset exceeds training data")
    dpo_ids = [train_ids[i] for i in np.random.default_rng(config["seed"] + 2).permutation(len(train_ids))[:config["dpo_examples"]]]
    same(read_json(root / "dpo_training_ids.json"), dpo_ids, "dpo_training_ids")
    dpo_rows = [rows_by_key["train", ticket] for ticket in dpo_ids]
    same(read_json(root / "training_mixture.json"), {
        "sft_categories": dict(Counter(row["category"] for row in data["train"])),
        "dpo_categories": dict(Counter(row["category"] for row in dpo_rows)),
        "dpo_rejection_types": dict(Counter(row["rejection_type"] for row in dpo_rows)),
        "subset_selection": "Fixed seeded random subset of training rows; no held-out information used"}, "training_mixture")
    refs = read_jsonl(root / "dpo_reference_logps.jsonl")
    require([record.get("id") for record in refs] == dpo_ids, "DPO references differ from training subset")
    for record in refs:
        require(set(record) == {"id", "chosen_logp", "rejected_logp"}, "DPO reference fields differ")
        for field in ("chosen_logp", "rejected_logp"):
            finite_number(record[field], field)
            require(record[field] <= 0, "Completion log probability cannot be positive")
    ref_info = read_json(root / "dpo_reference.json")
    require(ref_info.get("sha256") == digest(root / "dpo_reference_logps.jsonl"), "DPO reference hash differs")
    require(ref_info.get("reference") == "Frozen SFT adapter on the same NF4 base"
            and ref_info.get("reduction") == "completion token sum including EOS", "DPO reference description differs")
    finite_number(ref_info.get("seconds"), "reference seconds", True)
    timings = read_json(root / "timing.json")
    for stage, ids, seed in (("sft", train_ids, config["seed"]), ("dpo", dpo_ids, config["seed"] + 1)):
        steps = config[stage + "_steps"]
        logs = read_jsonl(root / (stage + "_training_log.jsonl"))
        require([row.get("step") for row in logs] == list(range(1, steps + 1)), f"{stage} log steps differ")
        for row in logs:
            for field in ("loss", "gradient_norm_before_clip"):
                finite_number(row.get(field), f"{stage} {field}", True)
        same(read_json(root / (stage + "_training_order.json")),
             expected_order(ids, steps, config["gradient_accumulation"], seed), stage + "_training_order")
        require(timings[stage].get("optimizer_steps") == steps
                and timings[stage].get("training_example_presentations") == steps * config["gradient_accumulation"],
                f"{stage} timing counts differ")
        for field in ("seconds", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes"):
            finite_number(timings[stage].get(field), stage + " " + field, True)

    review_rng = random.Random(config["seed"])
    review_ids = set()
    for category in sorted(CATEGORIES):
        candidates = sorted(row["id"] for row in data["test"] if row["category"] == category)
        review_ids.update(review_rng.sample(candidates, 2))
    expected_review = {(method, ticket) for method in METHODS[:3] for ticket in review_ids}
    mapping = read_json(root / "review_key.json")["mapping"]
    require(len(mapping) == 36 and {(row.get("method"), row.get("id")) for row in mapping} == expected_review,
            "Review key must cover the fixed balanced 12 tickets and three models")
    by_review = {row["review_id"]: row for row in mapping}
    require(len(by_review) == 36 and all(isinstance(key, str) and key.startswith("R") and key[1:].isdigit()
            for key in by_review), "Review IDs are missing, duplicated, or malformed")
    fields, reviews = read_csv(root / "manual_review.csv")
    require(fields == ["review_id", "prompt", "response", "reply_factual", "instruction_following", "unsupported_advice", "notes"],
            "Review CSV columns differ or reveal method names")
    require(len(reviews) == 36 and Counter(row["review_id"] for row in reviews) == Counter(by_review.keys()),
            "Review CSV coverage differs")
    for row in reviews:
        identity = by_review[row["review_id"]]
        require(row["prompt"] == rows_by_key["test", identity["id"]]["prompt"], "Review prompt differs")
        require(row["response"] == indexed[identity["method"], "test", identity["id"]]["response"], "Review response differs")
    rated = sum(all(row[k].strip() for k in ("reply_factual", "instruction_following", "unsupported_advice")) for row in reviews)
    return {"status": "verified", "evidence_files": len(files), "predictions": len(predictions),
            "test_examples_per_method": len(data["test"]), "review_rows": len(reviews), "review_rows_rated": rated,
            "scope": "Internal evidence consistency only; does not authenticate GPU execution or validate reply factuality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    try:
        report = verify(args.run_folder)
    except (VerificationError, ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
