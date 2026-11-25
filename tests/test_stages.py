"""Durable-stage tests with a fabricated CPU backend, never model benchmarks.

Each stage runs in a fresh Python process. Only model/GPU operations are mocked;
the runner's checkpoints, guards, JSON persistence, exports and retry logic run.
"""
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import train_evaluate as runner
from support_data import rule_response


def read(path):
    return json.loads(Path(path).read_text())


def event(path, kind, **fields):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"kind": kind, **fields}) + "\n")


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


class FakeTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    eos_token = "</s>"
    pad_token = "</s>"
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 3, 4]

    def __call__(self, text, **kwargs):
        return {"input_ids": [5, 6]}

    def save_pretrained(self, directory):
        runner.write_json(Path(directory) / "tokenizer_config.json", {"fabricated_cpu_fixture": True})


class FakeCPUModel(torch.nn.Module):
    """A scalar adapter state; not a language model or real safetensors file."""
    def __init__(self, config, stage="base", value=0.0):
        super().__init__()
        self.lora_fixture = torch.nn.Parameter(torch.tensor(float(value)))
        self.config = SimpleNamespace(use_cache=False)
        self.peft_config = {"default": SimpleNamespace(revision=config["model_revision"])}
        self.project_config = config
        self.stage = stage

    def get_nb_trainable_parameters(self):
        return 1, 2

    def save_pretrained(self, directory, **kwargs):
        runner.write_json(Path(directory) / "adapter_model.safetensors", {
            "fabricated_cpu_fixture": True, "value": self.lora_fixture.item(), "stage": self.stage})
        runner.write_json(Path(directory) / "adapter_config.json", {
            "base_model_name_or_path": self.project_config["model_id"],
            "revision": self.project_config["model_revision"], "peft_type": "LORA", "r": 16,
            "lora_alpha": 32, "lora_dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})


@contextlib.contextmanager
def fake_backend(action, event_path, fail_after=None):
    generated = 0

    def reload_model(config, adapter_path=None):
        if adapter_path is None:
            model = FakeCPUModel(config)
        else:
            saved = read(Path(adapter_path) / "adapter_model.safetensors")
            model = FakeCPUModel(config, saved["stage"], saved["value"])
        event(event_path, "load_training", stage=model.stage, value=model.lora_fixture.item())
        return model

    def evaluation_model(config, method, run_dir):
        if method == "base":
            model = FakeCPUModel(config)
        else:
            saved = read(Path(run_dir) / "adapters" / method / "adapter_model.safetensors")
            model = FakeCPUModel(config, saved["stage"], saved["value"])
        event(event_path, "load_evaluation", method=method, value=model.lora_fixture.item())
        return model

    def train(model, tokenizer, encoded, rows, config, run_dir, stage, reference=None):
        if action == "evaluate":
            raise AssertionError("Evaluation or retry invoked training")
        expected = 0.0 if stage == "sft" else 0.25
        if model.lora_fixture.item() != expected:
            raise AssertionError(f"{stage} loaded the wrong saved adapter state")
        with torch.no_grad():
            model.lora_fixture.add_(0.25 if stage == "sft" else 0.5)
        model.stage = stage
        event(event_path, "train", stage=stage, value=model.lora_fixture.item())
        steps, accumulation = config[stage + "_steps"], config["gradient_accumulation"]
        rng = np.random.default_rng(config["seed"] + int(stage == "dpo"))
        count = steps * accumulation
        order = [rows[i]["id"] for _ in range(math.ceil(count / len(rows))) for i in rng.permutation(len(rows))][:count]
        runner.write_json(Path(run_dir) / (stage + "_training_order.json"), order)
        runner.write_jsonl(Path(run_dir) / (stage + "_training_log.jsonl"), [
            {"step": step, "loss": math.log(2), "gradient_norm_before_clip": 1.0}
            for step in range(1, steps + 1)])
        return {"seconds": 0.01, "optimizer_steps": steps, "training_example_presentations": count,
                "peak_allocated_gpu_bytes": 0, "peak_reserved_gpu_bytes": 0}

    def reference(model, encoded, rows, run_dir):
        if model.stage != "sft" or model.lora_fixture.item() != 0.25:
            raise AssertionError("Reference did not use the saved SFT state")
        event(event_path, "reference", stage=model.stage)
        records = [{"id": row["id"], "chosen_logp": -2.0, "rejected_logp": -3.0} for row in rows]
        path = Path(run_dir) / "dpo_reference_logps.jsonl"
        runner.write_jsonl(path, records)
        return records, {"sha256": runner.digest(path), "seconds": 0.01,
            "reference": "Frozen SFT adapter on the same NF4 base", "reduction": "completion token sum including EOS"}

    def generate(model, tokenizer, row, config):
        nonlocal generated
        if fail_after is not None and generated >= abs(fail_after):
            if fail_after < 0:
                os._exit(9)  # Deliberately bypass runner exception/finally handlers.
            raise RuntimeError("INTENTIONAL_TEST_EVALUATION_INTERRUPTION")
        generated += 1
        event(event_path, "generate", method=model.stage, split=row["split"], id=row["id"])
        return rule_response(row), {"generation_seconds": 0.01,
            "generated_tokens_including_eos": 10, "hit_token_limit": False}

    with contextlib.ExitStack() as stack:
        for name, replacement in {
            "require_cuda": lambda: None,
            "environment_info": lambda: {"fabricated_cpu_fixture": True, "gpu": "NONE", "cuda": None},
            "load_tokenizer": lambda config: FakeTokenizer(),
            "load_training_model": reload_model,
            "load_evaluation_model": evaluation_model,
            "training_loop": train,
            "reference_scores": reference,
            "generate": generate,
        }.items():
            stack.enter_context(patch.object(runner, name, replacement))
        for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
            stack.enter_context(patch.object(torch.cuda, name, lambda *args, **kwargs: None))
        stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", lambda *args: 0))
        stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", lambda *args: 0))
        if fail_after == -999:
            stack.enter_context(patch.object(runner, "finalize_evidence", lambda *args, **kwargs: os._exit(9)))
        yield


def worker(action, target, output_root, event_path, fail_after):
    try:
        with fake_backend(action, event_path, fail_after), contextlib.redirect_stdout(io.StringIO()):
            if action == "sft":
                result = runner.run_sft(target, output_root)
            elif action == "dpo":
                result = runner.run_dpo(target)
            elif action == "evaluate":
                result = runner.evaluate_saved(target)
            else:
                raise ValueError(action)
        print(json.dumps({"ok": True, "run_dir": str(result)}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "type": type(exc).__name__}))
        traceback.print_exc(file=sys.stderr)
        return 2


def process(action, target, output_root, event_path, fail_after=None):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", action, str(target),
               str(output_root), str(event_path), "none" if fail_after is None else str(fail_after)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode == 9 and not result.stdout:
        return result, {"ok": False, "error": "INTENTIONAL_TEST_HARD_EXIT"}
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise AssertionError(result.stdout + "\n" + result.stderr)
    return result, payload


class DurableStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="fabricated_stage_tests_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.template = Path(cls.temp.name)
        config = read(PROJECT / "config.json")
        config.update(sft_steps=2, dpo_steps=2, gradient_accumulation=1, dpo_examples=24, bootstrap_resamples=101)
        runner.write_json(cls.template / "config.json", config)
        cls.event_file = cls.template / "fake_backend_events.jsonl"
        first, first_payload = process("sft", cls.template / "config.json", cls.template / "artifacts", cls.event_file)
        if first.returncode:
            raise AssertionError(first.stdout + first.stderr)
        cls.run_name = Path(first_payload["run_dir"]).name
        cls.run_dir = cls.template / "artifacts" / cls.run_name
        cls.sft_status = read(cls.run_dir / "run_status.json")
        cls.sft_timings = read(cls.run_dir / "timing.json")
        second, _ = process("dpo", cls.run_dir, cls.template / "artifacts", cls.event_file)
        if second.returncode:
            raise AssertionError(second.stdout + second.stderr)

    def setUp(self):
        self.case_temp = tempfile.TemporaryDirectory(prefix="fabricated_stage_case_")
        self.addCleanup(self.case_temp.cleanup)
        self.root = Path(self.case_temp.name)
        shutil.copytree(self.template, self.root, dirs_exist_ok=True)
        self.run_dir = self.root / "artifacts" / self.run_name
        self.event_file = self.root / "fake_backend_events.jsonl"

    def run_worker(self, action, fail_after=None):
        return process(action, self.run_dir, self.root / "artifacts", self.event_file, fail_after)

    def test_stages_reconstruct_saved_state_in_fresh_processes(self):
        self.assertEqual(self.sft_status["completed_stages"], ["sft"])
        self.assertFalse(self.sft_status["benchmark_completed"])
        self.assertEqual(self.sft_timings["sft"]["optimizer_steps"], 2)
        status = read(self.run_dir / "run_status.json")
        self.assertEqual(status["completed_stages"], ["sft", "dpo"])
        self.assertFalse(status["benchmark_completed"])
        self.assertEqual(read(self.run_dir / "timing.json")["dpo"]["optimizer_steps"], 2)
        self.assertEqual(read(self.run_dir / "adapters/sft/adapter_model.safetensors")["value"], 0.25)
        self.assertEqual(read(self.run_dir / "adapters/dpo/adapter_model.safetensors")["value"], 0.75)
        history = events(self.event_file)
        self.assertEqual([(e["stage"], e["value"]) for e in history if e["kind"] == "load_training"],
                         [("base", 0.0), ("sft", 0.25)])
        self.assertEqual([e["stage"] for e in history if e["kind"] == "train"], ["sft", "dpo"])
        self.assertFalse((self.run_dir / "predictions.jsonl").exists())
        self.assertEqual(set(read(self.run_dir / "adapter_backups.json")), {"sft", "dpo"})

    def test_interrupted_evaluation_retries_only_missing_predictions_and_never_trains(self):
        interrupted, payload = self.run_worker("evaluate", fail_after=7)
        self.assertEqual(interrupted.returncode, 2, interrupted.stdout + interrupted.stderr)
        self.assertIn("INTENTIONAL_TEST_EVALUATION_INTERRUPTION", payload["error"])
        path = self.run_dir / "predictions.jsonl"
        prior = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(prior), 7)
        self.assertEqual(read(self.run_dir / "run_status.json")["completed_stages"], ["sft", "dpo"])
        self.assertEqual(read(self.run_dir / "timing.json")["dpo"]["optimizer_steps"], 2)
        resumed, result = self.run_worker("evaluate")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertTrue(result["ok"])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        keys = [(r["method"], r["split"], r["id"]) for r in records]
        self.assertEqual(len(records), 720)
        self.assertEqual(len(set(keys)), 720)
        by_key = {key: row for key, row in zip(keys, records)}
        for old in prior:
            self.assertEqual(by_key[old["method"], old["split"], old["id"]], old)
        generated = [(e["method"], e["split"], e["id"]) for e in events(self.event_file) if e["kind"] == "generate"]
        self.assertEqual(len(generated), 540)
        self.assertEqual(len(set(generated)), 540)
        self.assertEqual([e["stage"] for e in events(self.event_file) if e["kind"] == "train"], ["sft", "dpo"])
        self.assertTrue(read(self.run_dir / "run_status.json")["benchmark_completed"])
        from scripts.verify_results import verify
        self.assertEqual(verify(self.run_dir)["status"], "verified")
        before = {p.relative_to(self.run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.run_dir.rglob("*") if p.is_file()}
        previous_events = self.event_file.read_bytes()
        again, again_payload = self.run_worker("evaluate")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertTrue(again_payload["ok"])
        after = {p.relative_to(self.run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.run_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(previous_events, self.event_file.read_bytes())
        archive = self.run_dir.parent / (self.run_dir.name + "_results.zip")
        for corruption in ("missing", "truncated"):
            with self.subTest(archive=corruption):
                if corruption == "missing":
                    archive.unlink()
                else:
                    archive.write_bytes(b"TRUNCATED ZIP TEST FIXTURE")
                rejected, rejected_payload = self.run_worker("evaluate")
                self.assertNotEqual(rejected.returncode, 0, rejected.stdout + rejected.stderr)
                self.assertFalse(rejected_payload["ok"])
                self.assertIn("Finalized evidence or its archive changed", rejected_payload["error"])
                if corruption == "missing":
                    self.assertFalse(archive.exists())
                else:
                    self.assertEqual(archive.read_bytes(), b"TRUNCATED ZIP TEST FIXTURE")
                self.assertEqual(previous_events, self.event_file.read_bytes())
                self.assertEqual(before, {p.relative_to(self.run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.run_dir.rglob("*") if p.is_file()})

    def test_changed_config_is_rejected_before_model_loading(self):
        config = read(self.run_dir / "config.json")
        config["max_new_tokens"] += 1
        runner.write_json(self.run_dir / "config.json", config)
        before = self.event_file.read_bytes()
        result, payload = self.run_worker("evaluate")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("config.json", payload["error"])
        self.assertEqual(before, self.event_file.read_bytes())
        self.assertFalse((self.run_dir / "predictions.jsonl").exists())

    def test_hard_process_exit_reuses_durable_prefix_only(self):
        killed, _ = self.run_worker("evaluate", fail_after=-13)
        self.assertEqual(killed.returncode, 9, killed.stdout + killed.stderr)
        path = self.run_dir / "predictions.jsonl"
        durable = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(durable), 10)
        resumed, _ = self.run_worker("evaluate")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        final = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(final[:10], durable)
        generated = [(e["method"], e["split"], e["id"]) for e in events(self.event_file) if e["kind"] == "generate"]
        self.assertEqual(len(generated), 543)  # Three unsaved responses are retried.
        self.assertEqual(len(set(generated)), 540)
        for row in durable:
            self.assertEqual(generated.count((row["method"], row["split"], row["id"])), 1)
        self.assertEqual([e["stage"] for e in events(self.event_file) if e["kind"] == "train"], ["sft", "dpo"])

    def test_hard_exit_before_final_export_recovers_without_generation(self):
        killed, _ = self.run_worker("evaluate", fail_after=-999)
        self.assertEqual(killed.returncode, 9, killed.stdout + killed.stderr)
        predictions_path = self.run_dir / "predictions.jsonl"
        prediction_bytes = predictions_path.read_bytes()
        self.assertEqual(len(prediction_bytes.splitlines()), 720)
        orphan = self.run_dir / ("metrics.json.tmp-" + "a" * 32)
        orphan.write_text("INCOMPLETE ATOMIC WRITE TEST FIXTURE")
        before_generations = [e for e in events(self.event_file) if e["kind"] == "generate"]
        resumed, _ = self.run_worker("evaluate")
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(predictions_path.read_bytes(), prediction_bytes)
        self.assertEqual([e for e in events(self.event_file) if e["kind"] == "generate"], before_generations)
        self.assertEqual(len(before_generations), 540)
        self.assertFalse(orphan.exists())
        self.assertNotIn(orphan.name, read(self.run_dir / "manifest_sha256.json"))
        self.assertTrue((self.run_dir.parent / (self.run_dir.name + "_results.zip")).is_file())
        from scripts.verify_results import verify
        self.assertEqual(verify(self.run_dir)["status"], "verified")

    def test_corrupt_adapter_is_rejected_before_model_loading(self):
        path = self.run_dir / "adapters/dpo/adapter_model.safetensors"
        path.write_bytes(path.read_bytes() + b"CORRUPTION")
        before = self.event_file.read_bytes()
        result, payload = self.run_worker("evaluate")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("dpo adapter", payload["error"].lower())
        self.assertEqual(before, self.event_file.read_bytes())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, action, target, output_root, event_path, fail_after = sys.argv
        raise SystemExit(worker(action, target, output_root, event_path,
                               None if fail_after == "none" else int(fail_after)))
    unittest.main()
