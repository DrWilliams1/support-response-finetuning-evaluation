"""Fixed SFT/DPO experiment; evidence exports never include model weights.

Run on a fresh CUDA runtime. --smoke uses a tiny RANDOM CPU model and is not a
benchmark. Adapter snapshots are backups, not an automatic resume mechanism.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def atomic_text(path, text):
    """Replace only after the complete new file has been flushed on the same mount."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_logps(logits, labels):
    """Sum causal next-token log probabilities; prompt/pad labels are -100."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected batch x sequence x vocabulary logits and aligned labels")
    targets = labels[:, 1:]
    mask = targets != -100
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    chosen = logp.gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return (chosen * mask).sum(-1)


def dpo_loss(pi_chosen, pi_rejected, reference_chosen, reference_rejected, beta):
    """Original sigmoid DPO objective using summed completion log probabilities."""
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    margin = (pi_chosen - pi_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(beta * margin).mean()


def encode_completion(tokenizer, prompt, completion, max_length):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
    response_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token")
    if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
        response_ids.append(tokenizer.eos_token_id)
    ids = list(prompt_ids) + response_ids
    if not prompt_ids or len(ids) > max_length:
        raise ValueError(f"Sequence length {len(ids)} exceeds limit {max_length}; no truncation performed")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + response_ids}


def tensor_batch(example, device):
    # Batch size is one: there is no artificial EOS padding to score.
    return {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in example.items()}


def forward_logps(model, example, device):
    batch = tensor_batch(example, device)
    labels = batch.pop("labels")
    outputs = model(**batch, use_cache=False)
    return completion_logps(outputs.logits, labels), int((labels[:, 1:] != -100).sum())


def load_base(config, training=False):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"],
        quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16,
        attn_implementation="eager", trust_remote_code=False)
    model.config.use_cache = not training
    if training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def save_adapter(model, tokenizer, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    if not (directory / "adapter_model.safetensors").is_file():
        raise RuntimeError("Adapter weights were not saved")
    files = {p.name: digest(p) for p in directory.iterdir() if p.is_file() and p.name != "adapter_manifest.json"}
    write_json(directory / "adapter_manifest.json", files)


def stage_backup(model, tokenizer, run_dir, stage, callback):
    path = run_dir / "adapters" / stage
    save_adapter(model, tokenizer, path)
    archive = run_dir.parent / (run_dir.name + "_" + stage + "_adapter.zip")
    temporary = Path(shutil.make_archive(str(archive.with_suffix(".tmp-" + uuid.uuid4().hex)),
        "zip", root_dir=path.parent, base_dir=path.name))
    temporary.replace(archive)
    print(f"{stage.upper()} adapter backup: {archive}", flush=True)
    if callback is not None:
        callback(stage, run_dir, archive)
    return {"directory": str(path.relative_to(run_dir)), "zip_name": archive.name,
            "zip_sha256": digest(archive), "adapter_weights_sha256": digest(path / "adapter_model.safetensors"),
            "adapter_manifest_sha256": digest(path / "adapter_manifest.json")}


def training_loop(model, tokenizer, encoded, rows, config, run_dir, stage, reference=None):
    """Fixed optimizer-step budget; deterministic shuffled cycling over training rows."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0  # Fixed choice: eliminate policy/reference dropout mismatch.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.requires_grad and "lora_" not in n for n, p in model.named_parameters()):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate_" + stage], weight_decay=0.0)
    rng = np.random.default_rng(config["seed"] + (1 if stage == "dpo" else 0))
    order = []; position = 0; logs = []; used_ids = []
    steps = config[stage + "_steps"]; accumulation = config["gradient_accumulation"]
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True); losses = []
        for _ in range(accumulation):
            if position == len(order):
                order = rng.permutation(len(rows)).tolist(); position = 0
            index = order[position]; position += 1; used_ids.append(rows[index]["id"])
            chosen, n_tokens = forward_logps(model, encoded[index][0], "cuda:0")
            if stage == "sft":
                loss = -chosen.mean() / n_tokens
            else:
                rejected, _ = forward_logps(model, encoded[index][1], "cuda:0")
                ref = reference[index]
                loss = dpo_loss(chosen, rejected,
                    chosen.new_tensor([ref["chosen_logp"]]), chosen.new_tensor([ref["rejected_logp"]]), config["beta"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss at step {step}")
            (loss / accumulation).backward(); losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite adapter gradient")
        optimizer.step()
        logs.append({"step": step, "loss": float(np.mean(losses)), "gradient_norm_before_clip": float(grad_norm)})
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"{stage.upper()} {step}/{steps} loss={logs[-1]['loss']:.6f}", flush=True)
            write_jsonl(run_dir / (stage + "_training_log.jsonl"), logs)
        if step % config["save_every"] == 0:
            checkpoint = run_dir / "checkpoints" / f"{stage}_step_{step}"
            save_adapter(model, tokenizer, checkpoint)
            write_json(checkpoint / "snapshot.json", {"stage": stage, "optimizer_step": step,
                "automatic_resume_supported": False})
    torch.cuda.synchronize()
    timing = {"seconds": time.perf_counter() - started, "optimizer_steps": steps,
              "training_example_presentations": len(used_ids),
              "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved()}
    write_jsonl(run_dir / (stage + "_training_log.jsonl"), logs)
    write_json(run_dir / (stage + "_training_order.json"), used_ids)
    del optimizer
    return timing


def reference_scores(model, encoded, rows, run_dir):
    """Cache the frozen SFT policy BEFORE DPO changes its single adapter."""
    model.eval(); records = []; started = time.perf_counter()
    with torch.no_grad():
        for row, pair in zip(rows, encoded):
            chosen, _ = forward_logps(model, pair[0], "cuda:0")
            rejected, _ = forward_logps(model, pair[1], "cuda:0")
            records.append({"id": row["id"], "chosen_logp": float(chosen), "rejected_logp": float(rejected)})
    path = run_dir / "dpo_reference_logps.jsonl"; write_jsonl(path, records)
    return records, {"sha256": digest(path), "seconds": time.perf_counter() - started,
                     "reference": "Frozen SFT adapter on the same NF4 base", "reduction": "completion token sum including EOS"}


def generate(model, tokenizer, row, config):
    from transformers import GenerationConfig
    prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
        tokenize=True, add_generation_prompt=True)
    if len(prompt_ids) + config["max_new_tokens"] > config["max_length"]:
        raise ValueError("Generation prompt plus output budget exceeds max_length")
    ids = torch.tensor([prompt_ids], device="cuda:0"); attention = torch.ones_like(ids)
    decoding = GenerationConfig(do_sample=False, max_new_tokens=config["max_new_tokens"],
        use_cache=True, bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=attention, generation_config=decoding)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started; new_ids = output[0, ids.shape[1]:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True), {
        "generation_seconds": elapsed, "generated_tokens_including_eos": len(new_ids),
        "hit_token_limit": len(new_ids) == config["max_new_tokens"] and new_ids[-1] != tokenizer.eos_token_id}


def paired_comparisons(predictions, seed, resamples=2000):
    by_method = {}
    for p in predictions:
        if p["split"] == "test" and p["method"] != "rule_engine":
            by_method.setdefault(p["method"], {})[p["id"]] = int(p["scores"]["structured_pass"])
    output = {}
    for a, b in [("base", "sft"), ("sft", "dpo"), ("base", "dpo")]:
        if by_method[a].keys() != by_method[b].keys():
            raise ValueError("Unpaired test rows")
        differences = np.array([by_method[b][i] - by_method[a][i] for i in sorted(by_method[a])])
        counts = np.array([(differences == k).sum() for k in [-1, 0, 1]])
        draws = np.random.default_rng(seed).multinomial(len(differences), counts / len(differences), size=resamples)
        deltas = (draws[:, 2] - draws[:, 0]) / len(differences)
        output[b + "_minus_" + a] = {"mean_delta": float(differences.mean()),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(), "b_better": int(counts[2]),
            "a_better": int(counts[0]), "ties": int(counts[1]), "resamples": resamples, "seed": seed,
            "interpretation": "Descriptive paired case bootstrap; no multiplicity or domain-shift adjustment"}
    return output


def finalize_evidence(run_dir, suffix="results"):
    # A hard runtime loss can strand a same-directory atomic-write temporary.
    for candidate in run_dir.rglob("*"):
        if candidate.is_file() and re.fullmatch(r".+\.tmp-[0-9a-f]{32}", candidate.name):
            candidate.unlink()
    files = sorted(p for p in run_dir.rglob("*") if p.is_file()
        and p.relative_to(run_dir).parts[0] not in ("adapters", "checkpoints")
        and p.name != "manifest_sha256.json")
    write_json(run_dir / "manifest_sha256.json", {str(p.relative_to(run_dir)): digest(p) for p in files})
    archive = run_dir.parent / (run_dir.name + "_" + suffix + ".zip")
    temporary = archive.with_name(archive.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files + [run_dir / "manifest_sha256.json"]:
                z.write(p, run_dir.name + "/" + str(p.relative_to(run_dir)))
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{suffix} evidence ZIP: {archive}", flush=True)
    return archive



def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This stage requires CUDA. --smoke validates mechanics on CPU only.")


def environment_info():
    return {"python": platform.python_version(), "packages": {
        p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "bitsandbytes", "numpy"]},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


def load_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = "right"
    return tokenizer


def load_training_model(config, adapter_path=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    model = load_base(config, training=True)
    if adapter_path is None:
        model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
            lora_dropout=0.0, target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            task_type="CAUSAL_LM", bias="none"))
    else:
        # The saved SFT adapter is the one trainable adapter, on the original base.
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.peft_config["default"].revision = config["model_revision"]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name: raise RuntimeError("Only LoRA parameters may be trained")
            parameter.data = parameter.data.float()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    return model


def load_evaluation_model(config, method, run_dir):
    from peft import PeftModel
    model = load_base(config)
    if method != "base": model = PeftModel.from_pretrained(model, run_dir / "adapters" / method, is_trainable=False)
    return model.eval()


def validate_config(config):
    revision = config.get("model_revision", "")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin a 40-character hexadecimal model commit")
    for name in ["sft_steps", "dpo_steps", "gradient_accumulation", "save_every", "max_new_tokens", "max_length", "dpo_examples", "bootstrap_resamples"]:
        if type(config.get(name)) is not int or config[name] < 1: raise ValueError(f"{name} must be positive integer")
    if config.get("eval_limit") is not None: raise ValueError("Evaluate the complete fixed splits")
    for name in ["learning_rate_sft", "learning_rate_dpo", "beta"]:
        if not math.isfinite(config[name]) or config[name] <= 0: raise ValueError(f"Invalid {name}")


def protocol_for(config):
    return {"methods": ["base", "sft", "dpo"], "reference_baseline": "rule_engine",
        "selection": "None: fixed steps and unchanged decoding; no development or test tuning",
        "primary_metric": "structured_pass", "test_evaluation_after_both_training_stages": True,
        "max_new_tokens": config["max_new_tokens"], "do_sample": False, "adapter_merging": False,
        "reply_semantic_quality_measured_by_primary_metric": False}


def validate_inputs(run_dir):
    """Reject edited configuration, data, source, or a changed software/GPU environment."""
    from support_data import validate_dataset
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "inputs_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Run has no staged input seal; legacy recovery must be explicitly audited before continuation")
    manifest = read_json(manifest_path)
    required = {"config.json", "protocol.json", "environment.json", "data_manifest.json", "data_audit.json",
        "code/train_evaluate.py", "code/support_data.py", "data/train.jsonl", "data/development.jsonl", "data/test.jsonl"}
    if set(manifest) != required: raise ValueError("Input manifest coverage differs")
    for relative, expected in manifest.items():
        if digest(run_dir / relative) != expected: raise ValueError(f"Saved input changed: {relative}")
    for filename in ["train_evaluate.py", "support_data.py"]:
        if digest(Path(__file__).with_name(filename)) != manifest["code/" + filename]:
            raise ValueError(f"Runner/source differs from saved experiment: {filename}; restore run/code version")
    config = read_json(run_dir / "config.json"); validate_config(config)
    if read_json(run_dir / "protocol.json") != protocol_for(config): raise ValueError("Protocol differs")
    data = {split: read_jsonl(run_dir / "data" / (split + ".jsonl")) for split in ["train", "development", "test"]}
    if validate_dataset(data) != read_json(run_dir / "data_audit.json"): raise ValueError("Dataset audit differs")
    if config["dpo_examples"] > len(data["train"]): raise ValueError("DPO subset exceeds training data")
    return config, data, read_json(run_dir / "run_status.json")


def validate_environment(run_dir):
    current = environment_info(); original = read_json(run_dir / "environment.json")
    if current != original:
        raise ValueError("Runtime differs from saved experiment (Python/packages/GPU/CUDA); use the recorded environment")
    return current


def validate_adapter(run_dir, stage):
    directory = run_dir / "adapters" / stage
    manifest = read_json(directory / "adapter_manifest.json")
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "adapter_manifest.json"}
    if actual != set(manifest) or "adapter_model.safetensors" not in actual: raise ValueError(f"Incomplete {stage} adapter")
    for filename, expected in manifest.items():
        if digest(directory / filename) != expected: raise ValueError(f"Changed {stage} adapter: {filename}")
    info = read_json(run_dir / "adapter_backups.json")[stage]
    if digest(directory / "adapter_manifest.json") != info["adapter_manifest_sha256"]:
        raise ValueError(f"{stage} adapter manifest changed")
    config = read_json(run_dir / "config.json"); adapter_config = read_json(directory / "adapter_config.json")
    if adapter_config.get("revision") != config["model_revision"] or adapter_config.get("base_model_name_or_path") != config["model_id"]:
        raise ValueError(f"{stage} adapter base/revision differs")
    if digest(directory / "adapter_model.safetensors") != info["adapter_weights_sha256"]:
        raise ValueError(f"{stage} adapter disagrees with completed-stage record")
    archive = run_dir.parent / info["zip_name"]
    if not archive.is_file() or digest(archive) != info["zip_sha256"]: raise ValueError(f"Missing or changed {stage} backup ZIP")
    return info


def validate_saved_run(run_dir, required_stages=("sft",)):
    """Read-only validation for notebook reattachment; does not load models."""
    out = Path(run_dir).resolve(); config, data, status = validate_inputs(out)
    for stage in required_stages:
        if stage not in ("sft", "dpo") or stage not in status["completed_stages"]:
            raise ValueError(f"Required completed stage is missing: {stage}")
        validate_adapter(out, stage)
        logs = read_jsonl(out / (stage + "_training_log.jsonl"))
        if [r.get("step") for r in logs] != list(range(1, config[stage + "_steps"] + 1)):
            raise ValueError(f"Incomplete {stage} training log")
        timing = read_json(out / "timing.json")[stage]
        if timing.get("optimizer_steps") != config[stage + "_steps"]:
            raise ValueError(f"Incomplete {stage} timing")
    return out


def stage_status(run_dir, status, stage, state, **extra):
    status.update(status=state, active_stage=stage, benchmark_completed=False, **extra)
    status.pop("error", None); status.pop("error_type", None); status.pop("failed_stage", None)
    write_json(run_dir / "run_status.json", status)


def interrupted(run_dir, status, stage, exc):
    status.update(status="interrupted", active_stage=stage, failed_stage=stage,
        error_type=type(exc).__name__, error=str(exc), benchmark_completed=False)
    write_json(run_dir / "run_status.json", status)
    try: finalize_evidence(run_dir, "partial_evidence")
    except Exception as export_error: print(f"Partial evidence export failed: {export_error}", flush=True)


def release_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_sft(config_path="config.json", output_root="artifacts"):
    """Start one new experiment, persist the SFT adapter and return before DPO."""
    from transformers import set_seed
    from support_data import build_dataset, validate_dataset
    require_cuda()
    config = read_json(config_path); validate_config(config)
    set_seed(config["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
    output_root = Path(output_root).resolve()
    if (output_root / "run_status.json").exists(): raise ValueError("output_root is an existing run; do not retrain it")
    name = "generation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = output_root / name; out.mkdir(parents=True)
    print(f"Persistent run folder: {out}", flush=True)
    write_json(out / "config.json", config); write_json(out / "environment.json", environment_info())
    write_json(out / "protocol.json", protocol_for(config))
    (out / "code").mkdir(); (out / "data").mkdir()
    for filename in ["train_evaluate.py", "support_data.py"]:
        shutil.copy2(Path(__file__).with_name(filename), out / "code" / filename)
    status = {"status": "running", "active_stage": "sft", "completed_stages": [],
              "benchmark_completed": False, "evaluation_complete": False}
    write_json(out / "run_status.json", status)
    write_json(out / "timing.json", {}); write_json(out / "adapter_backups.json", {})
    model = None
    try:
        data = build_dataset(config["seed"])
        if config["dpo_examples"] > len(data["train"]): raise ValueError("DPO subset exceeds training data")
        write_json(out / "data_audit.json", validate_dataset(data))
        for split, rows in data.items(): write_jsonl(out / "data" / (split + ".jsonl"), rows)
        write_json(out / "data_manifest.json", {s: {"n": len(rows), "sha256": digest(out / "data" / (s + ".jsonl"))} for s, rows in data.items()})
        sealed = ["config.json", "protocol.json", "environment.json", "data_manifest.json", "data_audit.json",
            "code/train_evaluate.py", "code/support_data.py", "data/train.jsonl", "data/development.jsonl", "data/test.jsonl"]
        write_json(out / "inputs_manifest.json", {name: digest(out / name) for name in sealed})
        tokenizer = load_tokenizer(config)
        encoded = [(encode_completion(tokenizer, r["prompt"], r["completion"], config["max_length"]),
                    encode_completion(tokenizer, r["prompt"], r["rejected"], config["max_length"])) for r in data["train"]]
        for split in ["development", "test"]:
            for row in data[split]:
                length = len(tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True))
                if length + config["max_new_tokens"] > config["max_length"]: raise ValueError(f"{split}/{row['id']}: generation budget too long")
        started = time.perf_counter(); model = load_training_model(config)
        timings = {"training_model_loading_seconds": time.perf_counter() - started}
        trainable, total = model.get_nb_trainable_parameters()
        write_json(out / "parameters.json", {"trainable_adapter_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / total, "lora_dropout": 0.0, "base_storage": "NF4 with double quantization",
            "compute_dtype": "float16", "adapter_dtype": "float32"})
        timings["sft"] = training_loop(model, tokenizer, encoded, data["train"], config, out, "sft")
        write_json(out / "timing.json", timings)
        backup = stage_backup(model, tokenizer, out, "sft", None)
        write_json(out / "adapter_backups.json", {"sft": backup})
        status["completed_stages"] = ["sft"]
        stage_status(out, status, "sft", "sft_complete")
        finalize_evidence(out, "sft_evidence")
        print(f"SFT saved. Next: run_dpo({str(out)!r})", flush=True)
        return out
    except BaseException as exc:
        interrupted(out, status, "sft", exc); raise
    finally:
        del model; release_gpu()


def run_dpo(run_dir):
    """Continue exactly the saved SFT adapter; never repeat an attempted DPO stage."""
    from transformers import set_seed
    out = Path(run_dir).resolve(); config, data, status = validate_inputs(out)
    if "dpo" in status["completed_stages"]:
        validate_saved_run(out, ("sft", "dpo"))
        print(f"DPO already complete; unchanged: {out}", flush=True)
        return out
    if status["completed_stages"] != ["sft"]: raise ValueError("A completed SFT stage is required")
    if (out / "dpo_attempt.json").exists(): raise ValueError("DPO was already attempted; automatic optimizer-state resume is not supported")
    validate_adapter(out, "sft"); require_cuda(); validate_environment(out)
    set_seed(config["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = load_tokenizer(config)
    subset = np.random.default_rng(config["seed"] + 2).permutation(len(data["train"]))[:config["dpo_examples"]].tolist()
    rows = [data["train"][i] for i in subset]
    encoded = [(encode_completion(tokenizer, r["prompt"], r["completion"], config["max_length"]),
                encode_completion(tokenizer, r["prompt"], r["rejected"], config["max_length"])) for r in rows]
    write_json(out / "dpo_training_ids.json", [r["id"] for r in rows])
    write_json(out / "training_mixture.json", {
        "sft_categories": dict(Counter(r["category"] for r in data["train"])),
        "dpo_categories": dict(Counter(r["category"] for r in rows)),
        "dpo_rejection_types": dict(Counter(r["rejection_type"] for r in rows)),
        "subset_selection": "Fixed seeded random subset of training rows; no held-out information used"})
    stage_status(out, status, "reference_scoring", "running")
    model = None; timings = read_json(out / "timing.json")
    try:
        started = time.perf_counter(); model = load_training_model(config, out / "adapters" / "sft")
        timings["dpo_model_loading_seconds"] = time.perf_counter() - started
        reference, info = reference_scores(model, encoded, rows, out)
        write_json(out / "dpo_reference.json", info)
        write_json(out / "dpo_attempt.json", {"started_at": datetime.now(timezone.utc).isoformat(),
            "sft_weights_sha256": digest(out / "adapters/sft/adapter_model.safetensors"),
            "reference_sha256": info["sha256"], "continued_adapter": "saved SFT on original pinned base; no merge"})
        stage_status(out, status, "dpo", "running")
        timings["dpo"] = training_loop(model, tokenizer, encoded, rows, config, out, "dpo", reference)
        write_json(out / "timing.json", timings)
        if digest(out / "dpo_reference_logps.jsonl") != info["sha256"]: raise ValueError("Frozen reference changed")
        backup = stage_backup(model, tokenizer, out, "dpo", None)
        backups = read_json(out / "adapter_backups.json"); backups["dpo"] = backup
        write_json(out / "adapter_backups.json", backups)
        status["completed_stages"] = ["sft", "dpo"]
        stage_status(out, status, "dpo", "training_complete")
        finalize_evidence(out, "training_evidence")
        print(f"DPO saved. Next: evaluate_saved({str(out)!r})", flush=True)
        return out
    except BaseException as exc:
        write_json(out / "timing.json", timings)
        interrupted(out, status, status.get("active_stage", "dpo"), exc); raise
    finally:
        del model; release_gpu()


def record_hash(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != "record_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evaluation_plan(data):
    return [(method, split, row) for method in ["base", "sft", "dpo", "rule_engine"]
            for split in ["development", "test"] for row in data[split]]


def validate_predictions(records, plan, config, provenance):
    from support_data import score_response, rule_response
    if len(records) > len(plan): raise ValueError("Too many predictions")
    for record, (method, split, row) in zip(records, plan):
        if (record.get("method"), record.get("split"), record.get("id")) != (method, split, row["id"]):
            raise ValueError("Saved predictions are not the canonical uninterrupted prefix")
        if record.get("provenance_sha256") != provenance or record.get("record_sha256") != record_hash(record):
            raise ValueError("Saved prediction contents/provenance changed")
        response = record.get("response")
        if not isinstance(response, str) or record.get("category") != row["category"]:
            raise ValueError("Malformed prediction")
        if record.get("scores") != score_response(row, response) or type(record.get("hit_token_limit")) is not bool:
            raise ValueError("Prediction scores or limit flag differ")
        tokens = record.get("generated_tokens_including_eos"); seconds = record.get("generation_seconds")
        if method == "rule_engine":
            if response != rule_response(row) or tokens is not None or seconds is not None or record["hit_token_limit"]:
                raise ValueError("Rule baseline differs")
        elif type(tokens) is not int or not 1 <= tokens <= config["max_new_tokens"] or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid generation timing/token count")
        elif record["hit_token_limit"] and tokens != config["max_new_tokens"]:
            raise ValueError("Invalid token-limit count")


def export_evaluation(out, config, data, predictions):
    from support_data import summarize
    indexed = {(p["method"], p["split"], p["id"]): p for p in predictions}
    metrics = {method: {split: summarize(data[split], [indexed[method, split, r["id"]]["response"] for r in data[split]])
        for split in ["development", "test"]} for method in ["base", "sft", "dpo", "rule_engine"]}
    write_json(out / "metrics.json", metrics)
    write_json(out / "paired_comparisons.json", paired_comparisons(predictions, config["seed"], config["bootstrap_resamples"]))
    import io
    buffer = io.StringIO(newline="")
    fields = ["method", "split"] + [k for k, v in metrics["base"]["test"].items() if not isinstance(v, (dict, list))]
    writer = csv.DictWriter(buffer, fieldnames=fields); writer.writeheader()
    for method, splits in metrics.items():
        for split, values in splits.items(): writer.writerow({"method": method, "split": split, **{k: values[k] for k in fields[2:]}})
    atomic_text(out / "metrics_summary.csv", buffer.getvalue())
    rng = random.Random(config["seed"]); ids = set()
    for category in sorted({r["category"] for r in data["test"]}):
        ids.update(rng.sample(sorted(r["id"] for r in data["test"] if r["category"] == category), 2))
    review = [p for p in predictions if p["split"] == "test" and p["method"] != "rule_engine" and p["id"] in ids]
    rng.shuffle(review); opaque = rng.sample(range(10**8, 10**9), len(review))
    mapping = [{"review_id": f"R{opaque[i]}", "method": p["method"], "id": p["id"]} for i, p in enumerate(review)]
    write_json(out / "review_key.json", {"instructions": "Open only after recording blinded ratings; method identity is hidden in the review CSV, not cryptographically protected.", "mapping": mapping})
    row_map = {r["id"]: r for r in data["test"]}; buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["review_id", "prompt", "response", "reply_factual", "instruction_following", "unsupported_advice", "notes"])
    writer.writeheader()
    for i, p in enumerate(review): writer.writerow({"review_id": mapping[i]["review_id"], "prompt": row_map[p["id"]]["prompt"], "response": p["response"]})
    atomic_text(out / "manual_review.csv", buffer.getvalue())


def evaluate_saved(run_dir):
    """Evaluate saved stages, checkpoint every ten responses, and safely retry gaps."""
    from support_data import rule_response, score_response
    from transformers import set_seed
    out = Path(run_dir).resolve(); config, data, status = validate_inputs(out)
    if status["completed_stages"] != ["sft", "dpo"]: raise ValueError("Both saved training stages are required")
    adapter_info = {stage: validate_adapter(out, stage) for stage in ["sft", "dpo"]}
    receipt_path = out.parent / (out.name + "_completion.json")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        archive = out.parent / (out.name + "_results.zip")
        if status.get("status") != "complete" or digest(out / "manifest_sha256.json") != receipt["manifest_sha256"] or not archive.is_file() or digest(archive) != receipt["results_zip_sha256"]:
            raise ValueError("Finalized evidence or its archive changed; refusing to overwrite it")
        from scripts.verify_results import verify
        verify(out)
        print(f"Already complete; unchanged: {out}", flush=True)
        return out
    # The receipt is the final commit. Without it, an interrupted ZIP export can
    # be completed using validated saved predictions, without loading a model.
    environment = read_json(out / "environment.json")
    provenance = {"inputs_manifest_sha256": digest(out / "inputs_manifest.json"),
        "adapter_weights_sha256": {s: v["adapter_weights_sha256"] for s, v in adapter_info.items()},
        "decoding": {"do_sample": False, "max_new_tokens": config["max_new_tokens"], "max_length": config["max_length"],
            "use_cache": True, "batch_size": 1, "generated_continuation_only": True, "allow_tf32": False},
        "environment": environment, "case_order": "base/sft/dpo/rule_engine; development/test; saved data order"}
    path = out / "evaluation_manifest.json"
    if path.exists() and read_json(path) != provenance: raise ValueError("Evaluation settings/provenance changed")
    if not path.exists(): write_json(path, provenance)
    provenance_hash = digest(path); plan = evaluation_plan(data)
    predictions = read_jsonl(out / "predictions.jsonl") if (out / "predictions.jsonl").exists() else []
    validate_predictions(predictions, plan, config, provenance_hash)
    if len(predictions) < len(plan):
        require_cuda(); validate_environment(out)
        set_seed(config["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = load_tokenizer(config)
    model = None; current_method = None
    timings = read_json(out / "timing.json")
    sessions_path = out / "evaluation_sessions.json"
    sessions = read_json(sessions_path) if sessions_path.exists() else []
    for previous in sessions:
        if previous.get("status") == "running":
            previous.update(status="interrupted", error_type="UncleanSessionEnd", ending_predictions=len(predictions))
    session = {"started_at": datetime.now(timezone.utc).isoformat(), "starting_predictions": len(predictions), "status": "running"}
    sessions.append(session); write_json(sessions_path, sessions)
    stage_status(out, status, "evaluation", "running", evaluation_complete=False)
    print(f"Evaluation: {len(predictions)}/{len(plan)} saved; resuming missing responses only.", flush=True)
    try:
        for method, split, row in plan[len(predictions):]:
            if method != current_method:
                if model is not None: del model; model = None; release_gpu()
                current_method = method
                if method != "rule_engine":
                    started = time.perf_counter(); model = load_evaluation_model(config, method, out)
                    key = method + "_evaluation_loading_seconds"
                    timings[key] = timings.get(key, 0.0) + time.perf_counter() - started
                    torch.cuda.reset_peak_memory_stats()
                    write_json(out / "timing.json", timings)
                print(f"Evaluating {method}: {split}", flush=True)
            if method == "rule_engine":
                text, timing = rule_response(row), {"generation_seconds": None, "generated_tokens_including_eos": None, "hit_token_limit": False}
            else:
                text, timing = generate(model, tokenizer, row, config)
                key = method + "_evaluation_peak_allocated_gpu_bytes"
                timings[key] = max(timings.get(key, 0), torch.cuda.max_memory_allocated())
            record = {"method": method, "split": split, "id": row["id"], "category": row["category"],
                "response": text, "scores": score_response(row, text), **timing, "provenance_sha256": provenance_hash}
            record["record_sha256"] = record_hash(record); predictions.append(record)
            if len(predictions) % 10 == 0 or len(predictions) == len(plan):
                write_jsonl(out / "predictions.jsonl", predictions); write_json(out / "timing.json", timings)
                print(f"Evaluation saved: {len(predictions)}/{len(plan)} ({method}/{split})", flush=True)
        validate_predictions(predictions, plan, config, provenance_hash)
        export_evaluation(out, config, data, predictions)
        timings["scope"] = "Single-run wall times; generation includes GPU synchronization; recovery loading times accumulated; no production cost measured"
        write_json(out / "timing.json", timings)
        session.update(status="complete", ending_predictions=len(predictions)); write_json(sessions_path, sessions)
        status.update(status="complete", active_stage="evaluation", benchmark_completed=True, evaluation_complete=True, human_review_completed=False)
        write_json(out / "run_status.json", status)
        archive = finalize_evidence(out)
        write_json(receipt_path, {"manifest_sha256": digest(out / "manifest_sha256.json"),
            "results_zip_sha256": digest(archive), "status": "complete"})
        return out
    except BaseException as exc:
        write_jsonl(out / "predictions.jsonl", predictions); write_json(out / "timing.json", timings)
        session.update(status="interrupted", ending_predictions=len(predictions), error_type=type(exc).__name__)
        write_json(sessions_path, sessions); interrupted(out, status, "evaluation", exc); raise
    finally:
        del model; release_gpu()


def run(config_path="config.json", output_root="artifacts", stage_callback=None):
    """Legacy all-stage wrapper. Prefer separate stage calls for recoverable notebooks."""
    out = run_sft(config_path, output_root)
    if stage_callback is not None: stage_callback("sft", out, out.parent / read_json(out / "adapter_backups.json")["sft"]["zip_name"])
    run_dpo(out)
    if stage_callback is not None: stage_callback("dpo", out, out.parent / read_json(out / "adapter_backups.json")["dpo"]["zip_name"])
    return evaluate_saved(out)

def smoke():
    """Execute the integration test against random weights; never download a model."""
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), pattern="test_training.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Training smoke did not complete every test; install the required packages")
    return {"status": "passed", "tests_run": result.testsRun, "model": "tiny randomly initialized Llama",
            "scope": "CPU mechanics only; not TinyLlama, GPU quantization, benchmark quality, or runtime validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke(), indent=2) if args.smoke else str(run(args.config, args.output_root)))
