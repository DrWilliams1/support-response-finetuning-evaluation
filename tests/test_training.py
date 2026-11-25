"""Training-mechanics checks; no model download or benchmark inference."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_evaluate import completion_logps, dpo_loss, encode_completion


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self, completion_ids=None):
        self.completion_ids = [4, 5] if completion_ids is None else completion_ids
        self.chat_calls = []
        self.token_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_calls.append((messages, tokenize, add_generation_prompt))
        return [1, 2, 3]

    def __call__(self, text, *, add_special_tokens):
        self.token_calls.append((text, add_special_tokens))
        return {"input_ids": self.completion_ids.copy()}


class CompletionEncodingTests(unittest.TestCase):
    def test_chat_prefix_and_completion_only_labels(self):
        tokenizer = FakeTokenizer()
        encoded = encode_completion(tokenizer, "Policy and ticket", '{"decision":"yes"}', 6)
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 9])
        self.assertEqual(encoded["attention_mask"], [1] * 6)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5, 9])
        self.assertEqual(tokenizer.chat_calls, [
            ([{"role": "user", "content": "Policy and ticket"}], True, True)])
        self.assertEqual(tokenizer.token_calls, [('{"decision":"yes"}', False)])

    def test_existing_eos_is_not_duplicated(self):
        encoded = encode_completion(FakeTokenizer([4, 5, 9]), "prompt", "answer", 6)
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 9])
        self.assertEqual(encoded["labels"][-1], 9)

    def test_overlong_example_fails_instead_of_truncating(self):
        with self.assertRaises(ValueError):
            encode_completion(FakeTokenizer(), "prompt", "answer", 5)


class CompletionLogProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[
            [0.2, -0.1, 0.3, 1.1],
            [1.2, 0.3, -0.5, 0.7],
            [-0.3, 1.4, 0.1, 0.0],
            [0.5, 0.2, 0.1, 0.3],
            [0.1, 0.5, 0.4, 0.8],
        ]], dtype=torch.float64)
        self.labels = torch.tensor([[-100, -100, 2, 1, -100]])

    def test_next_token_shift_and_sum_not_mean(self):
        expected = sum([
            -0.5 - math.log(sum(math.exp(x) for x in [1.2, 0.3, -0.5, 0.7])),
            1.4 - math.log(sum(math.exp(x) for x in [-0.3, 1.4, 0.1, 0.0])),
        ])
        actual = completion_logps(self.logits, self.labels)
        self.assertEqual(tuple(actual.shape), (1,))
        self.assertAlmostEqual(actual.item(), expected, places=6)
        self.assertNotAlmostEqual(actual.item(), expected / 2, places=3)

    def test_prompt_and_padding_scores_are_ignored(self):
        before = completion_logps(self.logits, self.labels)
        changed = self.logits.clone()
        changed[:, [0, 3, 4], :] = changed.new_tensor([100.0, -100.0, 0.0, 1.0])
        torch.testing.assert_close(completion_logps(changed, self.labels), before)

    def test_gradient_reaches_only_scored_prediction_positions(self):
        logits = self.logits.clone().requires_grad_()
        completion_logps(logits, self.labels).sum().backward()
        self.assertTrue(torch.equal(logits.grad[:, [0, 3, 4]], torch.zeros_like(logits.grad[:, [0, 3, 4]])))
        self.assertGreater(logits.grad[:, [1, 2]].abs().sum().item(), 0)

    def test_independent_batch_rows_and_zero_scored_tokens(self):
        logits = self.logits.repeat(2, 1, 1)
        labels = self.labels.repeat(2, 1)
        labels[1] = -100
        result = completion_logps(logits, labels)
        self.assertAlmostEqual(result[0].item(), completion_logps(self.logits, self.labels)[0].item())
        self.assertEqual(result[1].item(), 0.0)


class DPOLossTests(unittest.TestCase):
    def test_reference_policy_has_log_two_loss(self):
        chosen = torch.tensor([-7.0, -10.0])
        rejected = torch.tensor([-8.0, -12.0])
        loss = dpo_loss(chosen, rejected, chosen.clone(), rejected.clone(), 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)

    def test_gradient_favors_chosen_over_rejected(self):
        chosen = torch.tensor([-7.0], requires_grad=True)
        rejected = torch.tensor([-8.0], requires_grad=True)
        loss = dpo_loss(chosen, rejected, torch.tensor([-7.0]), torch.tensor([-8.0]), 0.2)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        self.assertAlmostEqual(chosen.grad.item(), -0.1, places=6)
        self.assertAlmostEqual(rejected.grad.item(), 0.1, places=6)

    def test_reference_adjustment_and_batch_mean(self):
        policy_chosen = torch.tensor([-4.0, -8.0], dtype=torch.float64)
        policy_rejected = torch.tensor([-6.0, -7.0], dtype=torch.float64)
        ref_chosen = torch.tensor([-5.0, -8.0], dtype=torch.float64)
        ref_rejected = torch.tensor([-6.0, -8.0], dtype=torch.float64)
        expected = (math.log1p(math.exp(-0.3)) + math.log1p(math.exp(0.3))) / 2
        actual = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, 0.3)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_invalid_beta_is_rejected(self):
        x = torch.tensor([-1.0])
        for beta in [0, -0.1, float("nan"), float("inf")]:
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                dpo_loss(x, x, x, x, beta)


@unittest.skipUnless(importlib.util.find_spec("peft") and importlib.util.find_spec("transformers"),
                     "Tiny-model integration requires installed transformers and peft")
class TinyLlamaAdapterIntegrationTests(unittest.TestCase):
    def test_sft_then_dpo_preserves_base_and_roundtrips_adapter(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

        torch.manual_seed(2026)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=64,
                             bos_token_id=1, eos_token_id=2, pad_token_id=0,
                             attention_dropout=0.0)
        base = LlamaForCausalLM(config)
        base_initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
        adapter_initial = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(adapter_initial)
        self.assertTrue(all("lora_" in k for k in adapter_initial))
        chosen_ids = torch.tensor([[1, 3, 5, 7, 9, 2]])
        rejected_ids = torch.tensor([[1, 3, 5, 8, 10, 2]])
        chosen_labels, rejected_labels = chosen_ids.clone(), rejected_ids.clone()
        chosen_labels[:, :3] = rejected_labels[:, :3] = -100
        mask = torch.ones_like(chosen_ids)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model.train()
        loss = model(input_ids=chosen_ids, attention_mask=mask, labels=chosen_labels).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(p, adapter_initial[k]) for k, p in model.named_parameters()
                            if k in adapter_initial))

        model.eval()  # Reference and DPO policy scores use dropout disabled.
        with torch.no_grad():
            ref_chosen = completion_logps(model(input_ids=chosen_ids, attention_mask=mask).logits, chosen_labels)
            ref_rejected = completion_logps(model(input_ids=rejected_ids, attention_mask=mask).logits, rejected_labels)
        sft_adapter = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        pi_chosen = completion_logps(model(input_ids=chosen_ids, attention_mask=mask).logits, chosen_labels)
        pi_rejected = completion_logps(model(input_ids=rejected_ids, attention_mask=mask).logits, rejected_labels)
        dpo = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, 0.1)
        self.assertAlmostEqual(dpo.item(), math.log(2), places=6)
        dpo.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(p, sft_adapter[k]) for k, p in model.named_parameters()
                            if k in sft_adapter))
        for k, p in model.named_parameters():
            if k in frozen:
                self.assertTrue(torch.equal(p, frozen[k]), k)
        with torch.no_grad():
            expected_logits = model(input_ids=chosen_ids, attention_mask=mask).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored_base = LlamaForCausalLM(config)
            restored_base.load_state_dict(base_initial)
            restored = PeftModel.from_pretrained(restored_base, directory).eval()
            with torch.no_grad():
                actual_logits = restored(input_ids=chosen_ids, attention_mask=mask).logits
            torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
            with torch.inference_mode():
                generated = restored.generate(input_ids=chosen_ids[:, :3], attention_mask=mask[:, :3],
                    generation_config=GenerationConfig(do_sample=False, max_new_tokens=2,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0))
            self.assertTrue(torch.equal(generated[:, :3], chosen_ids[:, :3]))
            self.assertGreater(generated.shape[1], 3)
            self.assertLessEqual(generated.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
