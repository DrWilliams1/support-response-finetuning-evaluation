# Code walkthrough

`support_data.py` is the business contract. `rule_decision` applies rules in priority order; damage overrides missing receipts, which override deadline and unused-item checks. The deadline itself is included. `rule_response` turns that deterministic result into a complete response and supplies the non-model baseline.

`build_dataset` samples reproducible cases from a finite pool. `fact_key` groups the actual policy inputs without ticket IDs, preventing a renamed copy of the same case from crossing splits. Categories are balanced. This controlled task makes errors interpretable, but its repeated language and simple rules limit generalization claims.

`encode_completion` uses the tokenizer's chat template and masks prompt tokens with `-100`. SFT therefore learns the reference answer rather than predicting the supplied policy. Overlong examples fail instead of silently dropping facts or answer tokens. LoRA updates a small set of matrices while the quantized backbone stays frozen; that lowers training memory at the cost of a restricted update space.

The SFT adapter is saved before DPO. `run_sft` and `run_dpo` are separate stages. DPO reloads and continues **the same saved SFT adapter**, so its final checkpoint includes both stages' changes. There is no merge between stages. Each saved adapter loads against the same pinned original chat backbone; the final adapter must not be stacked onto the SFT adapter.

Before DPO updates, the runner computes and freezes the SFT model's chosen and rejected completion log probabilities for the 240 training pairs. `completion_logps` shifts logits for next-token prediction, ignores prompt/padding positions, and **sums** completion scores. Precomputing this fixed reference saves memory; it is valid for these fixed pairs and is not a changing reference policy.

`dpo_loss` compares the policy's chosen-minus-rejected score with the frozen SFT reference difference, scaled by beta, using a negative log-sigmoid loss. It rewards the supplied preference relationship without a separate reward model. At the unchanged reference policy, the loss should be approximately `log(2)`. Lower loss does not guarantee better test responses.

`score_response` rejects invalid JSON and duplicate keys, checks the exact four-field schema and reply length, then compares structured fields with the rule-derived answer. The `unsupported_reply` rejection intentionally exposes its boundary: correct fields can coexist with an invented payment promise. Blinded review assesses factual consistency and helpfulness beyond the automatic metric.

`evaluate_saved` validates saved artifacts and runs inference without a training path. Progress and predictions are persisted during evaluation. On a retry, validated existing responses are retained and only missing cases are generated. Saved code, configuration, data, adapters, and environment constrain continuation so changed settings cannot silently produce a mixed comparison. Training timing is saved before evaluation begins.

The notebook writes run artifacts directly to mounted Drive. Browser downloads are an optional action after evaluation and are not used to establish that a stage was saved. This separates compute completion, persistent saves, and delivery to the user's computer.

The tests target token masking, score shifting/summing, DPO gradients, split isolation, rule boundaries, adapter save/reload, and staged interruption/retry behavior. A tiny randomly initialized CPU model checks the update path; mocked stage tests check orchestration. Neither establishes TinyLlama benchmark quality or validates Colab GPU/Drive behavior.
