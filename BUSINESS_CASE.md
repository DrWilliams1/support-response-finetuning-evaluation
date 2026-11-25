# Business application

A support team needs consistent return decisions and clear customer responses. This experiment measured whether adapting a small chat model improved formatting and adherence to a fictional return policy. It does not demonstrate lower handling time, higher satisfaction, or financial return.

For the exact structured inputs used here, **a rule engine plus templates is the practical first option**. The rules already determine the answer completely. That baseline achieved 100% structured success by construction and is transparent and easy to update when the policy changes.

| Option | Observed structured success | Business consideration |
|---|---:|---|
| Rules + templates | 100% (120/120) | Strong fit for these structured tickets |
| Released chat model | 0% (0/120) | Failed this output contract and token budget; not evidence of zero policy knowledge |
| SFT | 72.5% (87/120) | Improved structured responses but still failed 33 cases |
| SFT + DPO | 30.8% (37/120) | Added training and reduced success under this configuration |

These are results from 120 held-out synthetic tickets and one training seed. Structured success checks the schema, ticket ID, decision, and reason; it does not verify the factuality of the written reply. DPO reduced structured success by 41.7 percentage points relative to SFT, with 11 improved and 61 worsened cases. This result applies to the fixed configuration, not to DPO in general.

Category failures matter for a support workflow. SFT passed all damaged-item, missing-receipt, eligible, and deadline-boundary cases, but only 7/20 outside-window cases and 0/20 used-item cases. DPO passed 20/20 damaged-item and 17/20 outside-window cases, with no successes in the other categories. Its decline was not only a formatting problem.

**Human review remains pending: 0/36 responses rated.** Use a separate copy of the blinded worksheet to judge whether replies introduce promises or contradict facts. Preserve the original evidence. The measured structured results do not establish reliable free-text support or deployment readiness.

A useful next phase would add varied customer wording and missing or conflicting information, while leaving final policy decisions with the rule engine. Use separately collected, appropriately permitted examples, a new held-out set, and human review. Measure editing time, unsupported statements, escalation accuracy, latency, and operating cost before deciding whether a drafting model earns its place. No ROI estimate is claimed from this synthetic experiment.
