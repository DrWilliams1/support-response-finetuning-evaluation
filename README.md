# Customer-Support Response Fine-Tuning: SFT vs DPO

A controlled experiment found that SFT improved structured return-policy responses, while subsequent DPO reduced success. Rules plus templates solved the structured task exactly by construction.

**Status: completed run; exported evidence independently recomputed and checked. Human review remains pending (0/36 responses rated).**

## Test results

120 held-out synthetic tickets per method; one training seed. Structured success requires valid schema plus correct ticket ID, decision, and reason; it does not establish reply factuality.

| Method | Valid JSON | Structured success | Successful cases |
|---|---:|---:|---:|
| Released TinyLlama chat model | 0% | 0% | 0/120 |
| SFT | 100% | 72.5% | 87/120 |
| SFT followed by DPO | 90% | 30.8% | 37/120 |
| Rules + templates | 100% | 100% | 120/120 |

DPO minus SFT was −41.7 percentage points (descriptive paired bootstrap 95% interval: −53.3 to −30.0 points). DPO improved 11 cases and worsened 61; 48 tied. This is evidence about this fixed training configuration, not a general conclusion against DPO.

The base model reached the 128-token output limit on 118/120 test cases. Its zero structured score therefore reflects this particular output contract and decoding budget; it does not establish zero underlying policy knowledge. SFT had no token-limit hits; DPO had one.

## Error analysis and business meaning

SFT passed all damage, missing-receipt, eligible, and deadline-boundary cases, but only 7/20 outside-window cases and 0/20 used-item cases. DPO passed 20/20 damage and 17/20 outside-window cases but 0/20 in each remaining category. Its decline was not merely a formatting problem.

For structured policy inputs, retain deterministic decisions and templates as the business baseline. The model adds training and failure modes without beating it here. A separate future experiment could evaluate extracting facts from natural customer language or drafting replies with human review. No deployment-readiness or staffing-savings claim is supported.

Recorded SFT training took 198.0 seconds; DPO took 192.6 seconds, excluding separate model loading. Mean test generation time was 7.808 seconds for base, 5.257 for SFT, and 5.130 for DPO. Different output lengths affect latency; these are not production throughput or billing measurements.

## Reproduce and inspect

- [Executed notebook](Executed_Support_Response_FineTuning_Evaluation.ipynb): original displayed outputs preserved unchanged, with a publication-status note explaining historical preparation language.
- [Runnable notebook](Support_Response_FineTuning_Evaluation.ipynb): fixed protocol, staged Drive saves, recovery, and current benchmark status.
- [Evidence](results/generation_20261125T164816Z-6faa0a/): all predictions, metrics, configuration, source snapshots, timing, review worksheet, and original integrity manifest.
- [Code walkthrough](CODE_WALKTHROUGH.md), [business case](BUSINESS_CASE.md), [recovery](RECOVERY.md), and [attribution](ATTRIBUTION.md).

Install requirements in a CUDA environment to rerun. To verify existing results without loading models:

```bash
python scripts/verify_results.py results/generation_20261125T164816Z-6faa0a
```

The evidence contains 720 predictions: four methods × (60 development + 120 test cases). Training used 480 synthetic tickets and 240 training preference pairs. Fact combinations are disjoint, but wording templates are shared. The base is the released TinyLlama-1.1B-Chat-v1.0 checkpoint, pinned in config.json; SFT uses 120 steps and DPO 60. Model weights are excluded.

Verification recomputes scores, metrics and intervals, checks data and training orders, and validates file hashes. It checks internal consistency, not independent authentication of GPU execution. Complete a separate copy of manual_review.csv before claiming written-reply quality; leave original evidence unchanged.

[Local validation](validation/local_validation.json) records historical pre-run checks. Its pending benchmark status and unexecuted GPU/Drive items describe that earlier validation session; they do not supersede the completed-run evidence and verified results reported here.
