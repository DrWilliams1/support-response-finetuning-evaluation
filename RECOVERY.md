# Continue a saved experiment

The revised notebook saves directly under `My Drive/Support_Response_FineTuning/runs`. It does not depend on a browser download finishing while computation is running. A download progress bar is not confirmation that a file reached your computer.

## New run

Use a Colab T4 runtime, run the setup cells, and authorize the Drive mount in 6A. Leave `EXISTING_RUN_NAME` empty for a first run. If folders already exist, either select one or explicitly set `START_ANOTHER_RUN=True` to create an additional experiment. Nothing is deleted.

Run 6B for SFT and wait for its saved message. Run 6C for DPO and wait for its saved message. Each completed stage has its adapter, tokenizer, hash manifest, and training records saved before evaluation. Run 6D for evaluation. It prints progress and checkpoints predictions; there are 540 model responses, so this stage can take longer than training.

## Runtime disconnected during evaluation

1. Reopen the same revised notebook in a fresh GPU session and run setup.
2. Mount Drive in 6A. Copy the exact saved `generation_...` folder name into `EXISTING_RUN_NAME`, then rerun 6A.
3. Run 6D. The runner checks saved configuration, source, data, adapters, environment, and predictions. It retains valid responses and generates missing cases without training.
4. Continue the results, review, and verification cells. Download the evidence ZIP from Drive and save the executed notebook separately.

Do not change code, model revision, dependencies, data, or decoding settings to bypass a continuation failure. Keep the original run and inspect the reported mismatch. This recovery path applies to runs made with this revised workflow; importing adapters from a different notebook requires separate provenance checks.

A hard disconnect may lose up to nine responses since the last checkpoint; those responses are regenerated. Reported generation timings describe retained responses, not complete billed time including work lost before a checkpoint.

## Runtime disconnected during training

A completed SFT stage can be used to start DPO. A completed DPO stage can be evaluated. A stage stopped before its completion record cannot be treated as complete, even if an intermediate adapter exists. Snapshots do not include optimizer state, so automatic mid-stage training resume is not implemented. Keep the interrupted run and deliberately start a new run if a completed stage cannot be recovered.

## Original temporary run folder is gone

Check completed downloads and any Drive backup made before the loss. Valid SFT/DPO adapter ZIPs can support a separately verified evaluation-only recovery, although unavailable training timings or logs must remain unavailable. Screenshots and progress bars cannot recreate model weights. If no saved adapters remain, a new training run is required; the revised workflow protects completed stages in Drive.

## Download after completion

Open Google Drive, navigate to `Support_Response_FineTuning/runs`, and download the final `*_results.zip`. Adapter ZIPs sit beside the run folder. Human review files are inside the run folder; rate a copy outside the original evidence folder so its manifest stays intact. An optional standalone notebook cell can request the results ZIP through the browser after all computation finishes.
