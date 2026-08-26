# MSc Machine-Unlearning Research Project

This repository contains the complete MSc project workspace, including research documentation, the synthetic kidney-transplant study, implementation code, notebooks and saved reproducibility artefacts.

The self-contained kidney-transplant submission is under `code/final_submission/`. Its authoritative classifier inputs are the synthetic assessment dataset, approved feature contract, frozen split assignments and saved MLP baseline metrics. Large Qwen model weights and training checkpoints are intentionally excluded from Git and saved to Google Drive.

## Google Colab Qwen Workflow

```text
Codex/local development
        ↓
Git commit and push
        ↓
Colab clone or pull
        ↓
run Notebook 09 on a CUDA GPU
        ↓
save large Qwen outputs to Google Drive
```

Notebook 09 reads its dataset, feature contract, frozen splits and MLP comparison metrics from the GitHub clone. Google Drive is used only for Qwen checkpoints, adapters, predictions and run results. See `code/final_submission/notebooks/09_QWEN35_COLAB_INSTRUCTIONS.md` for the beginner-friendly run procedure.
