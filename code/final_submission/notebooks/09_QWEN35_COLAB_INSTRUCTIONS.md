# Running Notebook 09 from GitHub in Google Colab

Notebook: `code/final_submission/notebooks/09_qwen35_classifier_colab.ipynb`

## How the new workflow works

The complete `Research Proj` workspace lives in GitHub. Notebook 09 reads its established dataset, feature contract, frozen split assignments and MLP metrics from a GitHub clone under `/content`.

Google Drive has a different purpose: it stores the large Qwen checkpoint, LoRA adapter, classification head and run outputs. Project inputs are not read from Drive.

```text
Local project -> GitHub -> Colab clone -> CUDA training -> Google Drive outputs
```

## 1. Push the complete project to GitHub

Create an empty repository on GitHub. Do not add a GitHub README or `.gitignore`, because this project already contains them.

From the top-level `Research Proj` folder on your Mac, run:

```bash
git init
git add .
git status
git commit -m "Prepare MSc project for Qwen Colab workflow"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY`. Review `git status` before committing: `.env` files, model weights, checkpoints and caches should not appear.

## 2. Open Notebook 09 in Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **GitHub**.
3. Paste the repository URL.
4. Open `code/final_submission/notebooks/09_qwen35_classifier_colab.ipynb`.

## 3. Select a CUDA GPU

Choose **Runtime > Change runtime type > T4 GPU** or another NVIDIA GPU, then press **Save**.

Do this before running the package setup. The notebook deliberately stops if CUDA is unavailable; it never falls back to CPU or Apple MPS.

## 4. Configure and clone the GitHub repository

Near the top, find **Load Project from GitHub** and edit:

```python
GITHUB_REPO_URL = "https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git"
REPO_ROOT = Path("/content/YOUR-REPOSITORY")
```

Use the same repository name in both places. The cell clones the repository if it is absent. If it already exists and `PULL_LATEST = True`, it runs a fast-forward-only `git pull`.

The notebook then verifies these files in the clone:

- `code/final_submission/data/final/kidney_transplant_assessments.csv`
- `code/final_submission/data/final/classifier_feature_list.json`
- `code/final_submission/processed_data/split_assignments.csv`
- `code/final_submission/results/baseline/test_metrics.csv`

Do not change or regenerate these files in Colab.

## 5. Private repositories and secrets

For a public repository, the normal HTTPS URL works without credentials.

For a private repository, do not put a GitHub token in the notebook, clone URL or saved output. Use Colab's **Secrets** panel or a temporary Git credential method. Never commit `.env` files, tokens, API keys or credentials.

Notebook 09 does not require a Qwen API key. It downloads the public model weights into the temporary Colab runtime and trains locally on the selected GPU.

## 6. Mount Google Drive

Run the `drive.mount("/content/drive")` cell and approve access to your own Drive account. The default output folder is:

```text
/content/drive/MyDrive/Qwen Experiment
```

You may change `QWEN_OUTPUT_ROOT` if desired. Do not point it back into the GitHub clone.

## 7. Run the notebook

Run cells from top to bottom. Expected user interactions are:

1. Edit the GitHub URL and repository-name placeholders.
2. Wait for package installation.
3. Approve Google Drive mounting.
4. Wait for the Qwen model download and initial GPU kernel compilation.
5. Wait for baseline training.

Do not start a second copy of the same training run in another Colab tab.

The input verification table must show `True` for all four required repository artefacts before model loading begins.

## 8. Recognise a successful CUDA setup

The environment table should show:

- `CUDA available: True`;
- an NVIDIA GPU such as `Tesla T4`;
- GPU memory;
- PyTorch, Transformers and Unsloth versions.

If CUDA is unavailable, select a GPU runtime, restart the session and rerun from the top.

## 9. Expected Google Drive outputs

Each run receives a UTC timestamp. Under `Qwen Experiment`, expect:

- the best LoRA adapter and saved two-class head;
- tokenizer/processor files;
- experiment configuration and feature order;
- serialisation specification and frozen threshold;
- frozen split-membership copy;
- training history and validation metrics;
- test metrics, probabilities and logits;
- MLP-versus-Qwen comparison;
- runtime and GPU-memory information;
- an artefact manifest containing all saved Drive paths.

The final completion table should contain only `True` values.

## 10. Pull later Codex changes into Colab

After Codex changes the local project:

```bash
git add .
git status
git commit -m "Describe the project update"
git push
```

In Colab, rerun the **Load Project from GitHub** cell. With `PULL_LATEST = True`, it pulls the new commit. Rerun only the setup or analysis cells affected by the change; restart from the top if model construction, data handling or package setup changed.

## 11. If the notebook fails

Copy these details back to Codex:

1. The section heading and complete traceback.
2. The environment table and GPU name.
3. The input verification table.
4. The last printed epoch or progress message.
5. `!nvidia-smi` output if CUDA or memory is mentioned.
6. Whether this was a fresh clone or a later `git pull`.
7. Any configuration values you edited, excluding credentials.

Never paste a token or API key into an error report. Do not bypass assertions, reduce the dataset, change the split or switch models without recording the methodological change first.
