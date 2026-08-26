# Intended EchoForge project structure

This structure is added beneath the existing `code/` directory. Existing Adult
Income files remain where they are.

```text
code/
├── AGENTS.md
├── README.md
├── requirements.txt
├── incoming/
│   └── EchoForge_Synthetic_Dataset_CSV_Package.zip
├── data/
│   └── echoforge/
│       ├── README.md
│       ├── config.json
│       ├── data_dictionary.csv
│       ├── file_manifest.csv
│       ├── data/
│       │   ├── raw/
│       │   ├── processed/
│       │   ├── splits/
│       │   └── manifests/
│       └── validation/
├── src/
│   └── echoforge/
│       ├── __init__.py
│       ├── paths.py
│       ├── data.py
│       ├── model.py
│       ├── training.py
│       ├── metrics.py
│       ├── unlearning.py
│       └── sisa.py
├── notebooks/
│   └── echoforge/
│       ├── 00_data_validation.ipynb
│       ├── 01_original_mlp.ipynb
│       ├── 02_full_retraining.ipynb
│       ├── 03_retain_finetuning.ipynb
│       ├── 04_gradient_ascent.ipynb
│       ├── 05_gradient_difference.ipynb
│       ├── 06_sisa.ipynb
│       └── 07_results_comparison.ipynb
├── tests/
│   └── echoforge/
├── models/
│   └── echoforge/
│       ├── original/
│       ├── full_retraining/
│       └── sisa/
└── results/
    └── echoforge/
        ├── metrics/
        ├── predictions/
        ├── figures/
        ├── configurations/
        └── logs/
```

Empty result/model directories may contain `.gitkeep` files. Do not add
generated checkpoints or fake result CSVs during scaffolding.
