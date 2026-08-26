# EchoForge Codex story index

Run one story at a time in this exact order. Copy the full contents of the
corresponding file into Codex. Do not ask Codex to implement the whole index in
one task.

| Order | Story | Depends on | Main result |
|---:|---|---|---|
| 00 | Repository audit and scaffold | Starter pack installed | Safe EchoForge folder structure |
| 01 | Install and verify dataset | US-00 | Immutable validated dataset |
| 02 | Data access layer | US-01 | Reusable scenario/split loaders |
| 03 | Preprocessing and MLP core | US-02 | Saved preprocessing and training APIs |
| 04 | Data-validation notebook | US-02 | Executed `00_data_validation.ipynb` |
| 05 | Original MLP notebook | US-03, US-04 | Seed-42 original checkpoint/predictions |
| 06 | Full-retraining references | US-05 | Four retrained references |
| 07 | Reference-based metrics | US-06 | Tested forgetting metrics |
| 08 | Retain fine-tuning | US-07 | Four-scenario results |
| 09 | Gradient ascent | US-07 | Checked ascent implementation/results |
| 10 | Gradient difference | US-07 | Checked balanced objective/results |
| 11 | SISA partition and original ensemble | US-05 | Reproducible SISA training structure |
| 12 | SISA deletion and reference | US-11 | Four SISA unlearning comparisons |
| 13 | Seed-42 comparison | US-08–US-10, US-12 | Core tables and plots |
| 14 | Repeat frozen configurations | US-13 | Seeds 42, 43 and 44 |
| 15 | Reproduction and handoff | US-14 | Clean reproducible research artefact |

## Working rule

After each story:

- read Codex's changed-file list;
- run or inspect its verification evidence;
- commit or otherwise checkpoint the working version;
- update the checkbox below;
- only then continue.

## Progress

- [ ] US-00 Repository audit and scaffold
- [ ] US-01 Install and verify dataset
- [ ] US-02 Data access layer
- [ ] US-03 Preprocessing and MLP core
- [ ] US-04 Data-validation notebook
- [ ] US-05 Original MLP notebook
- [ ] US-06 Full-retraining references
- [ ] US-07 Reference-based metrics
- [ ] US-08 Retain fine-tuning
- [ ] US-09 Gradient ascent
- [ ] US-10 Gradient difference
- [ ] US-11 SISA partition and original ensemble
- [ ] US-12 SISA deletion and reference
- [ ] US-13 Seed-42 comparison
- [ ] US-14 Repeat frozen configurations
- [ ] US-15 Reproduction and handoff

