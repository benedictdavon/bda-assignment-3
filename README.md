# bda-assignment-3

Wine-quality prediction for **Big Data Analytics Techniques and Applications, Fall 2026, NYCU**. The objective is an accurate, reproducible model with meaningful explanations, evaluated using **quadratic weighted kappa (QWK)**.

## Project status

This repository currently provides dataset diagnostics, an independently checked QWK implementation, reproducible validation partitions, and audit figures. Predictive training, hyperparameter searches, and Kaggle submissions have not been run. No competition score is claimed.

## Task

Predict integer wine quality from 11 physicochemical measurements. The competition supplies 15,000 labeled training rows and 10,000 test rows. The theoretical score range is 0–10; observed training labels span 3–8. Submissions use `id,quality`, preserving the actual test IDs and sample order.

[Competition](https://www.kaggle.com/competitions/nycu-big-data-2026-fall-assignment-3) · [Data](https://www.kaggle.com/competitions/nycu-big-data-2026-fall-assignment-3/data) · [Rules](https://www.kaggle.com/competitions/nycu-big-data-2026-fall-assignment-3/rules)

## Setup

Use Python 3.12. Audit dependencies are pinned in [requirements-audit.txt](requirements-audit.txt). Training dependencies will be added when models are implemented.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements-audit.txt
```

## Obtain the local inputs

Download the competition archive through your authorized Kaggle account. Place it in the project root as `nycu-big-data-2026-fall-assignment-3.zip`, then extract it:

```bash
python -m zipfile -e nycu-big-data-2026-fall-assignment-3.zip data/raw
```

This produces `train.csv`, `test.csv`, and `sample_submission.csv` under `data/raw/`. Also obtain `BigData_Assignment 3.pdf` from the course and place it in the project root: the full audit records hashes for that document and the original ZIP.

Data, downloaded PDFs, archives, and generated outputs are ignored by Git. Do not redistribute the private competition files through this repository.

## Reproduce the preparation

Run commands from the repository root. Metric checks require no dataset:

```bash
python scripts/check_metric.py
```

With the local inputs in place, reproduce diagnostics, partitions, and figures:

```bash
python scripts/audit_data.py
python scripts/prepare_splits.py
python scripts/plot_audit.py
```

- **Audit:** schema, missingness, class balance, exact/near duplicates, correlations, marginal train/test differences, and input hashes. Outputs go to `reports/data_audit/`.
- **Metric:** count-matrix QWK versus scikit-learn on 105 cases, the population-moment identity, and rejection of invalid labels.
- **Partitions:** 3,000 stratified confirmation rows; 12,000 development rows with five folds for each of two fixed seeds. Version-1 membership cannot be silently changed. Generated membership is ignored; its versioned manifest records hashes and seeds.
- **Figures:** target balance and marginal target associations from recorded audit tables.

The confirmation partition has not been used for predictive evaluation. Descriptive full-data analysis preceded its reservation. Future preprocessing, thresholds, and ensembles must respect training/validation boundaries.

`scripts/verify_preparation.py` is an additional **local-workspace** check. It requires untracked research documents, source PDFs, and preparation outputs, so it is not part of the clean-clone workflow above.

## Repository layout

```text
configs/                   Versioned split manifest and future model configs
scripts/                   Audit, metric, partition, figure, and verification tools
requirements-audit.txt     Pinned audit dependencies
data/                      Local competition files; ignored
docs/                      Local research and agent knowledge base; ignored
reports/                   Generated diagnostics and source reviews; ignored
runs/                      Future experiment artifacts; ignored
```

Source code and configuration are committed. The local research notes under `docs/` are intentionally excluded, along with data, PDFs, models, credentials, submissions, and generated results.

## Competition and reproducibility

- Train only on the supplied competition files under the current rules. External labeled-data augmentation is excluded; pretrained-model exceptions and software/model licenses need verification before use.
- Three submissions per day reset at 00:00 UTC (08:00 Taiwan); two final selections are allowed. Public leaderboard feedback is a secondary diagnostic, not a threshold-tuning set.
- Deadline wording conflicts: Rules state December 16, 2026; the countdown and assignment PDF state December 22. Both refer to 23:55 UTC+8. Plan against the earlier date until teaching staff clarify.
- The PDF specifies an English A4 report of at most five pages. Rules also require reproducible public GitHub code and refer to a slide; report/slide wording needs clarification.
- Record the code version, input/split hashes, package versions, settings, seeds, and row-aligned out-of-fold predictions for every future experiment. Report only executed, verified results.

## References

Competition data are synthetic; original wine-data distributions and labels are not assumed to transfer unchanged.

- Cortez et al. (2009), [Wine Quality dataset](https://doi.org/10.24432/C56S3T), UCI Machine Learning Repository.
- Cortez et al. (2009), [Modeling wine preferences by data mining from physicochemical properties](https://doi.org/10.1016/j.dss.2009.05.016).
- [scikit-learn QWK reference](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.cohen_kappa_score.html).
