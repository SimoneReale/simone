# DREAM-Q PGK2 classical molecular prediction

An implemented Python pipeline for extremely imbalanced binary molecular activity prediction. It includes sparse molecular features, an activity-free denoising encoder, nested supervised selection, shared molecular/scaffold folds, screening metrics, final refitting and validated inference. No quantum component or external pretrained model is included.

The supplied files in `retrospective/` are the only molecular data inputs. The local manifest records source files and checksums; it does **not** specify the challenge's official submission schema. The configured `CatalogID,ActivityScore` output is a configurable candidate format, not a verified official format.

## Independent model suite

The project now provides separate, serializable models inspired by the reviewed challenge workflows. They are independently evaluated with the **same persisted identity/scaffold folds** and training-only inner selection. These are implementations of architectural ideas, not claimed reproductions of the teams' full data preparation, structural workflows or performance. The [new suite execution record](docs/model_suite_execution.md) lists the actual tests and real-data smoke runs.

| ID | Representation and classifier |
| --- | --- |
| M0 | Binary Morgan + L2 logistic regression |
| M1 | Binary Morgan + descriptors + L2 logistic regression |
| M2 | Existing frozen encoder + descriptors + L2 logistic regression |
| M3 | Inner-OOF-selected logit fusion of M1/M2 |
| M4 | True Morgan counts + L2 logistic regression |
| M5 | MACCS167 + L2 logistic regression |
| M6 | Atom Pair2048 + L2 logistic regression |
| M7 | Direct hashed Pharm2D + L2 logistic regression |
| M8 | Morgan + Pharm2D + descriptors + L2 logistic regression |
| M9 | Binary Morgan + Random Forest |
| M10 | Morgan counts + Extra Trees |
| M11 | Morgan counts + descriptors + XGBoost |
| M12 | Morgan counts + descriptors + LightGBM |
| M13 | Morgan/MACCS/Atom Pair/RDKit/Avalon + Extra Trees |
| M14 | Binary Morgan Tanimoto-kernel SVM |
| M15 | Supervised Chemprop graph ensemble, five seeds by default |
| M16 | Fixed equal-weight rank ensemble across configured base models |
| M17 | Single model or pair/weight selected exclusively using inner OOF AP |
| similarity | Training-active versus training-inactive Tanimoto reference |

Install CPU tree backends and inspect the catalogue:

```bash
source .venv/bin/activate
python -m pip install -e '.[trees,test]'
python -m dreamq.cli list-models
```

To evaluate, refit every model and generate a separate leaderboard CSV for each:

```bash
python -u -m dreamq.cli run-models --config configs/comparison.yaml
```

This preset runs all 18 non-Chemprop models and reuses the existing frozen encoder for M2/M3. If no encoder is present, first run the original `pretrain` command with matching configuration. All other models can run without an encoder; select a subset as shown below. The runner **does not retrain the encoder unnecessarily**.

The commands can also be executed separately:

```bash
python -m dreamq.cli compare --config configs/comparison.yaml
python -m dreamq.cli fit-models --config configs/comparison.yaml
python -m dreamq.cli predict-models --config configs/comparison.yaml
```

New outputs are separate from the original benchmark:

- `artifacts/comparison/classical/results.csv`: model comparison with AP, ROC-AUC, EF and uncertainty.
- `artifacts/comparison/classical/{stratified,scaffold}/`: exact outer OOF predictions, inner model selection/correlations, fold models, metrics and plots.
- `artifacts/comparison/classical/{stratified,scaffold}/portfolio_metrics.csv`: fixed diversity-policy hit/precision/recall and scaffold counts, evaluated separately in outer folds.
- `artifacts/comparison/classical/final/M0.joblib`, etc.: individual fitted models; `predictor.joblib` contains the complete selected suite and optional frozen encoder.
- `predictions/comparison/classical/M0_submission.csv`, etc.: one two-column candidate submission per model, preserving all source IDs and ordering.
- `predictions/comparison/classical/M0_diversity_0.25_submission.csv`, etc.: diagnostic copies with a `Selected` flag; these extra-column files are **not certified official challenge submissions**.

For a smaller independent experiment with no encoder dependency:

```bash
python -m dreamq.cli run-models --config configs/comparison.yaml \
  --models M0,M4,M7,M9,M10,M11,M12,M13,M14,M16,M17 --run-name fingerprints
```

An ensemble automatically fits its configured dependencies. To run only a single model use, for example, `--models M11 --run-name xgboost`. A run directory is protected against replacement by a different model set or training dataset: use a new `--run-name`. A prediction-only call can select an already-fitted subset with `--models M0,M11`. For arbitrary molecules add `--input compounds.csv --id-column CompoundID --smiles-column SMILES` to `predict-models`.

Append `--dev` for a smoke experiment with reduced grids and 2-fold CV. It retains **all 6,580 labels**, reduces tree counts, and writes under `artifacts/dev/comparison/...` and `predictions/dev/comparison/...`. Use the complete configuration to report benchmarks.

To leave the new suite working on the remote server:

```bash
screen -S dreamq-models
cd /home/icse/simone
source .venv/bin/activate
mkdir -p logs
set -o pipefail
python -u -m dreamq.cli run-models --config configs/comparison.yaml 2>&1 | tee logs/comparison.log
```

Detach with **Ctrl+A**, then **D**. Reconnect with `screen -r dreamq-models`.

### Graph model M15 and all-model experiments

Chemprop is optional and must be installed into the interpreter executing the run:

```bash
python -m pip install -e '.[trees,graph,test]'
python -m dreamq.cli run-models --config configs/graph.yaml
# Or all 19 models together:
python -m dreamq.cli run-models --config configs/all_models.yaml
```

The graph preset has output name `graph`; the all-model preset uses `all_models`, keeping their checkpoints/results separate. A five-seed nested Chemprop experiment is substantially more expensive than tree models. Start with `--dev` for one seed/one epoch. [M15 documentation](docs/chemprop.md) explains the fixed-epoch training protocol and how it preserves the shared caller-controlled folds.

Install optional dependencies **before** a scientific comparison: dependency resolution can change RDKit versions. The feature caches record these versions. If a dependency update changes RDKit, regenerate the original features and pretrained encoder in that environment before requesting M2/M3. All compared models must use the same molecular preprocessing and persisted fold signatures; results from differing dependency environments are not automatically a controlled comparison. The local isolated `.venv-chemprop` was created solely to test real optional graph training while leaving the existing environment and benchmark intact.

### Ensemble and candidate-selection protocol

M16/M17 transform base logits using a smoothed empirical midpoint CDF fitted to **training-only inner OOF predictions**. The reference stays fixed for inference, so predictions do not change when unrelated query molecules are added. M17 searches configured pairs/weights and single-model alternatives; it can reject an unhelpful fusion. Correlations are diagnostic training statistics, not proof of complementarity. No flexible learned stacking layer is introduced.

The selection policies use `percentile_rank(score) - diversity_weight * max_Tanimoto_to_selected` and unique canonical chemical entities. Weight zero is ordinary top-k; other fixed weights trade score against diversity. All configured weights are reported rather than choosing a winner on held-out labels. Selection is label-free, and it changes the portfolio, not the model's activity score. Training/validation budgets smaller than k are explicitly marked with `effective_k` and `budget_reduced`. Scaffold diversity is a proxy and **does not reproduce official MCS-defined hit series**. The policy weights and pool size should be declared before evaluating a final test.

Morgan counts are cached as sparse uint32 arrays; binary representations remain unscaled. Supplementary fingerprints have independent cache/version keys and preserve the original encoder cache format. M14 limits its dense training kernel by configurable memory and row guards and batches inference. Hyperparameter grids are deliberately small; expanding them aggressively with 39 positives increases selection risk.

## Install

Python 3.11 or newer is required. The implementation was tested in the local Python 3.12 environment; exact installed versions are recorded in `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# For CPU execution, install the CPU Torch distribution first:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
pytest -q
```

To recreate the recorded CPU package versions, use `python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu` and then `python -m pip install --no-deps -e .`. The lock records this tested Python/platform combination rather than guaranteeing availability of identical wheels on every platform.

For CUDA, install an appropriate PyTorch build for the machine instead of the CPU wheel. Set `pretraining.device: auto`, `cpu`, or `cuda`. CUDA reproducibility also requires a deterministic BLAS workspace, for example `CUBLAS_WORKSPACE_CONFIG=:4096:8`, before starting Python. The code requests deterministic Torch algorithms and will surface unsupported deterministic operations.

## Run

Paths in the YAML resolve relative to `project_root`, itself relative to the config file. Column mappings are explicit for every input; identifiers are read as strings. Source CSVs are never modified.

```bash
python -m dreamq.cli audit --config configs/default.yaml
python -m dreamq.cli featurize --config configs/default.yaml
python -m dreamq.cli pretrain --config configs/default.yaml
python -m dreamq.cli evaluate --config configs/default.yaml
python -m dreamq.cli fit-final --config configs/default.yaml
python -m dreamq.cli predict --config configs/default.yaml
```

The equivalent all-stage command is `python -m dreamq.cli run --config configs/default.yaml`. Full pretraining covers the permitted unlabeled collection and up to 50 epochs; full evaluation uses 5 outer folds × 3 repetitions for each of the two split strategies, plus inner selection. This is a substantially larger computation than the smoke run.

For a development run:

```bash
python -m dreamq.cli run --config configs/default.yaml --dev
```

Development mode retains **all** labeled and leaderboard rows, uses at most 1,024 unlabeled structures and 2 pretraining epochs, and reduces the parameter grid and folds. It writes exclusively under `cache/dev/`, `artifacts/dev/`, and `predictions/dev/`. Its metrics and final refit are marked development; they must not be reported as final benchmark results. The full audit is still performed.

Branch A, Branch B and fusion can each be used for inference:

```bash
python -m dreamq.cli predict --config configs/default.yaml --mode M1
python -m dreamq.cli predict --config configs/default.yaml --mode M2
python -m dreamq.cli predict --config configs/default.yaml --mode M3
```

Append `--dev` to use the development checkpoint. `fit-final` performs internal CV on all available labels independently of the reported outer evaluations. Its split strategy defaults to `stratified` and can be set explicitly with `cv.final_strategy: scaffold`. The default production output is fusion; no model family is automatically declared the winner based on the evaluation table.

Arbitrary compound sets can export scores, all branch logits, and embeddings:

```bash
python -m dreamq.cli predict --config configs/default.yaml --dev \
  --input my_compounds.csv --id-column CompoundID --smiles-column SMILES
```

This produces `arbitrary_scores.csv` and `arbitrary_diagnostics.csv`. Input activity columns, if present, are ignored. Invalid SMILES cause a diagnostic error, not a shorter prediction file.

## Scientific protocol

- Canonical isomeric RDKit SMILES, SHA-256 structure identifiers, and non-isomeric Bemis–Murcko scaffolds define identity and split groups. Salts, protonation states, tautomers and stereoisomers are not silently standardized into one entity. All acyclic molecules share a conservative scaffold group.
- Both ID and canonical-structure overlaps are audited. Source rows, including consistent labeled duplicates, are retained. Identical structures always stay in one fold. Conflicting labels fail by default; set `data.fail_on_conflicts: false` only for an explicitly justified sensitivity analysis.
- Unlabeled molecules never receive activity labels. Only the mapped labeled CSV supplies supervised targets; `usecols` prevents reading activity columns from the other files. The checksum audit reads file bytes only to verify integrity.
- The default is **transductive**: all permitted unique structures from the unlabeled CSV may be pretrained, including labeled/outer-validation molecules and leaderboard structures already in that collection. No extra leaderboard structures are appended to pretraining. The checkpoint records the actual included counts and exact structure identifiers.
- Set `pretraining.include_leaderboard: false` to exclude canonical leaderboard structures. Set `pretraining.mode: inductive` to exclude **all** labeled and leaderboard structures from unsupervised pretraining. This conservative inductive sensitivity experiment reuses a single encoder trained exclusively on other molecules; it does not exploit outer-training labeled structures for pretraining. It is stricter than an encoder retrained separately for each outer fold. ID/structure inconsistencies in the source are reported separately and are not assumed to be chemical equivalences.
- The encoder's imputation, scaling, bit weights and optimization use its activity-free unsupervised training subset only. Its validation subset determines early stopping. The supervised descriptor and embedding transformations are separate and fitted anew on each corresponding inner/outer training partition.
- `StratifiedGroupKFold` is used for both identity and scaffold splits. Requested fold counts are reduced if necessary to retain both classes in every training and validation partition. Positive scaffold counts and split limitations are saved. If no valid split exists, the run fails explicitly. No scaffold experiment is silently converted to a random split.
- All M0–M3 and similarity comparisons use identical persisted outer assignments. Inner assignments, selected parameters and inner OOF logits are saved. Alpha is selected from training-only inner OOF logits, never from the outer predictions used to report performance. Inner OOF values are selection statistics, not unbiased evaluation estimates.
- Logistic sigmoids and fused scores are **ranking scores, not calibrated activity probabilities**. Threshold 0.5 is explicitly configured for secondary confusion metrics; changing it does not establish calibration.

## Models and metrics

| Model | Input / score |
| --- | --- |
| M0 | 2,048 Morgan radius-2 binary bits; L2 logistic regression |
| M1 / A | Morgan bits plus 20 training-normalized RDKit descriptors; L2 logistic regression |
| M2 / B | Frozen 128-dimensional encoder embedding plus training-normalized descriptors; L2 logistic regression |
| M3 / C | `(1-alpha) * z_A + alpha * z_B`, followed by sigmoid |
| similarity | `5 * (max Tanimoto to training actives - max Tanimoto to training inactives)` followed by sigmoid |

The small C/positive-weight grids and alpha candidates are configurable. Binary fingerprint bits are never standardized. The similarity reference has no fitted hyperparameter and never sees validation activity labels.

Reported metrics include AP, ROC-AUC, EF@1/2/5/10%, precision/recall@k, balanced accuracy, MCC and confusion matrices. The observed prevalence is the random-ranking AP reference. AP is scikit-learn's average precision, not trapezoidal integration of the PR curve. AP/ROC-AUC are explicitly undefined when a class is absent.

For each screening fraction, `k = ceil(fraction * n)`, clamped to `[1,n]`. A tie crossing the cutoff gets its **expected** positive count under uniform ordering of tied molecules. Thus screening statistics are invariant to input ordering; fractional hit counts are possible. No row-index tie breaker is used.

Repeated OOF logits are averaged **per source molecule before pooling** so each row contributes once to pooled performance. Consistent duplicate source rows remain separate observations and stay grouped for splitting/bootstrap. Fold AP mean/SD and repetition AP are also reported; correlated CV folds/repetitions are not independent replicates. The 95% percentile bootstrap samples whole canonical chemical entities and reports skipped single-class replicates. It conditions on fitted OOF predictions, does not refit models, and does not capture full training/selection uncertainty. With only 39 positives, intervals and apparent improvements can be unstable.

## Artifacts and code

| Location | Contents |
| --- | --- |
| `retrospective/` | Original input data and manifest |
| `cache/` | Versioned canonicalization and sparse feature caches |
| `artifacts/audit/` | Dataset counts, duplicate/conflict/invalid/overlap CSVs, checksum provenance |
| `artifacts/encoder/` | Encoder weights, normalization, pretraining history and exact structure split |
| `artifacts/folds/` | Persisted common folds and feasibility reports |
| `artifacts/evaluation/` | Nested selection, per-fold models, OOF predictions, metrics, PR/ROC/enrichment PNGs, comparison table |
| `artifacts/final/` | Self-contained predictor, internal OOF logits, selection, training provenance |
| `artifacts/logs/` | Structured JSONL command logs |
| `predictions/` | Configurable candidate submission, diagnostics with 128 embedding columns, validation metadata |

Development artifacts use the same layout under `dev/`. Pickle/joblib artifacts must only be loaded from trusted local runs. Feature caches incorporate chemical identities, representation settings, implementation version and RDKit version. Encoder loading checks the exact permitted pretraining set, representation configuration and training settings; stale checkpoints fail explicitly.

Source modules follow the requested `src/dreamq/` layout. [Architecture and formulation](docs/architecture.md) explain the encoder objective, data boundaries and future extension contract. [Execution record](docs/execution.md) records what was actually executed locally.

## Future residual learning

`MolecularPredictor` exposes `classical_logit(smiles)`, `molecular_embedding(smiles)`, `pharmacophore_features(smiles)` and `predict_score(smiles)`. Bulk `predict_features` returns branch/fused logits and embeddings. These are sufficient inputs for a future `z_hybrid = z_classical + lambda * r_quantum` learner; no residual is fabricated here.

`CompactProjector(dimensions=4|6|8)` provides optional SVD for sparse pharmacophore bits or PCA for dense embeddings, followed by training-fitted, clipped `[0, pi]` scaling. Fit it exclusively on the permitted training partition, persist it, and transform validation/inference data without refitting. The angle range is a possible convention for a future feature map, not a quantum computation. Fit any residual model, its kernel parameters and lambda within the same nested split protocol.
