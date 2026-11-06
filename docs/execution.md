# Local execution record

This records the initial implementation smoke run. A subsequent full user run is available under `artifacts/evaluation/`; its results supersede the development scores for scientific interpretation. New independent-model experiments have their own record in `docs/model_suite_execution.md`.

Executed on 5 November 2026 (Europe/Zagreb), using the supplied local retrospective CSVs. No external molecular datasets or pretrained models were downloaded.

## Completed execution

- Created `.venv`, installed the project and scientific dependencies; exact versions are recorded in `requirements-lock.txt`.
- Executed the full real-data checksum and chemical audit.
- Executed the complete synthetic test suite: **29 passed**; output is preserved in `artifacts/test-results.txt`.
- Constructed and saved the full requested 5-fold × 3-repeat identity and scaffold assignments. All 30 outer folds retain 7–8 validation positives; 3-fold inner CV is feasible in every outer training partition. See `artifacts/folds/real_data_feasibility.json`.
- Executed `python -m dreamq.cli run --config configs/default.yaml --dev` on the real data, including feature extraction, pretraining, nested evaluation, final refitting and leaderboard inference.
- Executed the separate `fit-final --dev` and `predict --dev` commands to verify checkpoint reloading and inference after the final provenance checks were implemented.
- Verified 300 finite scores in [0,1], unique unchanged leaderboard identifiers and original ordering, complete OOF coverage for all five models, saved PR/ROC/enrichment plots and a frozen encoder. Machine-readable verification and artifact hashes are in `artifacts/dev/verification.json`.

## Dataset findings

| Dataset | Rows | Unique canonical structures | Invalid |
| --- | ---: | ---: | ---: |
| labeled | 6580 | 6579 | 0 |
| unlabeled | 184632 | 184632 | 0 |
| leaderboard | 300 | 300 | 0 |

The labeled table contains 39 positives and 6,541 negatives (prevalence 0.005927, approximately 0.593%). One consistent chemical entity occurs twice with different IDs; both source rows are retained and always share a fold. No conflicting structure labels were found. Twelve compound IDs map to multiple canonical structures across source files; details are in `artifacts/audit/inconsistent_ids.csv`. They are reported without silently changing molecular identity.

Canonical-structure intersections are 6,573 for labeled/unlabeled and 298 for unlabeled/leaderboard, despite all labeled and leaderboard IDs occurring in the unlabeled table. Labeled/leaderboard overlap is zero by both ID and canonical structure.

## Development evaluation only

This smoke run used 1,024 unique unlabeled structures, 921 for optimization and 103 for label-free validation, with 2 pretraining epochs. The combined feature cache covers 7,861 unique structures, including every labeled and leaderboard structure. The encoder pretraining subset actually contains 41 labeled structures and 1 leaderboard structure. Their activity labels were not supplied to pretraining. The protocol is explicitly transductive.

Evaluation used 2 outer folds × 1 repetition per strategy, 2 inner folds, C=0.1, positive weights {1,10}, the configured alpha grid, and 100 cluster-bootstrap replicates per model/strategy. These are **development results, not the final challenge benchmark**. No claim of improvement follows from this run.

| Model | Identity split AP | Scaffold split AP |
| --- | ---: | ---: |
| M0 | 0.426728 | 0.425751 |
| M1 | 0.402164 | 0.409499 |
| M2 | 0.086723 | 0.091329 |
| M3 | 0.402164 | 0.373184 |
| similarity | 0.197443 | 0.232616 |

All other metrics, intervals and variability are in `artifacts/dev/evaluation/results.csv` and each strategy’s `metrics.json`. Individual fold metrics, alpha selections, model artifacts, inner/outer assignments and OOF logits are retained. PR, ROC and enrichment plots are under each strategy’s `plots/` directory. With 39 positives, the apparent ordering among models is uncertain and can change with splits, training and selection.

## Saved models and predictions

- `artifacts/dev/encoder/encoder.pt`: actual best denoising autoencoder weights from the two-epoch development run.
- `artifacts/dev/encoder/normalization.joblib`: exact pretraining normalization and provenance.
- `artifacts/dev/final/predictor.joblib`: actual frozen encoder plus both final logistic branches fitted on all 6,580 labeled rows, with internally selected hyperparameters and alpha.
- `predictions/dev/submission.csv`: 300 leaderboard scores with configurable columns `CatalogID,ActivityScore`.
- `predictions/dev/leaderboard_diagnostics.csv`: individual branch logits, fused and selected classical logits, scores and 128-dimensional embeddings.
- `predictions/dev/prediction_metadata.json`: development marker, fusion coefficient and submission schema caveat.

Scores are ranking scores and have not been probability-calibrated. The candidate CSV format cannot be certified as the challenge official format from the supplied manifest. No submission was uploaded.

## Not executed

Full-collection feature extraction/pretraining, the complete supervised 5-fold × 3-repeat benchmark, and the conservative inductive sensitivity training were not executed. Their implementations and configurations are provided; the full split feasibility was checked without training those models. There is no production checkpoint or production leaderboard file outside the explicitly marked development directories. No quantum component was implemented.
