# Independent model suite execution record

Executed on 6 November 2026 (Europe/Zagreb). Existing source CSVs were used; no molecular dataset, pretrained weights or protein structures were downloaded.

## Verified implementation

- Implemented 19 individually addressable models, including the legacy baseline definitions, new fingerprint classifiers, tree learners, Tanimoto SVM, Chemprop graph ensemble and two rank ensembles.
- Complete core environment test suite: **79 passed, 2 explicitly skipped optional graph tests**.
- Complete isolated Chemprop environment test suite: **81 passed**. Nine expected Lightning warnings concern absent internal validation, worker count and a library deprecation; no test failures.
- XGBoost 3.4.1 and LightGBM 4.7.0 were actually trained in tests and the real-data smoke run. Chemprop 2.3.1 was actually trained on CPU in tests and the separate real-data smoke.

## Real-data classical smoke

Command: `python -m dreamq.cli run-models --config configs/comparison.yaml --dev`.

All 18 non-Chemprop models completed nested evaluation on **all 6,580 labeled rows**, with 2 outer folds, 1 repetition and 2 inner folds for both identity and scaffold strategies. Each model/strategy has complete OOF coverage. Reduced development parameter grids and 100 bootstrap replicates were used. The existing development frozen encoder was reused for M2/M3.

Every requested model was refitted and saved individually, and each generated 300 valid leaderboard scores in original source order. Each has three separate diversity-policy diagnostic files, selecting 50 unique molecular structures. A fresh reload of the full serialized suite and the existing automated serialization tests verify artifact usability.

These are **development execution results, not the full new-model benchmark**. Do not interpret them as evidence that a new architecture is superior. The previous full M0–M3 benchmark remains under `artifacts/evaluation/`.

- New comparison: `artifacts/dev/comparison/classical/results.csv`.
- Per-fold models, nested selection, OOF, metrics and plots: `artifacts/dev/comparison/classical/{stratified,scaffold}/`.
- Final models: `artifacts/dev/comparison/classical/final/`.
- Per-model leaderboard files: `predictions/dev/comparison/classical/`.
- Machine-readable checks and SHA-256 digests: `artifacts/dev/comparison/classical/verification.json`.

## Real-data graph smoke

M15 additionally completed inner selection/refitting on a deliberately small sample of **200 real labeled molecules: all 39 positives plus 161 sampled negatives**. One epoch and one seed were used. It produced all 300 leaderboard scores, with no leaderboard labels read. No outer evaluation was reported for this deliberately enriched sample; its internal selection scores are not a performance benchmark.

- Checkpoint: `artifacts/dev/comparison/graph_smoke/final/M15.joblib`.
- Scores: `predictions/dev/comparison/graph_smoke/M15_submission.csv`.
- Exact scope: `artifacts/dev/comparison/graph_smoke/smoke_metadata.json`.

The isolated graph environment contains Chemprop 2.3.1 and its resolved dependencies; its RDKit version differs from the existing core environment. The real graph smoke used a frozen snapshot of canonical inputs prepared by the core environment. It demonstrates execution, not controlled performance comparability. For a scientific comparison, install dependencies first and run `configs/all_models.yaml` using one consistent environment and matching regenerated encoder/features. Package snapshots are in `requirements-lock.txt` and `requirements-graph-lock.txt`.

## Preservation and limits

SHA-256 checks confirm that `artifacts/dev/encoder/encoder.pt`, `artifacts/dev/final/predictor.joblib` and `predictions/dev/submission.csv` are unchanged. Legacy full results also remain in their original directories. New outputs are protected by model-set/data signatures and optional run names.

The full 5-fold × 3-repeat evaluation of the expanded suite, full graph training, any docking/3D workflow and synthetic bioisosteric labeling were not executed or fabricated. Scores are uncalibrated rankings. Diversity policies are label-free, fixed in configuration and use scaffold counts only as a diversity proxy, not official chemical-series reconstruction.
