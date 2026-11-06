# Optional M15 graph ensemble

`dreamq.chemprop_model.ChempropClassifier` uses Chemprop 2.x's supervised bond-message-passing network and a binary classification head. It consumes canonical molecular graphs; the frozen autoencoder and fingerprint matrices are not used by this model. No external molecular dataset or pretrained weights are downloaded.

The default message dimension is 200, depth is 3, and five members use seed offsets `[0,1,2,3,4]` from the experiment seed. Their sigmoid scores are averaged. `decision_function` returns a finite logit of that consensus, not a calibrated probability or a mean of member logits. Optional positive class weighting changes training only.

The adapter trains for a **fixed number of epochs on every caller training row**. It makes no internal split, performs no activity-driven early stopping and never receives outer-validation targets. The runner chooses any configured epoch/architecture settings using training-only inner CV. Outer and inner molecular/scaffold splits remain those shared by the other models; this intentionally differs from independently running Chemprop's default scaffold split for each member.

Batch normalization is disabled and manual data loaders retain singleton batches, so no training or prediction row is silently dropped. The serialized classifier stores architecture parameters and CPU state dictionaries as bytes and does not depend on a temporary training directory. Inference requires a compatible Chemprop installation but defaults to CPU regardless of training device.

Install in the environment used to execute the pipeline:

```bash
python -m pip install -e '.[graph]'
```

The implementation follows the official [classification example](https://chemprop.readthedocs.io/en/latest/training_classification.html), [MPNN composition](https://chemprop.readthedocs.io/en/latest/tutorial/python/models/basic_mpnn_model.html), and [data loader documentation](https://chemprop.readthedocs.io/en/latest/tutorial/python/data/dataloaders.html). Chemprop 2.3.1 is the reference API. The adapter raises an explicit installation error when the optional graph stack is unavailable rather than replacing M15 with a different estimator.

The actual-training optional test is run with:

```bash
python -m pytest -q tests/test_chemprop_model.py
```

It performs real CPU graph training, checks repeatability, singleton prediction retention, and portable joblib serialization. In the core environment without Chemprop the two actual-training tests are explicitly skipped while parameter and missing-dependency tests still run. Five-seed nested graph evaluation can be much more expensive than the classical fingerprint baselines; start with one seed and a few epochs for development checks, and keep those results marked development.
