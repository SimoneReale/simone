"""Optional supervised Chemprop v2 graph ensemble, with caller-owned CV splits.

All rows passed to ``fit`` are used for optimization for a fixed number of
epochs. No implicit Chemprop train/test split and no label-bearing validation
set is created here: hyperparameters, including epochs, are chosen by the
outer runner's training-only inner CV. The default five members differ in seed.
"""
from __future__ import annotations

import importlib.metadata
import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import logit

if TYPE_CHECKING:
    from .representations import MolecularInputs

LOG = logging.getLogger(__name__)
INSTALL_MESSAGE = (
    "M15 requires Chemprop 2.x and its Lightning dependencies. "
    "Install the optional dependency with `python -m pip install -e '.[graph]'` "
    "in the Python environment used for training and inference. No pretrained model is required."
)


def _chemprop_modules() -> tuple[Any, Any, Any, Any, Any]:
    """Keep the core pipeline importable without optional graph dependencies."""
    try:
        from chemprop import data, featurizers, models, nn
        from lightning import pytorch as pl
        version = importlib.metadata.version("chemprop")
    except ImportError as error:
        raise ImportError(INSTALL_MESSAGE) from error
    if version.split(".")[0] != "2":
        raise ImportError(f"Unsupported Chemprop version {version}. {INSTALL_MESSAGE}")
    return data, featurizers, models, nn, pl


class ChempropClassifier:
    """Mean-score ensemble of independently trained bond-message-passing nets.

    Only molecule graphs and the supplied activity labels enter training. The
    descriptor/fingerprint/embedding matrices in MolecularInputs are not used.
    Serialized artifacts contain CPU tensor state bytes, not temporary file
    paths or trainer instances, and can be moved with their parent artifact.
    Scores are sigmoid ranking scores and are not probability-calibrated.
    """
    DEFAULTS = {
        "epochs": 30, "batch_size": 64, "hidden_dim": 200, "depth": 3,
        "ffn_hidden_dim": 200, "ffn_layers": 1, "dropout": 0.1,
        "seeds": [0, 1, 2, 3, 4], "device": "cpu", "learning_rate": 0.001,
        "positive_weight": 1.0,
    }

    def __init__(self, parameters: dict | None = None, seed: int = 2035, threads: int = 2):
        self.parameters = dict(parameters or {})
        self.seed = int(seed)
        self.threads = int(threads)

    def _settings(self) -> dict:
        unknown = set(self.parameters) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Chemprop parameters: {sorted(unknown)}")
        settings = {**self.DEFAULTS, **self.parameters}
        for name in ["epochs", "batch_size", "hidden_dim", "depth", "ffn_hidden_dim", "ffn_layers"]:
            if int(settings[name]) != settings[name] or settings[name] < 1:
                raise ValueError(f"Chemprop {name} must be a positive integer")
            settings[name] = int(settings[name])
        seeds = settings["seeds"]
        if not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Chemprop seeds must be a nonempty list of distinct integer offsets")
        if any(not isinstance(s, (int, np.integer)) or s < 0 for s in seeds):
            raise ValueError("Chemprop seeds must be nonnegative integer offsets")
        if not 0 <= settings["dropout"] < 1:
            raise ValueError("Chemprop dropout must be in [0, 1)")
        if (not np.isfinite(settings["learning_rate"]) or not np.isfinite(settings["positive_weight"])
                or settings["learning_rate"] <= 0 or settings["positive_weight"] <= 0):
            raise ValueError("Chemprop learning_rate and positive_weight must be positive")
        if settings["device"] not in {"cpu", "cuda", "auto"}:
            raise ValueError("Chemprop device must be cpu, cuda, or auto")
        if self.threads < 1:
            raise ValueError("Chemprop threads must be positive")
        return settings

    @staticmethod
    def _network(settings: dict, models: Any, nn: Any):
        mp = nn.BondMessagePassing(d_h=settings["hidden_dim"], depth=settings["depth"], dropout=settings["dropout"])
        predictor = nn.BinaryClassificationFFN(
            input_dim=settings["hidden_dim"], hidden_dim=settings["ffn_hidden_dim"],
            n_layers=settings["ffn_layers"], dropout=settings["dropout"],
        )
        # No batch norm: preserve singleton batches instead of dropping samples.
        return models.MPNN(
            mp, nn.MeanAggregation(), predictor, batch_norm=False,
            warmup_epochs=min(2, max(0, settings["epochs"] - 1)),
            init_lr=settings["learning_rate"] / 10,
            max_lr=settings["learning_rate"], final_lr=settings["learning_rate"] / 10,
        )

    @staticmethod
    def _loader(inputs: MolecularInputs, data: Any, featurizers: Any,
                settings: dict, y: np.ndarray | None = None, seed: int = 0):
        import torch
        from chemprop.data.collate import collate_batch
        targets = [None] * len(inputs.structure_ids) if y is None else y
        datapoints = [data.MoleculeDatapoint.from_smi(
            str(smiles), None if target is None else np.asarray([target], dtype=np.float32),
            weight=settings["positive_weight"] if target == 1 else 1.0,
        ) for smiles, target in zip(inputs.canonical_smiles, targets)]
        dataset = data.MoleculeDataset(datapoints, featurizers.SimpleMoleculeMolGraphFeaturizer())
        # build_dataloader may discard a singleton batch; manual DataLoader
        # deliberately retains every caller row in both training and inference.
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=settings["batch_size"], shuffle=y is not None,
            drop_last=False, collate_fn=collate_batch, num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        return loader

    def fit(self, inputs: MolecularInputs, y: np.ndarray) -> ChempropClassifier:
        import torch
        settings = self._settings()
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(inputs.structure_ids) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("Chemprop requires matching binary activity labels with both classes")
        if len(inputs.canonical_smiles) != len(y):
            raise ValueError("Chemprop molecular identities and SMILES have different lengths")
        data, featurizers, models, nn, pl = _chemprop_modules()
        requested = settings["device"]
        device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Chemprop CUDA requested but CUDA is unavailable")
        torch.set_num_threads(self.threads)
        self.member_states_: list[bytes] = []
        self.member_seeds_ = [self.seed + int(offset) for offset in settings["seeds"]]
        for member_seed in self.member_seeds_:
            pl.seed_everything(member_seed, workers=True)
            model = self._network(settings, models, nn)
            loader = self._loader(inputs, data, featurizers, settings, y, member_seed)
            trainer = pl.Trainer(
                max_epochs=settings["epochs"], accelerator="gpu" if device == "cuda" else "cpu",
                devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
                deterministic=True,
            )
            LOG.info("Chemprop fixed-epoch member seed=%d epochs=%d rows=%d", member_seed, settings["epochs"], len(y))
            trainer.fit(model, train_dataloaders=loader)
            model.cpu().eval()
            buffer = io.BytesIO()
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, buffer)
            self.member_states_.append(buffer.getvalue())
        self.settings_ = settings
        self.feature_signature_ = inputs.signature
        self.training_structure_ids_ = np.asarray(inputs.structure_ids).copy()
        self.training_row_count_ = len(y)
        self.chemprop_version_ = importlib.metadata.version("chemprop")
        self.training_protocol_ = "fixed epochs on all caller training rows; no internal validation or early stopping"
        return self

    def predict_score(self, inputs: MolecularInputs) -> np.ndarray:
        import torch
        if not hasattr(self, "member_states_"):
            raise ValueError("Chemprop classifier has not been fitted")
        if inputs.signature != self.feature_signature_:
            raise ValueError("Chemprop training and inference feature configurations differ")
        data, featurizers, models, nn, _ = _chemprop_modules()
        if not len(inputs.structure_ids):
            return np.empty(0, dtype=float)
        loader = self._loader(inputs, data, featurizers, self.settings_)
        members = []
        torch.set_num_threads(self.threads)
        for state in self.member_states_:
            model = self._network(self.settings_, models, nn)
            model.load_state_dict(torch.load(io.BytesIO(state), map_location="cpu", weights_only=True))
            model.eval()
            values = []
            with torch.inference_mode():
                for batch in loader:
                    bmg, V_d, X_d, *_ = batch
                    values.append(model(bmg, V_d, X_d).cpu().numpy().ravel())
            members.append(np.concatenate(values).astype(float))
        scores = np.mean(members, axis=0)
        if len(scores) != len(inputs.structure_ids) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("Chemprop returned incomplete, non-finite, or invalid sigmoid scores")
        return scores

    def decision_function(self, inputs: MolecularInputs) -> np.ndarray:
        # Consensus is the arithmetic mean of member sigmoid scores, represented
        # as a finite logit for compatibility with downstream classical fusion.
        return logit(np.clip(self.predict_score(inputs), 1e-7, 1 - 1e-7))
