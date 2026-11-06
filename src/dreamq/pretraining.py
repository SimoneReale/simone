"""Activity-free denoising pretraining with train-only normalization and early stopping."""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .encoders import MolecularAutoencoder, corrupt_bits
from .features import FeatureBundle, descriptor_transformer
from .utils import digest, seed_all, versions, write_json

LOG = logging.getLogger(__name__)


class SparseMoleculeDataset(Dataset):
    def __init__(self, fingerprints, descriptors: np.ndarray, indices: np.ndarray):
        self.fingerprints = fingerprints
        self.descriptors = descriptors
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        return (torch.from_numpy(self.fingerprints[idx].toarray().ravel().astype(np.float32)),
                torch.from_numpy(self.descriptors[idx].astype(np.float32)))


class FrozenEncoder:
    def __init__(self, model: MolecularAutoencoder, normalization, metadata: dict, device: str = "cpu"):
        self.model = model.to(device).freeze()
        self.normalization = normalization
        self.metadata = metadata
        self.device = device

    @torch.no_grad()
    def transform(self, features: FeatureBundle, batch_size: int = 512) -> np.ndarray:
        if features.signature != self.metadata["feature_signature"]:
            raise ValueError("Encoder and molecular feature configurations do not match")
        fp = features.fingerprints()
        desc = self.normalization.transform(features.descriptors).astype(np.float32)
        outputs = []
        for start in range(0, len(desc), batch_size):
            bits = torch.from_numpy(fp[start:start + batch_size].toarray().astype(np.float32)).to(self.device)
            values = torch.from_numpy(desc[start:start + batch_size]).to(self.device)
            outputs.append(self.model.encode(bits, values).cpu().numpy())
        return np.concatenate(outputs)

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> FrozenEncoder:
        directory = Path(directory)
        metadata, normalization = joblib.load(directory / "normalization.joblib")
        model = MolecularAutoencoder(**metadata["architecture"])
        model.load_state_dict(torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True))
        return cls(model, normalization, metadata, device)


def pretrain(features: FeatureBundle, config: dict, directory: str | Path,
             provenance: dict | None = None) -> FrozenEncoder:
    if len(np.unique(features.structure_ids)) != len(features.structure_ids):
        raise ValueError("Pretraining requires unique molecular structures to avoid validation leakage")
    if len(features.structure_ids) < 4:
        raise ValueError("Pretraining needs at least four unique molecular structures")
    seed_all(config["seed"], config.get("threads", 2))
    settings = config["pretraining"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    architecture = {"fingerprint_dim": features.ecfp.shape[1] + features.pharmacophore.shape[1],
                    "descriptor_dim": features.descriptors.shape[1], "latent_dim": settings["latent_dim"],
                    "dropout": settings["dropout"]}
    train, valid = train_test_split(np.arange(len(features.structure_ids)), test_size=settings["validation_fraction"],
                                   random_state=config["seed"])
    normalization = descriptor_transformer().fit(features.descriptors[train])
    desc = normalization.transform(features.descriptors).astype(np.float32)
    fp = features.fingerprints()
    counts = np.asarray(fp[train].sum(axis=0)).ravel()
    weights = np.clip((len(train) - counts) / np.maximum(counts, 1), 1, settings["positive_weight_cap"]).astype(np.float32)
    # Bits never observed in training have no positive reconstruction supervision.
    weights[counts == 0] = 1
    requested = settings["device"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = MolecularAutoencoder(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    binary_loss = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).to(device))
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {name: DataLoader(SparseMoleculeDataset(fp, desc, indices), batch_size=settings["batch_size"],
                               shuffle=name == "train", num_workers=0, generator=generator)
               for name, indices in [("train", train), ("validation", valid)]}
    best_loss, stale, best_state = float("inf"), 0, None
    history = []
    for epoch in range(settings["max_epochs"]):
        record = {"epoch": epoch + 1}
        for phase, loader in loaders.items():
            model.train(phase == "train")
            total, n = 0.0, 0
            with torch.set_grad_enabled(phase == "train"):
                for bits, values in loader:
                    bits, values = bits.to(device), values.to(device)
                    corrupted = corrupt_bits(bits, settings["mask_probability"], settings["bit_noise"]) if phase == "train" else bits
                    noisy_values = values + torch.randn_like(values) * settings["descriptor_noise"] if phase == "train" else values
                    reconstructed, desc_reconstructed = model(corrupted, noisy_values)
                    loss = binary_loss(reconstructed, bits) + settings["descriptor_loss_weight"] * nn.functional.mse_loss(desc_reconstructed, values)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite self-supervised loss")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    total += loss.item() * len(bits)
                    n += len(bits)
            record[phase] = total / n
        history.append(record)
        LOG.info("Pretraining epoch %d: train=%.5f validation=%.5f", epoch + 1, record["train"], record["validation"])
        if record["validation"] < best_loss - 1e-5:
            best_loss, stale = record["validation"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("Pretraining must run at least one epoch")
    model.load_state_dict(best_state)
    metadata = {"architecture": architecture, "settings": deepcopy(settings), "seed": config["seed"],
                "feature_signature": features.signature, "development": config.get("development", False),
                "versions": versions(), "provenance": provenance or {}, "history": history,
                "training_structures": features.structure_ids[train].tolist(),
                "validation_structures": features.structure_ids[valid].tolist()}
    metadata["identity"] = digest({"features": features.structure_ids.tolist(), "config": settings,
                                   "feature_signature": features.signature, "seed": config["seed"]})
    torch.save(best_state, directory / "encoder.pt")
    joblib.dump((metadata, normalization), directory / "normalization.joblib")
    write_json(directory / "metadata.json", metadata)
    # Compare against a constant near-zero reconstruction using the same objective.
    valid_bits = fp[valid]
    positive_rate = np.asarray(valid_bits.mean(axis=0)).ravel()
    zeros_reference = float(np.mean(np.logaddexp(0, -10) * (1 - positive_rate)
                                    + weights * np.logaddexp(0, 10) * positive_rate))
    write_json(directory / "reconstruction_diagnostics.json", {"best_validation_loss": best_loss,
               "near_zero_fingerprint_BCE_reference": zeros_reference,
               "training_fingerprint_density": float(fp[train].nnz / (len(train) * fp.shape[1])),
               "positive_weight_min": float(weights.min()), "positive_weight_max": float(weights.max())})
    return FrozenEncoder(model, normalization, metadata, device)
