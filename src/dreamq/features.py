"""Deterministic sparse Morgan/Gobbi Pharm2D features and RDKit descriptors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import digest

LOG = logging.getLogger(__name__)
FEATURE_VERSION = 1
DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt), ("LogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA), ("HBD", Lipinski.NumHDonors),
    ("HBA", Lipinski.NumHAcceptors), ("RotatableBonds", Lipinski.NumRotatableBonds),
    ("HeavyAtomCount", Lipinski.HeavyAtomCount), ("RingCount", Lipinski.RingCount),
    ("AromaticRings", Lipinski.NumAromaticRings), ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("AliphaticRings", Lipinski.NumAliphaticRings), ("HeteroAtoms", Lipinski.NumHeteroatoms),
    ("NHOHCount", Lipinski.NHOHCount), ("NOCount", Lipinski.NOCount),
    ("MolMR", Crippen.MolMR), ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ("BalabanJ", Descriptors.BalabanJ), ("BertzCT", Descriptors.BertzCT),
    ("FormalCharge", Chem.GetFormalCharge), ("ValenceElectrons", Descriptors.NumValenceElectrons),
]


def feature_signature(config: dict) -> str:
    settings = {k: v for k, v in config.items() if k != "n_jobs"}
    return digest({"settings": settings, "rdkit": rdBase.rdkitVersion, "version": FEATURE_VERSION,
                   "descriptors": [name for name, _ in DESCRIPTORS], "pharm_factory": "Gobbi_Pharm2D",
                   "hash": "sha256(str(position)) first 8 bytes big-endian modulo bits"})


def molecule_features(smiles: str, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Cannot extract features for invalid SMILES: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config["morgan_radius"], fpSize=config["morgan_bits"])
    ecfp = np.asarray(generator.GetFingerprint(mol).GetOnBits(), dtype=np.int32)
    native = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    pharm = np.unique([int.from_bytes(hashlib.sha256(str(int(i)).encode("ascii")).digest()[:8], "big")
                       % config["pharmacophore_bits"] for i in native.GetOnBits()]).astype(np.int32)
    values = []
    for _, function in DESCRIPTORS:
        try:
            value = float(function(mol))
            values.append(value if np.isfinite(value) else np.nan)
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            values.append(np.nan)
    return ecfp, pharm, np.asarray(values, dtype=np.float32)


def _csr(rows: list[np.ndarray], width: int) -> sparse.csr_matrix:
    indptr = np.cumsum([0] + [len(row) for row in rows])
    indices = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    return sparse.csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr), shape=(len(rows), width))


@dataclass
class FeatureBundle:
    structure_ids: np.ndarray
    ecfp: sparse.csr_matrix
    pharmacophore: sparse.csr_matrix
    descriptors: np.ndarray
    signature: str

    def take(self, indices: np.ndarray | list) -> FeatureBundle:
        idx = np.asarray(indices, dtype=int)
        return FeatureBundle(self.structure_ids[idx], self.ecfp[idx], self.pharmacophore[idx], self.descriptors[idx], self.signature)

    def for_frame(self, frame: pd.DataFrame) -> FeatureBundle:
        lookup = {s: i for i, s in enumerate(self.structure_ids)}
        return self.take([lookup[s] for s in frame.structure_id])

    def fingerprints(self) -> sparse.csr_matrix:
        return sparse.hstack([self.ecfp, self.pharmacophore], format="csr", dtype=np.uint8)


def featurize(entities: pd.DataFrame, config: dict, cache_dir: str | Path) -> FeatureBundle:
    if entities.structure_id.isna().any():
        raise ValueError("Feature requests include invalid molecules; audit must resolve them first")
    entities = entities.drop_duplicates("structure_id").sort_values("structure_id")
    signature = feature_signature(config)
    key = digest({"signature": signature, "entities": entities[["structure_id", "canonical_smiles"]].values.tolist()})
    cache = Path(cache_dir) / "features"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{key}.joblib"
    if target.exists():
        LOG.info("Loading feature cache %s", target)
        return joblib.load(target)
    LOG.info("Extracting features for %d unique structures", len(entities))
    # Process chunks: no dense float32 matrix for the molecular collection.
    ecfp_rows, pharm_rows, descriptors = [], [], []
    smiles = entities.canonical_smiles.tolist()
    with joblib.Parallel(n_jobs=config.get("n_jobs", 1)) as parallel:
        for start in range(0, len(smiles), 1024):
            result = parallel(joblib.delayed(molecule_features)(s, config) for s in smiles[start:start + 1024])
            for ecfp, pharm, desc in result:
                ecfp_rows.append(ecfp)
                pharm_rows.append(pharm)
                descriptors.append(desc)
            LOG.info("Featurized %d / %d", min(start + 1024, len(smiles)), len(smiles))
    bundle = FeatureBundle(entities.structure_id.to_numpy(), _csr(ecfp_rows, config["morgan_bits"]),
                           _csr(pharm_rows, config["pharmacophore_bits"]), np.stack(descriptors), signature)
    temporary = target.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(target)
    return bundle


def descriptor_transformer() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scale", StandardScaler())])


class CompactProjector(BaseEstimator, TransformerMixin):
    """Fit on permitted training rows; project to [0, pi] for future feature maps."""
    def __init__(self, dimensions: int = 6, seed: int = 2035):
        self.dimensions = dimensions
        self.seed = seed

    def fit(self, X, y=None):
        if self.dimensions not in (4, 6, 8):
            raise ValueError("Compact representation dimensions must be 4, 6, or 8")
        if min(X.shape) < self.dimensions:
            raise ValueError("Not enough training rows/features for requested projection")
        self.reducer_ = (TruncatedSVD(self.dimensions, random_state=self.seed) if sparse.issparse(X)
                         else PCA(self.dimensions, random_state=self.seed))
        reduced = self.reducer_.fit_transform(X)
        self.scaler_ = MinMaxScaler(feature_range=(0, np.pi), clip=True).fit(reduced)
        return self

    def transform(self, X):
        return self.scaler_.transform(self.reducer_.transform(X))
