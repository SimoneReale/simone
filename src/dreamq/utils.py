"""Configuration, provenance, deterministic execution, and JSON logging."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False) + "\n")


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ["dreamq-pgk2", "rdkit", "numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib", "xgboost-cpu", "xgboost", "lightgbm", "chemprop"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "source-tree" if name == "dreamq-pgk2" else "not-installed"
    return result


def seed_all(seed: int, threads: int = 2) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _merge(base: dict, update: dict) -> dict:
    from copy import deepcopy
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _yaml_config(path: Path, visited: set[Path] | None = None) -> dict:
    visited = set() if visited is None else visited
    if path in visited:
        raise ValueError(f"Cyclic YAML extends: {path}")
    visited.add(path)
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    if parent:
        config = _merge(_yaml_config((path.parent / parent).resolve(), visited), config)
    return config


def read_config(path: str | Path, dev: bool = False) -> dict:
    path = Path(path).resolve()
    config = _yaml_config(path)
    root = (path.parent / config.get("project_root", "..")).resolve()
    for key in ["data_dir", "cache_dir", "artifact_dir", "prediction_dir"]:
        config[key] = str((root / config[key]).resolve())
    config["development"] = bool(dev)
    if dev:
        config["cache_dir"] += "/dev"
        config["artifact_dir"] += "/dev"
        config["prediction_dir"] += "/dev"
        config["pretraining"].update(config.get("development_overrides", {}).get("pretraining", {}))
        config["cv"].update(config.get("development_overrides", {}).get("cv", {}))
        config["models"].update(config.get("development_overrides", {}).get("models", {}))
        config["zoo"] = _merge(config.get("zoo", {}), config.get("development_overrides", {}).get("zoo", {}))
    return config


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"time": self.formatTime(record), "level": record.levelname,
                           "logger": record.name, "message": record.getMessage()})


def configure_logging(log_path: str | Path | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handlers = [handler]
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        persistent = logging.FileHandler(log_path)
        persistent.setFormatter(JsonFormatter())
        handlers.append(persistent)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
