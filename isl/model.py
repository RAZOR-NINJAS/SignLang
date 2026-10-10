"""ISL Multi-Layer Perceptron (PyTorch training + pure NumPy live inference)."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .config import LABELS_PATH, WEIGHTS_PATH


def _gelu_np(x: np.ndarray) -> np.ndarray:
    """Accurate numerical approximation of GELU activation in NumPy."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def _layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """NumPy implementation of LayerNorm."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return norm * weight + bias


class ISL_MLP:
    """NumPy-native inference engine for ISL MLP.

    Executes forward passes without importing PyTorch at runtime, preserving RAM
    and reducing thermal load on memory-constrained systems.
    """

    def __init__(
        self,
        weights: List[Tuple[np.ndarray, np.ndarray]],
        ln_params: List[Tuple[np.ndarray, np.ndarray]],
        labels: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        temperature: float = 1.0,
    ):
        self.weights = weights
        self.ln_params = ln_params
        self.labels = list(labels)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.temperature = max(float(temperature), 0.01)

    def forward(self, feats: np.ndarray) -> np.ndarray:
        """Run forward inference and return probability distribution over classes."""
        x = (np.asarray(feats, dtype=np.float32) - self.mean) / self.std
        # Layer 1
        h = np.dot(x, self.weights[0][0]) + self.weights[0][1]
        h = _layer_norm_np(h, self.ln_params[0][0], self.ln_params[0][1])
        h = _gelu_np(h)

        # Layer 2
        h = np.dot(h, self.weights[1][0]) + self.weights[1][1]
        h = _layer_norm_np(h, self.ln_params[1][0], self.ln_params[1][1])
        h = _gelu_np(h)

        # Output projection
        logits = np.dot(h, self.weights[2][0]) + self.weights[2][1]
        logits = logits / self.temperature

        # Stable Softmax
        shift = logits - np.max(logits)
        exp_logits = np.exp(shift)
        probs = exp_logits / np.sum(exp_logits)
        return probs.astype(np.float32)

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        return self.forward(feats)


def get_torch_model(in_dim: int, n_classes: int, hidden: Tuple[int, int] = (192, 112), dropout: float = 0.28):
    """Factory creating PyTorch module for training (only imported during training)."""
    import torch
    import torch.nn as nn

    class PyTorchISLMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_dim = in_dim
            self.n_classes = n_classes
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden[0]),
                nn.LayerNorm(hidden[0]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]),
                nn.LayerNorm(hidden[1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden[1], n_classes),
            )

        def forward(self, x):
            return self.net(x)

    return PyTorchISLMLP()


def save_checkpoint(path: Optional[Path], model, labels: List[str], mean: np.ndarray, std: np.ndarray, meta: Optional[dict] = None) -> Path:
    """Save PyTorch training checkpoint and export companion labels.json."""
    import torch

    target_path = path or WEIGHTS_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()

    torch.save(
        {
            "state_dict": state,
            "labels": list(labels),
            "in_dim": getattr(model, "in_dim", 164),
            "n_classes": getattr(model, "n_classes", len(labels)),
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
            "temperature": float((meta or {}).get("temperature", 1.0) or 1.0),
            "meta": meta or {},
        },
        target_path,
    )
    LABELS_PATH.write_text(json.dumps(list(labels), indent=2))
    return target_path


def load_numpy_predictor(path: Optional[Path] = None) -> ISL_MLP:
    """Load model weights directly into NumPy inference engine."""
    import torch

    target_path = path or WEIGHTS_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"No trained ISL model found at {target_path}. "
            f"Run 'signlang isl collect' then 'signlang isl train' to record real signs and train the model."
        )

    ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # Extract weights in NumPy format (W.T because PyTorch Linear stores (out_features, in_features))
    weights = [
        (state["net.0.weight"].numpy().T, state["net.0.bias"].numpy()),
        (state["net.4.weight"].numpy().T, state["net.4.bias"].numpy()),
        (state["net.8.weight"].numpy().T, state["net.8.bias"].numpy()),
    ]
    ln_params = [
        (state["net.1.weight"].numpy(), state["net.1.bias"].numpy()),
        (state["net.5.weight"].numpy(), state["net.5.bias"].numpy()),
    ]

    return ISL_MLP(
        weights=weights,
        ln_params=ln_params,
        labels=ckpt["labels"],
        mean=ckpt["mean"],
        std=ckpt["std"],
        temperature=float(ckpt.get("temperature", 1.0) or 1.0),
    )


def load_checkpoint(path: Optional[Path] = None) -> Tuple[ISL_MLP, List[str], np.ndarray, np.ndarray, dict]:
    """Load model checkpoint and return (model, labels, mean, std, meta)."""
    import torch

    target_path = path or WEIGHTS_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {target_path}")

    ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
    model = load_numpy_predictor(target_path)
    return (
        model,
        ckpt["labels"],
        np.asarray(ckpt["mean"], dtype=np.float32),
        np.asarray(ckpt["std"], dtype=np.float32),
        ckpt.get("meta", {}),
    )


class Predictor:
    """Public predictor interface for ISL recognition."""

    def __init__(self, path: Optional[Path] = None):
        self._engine = load_numpy_predictor(path)
        self.labels = self._engine.labels

    def probs(self, feats: np.ndarray) -> np.ndarray:
        return self._engine.forward(feats)
