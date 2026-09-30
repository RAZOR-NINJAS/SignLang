import json

import numpy as np
import torch
import torch.nn as nn

from .config import LABELS_PATH, WEIGHTS_PATH


class SignMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(192, 112), dropout=0.28):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)


def save_checkpoint(path, model, labels, mean, std, meta=None):
    path = path or WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": list(labels),
            "in_dim": model.in_dim,
            "n_classes": model.n_classes,
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
            "temperature": float((meta or {}).get("temperature", 1.0) or 1.0),
            "meta": meta or {},
        },
        path,
    )
    LABELS_PATH.write_text(json.dumps(list(labels), indent=2))
    return path


def load_checkpoint(path=None):
    path = path or WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}. Record data with `signlang collect`, "
            f"then run `signlang train`."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SignMLP(ckpt["in_dim"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class Predictor:
    def __init__(self, path=None):
        self.model, self.ckpt = load_checkpoint(path)
        self.labels = self.ckpt["labels"]
        self.mean = self.ckpt["mean"]
        self.std = self.ckpt["std"]
        self.temperature = float(self.ckpt.get("temperature", 1.0) or 1.0)

    def probs_from_landmarks(self, landmark_coords):
        from .features import landmark_features

        feats = landmark_features(
            np.asarray(landmark_coords, dtype=np.float32).reshape(21, 3)
        )
        return self.probs(feats)

    def probs(self, feats):
        x = (np.asarray(feats, dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).unsqueeze(0))
            p = torch.softmax(logits / self.temperature, dim=1).squeeze(0).numpy()
        return p
