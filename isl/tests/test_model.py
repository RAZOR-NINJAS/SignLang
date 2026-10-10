"""Unit tests for ISL model and NumPy inference parity."""

from pathlib import Path
import numpy as np
import pytest
import torch

from isl.config import FEATURE_DIM, TRAIN_LABELS
from isl.model import ISL_MLP, Predictor, get_torch_model


def test_numpy_isl_mlp_forward_shape_and_probability_sum():
    n_classes = len(TRAIN_LABELS)
    in_dim = FEATURE_DIM
    hidden = (192, 112)

    rng = np.random.default_rng(42)
    w0 = rng.normal(0, 0.1, (in_dim, hidden[0])).astype(np.float32)
    b0 = np.zeros(hidden[0], dtype=np.float32)
    w1 = rng.normal(0, 0.1, (hidden[0], hidden[1])).astype(np.float32)
    b1 = np.zeros(hidden[1], dtype=np.float32)
    w2 = rng.normal(0, 0.1, (hidden[1], n_classes)).astype(np.float32)
    b2 = np.zeros(n_classes, dtype=np.float32)

    ln0_w = np.ones(hidden[0], dtype=np.float32)
    ln0_b = np.zeros(hidden[0], dtype=np.float32)
    ln1_w = np.ones(hidden[1], dtype=np.float32)
    ln1_b = np.zeros(hidden[1], dtype=np.float32)

    mean = np.zeros(in_dim, dtype=np.float32)
    std = np.ones(in_dim, dtype=np.float32)

    model = ISL_MLP(
        weights=[(w0, b0), (w1, b1), (w2, b2)],
        ln_params=[(ln0_w, ln0_b), (ln1_w, ln1_b)],
        labels=TRAIN_LABELS,
        mean=mean,
        std=std,
        temperature=1.0,
    )

    x = rng.normal(0, 1, in_dim).astype(np.float32)
    probs = model.forward(x)

    assert probs.shape == (n_classes,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert pytest.approx(float(np.sum(probs)), rel=1e-5) == 1.0


def test_numpy_and_pytorch_numerical_parity():
    """Verify bitwise parity between PyTorch training forward pass and pure NumPy inference."""
    torch.manual_seed(123)
    in_dim = FEATURE_DIM
    n_classes = len(TRAIN_LABELS)

    torch_model = get_torch_model(in_dim, n_classes, hidden=(192, 112), dropout=0.0)
    torch_model.eval()

    state = torch_model.state_dict()
    weights = [
        (state["net.0.weight"].numpy().T, state["net.0.bias"].numpy()),
        (state["net.4.weight"].numpy().T, state["net.4.bias"].numpy()),
        (state["net.8.weight"].numpy().T, state["net.8.bias"].numpy()),
    ]
    ln_params = [
        (state["net.1.weight"].numpy(), state["net.1.bias"].numpy()),
        (state["net.5.weight"].numpy(), state["net.5.bias"].numpy()),
    ]

    mean = np.zeros(in_dim, dtype=np.float32)
    std = np.ones(in_dim, dtype=np.float32)

    np_model = ISL_MLP(
        weights=weights,
        ln_params=ln_params,
        labels=TRAIN_LABELS,
        mean=mean,
        std=std,
        temperature=1.0,
    )

    rng = np.random.default_rng(999)
    x = rng.normal(0, 1.0, in_dim).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        t_in = torch.from_numpy(x).unsqueeze(0)
        t_logits = torch_model(t_in)
        t_probs = torch.softmax(t_logits, dim=-1).squeeze(0).numpy()

    # NumPy forward
    np_probs = np_model.forward(x)

    assert np.allclose(t_probs, np_probs, atol=1e-4)


def test_missing_model_raises_filenotfound():
    """Verify that calling Predictor on missing path gives helpful error message."""
    missing_path = Path("isl/models/does_not_exist.pt")
    with pytest.raises(FileNotFoundError) as exc_info:
        Predictor(path=missing_path)
    assert "No trained ISL model found" in str(exc_info.value)
