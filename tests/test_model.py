import pytest
import torch

from radar_cls.models import BasicCNN, available_models, build_model
from radar_cls.utils import count_parameters


def test_parameter_budget():
    model = build_model("rf_spectronet", num_classes=12)
    n = count_parameters(model)
    # Same count as legacy/Main.py with 12 classes. The old README said 89,196; that was wrong.
    assert n == 86_196, "architecture changed: update the README or revert"
    assert n < 100_000


@pytest.mark.parametrize("size", [32, 224])
def test_forward_shape(size):
    model = build_model("rf_spectronet", num_classes=12).eval()
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 12)
    assert torch.isfinite(out).all()


def test_eval_mode_is_deterministic():
    model = build_model("rf_spectronet", num_classes=12).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_legacy_state_dict_keys_are_compatible():
    # Weights saved by legacy/Main.py (class BasicCNN) must load into the new module.
    assert BasicCNN is build_model("rf_spectronet", 12).__class__
    keys = build_model("rf_spectronet", 12).state_dict().keys()
    assert "stem.0.weight" in keys and "blocks.4.ca.conv_h.weight" in keys and "fc.weight" in keys


def test_unknown_model_raises():
    assert "rf_spectronet" in available_models()
    with pytest.raises(KeyError):
        build_model("does_not_exist", 12)


def _legacy_basic_cnn():
    """Load the model classes from legacy/Main.py without running the training script."""
    from pathlib import Path

    src = (Path(__file__).parents[1] / "legacy" / "Main.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec("import torch\nimport torch.nn as nn\n" + src[src.index("class CoordAtt") : src.index("model = BasicCNN")], ns)
    return ns["BasicCNN"]


def test_matches_legacy_model_outputs():
    legacy = _legacy_basic_cnn()(12).eval()
    new = build_model("rf_spectronet", 12).eval()
    new.load_state_dict(legacy.state_dict())
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(new(x), legacy(x))


def test_ema_warmup_tracks_early_weights():
    from radar_cls.engine import ModelEMA

    model = torch.nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.999, warmup=True)
    assert ema.current_decay() == pytest.approx(0.1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    # With warmup the first update moves the EMA 90% of the way; without it, only 0.1%.
    assert ema.ema.weight.mean().item() > 0.8
    assert ModelEMA(model, 0.999, warmup=False).current_decay() == 0.999
