import torch

from ramanv2.core.config import ExecutionConfig, TrainingConfig
from ramanv2.training.checkpoint import TrainingState
from ramanv2.training.loop import _update_ema_class_weights
from ramanv2.training.losses import FocalLoss
from ramanv2.training.train_spec import build_training_spec


def test_difficulty_weights_recompute_from_base_without_compounding() -> None:
    base = torch.tensor([1.0, 1.0])
    ema = torch.tensor([2.0, 1.0])
    difficulty = ema / ema.mean()
    expected = base * difficulty
    expected = expected / expected.mean()

    criterion = FocalLoss(1.2, weight=base.clone())
    state = TrainingState(ema_class_ce=ema.clone(), class_weights=base.clone())
    spec = build_training_spec(
        TrainingConfig(use_ema=True, ema_difficulty_weight=1.0),
        ExecutionConfig(),
    )

    _update_ema_class_weights(criterion, state, spec, epoch=5)
    first = criterion.weight.clone()
    _update_ema_class_weights(criterion, state, spec, epoch=6)

    assert torch.allclose(first, expected)
    # 难度不变时重复更新应保持同一结果；旧实现会在上一轮权重上继续累乘。
    assert torch.allclose(criterion.weight, first)
