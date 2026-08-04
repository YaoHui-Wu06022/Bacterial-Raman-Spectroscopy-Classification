"""模型构建的唯一应用入口。"""

from ramanv2.modeling.classifier import RamanClassifier1D
from ramanv2.modeling.spec import ModelSpec, validate_model_spec


def build_model(num_classes: int, model_spec: ModelSpec) -> RamanClassifier1D:
    """校验规格后构建分类模型。"""
    if int(num_classes) <= 0:
        raise ValueError("num_classes 必须为正数")
    validate_model_spec(model_spec)
    return RamanClassifier1D(int(num_classes), model_spec)


def validate_model_input(model_spec: ModelSpec, input_spec) -> None:
    """校验输入规格能与模型通道、pooling 和位置编码长度共同工作。"""
    if int(input_spec.in_channels) != model_spec.in_channels:
        raise ValueError("InputSpec 通道数与 ModelSpec 不一致")
    point_count = int(input_spec.point_count)
    if point_count <= 0:
        raise ValueError("InputSpec point_count 必须为正数")
    encoded_points = point_count // 4 if model_spec.backbone_type == "cnn" else point_count
    if encoded_points <= 0:
        raise ValueError("输入长度不足以通过 CNN pooling")
    if model_spec.encoder_type == "transformer" and encoded_points > 1000:
        raise ValueError("输入长度超过 Transformer 位置编码上限")
