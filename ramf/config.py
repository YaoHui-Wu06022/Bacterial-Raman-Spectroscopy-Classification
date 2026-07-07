from dataclasses import asdict, dataclass


@dataclass
class RaMFConfig:
    """Hyperparameters for the Sun et al. 2026 style RaMF model."""

    in_channels: int = 1

    # 1D Raman transformer branch
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 2
    transformer_dropout: float = 0.2
    patch_size: int = 16
    pos_max_tokens: int = 512
    cffn_ratio: int = 2

    # GASF/MTF/RP image stack and 3D-CNN branch
    image_size: int = 64
    mtf_bins: int = 8
    mtf_quantile: bool = False
    rp_threshold: float = 0.10
    branch_channels: int = 32
    cube_pool_size: int = 4

    # Symmetric cross-attention fusion
    fusion_dim: int | None = None
    fusion_heads: int | None = None
    dropout: float = 0.5

    def resolved_fusion_dim(self):
        return int(self.fusion_dim or self.transformer_dim)

    def resolved_fusion_heads(self):
        return int(self.fusion_heads or self.transformer_heads)

    def to_dict(self):
        return asdict(self)
