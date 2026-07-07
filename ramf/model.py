import torch
import torch.nn as nn
import torch.nn.functional as F

from ramf.config import RaMFConfig


class CFFN1D(nn.Module):
    """CNN-enhanced feed-forward block used in the spectral transformer."""

    def __init__(self, dim, ratio=2, dropout=0.0):
        super().__init__()
        hidden = max(int(dim * ratio), dim)
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, hidden, kernel_size=1, bias=False),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel_size=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.net(x)
        return residual + x.transpose(1, 2)


class RamanTransformerBlock(nn.Module):
    """Pre-norm MSA block with CFFN replacing the normal transformer FFN."""

    def __init__(self, dim, heads, cffn_ratio, attn_dropout, ffn_dropout):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(attn_dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.cffn = CFFN1D(dim, ratio=cffn_ratio, dropout=ffn_dropout)

    def forward(self, x):
        attn_in = self.norm_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.drop_attn(attn_out)
        return self.cffn(self.norm_ffn(x))


class SpectralTransformerBranch(nn.Module):
    """1D branch: non-overlapping spectral tokens, MSA, and CFFN blocks."""

    def __init__(self, config):
        super().__init__()
        dim = int(config.transformer_dim)
        heads = int(config.transformer_heads)
        if dim % heads != 0:
            raise ValueError(f"transformer_dim={dim} must be divisible by heads={heads}")

        self.patch_size = max(int(config.patch_size), 1)
        self.patch_embed = nn.Conv1d(
            int(config.in_channels),
            dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, int(config.pos_max_tokens), dim))
        self.blocks = nn.ModuleList(
            [
                RamanTransformerBlock(
                    dim,
                    heads,
                    cffn_ratio=int(config.cffn_ratio),
                    attn_dropout=float(config.transformer_dropout),
                    ffn_dropout=float(config.dropout),
                )
                for _ in range(int(config.transformer_layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _positional_slice(self, length):
        if length <= self.pos_embed.size(1):
            return self.pos_embed[:, :length, :]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=length, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, x):
        if x.size(-1) < self.patch_size:
            x = F.pad(x, (0, self.patch_size - x.size(-1)))
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens + self._positional_slice(tokens.size(1))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class SpectralImageStack(nn.Module):
    """Build GASF, MTF, and RP maps from the first Raman input channel."""

    def __init__(self, image_size=64, mtf_bins=8, rp_threshold=0.10, mtf_quantile=False):
        super().__init__()
        self.image_size = max(int(image_size), 2)
        self.mtf_bins = max(int(mtf_bins), 2)
        self.rp_threshold = float(rp_threshold)
        self.mtf_quantile = bool(mtf_quantile)

    def _resize(self, values):
        values = values.unsqueeze(1)
        values = F.interpolate(
            values,
            size=self.image_size,
            mode="linear",
            align_corners=False,
        )
        return values.squeeze(1)

    def _scale01(self, values):
        low = values.amin(dim=1, keepdim=True)
        high = values.amax(dim=1, keepdim=True)
        return (values - low) / (high - low).clamp_min(1e-6)

    def _gasf(self, values01):
        values = values01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        phi = torch.acos(values)
        return torch.cos(phi.unsqueeze(2) + phi.unsqueeze(1))

    def _mtf_quantile(self, values01):
        quantiles = torch.linspace(
            0.0,
            1.0,
            self.mtf_bins + 1,
            device=values01.device,
            dtype=values01.dtype,
        )
        images = []
        for sample in values01:
            edges = torch.quantile(sample, quantiles)
            states = torch.bucketize(sample.contiguous(), edges[1:-1].contiguous())
            transitions = states[:-1] * self.mtf_bins + states[1:]
            counts = torch.bincount(
                transitions,
                minlength=self.mtf_bins * self.mtf_bins,
            ).to(values01.dtype)
            counts = counts.view(self.mtf_bins, self.mtf_bins)
            probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-6)
            images.append(probs[states][:, states])
        return torch.stack(images, dim=0)

    def _mtf_fast(self, values01):
        batch_size, length = values01.shape
        states = torch.clamp(
            (values01 * self.mtf_bins).to(torch.long),
            min=0,
            max=self.mtf_bins - 1,
        )
        transitions = states[:, :-1] * self.mtf_bins + states[:, 1:]
        counts = values01.new_zeros(batch_size, self.mtf_bins * self.mtf_bins)
        counts.scatter_add_(
            dim=1,
            index=transitions,
            src=torch.ones_like(transitions, dtype=values01.dtype),
        )
        probs = counts.view(batch_size, self.mtf_bins, self.mtf_bins)
        probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(1e-6)
        source_rows = probs.gather(
            dim=1,
            index=states.unsqueeze(2).expand(batch_size, length, self.mtf_bins),
        )
        return source_rows.gather(
            dim=2,
            index=states.unsqueeze(1).expand(batch_size, length, length),
        )

    def _mtf(self, values01):
        if self.mtf_quantile:
            return self._mtf_quantile(values01)
        return self._mtf_fast(values01)

    def _rp(self, values01):
        dist = (values01.unsqueeze(2) - values01.unsqueeze(1)).abs()
        return (dist <= self.rp_threshold).to(values01.dtype)

    def forward(self, x):
        values = self._resize(x[:, 0, :])
        values01 = self._scale01(values)
        return torch.stack(
            [
                self._gasf(values01),
                self._mtf(values01),
                self._rp(values01),
            ],
            dim=1,
        )


class SpatialAttention3D(nn.Module):
    """3D spatial attention over the multi-scale spectral cube."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class ResidualBlock3D(nn.Module):
    """Residual 3D convolution block for cube features."""

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CNN3DBranch(nn.Module):
    """3D-CNN branch over stacked GASF, MTF, and RP maps."""

    def __init__(self, config):
        super().__init__()
        out_dim = int(config.resolved_fusion_dim())
        branch_channels = int(config.branch_channels)
        self.image_stack = SpectralImageStack(
            image_size=int(config.image_size),
            mtf_bins=int(config.mtf_bins),
            rp_threshold=float(config.rp_threshold),
            mtf_quantile=bool(config.mtf_quantile),
        )
        self.inception = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        1,
                        branch_channels,
                        kernel_size=(3, kernel_size, kernel_size),
                        padding=(1, kernel_size // 2, kernel_size // 2),
                        bias=False,
                    ),
                    nn.BatchNorm3d(branch_channels),
                    nn.GELU(),
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.out_channels = branch_channels * len(self.inception)
        self.spatial_attention = SpatialAttention3D()
        self.residual = ResidualBlock3D(self.out_channels, dropout=float(config.dropout))
        self.pool = nn.AdaptiveAvgPool3d((1, int(config.cube_pool_size), int(config.cube_pool_size)))
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.Linear(self.out_channels, out_dim),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )

    def forward(self, x):
        cube = self.image_stack(x).unsqueeze(1)
        features = torch.cat([branch(cube) for branch in self.inception], dim=1)
        features = self.spatial_attention(features)
        features = self.residual(features)
        features = self.pool(features)
        tokens = features.flatten(2).transpose(1, 2)
        return self.token_proj(tokens)


class SymmetricCrossAttentionFusion(nn.Module):
    """Fuse 1D spectral tokens and 3D cube tokens with bidirectional attention."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"fusion_dim={dim} must be divisible by heads={heads}")
        self.spec_norm = nn.LayerNorm(dim)
        self.cube_norm = nn.LayerNorm(dim)
        self.spec_to_cube = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cube_to_spec = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, spec_tokens, cube_tokens):
        spec_tokens = self.spec_norm(spec_tokens)
        cube_tokens = self.cube_norm(cube_tokens)
        spec_ctx, _ = self.spec_to_cube(spec_tokens, cube_tokens, cube_tokens, need_weights=False)
        cube_ctx, _ = self.cube_to_spec(cube_tokens, spec_tokens, spec_tokens, need_weights=False)
        return self.out(torch.cat([spec_ctx.mean(dim=1), cube_ctx.mean(dim=1)], dim=1))


class RaMFNet(nn.Module):
    """Transformer and 3D-CNN feature-fusion network for Raman spectra."""

    def __init__(self, num_classes, config=None):
        super().__init__()
        self.config = config or RaMFConfig()
        self.num_classes = int(num_classes)
        self.dim = int(self.config.resolved_fusion_dim())
        self.spectral_branch = SpectralTransformerBranch(self.config)
        self.cube_branch = CNN3DBranch(self.config)
        self.fusion = SymmetricCrossAttentionFusion(
            self.dim,
            int(self.config.resolved_fusion_heads()),
            dropout=float(self.config.dropout),
        )
        self.head = nn.Linear(self.dim, self.num_classes)

    def forward(self, x, return_feat=False):
        spec_tokens = self.spectral_branch(x)
        cube_tokens = self.cube_branch(x)
        feat = self.fusion(spec_tokens, cube_tokens)
        logits = self.head(feat)
        if return_feat:
            return logits, feat
        return logits
