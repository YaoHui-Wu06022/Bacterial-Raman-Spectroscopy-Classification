# RaMF Standalone Model

This package keeps the Sun et al. 2026 style RaMF experiment separate from the existing `raman` package.

It reuses:

- `raman.data.RamanDataset` and the existing preprocessing pipeline
- `raman.training.split` for train/validation splits
- `raman.training.losses` and `raman.eval.common` for comparable training metrics

Run a validation performance test:

```powershell
python -X utf8 -m ramf.train --dataset GN --level level_1 --epochs 65 --batch-size 32
```

For a quick CPU smoke run:

```powershell
python -X utf8 -m ramf.train --dataset GN --level level_1 --epochs 1 --batch-size 2 --cpu
```

Outputs are written under:

```text
output/ramf/<dataset>/<level>/<timestamp>/
```

The model follows the paper at the architectural level: a 1D spectral Transformer with CFFN, a 3D-CNN branch over GASF/MTF/RP maps, and symmetric cross-attention fusion. The paper does not disclose every layer width and preprocessing detail, so the defaults here are configurable rather than a claimed exact reproduction.
