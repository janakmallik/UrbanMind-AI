# Dataset Card: HMDB-51

## Identity

| Field | Value |
|---|---|
| Source | HMDB: A Large Video Database for Human Motion Recognition |
| Original paper | Kuehne et al., ICCV 2011 |
| Download | https://huggingface.co/datasets/Serrelab/hmdb51 |
| Local path | `<URBANINTEL_DATA_ROOT>/hmdb51/` |
| Total clips | 6,766 (official release) |
| Total classes | 51 |
| License | CC BY 4.0 |

## Purpose

Human action recognition. Week 7 uses a 5-class subset for
tractable video model training on a 6 GB GPU.

## Class Subset for Week 7

| Class | Clips | % of Subset |
|---|---:|---:|
| walk | 548 | 47.4% |
| run | 232 | 20.1% |
| sit | 142 | 12.3% |
| clap | 130 | 11.2% |
| wave | 104 | 9.0% |
| **Total** | **1,156** | 100% |

**Class imbalance: 5.3×** (walk 548 vs. wave 104). Report macro-F1
as primary metric, plus per-class accuracy.

## Split

Uses official HMDB-51 **split 1** (`testTrainMulti_7030_splits`):
- 70% train / 30% test per class
- Files named `<class>_test_split1.txt`
- Each line: `<filename> <0 or 1>` (0=train, 1=test)
- Clips from the same source video do not cross split boundaries
- **Validation set:** carved from official train (15% of train) for
  checkpoint selection during training

## Video Format

- `.avi` files, ~30 fps, ~1–15 seconds per clip
- Resolution varies (typically 320×240)
- Preprocessing: 16 uniformly sampled frames → resize to 112×112
- Data augmentation: random crop, horizontal flip

## Known Limitations

- **HMDB-51 is a human action dataset**, not urban driving footage.
  Actions like walk, run, sit, wave, clap are urban-relevant but the
  dataset was not designed for urban scene understanding.
- **5.3× class imbalance** — dominated by walk (47%)
- **~1,156 clips total** — small for deep learning; 3D CNN may overfit
- **Variable clip length and quality** — raw web/movie footage
- **No urban context** — background, lighting, camera motion are arbitrary

**This is a video modeling learning exercise**, not a claim about urban
video understanding. Future work should use a real urban video dataset
(e.g., AI City Challenge, BDD100K video subset) if available.

## Citation

Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., & Serre, T. (2011).
*HMDB: A Large Video Database for Human Motion Recognition*.
IEEE International Conference on Computer Vision (ICCV), 2556-2563.
https://doi.org/10.1109/ICCV.2011.6126543