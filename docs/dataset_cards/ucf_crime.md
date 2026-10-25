# Dataset Card: UCF-Crime

## What this dataset is

UCF-Crime is a large-scale dataset of 1,610 real-world surveillance
videos, split into 13 anomaly classes plus normal activity. It is the
standard benchmark for video anomaly detection and is explicitly
designed for weakly-supervised learning: each video has a single label
(anomaly class or normal), not per-frame annotations.

**Source:** Sultani, W., Chen, C., & Shah, M. (2018). *Real-world
Anomaly Detection in Surveillance Videos*. CVPR.

**Accessed via:** Manual download from UCF CRCV
(https://www.crcv.ucf.edu/projects/real-world/), ~120 GB for the full
dataset; the project uses the **Test split** (~10 GB).

## Taxonomy (13 anomaly classes + normal)

The original dataset defines 13 anomaly categories:

| Category | Sub-types (original) |
|---|---|
| Normal | — |
| Abuse | — |
| Arrest | — |
| Arson | — |
| Assault | — |
| Burglary | — |
| Explosion | — |
| Fighting | — |
| Road Accidents | — |
| Robbery | — |
| Shooting | — |
| Shoplifting | — |
| Stealing | — |
| Vandalism | — |

## Week 7.5 grouping (4 classes)

For the project's anomaly classifier, the 13 categories were grouped
into 4 classes to reduce sparsity and match a "response-type" framing:

| Week 7.5 class | Original sub-types included | Class share |
|---|---|---:|
| `normal` | normal | ~48% |
| `violence` | Assault, Fighting, Abuse | ~12% |
| `criminal` | Arrest, Burglary, Robbery, Shoplifting, Stealing | ~30% |
| `public_hazard` | Arson, Road Accidents, Vandalism, Explosion, Shooting | ~10% |

**This grouping is the project's own decision**, not part of the
dataset. The 4-class labels are documented in
`src/urbanintel/data/ucf_crime.py::CATEGORIES`.

## Size and splits

| Item | Value |
|---|---:|
| Source videos | 1,610 |
| Total duration | ~128 hours |
| Week 7.5 clip extraction | 16 frames, stride 8 |
| Max clips per video | 5 |
| Total training clips | 5,518 |
| Training videos | 1,126 |
| Validation clips | 1,180 (240 videos) |
| Test clips (project-held-out) | 1,183 (244 videos) |
| Official test clips (UCF) | 1,406 (290 videos) |

**Split discipline:** Train/val/test splits are stratified by **(class,
video_id)** to prevent frame-level leakage. No video appears in more
than one split. This is critical — a random split would let frames from
the same video appear in train and test, inflating accuracy.

## Clip extraction pipeline

For each source video, clips are extracted at stride 8:

1. Read all frames with `decord` (or equivalent)
2. For each clip starting frame, sample 16 uniformly-spaced frames
3. Resize to 112×112
4. Save as (16, 112, 112, 3) uint8

**Why 16 frames:** matches the R(2+1)D-18 Kinetics-400 input format.
**Why 112×112:** matches Kinetics normalization + R(2+1)D-18 default
input resolution.

**Cached at:** `data/ucf_crime_clips/{train,val,test,official_test}_clips.parquet`
(gitignored; regenerable via `scripts/build_ucf_clips.py`)

## Known limitations of this dataset

1. **Not real urban surveillance.** UCF-Crime is compiled from YouTube
   videos, movies, and web uploads — not municipal CCTV. Scenes,
   camera angles, lighting, and image quality are not representative
   of real urban deployment.

2. **Weak supervision (video-level labels).** A video labeled "Assault"
   contains both assault segments and normal activity. The 4-class
   classifier learns "this clip came from an assault video," not "this
   clip contains an assault." Temporal localization is not attempted.

3. **Class imbalance.** `normal` is ~48% of the dataset; `violence` is
   ~12%. Class-weighted loss compensates, but per-class F1 on `violence`
   lags (0.27 in Week 7.5).

4. **Small per-class sample.** `public_hazard` is ~10% of videos. The
   model underperforms on rare hazards (explosion, road accidents).

5. **Labels are subjective.** "Violence" vs. "Fighting" vs. "Assault"
   overlap; the 4-class grouping mitigates but doesn't eliminate this.

6. **Copyright.** UCF-Crime videos are aggregated from various sources;
   the dataset is research-only and not redistributable.

## Why this dataset fits UrbanMind-AI

UCF-Crime is the natural video-anomaly benchmark for a project that
wants to demonstrate staged fine-tuning of a pretrained video model
(R(2+1)D-18 from Kinetics-400). It is not representative of urban
surveillance footage, but it is the standard benchmark for
weakly-supervised anomaly detection, and the honest model card documents
this gap.

The 4-class grouping (normal / violence / criminal / public_hazard) is
also directly consumable by the Week 8 report generator as one of its
6 evidence sources.

**Connection to Week 9a fusion:** UCF-Crime is *not* fused with any
other modality in the current project. Cross-modal fusion (video +
audio) would require a paired dataset like MAVAD (planned for Week 11).

## License

Research use only. UCF-Crime is provided by UCF CRCV for academic
research. See the original paper and site for terms. Not
redistributable.

## Citation

```bibtex
@inproceedings{sultani2018real,
  title={Real-world Anomaly Detection in Surveillance Videos},
  author={Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6479--6488},
  year={2018}
}