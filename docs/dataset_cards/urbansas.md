# Dataset Card: Urbansas

## What this dataset is

Urbansas is a paired audio-visual dataset for urban vehicle
understanding. It combines clips from two source corpora:

- **MAVD** (Madrid Audio-Visual Dataset) — 540 clips from 5 Madrid
  locations. Spanish urban traffic, daytime only (one location is
  the exception; see below).
- **TAU Urban Audio-Visual Scenes 2021** — 540 clips from 10+ European
  cities (Barcelona, Helsinki, Lisbon, Lyon, London, Milan, Paris,
  Prague, Stockholm, Vienna). Different microphones, different street
  acoustics.

Each clip is 10 seconds long, 48 kHz audio, 24 fps video, with
per-event audio annotations (`start`, `end`, `label`) and per-frame
video bounding-box annotations.

**Source:** https://zenodo.org/records/6658386

**Local path:** `<URBANSAS_DIR>` (default `C:\Users\Janak Mallik\ml-data\urbansas`)

**Not to be confused with:** MAVAD (Masked Audio-Visual Anomaly
Detection, Zenodo record 7950008, Malta cameras). `scripts/download_mavad.py`
downloads MAVAD, not Urbansas. The two datasets have unrelated
content and label schemes.

## Taxonomy

Audio annotations use 6 labels:

| Label | Meaning |
|---|---|
| `car` | Car engine/tire sound, source visible in frame |
| `bus` | Bus sound, source visible in frame |
| `motorbike` | Motorbike sound, source visible in frame |
| `truck` | Truck sound, source visible in frame |
| `offscreen` | Vehicle sound, source not visible in frame |
| `-1` | No sound event in this segment |

**A clip's label set can contain multiple labels.** A clip can have
`car` + `offscreen` simultaneously — meaning a visible car was heard
at one time window and a different (offscreen) vehicle was heard at
another. The Week 9c experiment's target is:

> `has_offscreen = "offscreen" in set(clip's audio labels)`

**This is not "is there a visible vehicle."** It is "does the clip
contain an offscreen vehicle sound." A clip with a clearly visible
car and no offscreen sound is a *negative* under this target.

## Class distribution (per-clip, MAVD + TAU combined)

| Label set | MAVD (540) | TAU (540) | Combined (1,080) |
|---|---:|---:|---:|
| Only `offscreen` | 66 (12.2%) | 26 (4.8%) | 92 (8.5%) |
| `offscreen` + visible-vehicle | 72 (13.3%) | 55 (10.2%) | 127 (11.8%) |
| Only visible-vehicle | 357 (66.1%) | 374 (69.3%) | 731 (67.7%) |
| Empty (only `-1`) | 45 (8.3%) | 85 (15.7%) | 130 (12.0%) |

**`has_offscreen` positive rate:** MAVD 25.6%, TAU 15.0%.

**Both percentages are structural facts about the dataset, not the
model.** Any model trained on MAVD will see a different positive
prior than it faces on TAU. This is documented; not corrected.

## Video annotations

`video_annotations.csv` provides per-frame bounding boxes for visible
vehicles, with `class_id`, `visibility`, `location_id`, `city`,
`subset`, `night`. **Not used in the Week 9c audio-only experiment.**
Will be used in the video-only baseline.

Notable: the `location_id` field is the source of the 5 MAVD locations
and the ~35 TAU locations used for CV splitting.

## Splits used (Week 9c)

| Split | Size | Source |
|---|---:|---|
| MAVD train (5-fold LOLA) | ~432 per fold | All MAVD minus one location |
| MAVD test (5-fold LOLA) | ~108 per fold | Held-out location |
| TAU external | 540 | All TAU, untouched until final eval |

**No random split.** Leave-one-location-out ensures no scene-level
leakage: a Madrid intersection that appears in training does not
appear in test.

## Preprocessing pipeline

**Audio (Week 9c):**

1. Load each `.wav` at 48 kHz (source is 48 kHz stereo; downmix to
   mono via `soundfile.read` + `.mean(axis=1)`)
2. Pad or trim to exactly 10.0 s (480,000 samples)
3. Compute 128-mel log-spectrogram:
   `n_fft=2048, hop_length=2048, n_mels=128, fmin=20, fmax=24000`
4. Convert to dB via `librosa.power_to_db(ref=np.max)`
5. Per-sample normalize: `(mel - mel.mean()) / (mel.std() + 1e-6)`
6. Cache all 1,080 clips to `data/urbansas_mels/mels.npy` (~130 MB)

**Note:** the original per-clip RMS is preserved in the annotations
(audio peak timestamp, RMS), but the mel normalization discards
absolute amplitude. This is the same trade-off as Week 7.6's
UrbanSound8K preprocessing — documented there as a limitation.

## Known limitations of the dataset

1. **`offscreen` coexists with visible-vehicle labels.** 52% of MAVD
   positives and 68% of TAU positives have both. The task is not a
   clean visible-vs-invisible dichotomy. Per-subclass ROC-AUC in the
   model card reflects this — `only_offscreen` 0.660 vs `both` 0.590.
2. **`rivera0922` has 2 positives.** Fold 3's F1 of 0.5000 is 1/2
   true positive. Report as noise; do not weight as equally reliable
   with `magallanes0308` (48 positives).
3. **MAVD night coverage is concentrated.** Four of five MAVD
   locations are 100% daytime. The only night clips (30, `rivera0922`)
   are also the fold with 2 positives. No night-vs-day analysis is
   possible with this fold structure.
4. **TAU is genuinely cross-city.** Different microphones, different
   street acoustics, different vehicle-type mixes. The TAU external
   test measures both domain shift and label-prevalence shift
   simultaneously. Do not interpret a TAU failure as a clean
   generalization measure.
5. **No overlapping-audio handling.** `offscreen` and `car` can
   co-occur; the model sees them as a single target. Whether
   overlapping offscreen + visible events are distinguishable in
   audio is untested.
6. **Short clips only.** 10 seconds; cannot capture sustained events
   like traffic jams or long construction periods.
7. **Label event durations are compatible MAVD↔TAU.** MAVD median
   3.85 s, TAU median 3.21 s; both span ~0–10 s. This is
   *reassuring* but does not prove label-definition equivalence —
   different annotators, different corpora.

## Why this dataset fits UrbanMind-AI

UrbanMind-AI's Week 9c goal is a cross-modal (audio + video) fusion
experiment. Previous fusion attempts (9a multi-view on BDD100K,
9b tabular+text on NYC 311) were same-modality or non-paired.
**Urbansas is the first genuinely paired audio+video dataset in the
project**, enabling the first true cross-modal fusion test.

The `offscreen` label is the natural fusion target: video alone
cannot detect offscreen vehicles (no visual signal by definition),
audio alone can (but weakly, as this experiment showed), and fusion
*tests the hypothesis* that the two modalities provide complementary
information. Whether they do is an empirical question — this dataset
makes the question testable.

## License

See the Urbansas Zenodo record for exact terms. Data is released
for research use.

## Citation

% TODO: Replace with exact citation from CITATIONS.md before final portfolio polish.

Full citation pending — replace with the exact Zenodo record
citation from `CITATIONS.md`.

## Reproduce the cache

```powershell
# one-time, ~5 min on a modern CPU
uv run python scripts/precompute_urbansas_mels.py

# verify the cache
Get-ChildItem data\urbansas_mels\
# expect: filenames.txt (~20 KB), mels.npy (~130 MB)
```

## Changelog

| Date | Notes |
|---|---|
| 2026-10-27 | Initial card. Audio-only baseline complete; video-only and fusion pending. |