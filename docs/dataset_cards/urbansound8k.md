# Dataset Card: UrbanSound8K

## What this dataset is

UrbanSound8K is a curated dataset of 8,732 labeled urban sound clips, each
up to 4 seconds long, drawn from field recordings in New York City. It is
the standard benchmark for urban sound classification and is specifically
designed to reflect the acoustic conditions of dense urban environments.

**Source:** Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and
Taxonomy for Urban Sound Research*. ACM MM '14.

**Accessed via:** HuggingFace parquet mirror (`ml-data/urbansound8k`),
16 shards, ~6.52 GiB raw audio.

## Taxonomy (10 classes)

| classID | Class | Urban relevance |
|---:|---|---|
| 0 | air_conditioner | Ambient HVAC noise — background layer of most urban scenes |
| 1 | car_horn | Traffic stress, aggressive driving, congestion |
| 2 | children_playing | Public spaces, parks, school zones |
| 3 | dog_bark | Residential, parks, pedestrian areas |
| 4 | drilling | Construction, roadwork — persistent urban hazard |
| 5 | engine_idling | Idling vehicles, congestion, air-quality indicator |
| 6 | gun_shot | **Emergency** — the primary anomaly class in this taxonomy |
| 7 | jackhammer | Construction, roadwork |
| 8 | siren | **Emergency** — police, ambulance, fire response |
| 9 | street_music | Public performance, street-level culture |

**Two classes are explicitly emergency-relevant for UrbanMind-AI:**
`gun_shot` (classID 6) and `siren` (classID 8). These are the audio signals
that a downstream alert system would act on. The remaining 8 classes are
context/background categories — useful for distinguishing urban ambience
from emergencies, not for flagging anomalies themselves.

## Class distribution (global, 8,732 clips)

| classID | Class | Count | Share | Notes |
|---:|---|---:|---:|---|
| 0 | air_conditioner | 1000 | 11.4% | majority |
| 1 | car_horn | 429 | 4.9% | **minority (2.3× under)** |
| 2 | children_playing | 1000 | 11.4% | majority |
| 3 | dog_bark | 1000 | 11.4% | majority |
| 4 | drilling | 1000 | 11.4% | majority |
| 5 | engine_idling | 1000 | 11.4% | majority |
| 6 | gun_shot | 374 | 4.3% | **minority (2.7× under) — the class that matters most** |
| 7 | jackhammer | 1000 | 11.4% | majority |
| 8 | siren | 929 | 10.6% | slightly under |
| 9 | street_music | 1000 | 11.4% | majority |

**Implication:** the two emergency-relevant classes (`gun_shot`, `siren`)
are the two most under-represented. `gun_shot` is the *single* smallest class.
This is a hard problem — an emergency-focused classifier on this dataset
is trained on the fewest examples of the class it cares about most. Any
classifier trained on UrbanSound8K must use class-weighted loss or
resampling to compensate.

## Predefined 10-fold cross-validation

UrbanSound8K ships with **10 predefined folds**, and the dataset authors
explicitly recommend using them instead of random splits. The folds are
designed to prevent "same recording session" leakage: clips from the same
original recording never appear in different folds.

**Critical caveat — folds are session-stratified, not class-stratified.**
The five majority classes (0, 2, 3, 4, 9) happen to have exactly 100
clips in each fold, but the minority classes vary substantially:

| classID | class | f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 | f10 | min/max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | air_conditioner | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 1.0× |
| 1 | car_horn | 36 | 42 | 43 | 59 | 98 | 28 | 28 | 30 | 32 | 33 | **3.5×** |
| 2 | children_playing | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 1.0× |
| 3 | dog_bark | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 1.0× |
| 4 | drilling | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 1.0× |
| 5 | engine_idling | 96 | 100 | 107 | 107 | 107 | 107 | 106 | 88 | 89 | 93 | 1.2× |
| 6 | gun_shot | 35 | 35 | 36 | 38 | 40 | 46 | 51 | 30 | 31 | 32 | 1.7× |
| 7 | jackhammer | 120 | 120 | 120 | 120 | 120 | 68 | 76 | 78 | 82 | 96 | **1.8×** |
| 8 | siren | 86 | 91 | 119 | 166 | 71 | 74 | 77 | 80 | 82 | 83 | **2.3×** |
| 9 | street_music | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 1.0× |

Two implications:

1. **Per-fold class weights would differ significantly from global weights**
   for the minority classes. Our training script uses global class weights
   (computed once from all 8,732 rows) — this is an approximation, and the
   per-fold weight for `car_horn` varies by up to ~3.5× depending on which
   fold is held out. A more rigorous protocol would recompute weights
   per fold. Documented as a limitation; not corrected in Week 7.6.

2. **10-fold CV mean ± std is the right metric** — and the std will be
   large for minority-class F1 because the minority classes appear in
   very different quantities per fold. A single 80/20 split would give a
   misleadingly confident-looking single number. **Report mean ± std
   across all 10 folds.** Any fold-level summary that isn't the mean
   ± std is not honest.

## Preprocessing pipeline

**Raw audio → cached mel-spectrograms:**

1. **Mono downmix + resample.** Source audio is **44.1 kHz stereo**;
   `precompute_audio.py` averages channels and resamples to
   **22,050 Hz mono** (standard for UrbanSound8K baselines).
   The mel-spectrogram pipeline then assumes 22.05 kHz throughout.
2. **Pad or trim** to exactly 4.0 seconds (88,200 samples)
3. **Mel-spectrogram** via `librosa.feature.melspectrogram`:
   - `n_fft=1024`, `hop_length=512`, `n_mels=128`
   - `fmin=20 Hz`, `fmax=8000 Hz`
   - Output shape: **(128 mel bins, 173 time frames)**
4. **Power-to-dB** via `librosa.power_to_db(ref=np.max)` → float32
5. **Cache** all 8,732 clips to a single `audio.npy` memmap
   - **2.87 GiB** on disk (verified)
   - Enables 3× training speedup vs. re-decoding mp3/ogg each epoch

The cache is created once by `scripts/precompute_audio.py` and lives at
`C:/Users/Janak Mallik/ml-data/urbansound8k-cache/` — **outside the repo**,
so it is not tracked by git.

**Runtime:** precompute took ~3-4 minutes end-to-end on the
author's machine (16 shards, 8,732 clips). Once cached, each training
epoch reads memmapped float32 from disk at ~3× the throughput of
re-decoding mp3/ogg from the parquet shards, which is why the training
script never touches the original shards.

**Per-sample normalization at load time:**
```python
mel = (mel - mel.mean()) / (mel.std() + 1e-6)
```
This is a **design choice**, not a standard: it discards absolute
amplitude information. A dataset-wide per-mel-bin normalization
(precomputed mean/std across training data) is the more common choice
for UrbanSound8K. The per-sample approach was used here for simplicity
and is documented as a limitation in the model card.

## Splits used

| Split source | Size | Purpose |
|---|---:|---|
| Global | 8,732 | full dataset |
| Per-fold train | ~6,700 | 9 folds minus 15% val |
| Per-fold val | ~1,180 | 15% of 9-fold train pool (stratified) |
| Per-fold test | ~873 | held-out fold |

**Important:** validation is carved from the 9-fold training pool,
not from the test fold. Test sets are the actual predefined folds.
This means **val and test are both disjoint from training** — there is
no leakage. However, the per-fold val set is ~35% larger than the
test set and shares 9-fold provenance with training rows, so val
F1 is *not* a good generalization estimate. Only test F1 is.

## Known limitations of the dataset

1. **Short clips only** — 4 seconds max; cannot capture sustained events
   like multi-minute traffic jams or long construction periods.
2. **Single-label** — each clip has one class label. Real urban audio
   is overlapped (siren + traffic + construction).
3. **Label granularity is coarse** — `dog_bark` covers all dog barks;
   `gun_shot` covers gunshots, fireworks, and backfiring cars (acoustic
   confusions that a real emergency system must handle).
4. **No speech** — UrbanSound8K does not include conversational speech
   or shouted commands. A system that needs to detect "call the police"
   must use a separate speech model (see Week 7.6b — Whisper keywords).
5. **No live deployment data** — all clips are field recordings; model
   performance on real-time, unnormalized, overlapping audio from a
   live feed is unknown and untested.
6. **Class imbalance** — the two emergency-relevant classes are the
   smallest. Reported macro-F1 (not accuracy) is the honest metric.
7. **Variable source length, zero-padded to 4 s.** Clips range from
   ~0.06 s to 4.0 s. The pipeline zero-pads shorter clips to exactly
   88,200 samples, so very short clips (e.g. a 0.06 s clip) become
   >98% silence after padding — the model effectively receives no
   signal for those rows. This affects a small fraction of the 8,732
   clips but is a real edge case worth documenting.

## Why this dataset fits UrbanMind-AI

UrbanMind-AI is a multi-modality platform for urban scene understanding.
UrbanSound8K is the canonical "urban audio" dataset — it was literally
designed for the task of classifying sounds in an urban environment.
Adding it brings the project to **7 modalities** (tabular, time series,
graph, text, image, video, audio) and provides the audio input for any
future multimodal fusion or alert system.

The two emergency-relevant classes (`gun_shot`, `siren`) also connect
to Week 7.5's UCF-Crime video anomaly work: both are anomaly detectors,
one audio, one video. Their combination is the natural Week 9 fusion
target — but *training a fused model on real paired data* is a separate
project from *training each modality independently* (which is what
7.5 and 7.6 do).

## License

UrbanSound8K is released under **Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0)**. It is intended for research and
non-commercial use. See:
https://urbansounddataset.weebly.com/urbansound8k.html

## Citation

```bibtex
@inproceedings{salamon2014urbansound,
  title={A Dataset and Taxonomy for Urban Sound Research},
  author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle={Proceedings of the 22nd ACM International Conference on Multimedia},
  pages={1041--1044},
  year={2014}
}
```

## Reproduce the cache

```powershell
# one-time, ~10-15 min on a modern CPU
uv run python scripts/precompute_audio.py

# verify the cache matches expectation (2.87 GiB, 8732 rows)
Get-ChildItem "C:\Users\Janak Mallik\ml-data\urbansound8k-cache"
```

---