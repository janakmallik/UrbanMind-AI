# Dataset Card: Week 8 Synthetic Incident Scenarios

## What this dataset is

31 hand-crafted evidence bundles, each representing a plausible incident
input to the Week 8 report generator. Each bundle combines records from
UrbanMind-AI's trained modalities (video anomaly, audio events, image
scene, segmentation, traffic, complaints).

**Each scenario is entirely synthetic.** No real incident data exists.

## Coverage (31 scenarios)

| Category | Count | Scenarios |
|---|---:|---|
| Fully normal | 3 | S01, S02, S03 |
| Single critical signal | 5 | S04–S08 |
| Multiple coincident critical | 5 | S09–S13 |
| Conflicting signals | 5 | S14–S18 |
| Low confidence | 5 | S19–S23 |
| Missing data | 4 | S24–S27 |
| Benign compound | 3 | S28–S30 |
| Insufficient (missing critical + normal-looking present) | 1 | S31 |

## Why synthetic

UrbanMind-AI has no paired multimodal incident dataset. The video model
was trained on HMDB-51 and UCF-Crime (movies, web videos). The audio model
was trained on UrbanSound8K (field recordings). No dataset exists in this
project where a video clip, audio clip, image, and complaint refer to the
same real-world event.

Therefore: co-occurrence in all 31 scenarios is **synthetic**. The
scenarios are designed to test the **reporting pipeline** — whether the
generator faithfully represents the evidence it receives — not whether the
underlying models detect real incidents (they're evaluated separately in
Weeks 5–7.6).

## Limitations

1. **No real paired data** — co-occurrence is invented
2. **No ground-truth incident labels** — we measure fidelity to evidence,
   not correctness about the world
3. **31 is small** — enough for a baseline comparison, not enough for
   statistical confidence
4. **Single-author bias** — the scenarios reflect one person's
   imagination of plausible incidents
5. **English-only** — no multilingual scenarios

## Why 31 is enough

The purpose is not to prove "the system is accurate on urban incidents."
It is to answer a narrow question: **"Does an LLM generate a more useful
field report from this evidence than a deterministic template?"**

For that question, 31 carefully-designed scenarios covering the failure
modes that matter (missing data, low confidence, conflicting signals) are
sufficient to reveal whether the LLM adds value. A larger set would
improve precision of the score but not the qualitative finding.

## What's in each scenario

Each scenario is an `EvidenceBundle` (see `src/urbanintel/reporting/schema.py`):

```python
{
  "incident_id": "S13-full-stack",
  "timestamp": "2026-10-22T15:20:00Z",
  "location": "Atlantic Terminal",
  "evidence": [
    {"source": "video_anomaly", "class_name": "violence", "confidence": 0.78, ...},
    {"source": "audio_event", "class_name": "gun_shot", "confidence": 0.85, ...},
    ...
  ]
}
```

## Licence

MIT — same as the parent project.

## Files

- `src/urbanintel/reporting/scenarios.py` — the 31 bundles
- `docs/week8_plan.md` — the plan using these scenarios
- `src/urbanintel/reporting/schema.py` — the schema they conform to