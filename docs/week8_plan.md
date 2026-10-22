# Week 8 — Incident Report Generation

## Task

Given structured evidence from UrbanMind-AI's trained modalities, generate a
natural-language field report that:
- Narrates what was observed in plain language
- Distinguishes observation from interpretation from unknown
- Acknowledges missing or conflicting evidence
- Recommends verification steps for a human responder

**Not:** classifying complaints. **Not:** RAG. **Not:** fine-tuning before
inspection.

## The intended output

For evidence bundle S13 (7 signals including violence + gunshot):

```
INCIDENT REPORT — S13-full-stack
Atlantic Terminal, Brooklyn — 2026-10-22 15:20 UTC
Severity: ESCALATE

OBSERVED
At approximately 15:20 UTC, near Atlantic Terminal in Brooklyn, the video
classifier detected a violent event (78% confidence). The audio track
independently captured a gunshot-like event (85% confidence) and emergency
sirens (90% confidence) in the same window. The scene was classified as an
urban street (86% confidence) with four pedestrians and two vehicles
visible. Traffic data projects a 55% speed reduction within 15 minutes. A
noise complaint was filed in Brooklyn.

INTERPRETATION
The coincidence of violence and gunshot is consistent with an active
violent incident. The presence of pedestrians in an urban street suggests
bystanders may be at risk. The traffic slowdown is consistent with roadway
disruption.

UNKNOWN
- Whether the two detected signals (video violence, audio gunshot) describe
  the same real event — the video and audio models were trained on different
  datasets and have no shared metadata proving temporal or geographic
  alignment.
- Whether the sirens are responding to this incident or another one.
- Whether the noise complaint is related or coincidental.

RECOMMENDED FOLLOW-UP
Escalate for immediate police and medical response. Verify location and
number of persons against source footage before dispatch. This is a
decision-support report, not verified evidence of a crime; confirm against
the source footage before any enforcement action.
```

The template baseline cannot produce this. The template enumerates; the
LLM interprets. That gap is the deliverable.

## Severity (four levels)

| Level | Rule |
|---|---|
| `escalate` | Critical signal at ≥0.5 (violence or gunshot) |
| `review` | Alert signal at ≥0.5, OR 2+ non-normal signals, OR critical signal BELOW 0.5 (needs eyes) |
| `insufficient` | Critical source missing AND no positive critical evidence |
| `normal` | All present sources normal ≥0.5 AND no critical missing |

**Missing evidence is NEVER treated as normal.** A missing video feed is not
"no violence" — it's "unknown."

## Evaluation

Automated constraint checks (not full reasoning evaluation):

| Metric | Weight | What it checks |
|---|---|---|
| evidence_fidelity | 0.30 | cited sources match present sources |
| signal_coverage | 0.15 | proportion of present sources cited |
| no_hallucination | 0.15 | no cited source absent from evidence |
| uncertainty_ack | 0.15 | acknowledges uncertainty when needed |
| recommendation_quality | 0.15 | action matches severity |
| epistemic_clarity | 0.10 | distinguishes observed/inferred/unknown |

**Human spot-check** (5 reports, 1–5 scale) accompanies the automated
scores: reasoning quality, uncertainty handling, action usefulness.

**No predicted scores.** Actual results are measured after running the
evaluation. See `docs/experiments.md` for measured results.

## Sessions

| Session | What | GPU | Tag |
|---|---|---|---|
| 1a | Schema + template + scenarios + evaluation | No | `v0.9.7-week8-schema` |
| 1b | Qwen2.5-1.5B zero-shot reports + comparison | Yes | `v0.9.8-week8-qwen` |
| 1c | LoRA fine-tune if 1b has documented weaknesses | Yes | `v0.9.9-week8-lora` |

## Limitations

- 31 scenarios are **synthetic** — hand-crafted combinations of evidence
- Cross-modal co-occurrence is **unverified** — no paired multimodal
  incident dataset exists in this project
- Video and audio models were trained on different datasets (HMDB-51/UCF-Crime
  for video, UrbanSound8K for audio)
- Not a real-time system
- Not evaluated on real incidents
- All reports state "decision-support, not verified evidence"

## Success criteria

- Template baseline scores measured and reported honestly
- Qwen zero-shot scores measured and compared
- Human spot-check performed on 5 reports
- All limitations documented in model card
- `pytest` passes, `ruff` clean, CI green