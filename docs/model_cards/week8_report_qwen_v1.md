# Model Card: Week 8 Incident Report Generator (v1)

## What this model does

Takes UrbanMind-AI's structured modality evidence (video anomaly, audio
events, image scene, segmentation, traffic forecast, complaints) and
generates a natural-language field report for a human responder. The
report is structured in four sections:

1. **OBSERVED** — model outputs, confidences, and raw evidence
2. **INTERPRETATION** — plausible readings of the evidence, including
   tensions and contradictions
3. **UNKNOWN** — explicit list of what cannot be concluded from the
   evidence, including missing sources and unverified cross-modal claims
4. **RECOMMENDED FOLLOW-UP** — verification steps, not enforcement orders

## What this model does NOT do

- **Does NOT detect incidents.** It translates existing classifier outputs
  into prose. All detection is done by the underlying modality models
  (Weeks 5-7.6), not by this model.
- **Does NOT establish cross-modal co-occurrence.** No paired multimodal
  incident dataset exists in this project. When the report says "video
  and audio agree," it means the two independent signals are consistent —
  not that they observed the same real event. This is stated explicitly
  in every report's UNKNOWN section.
- **Is NOT grounds for enforcement.** Every report explicitly states:
  "This is a decision-support report, not verified evidence of a crime;
  confirm against source footage before any enforcement action."
- **Is NOT a real-time system.** Takes a single evidence bundle and
  produces one report.
- **Is NOT validated on real incidents.** All evaluation is on 30
  synthetic scenarios with documented limitations (see Dataset Card).

## Architecture

- Base model: Qwen2.5-1.5B-Instruct (1.5B params)
- Quantization: 4-bit (bitsandbytes nf4)
- Fine-tuning: none for v1 (zero-shot); LoRA considered in v2 if needed
- Prompting: system prompt + 3 few-shot examples + evidence JSON
- Generation: temperature=0.3, max_new_tokens=512, top_p=0.9
- GPU: RTX 4050 6GB (fits in fp16 with 4-bit quant, ~3.5 GB peak)

## Training

- **No training in v1.** Zero-shot inference only.
- Baseline comparison: deterministic template (`template_baseline.py`).
- If v2 (LoRA) ships, training details go here.

## Results

Evaluation is on 31 synthetic scenarios (see `docs/week8_plan.md`).
Metrics are constraint checks (source availability, forbidden phrases,
scenario-specific claims, policy compliance, section separation), not
full reasoning-quality judges. Each metric can fail — the Week 8 evaluator
had two tautological checks that always passed.

| Metric | Template baseline | Qwen zero-shot | Δ |
|---|---:|---:|---:|
| Source availability | 0.763 | 0.839 | +0.076 |
| Forbidden phrase | 1.000 | 0.839 | −0.161 |
| Scenario specific | 0.952 | 0.984 | +0.032 |
| Policy compliance | 0.032 | 0.871 | +0.839 |
| Section separation | 0.000 | 1.000 | +1.000 |
| **Overall** | **0.626** | **0.890 ~ 0.902** | **+0.264** |

Measured under the Week 8.1 corrected evaluator (see
`docs/week8_1_policy.md`). The Week 8 evaluator's 0.956 score is
superseded — it was inflated by two tautological metrics and a
duplicate. The 0.890 here is the honest measurement and in another attempt, we obtained 0.902.

Human spot-check (5 reports, 1-5 scale):

Status: NOT PERFORMED — deferred to future work.

(The rubric is defined in `docs/week8_1_policy.md` §6. Future work
should read the S08, S13, S17, S24, S27 reports from both template and
Qwen outputs, and score them 1-5 on reasoning quality, uncertainty
handling, and action usefulness.)

## Known limitations

1. **Synthetic scenarios.** All 30 test scenarios are hand-crafted
   combinations of evidence. No real paired incident data exists.
2. **Cross-modal co-occurrence is unverified.** Video/audio/image
   classifiers were trained on different datasets; the report generator
   cannot establish they observed the same real event.
3. **Small model.** Qwen2.5-1.5B may produce generic language on complex
   combinations. LoRA fine-tuning is evaluated if needed.
4. **No temporal localization.** The report describes a bundle, not a
   specific moment in a stream.
5. **English only.** No multilingual support.
6. **Not tested for adversarial inputs.** A malicious operator could
   craft a bundle that produces misleading output.
7. **Forbidden phrase violations (5 of 31 scenarios).** Qwen-1.5B
   generated at least one prohibited phrase in S05, S06, S10, S16, and
   S23 despite the prompt's explicit prohibition. This is a small-model
   instruction-following limitation; a larger model or LoRA fine-tune
   would address it. Users must verify report text against
   `docs/week8_1_policy.md` before acting on any recommendation.

8. **Policy compliance is 0.871, not 1.0.** In 4 of 31 scenarios, Qwen
   recommended an action that doesn't match the rule-computed severity.
   Most failures are over-escalation on review-tier scenarios (recommending
   dispatch when the policy calls for review). The severity field itself
   is rule-computed and correct; only the LLM's recommendation language
   drifts.

## What this model demonstrates

- LLM integration with structured multimodal data
- Structured output (OBSERVED/INTERPRETATION/UNKNOWN/RECOMMENDED)
- Uncertainty-handling discipline (missing data ≠ normal)
- Baseline comparison (template vs. LLM)
- Honest limitations documentation
- Decision-support framing (not enforcement)

## Citations

- Qwen Team. (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
- This system's evidence schema: `src/urbanintel/reporting/schema.py`