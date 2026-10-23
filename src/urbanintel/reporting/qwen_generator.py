"""Qwen2.5-1.5B-Instruct incident report generator.

Loads Qwen with 4-bit quantization, prompts with a structured 4-section
template (OBSERVED / INTERPRETATION / UNKNOWN / RECOMMENDED FOLLOW-UP),
and parses the output into an IncidentReport.

Feasibility verified: 1.15 GB at load, 2.19 GB peak at 2048 tokens,
4.25 GB headroom on RTX 4050 6 GB.
"""

from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from urbanintel.reporting.header import build_evidence_header
from urbanintel.reporting.schema import (
    AudioEventEvidence,
    ComplaintEvidence,
    EvidenceBundle,
    ImageSceneEvidence,
    IncidentReport,
    SegmentationSummaryEvidence,
    TrafficForecastEvidence,
    VideoAnomalyEvidence,
    rule_based_severity,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9


SYSTEM_PROMPT = """You are a decision-support report generator for an urban \
incident-response system. You receive STRUCTURED EVIDENCE produced by \
independent machine-learning classifiers (video anomaly, audio events, \
image scene, segmentation, traffic forecast, complaints). Your job is to \
produce a field report for a HUMAN RESPONDER (police, dispatcher, or \
emergency coordinator).

WHAT YOU ARE NOT:
- You are NOT an incident detector. All detection is done by the classifiers.
- You are NOT a dispatch system. You do NOT send officers. You flag for \
  human review.
- You are NOT a source of verified facts. You describe what the classifiers \
  reported, not what happened.
- You are NOT authorized to conclude that a crime occurred.

CRITICAL RULES:

1. Four sections, exact headers: OBSERVED, INTERPRETATION, UNKNOWN, \
   RECOMMENDED FOLLOW-UP. In that order.

2. OBSERVED contains ONLY what the classifiers reported with their \
   confidences. INTERPRETATION contains ONLY plausible readings. Never \
   put inference in OBSERVED or observation in INTERPRETATION.

3. Severity-to-action mapping (determined by rule_based_severity, which \
   you do not compute — you receive it in the evidence):

   - ESCALATE → permitted recommendation: "flag for urgent human review", \
     "urgent human review", "escalate for human review", "review \
     immediately". FORBIDDEN: "dispatch", "send officers", "send police", \
     "immediate police response", "send units".

   - REVIEW → permitted recommendation: "verify", "review", "investigate", \
     "monitor", "follow up". FORBIDDEN: "dispatch", "escalate for \
     immediate", "immediate police response".

   - INSUFFICIENT → permitted recommendation: "verify feed", "recover \
     feed", "restore feed", "do not treat as normal". FORBIDDEN: "all \
     clear", "no incident", "safe".

   - NORMAL → permitted recommendation: "log", "monitor", "summarize". \
     FORBIDDEN: "no incident occurred", "scene is safe", "conditions are \
     normal".

4. FORBIDDEN CLAIMS (never use these phrases regardless of severity):
   "potential criminal activity", "suggesting criminal activity", \
   "identified a criminal activity", "criminal activity was detected", \
   "suspicious activity", "consistent with a potential criminal act", \
   "implies criminal behavior".

   A gunshot may be from a police officer, a licensed range, a firework, \
   or a misclassified sound. A violence detection may be a misclassified \
   sports event or rescue. Describe the signal, not the crime.

   For the video_anomaly class 'criminal': SAY "the video classifier \
   placed this event in its 'criminal' category at <number>% confidence". \
   Do NOT say "criminal activity was detected".

5. NEVER infer criminality from scene composition. Pedestrians, vehicles, \
   or an urban street do NOT imply criminality. NEVER state or imply that \
   the presence of people or cars suggests crime.

6. NEVER interpret a traffic SPEED DROP as "high-speed traffic". A speed \
   drop means traffic is slowing down, not speeding up.

7. SOURCE AVAILABILITY — critical:

   For EACH of video_anomaly and audio_event, check the evidence bundle:

   - If the source is PRESENT: mention it in OBSERVED as a detection \
     (e.g., "The audio event classifier detected a siren at 88% \
     confidence").
   - If the source is MISSING: explicitly state in OBSERVED that its feed \
     is unavailable (e.g., "The video anomaly feed was unavailable for \
     this window").

   DO NOT combine these statements. If audio is present but video is \
   missing, say ONLY "video feed unavailable" — NEVER say "audio feed \
   unavailable" if audio is present. NEVER say "all feeds unavailable" \
   unless both are actually missing.

8. MISSING EVIDENCE ≠ NORMAL CONDITIONS. If a critical source (video or \
   audio) is missing, state in UNKNOWN that this prevents confirmation or \
   ruling out of an emergency. Use phrases like "missing evidence is not \
   evidence of normal conditions" or "do not treat this as clear".

9. CROSS-MODAL CAVEAT: If BOTH video_anomaly AND audio_event are present \
   AND at least one reports a non-benign class (violence, gun_shot, \
   criminal, public_hazard, siren, car_horn, drilling, jackhammer), the \
   UNKNOWN section MUST include a bullet containing BOTH "different \
   datasets" AND "no shared metadata".

10. CONTRADICTORY SIGNALS: If video reports normal but audio reports \
    gun_shot (or vice versa), describe the contradiction in INTERPRETATION \
    and recommend verification in RECOMMENDED FOLLOW-UP. Do NOT pick one \
    signal as authoritative.

11. REQUIRED DISCLAIMER: Every report MUST end with exactly this sentence:
    "This is a decision-support report, not verified evidence of a crime; \
    confirm against the source footage before any enforcement action."

12. UNKNOWN must contain 3-5 bullets. Each bullet starts with "Whether" \
    or describes a specific absence or ambiguity.

OUTPUT FORMAT (use exactly these section headers):

OBSERVED
[2-4 sentences: only what the classifiers reported, with confidences]

INTERPRETATION
[2-3 sentences: plausible readings, tensions, contradictions]

UNKNOWN
- [bullet 1]
- [bullet 2]
- [bullet 3]

RECOMMENDED FOLLOW-UP
[2-3 sentences matching the severity-to-action mapping above. End with \
the required disclaimer.]
"""


FEW_SHOT_EXAMPLES = [
    {
        "user": """EVIDENCE BUNDLE:
Incident: EX-01
Timestamp: 2026-10-22T15:20:00Z
Location: Test Location A
Evidence:
  - video_anomaly: normal (confidence 0.92)
  - audio_event: air_conditioner (confidence 0.88)
  - image_scene: residential (confidence 0.85)""",
        "assistant": """OBSERVED
The video anomaly classifier reported normal activity at 92% confidence. The audio event classifier detected an air-conditioner hum at 88% confidence. The image scene classifier classified the location as residential at 85% confidence.

INTERPRETATION
All three signals are consistent with a quiet residential scene. No emergency signal was detected among the available sources.

UNKNOWN
- Whether any emergency event occurred during this window that the classifiers missed.
- Whether the video and audio signals describe the same real event — the classifiers were trained on different datasets and have no shared metadata.
- Whether all sensors were fully operational during this window.

RECOMMENDED FOLLOW-UP
Log this observation for the day's incident summary. No follow-up action is required at this time. This is a decision-support report, not verified evidence of a crime; confirm against the source footage before any enforcement action.""",
    },
    {
        "user": """EVIDENCE BUNDLE:
Incident: EX-02
Timestamp: 2026-10-22T15:20:00Z
Location: Test Location B
Evidence:
  - video_anomaly: violence (confidence 0.78)
  - audio_event: gun_shot (confidence 0.85)
  - image_scene: city_street (confidence 0.86)""",
        "assistant": """OBSERVED
The video anomaly classifier detected a violent event at 78% confidence. The audio event classifier separately detected a gunshot-like event at 85% confidence. The image scene classifier classified the location as an urban street at 86% confidence.

INTERPRETATION
The coincidence of a violent event on video and a gunshot-like event on audio is consistent with a serious incident. Neither classifier alone establishes what occurred.

UNKNOWN
- Whether the video and audio signals describe the same real event — the classifiers were trained on different datasets and have no shared metadata proving temporal or geographic alignment.
- Whether the gunshot detection is accurate or a false positive (firework, backfire, misclassification).
- The exact number of persons involved.

RECOMMENDED FOLLOW-UP
Flag this incident for urgent human review. Verify the incident location and the number of persons involved against the source footage before any response decision. This is a decision-support report, not verified evidence of a crime; confirm against the source footage before any enforcement action.""",
    },
    {
        "user": """EVIDENCE BUNDLE:
Incident: EX-03
Timestamp: 2026-10-22T15:20:00Z
Location: Test Location C
Evidence:
  - image_scene: city_street (confidence 0.85)
  # NOTE: video_anomaly and audio_event feeds are UNAVAILABLE""",
        "assistant": """OBSERVED
The image scene classifier classified the location as an urban street at 85% confidence. The video anomaly feed was unavailable for this window. The audio event feed was unavailable for this window.

INTERPRETATION
The available evidence does not confirm or rule out any incident. Missing video and audio prevent assessment of violence or other emergency signals.

UNKNOWN
- Whether any violence, gunshot, or emergency event occurred — video and audio are missing.
- Whether the missing feeds reflect equipment failure or a data gap.
- Whether the image scene classification captures the full context.

RECOMMENDED FOLLOW-UP
Verify feed availability and recover the missing video and audio if possible. Do not treat this window as clear — missing evidence is not evidence of normal conditions. This is a decision-support report, not verified evidence of a crime; confirm against the source footage before any enforcement action.""",
    },
]


@lru_cache(maxsize=1)
def load_qwen() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen 4-bit. Cached — second call is fast."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb,
        device_map="auto",
    )
    return model, tokenizer


def _evidence_to_text(bundle: EvidenceBundle) -> str:
    """Serialize the bundle into a compact evidence listing."""
    lines = [
        f"Incident: {bundle.incident_id}",
        f"Timestamp: {bundle.timestamp.isoformat()}",
        f"Location: {bundle.location or 'Unknown'}",
        "Evidence:",
    ]
    for ev in bundle.evidence:
        if isinstance(ev, VideoAnomalyEvidence):
            lines.append(f"  - video_anomaly: {ev.class_name} (confidence {ev.confidence:.2f})")
        elif isinstance(ev, AudioEventEvidence):
            lines.append(f"  - audio_event: {ev.class_name} (confidence {ev.confidence:.2f})")
        elif isinstance(ev, ImageSceneEvidence):
            lines.append(f"  - image_scene: {ev.class_name} (confidence {ev.confidence:.2f})")
        elif isinstance(ev, SegmentationSummaryEvidence):
            lines.append(
                f"  - segmentation: {ev.persons} person(s), "
                f"{ev.vehicles} vehicle(s), road_dominant={ev.road_dominant}"
            )
        elif isinstance(ev, TrafficForecastEvidence):
            if ev.speed_drop_fraction is not None:
                lines.append(
                    f"  - traffic_forecast ({ev.horizon_min} min): "
                    f"speed change {ev.speed_drop_fraction * 100:+.0f}%"
                )
            else:
                lines.append(f"  - traffic_forecast ({ev.horizon_min} min): available")
        elif isinstance(ev, ComplaintEvidence):
            lines.append(
                f"  - complaint: {ev.complaint_type} "
                f"(borough={ev.complaint_borough}; NOT incident location)"
            )

    # Explicitly note missing critical sources
    missing = bundle.critical_sources_missing()
    if missing:
        lines.append(f"  # NOTE: {', '.join(sorted(missing))} feeds are UNAVAILABLE")
    return "\n".join(lines)


def _build_messages(bundle: EvidenceBundle) -> list[dict]:
    """Build the chat messages for Qwen."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    messages.append(
        {
            "role": "user",
            "content": f"EVIDENCE BUNDLE:\n{_evidence_to_text(bundle)}",
        }
    )
    return messages


def _parse_sections(text: str) -> dict[str, str]:
    """Parse the four sections from the model output."""
    sections = {}
    patterns = {
        "OBSERVED": r"OBSERVED\s*\n(.*?)(?=\nINTERPRETATION|$)",
        "INTERPRETATION": r"INTERPRETATION\s*\n(.*?)(?=\nUNKNOWN|$)",
        "UNKNOWN": r"UNKNOWN\s*\n(.*?)(?=\nRECOMMENDED|$)",
        "RECOMMENDED FOLLOW-UP": r"RECOMMENDED FOLLOW-UP\s*\n(.*?)$",
    }
    for name, pattern in patterns.items():
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        sections[name] = m.group(1).strip() if m else ""
    return sections


def _infer_sources_cited(text: str, bundle: EvidenceBundle) -> list[str]:
    """Detect which evidence sources the report actually mentions."""
    present = bundle.sources_present()
    cited = []
    text_lower = text.lower()
    keyword_map = {
        "video_anomaly": ["video", "anomaly"],
        "audio_event": ["audio", "sound"],
        "image_scene": ["image", "scene"],
        "segmentation": ["segment", "person", "vehicle"],
        "traffic_forecast": ["traffic", "speed"],
        "complaint": ["complaint"],
    }
    for source in present:
        keywords = keyword_map.get(source, [source])
        if any(k in text_lower for k in keywords):
            cited.append(source)
    return cited


def generate_qwen_report(bundle: EvidenceBundle) -> IncidentReport:
    """Generate an incident report using Qwen2.5-1.5B-Instruct (zero-shot)."""
    model, tokenizer = load_qwen()
    messages = _build_messages(bundle)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    sources_cited = _infer_sources_cited(generated, bundle)
    header = build_evidence_header(bundle)

    return IncidentReport(
        incident_id=bundle.incident_id,
        severity=rule_based_severity(bundle),
        sources_cited=sources_cited,
        report_text=f"{header}\n\n{generated}",
    )
