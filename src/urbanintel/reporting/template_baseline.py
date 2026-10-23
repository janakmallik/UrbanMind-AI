"""Deterministic template report generator — the baseline to beat.

This is NOT the deliverable. It exists so we can measure whether Qwen
(or LoRA) adds value over a simple deterministic formatter. See
docs/week8_plan.md.
"""

from __future__ import annotations

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


def generate_template_report(bundle: EvidenceBundle) -> IncidentReport:
    severity = rule_based_severity(bundle)
    sentences: list[str] = []
    sources_cited: list[str] = []

    when = bundle.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    where = f" at {bundle.location}" if bundle.location else ""
    sentences.append(
        f"Incident {bundle.incident_id} observed on {when}{where}, severity: {severity}."
    )

    for ev in bundle.evidence:
        if isinstance(ev, VideoAnomalyEvidence):
            sentences.append(
                f"Video anomaly classifier: {ev.class_name} (confidence {ev.confidence:.2f})."
            )
            sources_cited.append("video_anomaly")
        elif isinstance(ev, AudioEventEvidence):
            sentences.append(
                f"Audio event classifier: {ev.class_name} (confidence {ev.confidence:.2f})."
            )
            sources_cited.append("audio_event")
        elif isinstance(ev, ImageSceneEvidence):
            sentences.append(
                f"Image scene classifier: {ev.class_name} (confidence {ev.confidence:.2f})."
            )
            sources_cited.append("image_scene")
        elif isinstance(ev, SegmentationSummaryEvidence):
            sentences.append(
                f"Segmentation summary: {ev.persons} person(s), "
                f"{ev.vehicles} vehicle(s), road_dominant={ev.road_dominant}."
            )
            sources_cited.append("segmentation")
        elif isinstance(ev, TrafficForecastEvidence):
            if ev.speed_drop_fraction is not None:
                sentences.append(
                    f"Traffic forecast ({ev.horizon_min} min horizon): "
                    f"speed expected to change "
                    f"{ev.speed_drop_fraction * 100:+.0f}%."
                )
            else:
                sentences.append(f"Traffic forecast ({ev.horizon_min} min horizon) available.")
            sources_cited.append("traffic_forecast")
        elif isinstance(ev, ComplaintEvidence):
            sentences.append(
                f"A {ev.complaint_type} complaint was filed in "
                f"{ev.complaint_borough} (may or may not relate to this incident)."
            )
            sources_cited.append("complaint")

    # Explicitly note missing critical sources
    missing = bundle.critical_sources_missing()
    if missing:
        sentences.append(
            f"Missing critical source(s): {', '.join(sorted(missing))}. "
            f"Absence of signal is NOT evidence of normal conditions."
        )

    header = build_evidence_header(bundle)
    return IncidentReport(
        incident_id=bundle.incident_id,
        severity=severity,
        sources_cited=list(dict.fromkeys(sources_cited)),
        report_text=f"{header}\n\n" + " ".join(sentences),
    )
