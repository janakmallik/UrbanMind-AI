"""Pydantic schema for multimodal incident evidence and reports.

Six evidence sources derived from existing UrbanMind-AI modalities.
`timestamp_source` distinguishes real from synthesized timestamps.

Co-occurrence is synthetic — no real paired multimodal incident dataset
exists in this project. See docs/week8_plan.md.

Severity is FOUR-LEVEL:
    normal       — all present sources signal normal at >=0.5; no critical
                   source missing
    insufficient — one or more critical sources missing, and no positive
                   evidence of a critical condition
    review       — alert-level signal at >=0.5, OR 2+ non-normal signals,
                   OR any critical signal BELOW threshold (needs eyes)
    escalate     — high-confidence critical signal (violence/gunshot >= 0.5)

Missing evidence is treated as UNKNOWN, not normal. This is a
safety-critical design decision — a missing video feed is not "no
violence".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["normal", "insufficient", "review", "escalate"]

_CRITICAL_SOURCES = {"video_anomaly", "audio_event"}
_NON_ALERT_AUDIO = {
    "air_conditioner",
    "engine_idling",
    "street_music",
    "children_playing",
    "dog_bark",
}


# --- Evidence records --------------------------------------------------------


class VideoAnomalyEvidence(BaseModel):
    source: Literal["video_anomaly"] = "video_anomaly"
    class_name: Literal["normal", "violence", "criminal", "public_hazard"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    timestamp_source: Literal["real", "synthetic"] = "synthetic"
    location: str | None = None


class AudioEventEvidence(BaseModel):
    source: Literal["audio_event"] = "audio_event"
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    timestamp_source: Literal["real", "synthetic"] = "synthetic"
    location: str | None = None


class ImageSceneEvidence(BaseModel):
    source: Literal["image_scene"] = "image_scene"
    class_name: Literal["city_street", "highway", "residential"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    timestamp_source: Literal["real", "synthetic"] = "synthetic"


class SegmentationSummaryEvidence(BaseModel):
    source: Literal["segmentation"] = "segmentation"
    persons: int = Field(ge=0)
    vehicles: int = Field(ge=0)
    road_dominant: bool
    timestamp: datetime
    timestamp_source: Literal["real", "synthetic"] = "synthetic"


class TrafficForecastEvidence(BaseModel):
    """Traffic signal — actual current signal, not model performance.

    `speed_mae_mph` is intentionally ABSENT: MAE is a model-evaluation
    metric, not evidence of a current incident. Only the predicted speed
    change is evidence.
    """

    source: Literal["traffic_forecast"] = "traffic_forecast"
    horizon_min: int = Field(gt=0)
    speed_drop_fraction: float | None = Field(default=None, ge=-1.0, le=1.0)
    timestamp: datetime
    timestamp_source: Literal["real", "synthetic"] = "synthetic"


class ComplaintEvidence(BaseModel):
    """A 311 complaint — a separate record, not incident location.

    `complaint_borough` is WHERE THE COMPLAINT WAS FILED, not where the
    incident occurred. This distinction is critical: a noise complaint in
    Brooklyn does NOT mean the incident is in Brooklyn.
    """

    source: Literal["complaint"] = "complaint"
    complaint_type: str
    complaint_borough: str
    timestamp: datetime
    timestamp_source: Literal["real"] = "real"


Evidence = (
    VideoAnomalyEvidence
    | AudioEventEvidence
    | ImageSceneEvidence
    | SegmentationSummaryEvidence
    | TrafficForecastEvidence
    | ComplaintEvidence
)


# --- Bundle + report ---------------------------------------------------------


class EvidenceBundle(BaseModel):
    """A set of evidence records attributed to (nominally) one incident.

    Co-occurrence is synthetic — no real paired multimodal incident
    dataset exists. See docs/week8_plan.md.
    """

    incident_id: str
    timestamp: datetime
    location: str | None = None
    evidence: list[Evidence]

    def sources_present(self) -> set[str]:
        return {e.source for e in self.evidence}

    def critical_sources_missing(self) -> set[str]:
        return _CRITICAL_SOURCES - self.sources_present()


class IncidentReport(BaseModel):
    incident_id: str
    severity: Severity
    sources_cited: list[str] = Field(default_factory=list)
    report_text: str


# --- Severity rules ----------------------------------------------------------


def rule_based_severity(bundle: EvidenceBundle) -> Severity:
    """Four-level severity.

    - escalate:     high-confidence critical signal (violence/gunshot >=0.5)
    - review:       alert-level signal >=0.5, OR 2+ non-normal signals, OR
                    critical signal BELOW threshold (needs human eyes)
    - insufficient: critical source missing AND no positive critical evidence
    - normal:       all present sources normal >=0.5 AND no critical missing

    Missing evidence is NEVER treated as normal.
    """
    escalate = False
    review = False
    non_normal = 0
    low_conf_critical = False

    for ev in bundle.evidence:
        if isinstance(ev, VideoAnomalyEvidence):
            if ev.class_name == "violence" and ev.confidence >= 0.5:
                escalate = True
            elif ev.class_name == "violence":
                low_conf_critical = True
            elif ev.class_name != "normal" and ev.confidence >= 0.5:
                review = True
            if ev.class_name != "normal":
                non_normal += 1

        elif isinstance(ev, AudioEventEvidence):
            if ev.class_name == "gun_shot" and ev.confidence >= 0.5:
                escalate = True
            elif ev.class_name == "gun_shot":
                low_conf_critical = True
            elif ev.class_name not in _NON_ALERT_AUDIO and ev.confidence >= 0.5:
                review = True
            if ev.class_name not in _NON_ALERT_AUDIO:
                non_normal += 1

        elif isinstance(ev, TrafficForecastEvidence):
            if ev.speed_drop_fraction is not None and ev.speed_drop_fraction <= -0.3:
                review = True
                non_normal += 1

    # Positive evidence always wins over missing data
    if escalate:
        return "escalate"
    if review or low_conf_critical or non_normal >= 2:
        return "review"
    if bundle.critical_sources_missing():
        return "insufficient"
    return "normal"
