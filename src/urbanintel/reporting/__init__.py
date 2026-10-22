"""Incident report generation from multimodal evidence."""

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
from urbanintel.reporting.template_baseline import generate_template_report

__all__ = [
    "AudioEventEvidence",
    "ComplaintEvidence",
    "EvidenceBundle",
    "ImageSceneEvidence",
    "IncidentReport",
    "SegmentationSummaryEvidence",
    "TrafficForecastEvidence",
    "VideoAnomalyEvidence",
    "generate_template_report",
    "rule_based_severity",
]
