"""Unit tests for report generation."""

from datetime import datetime

from urbanintel.reporting.schema import (
    AudioEventEvidence,
    ComplaintEvidence,
    EvidenceBundle,
    ImageSceneEvidence,
    VideoAnomalyEvidence,
    rule_based_severity,
)
from urbanintel.reporting.template_baseline import generate_template_report

_TS = datetime(2026, 10, 22, 15, 0, 0)


def _bundle(*evidence, location="Test"):
    return EvidenceBundle(
        incident_id="test", timestamp=_TS, location=location, evidence=list(evidence)
    )


# --- Severity ----------------------------------------------------------------


def test_severity_normal_when_all_normal():
    b = _bundle(
        VideoAnomalyEvidence(class_name="normal", confidence=0.9, timestamp=_TS),
        AudioEventEvidence(class_name="air_conditioner", confidence=0.88, timestamp=_TS),
        ImageSceneEvidence(class_name="residential", confidence=0.85, timestamp=_TS),
    )
    assert rule_based_severity(b) == "normal"


def test_severity_escalate_on_violence():
    b = _bundle(
        VideoAnomalyEvidence(class_name="violence", confidence=0.8, timestamp=_TS),
    )
    assert rule_based_severity(b) == "escalate"


def test_severity_escalate_on_gunshot():
    b = _bundle(
        AudioEventEvidence(class_name="gun_shot", confidence=0.85, timestamp=_TS),
    )
    assert rule_based_severity(b) == "escalate"


def test_severity_review_on_low_conf_violence():
    """Low-confidence violence is NOT normal — it needs human review."""
    b = _bundle(
        VideoAnomalyEvidence(class_name="violence", confidence=0.3, timestamp=_TS),
    )
    assert rule_based_severity(b) == "review"


def test_severity_insufficient_on_missing_critical():
    """Missing video and audio = insufficient evidence, not normal."""
    b = _bundle(
        ImageSceneEvidence(class_name="city_street", confidence=0.85, timestamp=_TS),
    )
    assert rule_based_severity(b) == "insufficient"


def test_severity_escalate_overrides_missing():
    """Positive critical evidence beats missing-source caution."""
    b = _bundle(
        AudioEventEvidence(class_name="gun_shot", confidence=0.9, timestamp=_TS),
    )
    assert rule_based_severity(b) == "escalate"


# --- Template report ---------------------------------------------------------


def test_template_report_cites_all_sources():
    b = _bundle(
        VideoAnomalyEvidence(class_name="violence", confidence=0.8, timestamp=_TS),
        AudioEventEvidence(class_name="siren", confidence=0.7, timestamp=_TS),
    )
    report = generate_template_report(b)
    assert "video_anomaly" in report.sources_cited
    assert "audio_event" in report.sources_cited


def test_template_report_text_includes_confidence():
    b = _bundle(
        VideoAnomalyEvidence(class_name="violence", confidence=0.78, timestamp=_TS),
    )
    report = generate_template_report(b)
    assert "0.78" in report.report_text


def test_template_report_handles_missing_sources():
    """Missing critical sources must be flagged explicitly."""
    b = _bundle(
        ImageSceneEvidence(class_name="city_street", confidence=0.85, timestamp=_TS),
    )
    report = generate_template_report(b)
    assert "Missing critical source" in report.report_text
    assert "NOT evidence of normal" in report.report_text


def test_complaint_not_used_as_incident_location():
    """Complaint borough must not appear as if it were incident location."""
    b = _bundle(
        ComplaintEvidence(complaint_type="noise", complaint_borough="Brooklyn", timestamp=_TS),
        location="Manhattan",
    )
    report = generate_template_report(b)
    # Incident location is Manhattan
    assert "Manhattan" in report.report_text
    # Complaint is Brooklyn, phrased as a separate record
    assert "complaint was filed in Brooklyn" in report.report_text
    # Must contain the disambiguation phrase
    assert "may or may not relate" in report.report_text


def test_traffic_forecast_no_mae_in_report():
    """MAE is a model metric, not incident evidence — must not appear."""
    from urbanintel.reporting.schema import TrafficForecastEvidence

    b = _bundle(
        TrafficForecastEvidence(horizon_min=15, speed_drop_fraction=-0.4, timestamp=_TS),
    )
    report = generate_template_report(b)
    assert "MAE" not in report.report_text
    assert "40%" in report.report_text or "-40" in report.report_text
