"""Deterministic evidence header shared by template and Qwen reports."""

from __future__ import annotations

from urbanintel.reporting.schema import (
    EvidenceBundle,
    rule_based_severity,
)


def build_evidence_header(bundle: EvidenceBundle) -> str:
    """Build a deterministic evidence header prepended to any report.

    Lists every source's status (present/missing/not-present), class,
    and confidence. This gives the reader raw facts before reading any
    generated interpretation.
    """
    missing = bundle.critical_sources_missing()
    all_sources = [
        "video_anomaly",
        "audio_event",
        "image_scene",
        "segmentation",
        "traffic_forecast",
        "complaint",
    ]
    present = bundle.sources_present()

    lines = [
        "=" * 70,
        "INCIDENT EVIDENCE SUMMARY",
        "=" * 70,
        f"Incident:      {bundle.incident_id}",
        f"Timestamp:     {bundle.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Location:      {bundle.location or 'Unknown'}",
        f"Severity:      {rule_based_severity(bundle)}",
        "",
        f"EVIDENCE SOURCES ({len(present)} present, {len(missing)} missing):",
    ]

    for source in all_sources:
        if source in missing:
            lines.append(f"  [MISSING] {source:<20s} (critical source unavailable)")
        elif source in present:
            for ev in bundle.evidence:
                if ev.source != source:
                    continue
                if hasattr(ev, "class_name") and hasattr(ev, "confidence"):
                    lines.append(
                        f"  [✓]       {source:<20s} {ev.class_name:<15s} conf={ev.confidence:.2f}"
                    )
                elif source == "segmentation":
                    lines.append(
                        f"  [✓]       {source:<20s} "
                        f"persons={ev.persons} vehicles={ev.vehicles} "
                        f"road_dominant={ev.road_dominant}"
                    )
                elif source == "traffic_forecast":
                    drop = (
                        f"{ev.speed_drop_fraction:+.0%}"
                        if ev.speed_drop_fraction is not None
                        else "n/a"
                    )
                    lines.append(
                        f"  [✓]       {source:<20s} horizon={ev.horizon_min}min speed_change={drop}"
                    )
                elif source == "complaint":
                    lines.append(
                        f"  [✓]       {source:<20s} {ev.complaint_type} in {ev.complaint_borough}"
                    )
                break
        else:
            lines.append(f"  [ ]       {source:<20s} (not present)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("FIELD REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)
