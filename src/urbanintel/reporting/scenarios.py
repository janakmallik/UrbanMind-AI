"""31 synthetic scenarios for incident report evaluation.

Each scenario is an EvidenceBundle. Co-occurrence of modalities is
synthetic — no real paired multimodal incident dataset exists in
UrbanMind-AI. See docs/dataset_cards/week8_synthetic_scenarios.md.

Coverage:
    - 3 fully normal
    - 5 single critical
    - 5 multiple coincident critical
    - 5 conflicting signals
    - 5 low-confidence
    - 4 missing data
    - 3 benign compound
    - 1 missing critical + normal-present (insufficient case)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from urbanintel.reporting.schema import (
    AudioEventEvidence,
    ComplaintEvidence,
    EvidenceBundle,
    ImageSceneEvidence,
    SegmentationSummaryEvidence,
    TrafficForecastEvidence,
    VideoAnomalyEvidence,
)

_BASE_TS = datetime(2026, 10, 22, 15, 0, 0)


def _ts(offset_min: int) -> datetime:
    return _BASE_TS + timedelta(minutes=offset_min)


def _common(bundle_id: str, offset_min: int = 0, location: str | None = None) -> dict:
    return dict(
        incident_id=bundle_id,
        timestamp=_ts(offset_min),
        location=location,
    )


def all_scenarios() -> list[EvidenceBundle]:
    bundles: list[EvidenceBundle] = []

    # --- 3 fully normal -------------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S01-normal-1", 0, "Bedford Ave & N 7th"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.92, timestamp=_ts(0)),
                AudioEventEvidence(class_name="air_conditioner", confidence=0.88, timestamp=_ts(0)),
                ImageSceneEvidence(class_name="residential", confidence=0.85, timestamp=_ts(0)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S02-normal-2", 10, "Atlantic Ave"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.94, timestamp=_ts(10)),
                AudioEventEvidence(class_name="engine_idling", confidence=0.83, timestamp=_ts(10)),
                SegmentationSummaryEvidence(
                    persons=1, vehicles=0, road_dominant=True, timestamp=_ts(10)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S03-normal-3", 20, "Prospect Park"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.90, timestamp=_ts(20)),
                AudioEventEvidence(
                    class_name="children_playing", confidence=0.79, timestamp=_ts(20)
                ),
                ImageSceneEvidence(class_name="residential", confidence=0.88, timestamp=_ts(20)),
            ],
        )
    )

    # --- 5 single critical ----------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S04-violence-only", 30, "Flatbush Ave"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.87, timestamp=_ts(30)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S05-gunshot-only", 35, "Crown Heights"),
            evidence=[
                AudioEventEvidence(class_name="gun_shot", confidence=0.91, timestamp=_ts(35)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S06-criminal-only", 40, "Downtown Brooklyn"),
            evidence=[
                VideoAnomalyEvidence(class_name="criminal", confidence=0.82, timestamp=_ts(40)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S07-hazard-only", 45, "BQE near Exit 27"),
            evidence=[
                VideoAnomalyEvidence(
                    class_name="public_hazard", confidence=0.76, timestamp=_ts(45)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S08-siren-only", 50, "Manhattan Bridge"),
            evidence=[
                AudioEventEvidence(class_name="siren", confidence=0.88, timestamp=_ts(50)),
            ],
        )
    )

    # --- 5 multiple coincident critical ---------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S09-violence-siren", 60, "Bedford-Stuyvesant"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.72, timestamp=_ts(60)),
                AudioEventEvidence(class_name="siren", confidence=0.89, timestamp=_ts(60)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S10-violence-gunshot", 65, "Brownsville"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.81, timestamp=_ts(65)),
                AudioEventEvidence(class_name="gun_shot", confidence=0.93, timestamp=_ts(65)),
                AudioEventEvidence(class_name="siren", confidence=0.77, timestamp=_ts(65)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S11-criminal-siren", 70, "Downtown Brooklyn"),
            evidence=[
                VideoAnomalyEvidence(class_name="criminal", confidence=0.68, timestamp=_ts(70)),
                AudioEventEvidence(class_name="siren", confidence=0.82, timestamp=_ts(70)),
                ComplaintEvidence(
                    complaint_type="noise",
                    complaint_borough="Brooklyn",
                    timestamp=_ts(70),
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S12-hazard-traffic", 75, "BQE near Exit 27"),
            evidence=[
                VideoAnomalyEvidence(
                    class_name="public_hazard", confidence=0.84, timestamp=_ts(75)
                ),
                TrafficForecastEvidence(
                    horizon_min=15, speed_drop_fraction=-0.42, timestamp=_ts(75)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S13-full-stack", 80, "Atlantic Terminal"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.78, timestamp=_ts(80)),
                AudioEventEvidence(class_name="gun_shot", confidence=0.85, timestamp=_ts(80)),
                AudioEventEvidence(class_name="siren", confidence=0.90, timestamp=_ts(80)),
                ImageSceneEvidence(class_name="city_street", confidence=0.86, timestamp=_ts(80)),
                SegmentationSummaryEvidence(
                    persons=4, vehicles=2, road_dominant=True, timestamp=_ts(80)
                ),
                TrafficForecastEvidence(
                    horizon_min=15, speed_drop_fraction=-0.55, timestamp=_ts(80)
                ),
                ComplaintEvidence(
                    complaint_type="noise",
                    complaint_borough="Brooklyn",
                    timestamp=_ts(80),
                ),
            ],
        )
    )

    # --- 5 conflicting --------------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S14-conflict-1", 90, "Bay Ridge"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.91, timestamp=_ts(90)),
                AudioEventEvidence(class_name="gun_shot", confidence=0.72, timestamp=_ts(90)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S15-conflict-2", 95, "Park Slope"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.75, timestamp=_ts(95)),
                AudioEventEvidence(
                    class_name="air_conditioner", confidence=0.88, timestamp=_ts(95)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S16-conflict-3", 100, "Red Hook"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.84, timestamp=_ts(100)),
                AudioEventEvidence(class_name="siren", confidence=0.86, timestamp=_ts(100)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S17-conflict-4", 105, "Gowanus"),
            evidence=[
                VideoAnomalyEvidence(class_name="criminal", confidence=0.66, timestamp=_ts(105)),
                ImageSceneEvidence(class_name="residential", confidence=0.87, timestamp=_ts(105)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S18-conflict-5", 110, "Sunset Park"),
            evidence=[
                AudioEventEvidence(class_name="gun_shot", confidence=0.78, timestamp=_ts(110)),
                ComplaintEvidence(
                    complaint_type="noise",
                    complaint_borough="Brooklyn",
                    timestamp=_ts(110),
                ),
            ],
        )
    )

    # --- 5 low-confidence -----------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S19-lowconf-1", 120, "Borough Park"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.42, timestamp=_ts(120)),
                AudioEventEvidence(class_name="siren", confidence=0.38, timestamp=_ts(120)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S20-lowconf-2", 125, "Kensington"),
            evidence=[
                VideoAnomalyEvidence(class_name="criminal", confidence=0.45, timestamp=_ts(125)),
                AudioEventEvidence(class_name="gun_shot", confidence=0.41, timestamp=_ts(125)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S21-lowconf-3", 130, "Midwood"),
            evidence=[
                VideoAnomalyEvidence(
                    class_name="public_hazard", confidence=0.35, timestamp=_ts(130)
                ),
                TrafficForecastEvidence(
                    horizon_min=15, speed_drop_fraction=-0.32, timestamp=_ts(130)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S22-lowconf-4", 135, "Sheepshead Bay"),
            evidence=[
                AudioEventEvidence(class_name="gun_shot", confidence=0.48, timestamp=_ts(135)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S23-lowconf-5", 140, "Brighton Beach"),
            evidence=[
                VideoAnomalyEvidence(class_name="normal", confidence=0.44, timestamp=_ts(140)),
                AudioEventEvidence(
                    class_name="air_conditioner", confidence=0.42, timestamp=_ts(140)
                ),
            ],
        )
    )

    # --- 4 missing data -------------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S24-missing-video", 150, "Williamsburg"),
            evidence=[
                AudioEventEvidence(class_name="gun_shot", confidence=0.89, timestamp=_ts(150)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S25-missing-audio", 155, "Greenpoint"),
            evidence=[
                VideoAnomalyEvidence(class_name="violence", confidence=0.83, timestamp=_ts(155)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S26-missing-seg", 160, "Bushwick"),
            evidence=[
                VideoAnomalyEvidence(class_name="criminal", confidence=0.71, timestamp=_ts(160)),
                AudioEventEvidence(class_name="siren", confidence=0.84, timestamp=_ts(160)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S27-missing-two", 165, "Ridgewood"),
            evidence=[
                AudioEventEvidence(class_name="car_horn", confidence=0.77, timestamp=_ts(165)),
            ],
        )
    )

    # --- 3 benign compound ----------------------------------------------------
    bundles.append(
        EvidenceBundle(
            **_common("S28-benign-1", 180, "Bed-Stuy residential"),
            evidence=[
                AudioEventEvidence(class_name="car_horn", confidence=0.61, timestamp=_ts(180)),
                ImageSceneEvidence(class_name="residential", confidence=0.91, timestamp=_ts(180)),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S29-benign-2", 185, "Prospect Heights"),
            evidence=[
                AudioEventEvidence(class_name="siren", confidence=0.66, timestamp=_ts(185)),
                ImageSceneEvidence(class_name="city_street", confidence=0.89, timestamp=_ts(185)),
                SegmentationSummaryEvidence(
                    persons=3, vehicles=1, road_dominant=True, timestamp=_ts(185)
                ),
            ],
        )
    )
    bundles.append(
        EvidenceBundle(
            **_common("S30-benign-3", 190, "Clinton Hill"),
            evidence=[
                AudioEventEvidence(class_name="drilling", confidence=0.72, timestamp=_ts(190)),
                ImageSceneEvidence(class_name="city_street", confidence=0.83, timestamp=_ts(190)),
                ComplaintEvidence(
                    complaint_type="noise",
                    complaint_borough="Brooklyn",
                    timestamp=_ts(190),
                ),
            ],
        )
    )

    # --- 1 insufficient (missing critical + normal-looking present) -----------
    bundles.append(
        EvidenceBundle(
            **_common("S31-missing-video-audio", 200, "Bushwick, Brooklyn"),
            evidence=[
                ImageSceneEvidence(class_name="city_street", confidence=0.85, timestamp=_ts(200)),
            ],
        )
    )

    return bundles
