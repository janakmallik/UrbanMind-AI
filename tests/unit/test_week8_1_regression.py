"""Regression tests for Week 8.1 corrections.

Each test maps to a specific failure from the Week 8 prototype. If any
of these fail, the corresponding bug has regressed.

Tests run against the template baseline and the schema — NOT against
Qwen (which requires GPU + 30s per scenario). Instead, we test the
*prompts*, *schema*, and *evaluator* directly so failures surface
without needing model inference.
"""

from __future__ import annotations

from datetime import datetime

from urbanintel.reporting.qwen_generator import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from urbanintel.reporting.schema import (
    AudioEventEvidence,
    EvidenceBundle,
    ImageSceneEvidence,
    VideoAnomalyEvidence,
    rule_based_severity,
)

_TS = datetime(2026, 10, 22, 15, 0, 0)


# --- Test 1: The policy document exists and contains required sections ------


def test_policy_document_exists():
    """Policy doc must exist — it's the reference point for both prompt and evaluator."""
    from pathlib import Path

    policy = Path(__file__).resolve().parents[2] / "docs" / "week8_1_policy.md"
    assert policy.exists(), "docs/week8_1_policy.md is missing"

    text = policy.read_text(encoding="utf-8")
    # Required sections
    assert "Severity tiers" in text or "severity" in text.lower()
    assert "forbidden" in text.lower()
    assert "decision-support" in text.lower()


# --- Test 2: SYSTEM_PROMPT enforces "escalate ≠ dispatch" -------------------


def test_prompt_forbids_dispatch_on_escalate():
    """The prompt must say escalate = human review, not dispatch."""
    assert "urgent human review" in SYSTEM_PROMPT.lower()
    # Rule 3 explicitly forbids "dispatch" for escalate
    assert "FORBIDDEN" in SYSTEM_PROMPT
    # The policy is stated
    assert (
        "you do not send officers" in SYSTEM_PROMPT.lower()
        or "not a dispatch system" in SYSTEM_PROMPT.lower()
    )


# --- Test 3: SYSTEM_PROMPT requires per-source missing handling -------------


def test_prompt_requires_per_source_availability():
    """Rule 7 must require per-source statement, not combined 'all unavailable'."""
    lower = SYSTEM_PROMPT.lower()
    # Must mention per-source handling
    assert "each of video_anomaly and audio_event" in lower
    # Must forbid the specific Week 8 bug
    assert (
        'never say "audio feed unavailable" if audio is present' in lower
        or "do not combine" in lower
    )


# --- Test 4: Few-shot examples follow the policy (no "dispatch") ------------


def test_few_shot_examples_no_dispatch_on_escalate():
    """The few-shot examples must demonstrate the escalate→review behavior."""
    for ex in FEW_SHOT_EXAMPLES:
        assistant = ex["assistant"].lower()
        # Escalate-tier example must NOT say "dispatch"
        if "escalate" in assistant or "urgent human review" in assistant:
            assert "dispatch" not in assistant, (
                f"Few-shot example says 'dispatch' despite Week 8.1 policy: "
                f"{ex['assistant'][:100]}..."
            )


# --- Test 5: Few-shot examples follow per-source missing handling -----------


def test_few_shot_missing_source_is_per_source():
    """The missing-source example must state each source separately."""
    missing_example = None
    for ex in FEW_SHOT_EXAMPLES:
        if "UNAVAILABLE" in ex["user"] and "video_anomaly" in ex["user"]:
            missing_example = ex
            break
    assert missing_example is not None, "Missing-source example not found"

    assistant = missing_example["assistant"].lower()
    # Must say each feed separately
    assert (
        "video anomaly feed was unavailable" in assistant
        or "video anomaly classifier" in assistant
        or "video feed was unavailable" in assistant
    ), "Missing-source example must state video status"
    assert (
        "audio event feed was unavailable" in assistant or "audio feed was unavailable" in assistant
    ), "Missing-source example must state audio status"
    # Must NOT combine
    assert "video and audio feeds are unavailable" not in assistant, (
        "Missing-source example must not combine both into one statement"
    )


# --- Test 6: Severity rules — low-confidence violence = review --------------


def test_severity_low_conf_violence_is_review():
    """Low-confidence violence (0.3) must be 'review', not 'normal'."""
    bundle = EvidenceBundle(
        incident_id="test",
        timestamp=_TS,
        evidence=[
            VideoAnomalyEvidence(class_name="violence", confidence=0.3, timestamp=_TS),
        ],
    )
    assert rule_based_severity(bundle) == "review"


# --- Test 7: Severity rules — missing critical sources = insufficient --------


def test_severity_missing_critical_is_insufficient():
    """Missing video AND audio with only benign image = insufficient, not normal."""
    bundle = EvidenceBundle(
        incident_id="test",
        timestamp=_TS,
        evidence=[
            ImageSceneEvidence(class_name="city_street", confidence=0.85, timestamp=_TS),
        ],
    )
    assert rule_based_severity(bundle) == "insufficient"


# --- Test 8: Severity rules — positive critical evidence beats missing data --


def test_severity_gunshot_overrides_missing_video():
    """Audio gunshot + missing video = escalate (positive beats missing)."""
    bundle = EvidenceBundle(
        incident_id="test",
        timestamp=_TS,
        evidence=[
            AudioEventEvidence(class_name="gun_shot", confidence=0.89, timestamp=_TS),
        ],
    )
    assert rule_based_severity(bundle) == "escalate"


# --- Test 9: Evaluator imports and metrics are unique (no dupes) ------------


def test_evaluator_has_five_unique_metrics():
    """The corrected evaluator must have exactly 5 unique metrics."""

    from pathlib import Path

    # Import evaluator as a module by path
    evaluator_path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_reports.py"
    spec = __import__(
        "importlib.util", fromlist=["spec_from_file_location"]
    ).spec_from_file_location("evaluate_reports", evaluator_path)
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics = module.METRICS
    assert len(metrics) == 5, f"Expected 5 metrics, got {len(metrics)}"

    names = [m[0] for m in metrics]
    assert len(names) == len(set(names)), f"Duplicate metric names: {names}"

    # Ensure the broken Week 8 metrics are gone
    for old_name in [
        "evidence_fidelity",
        "signal_coverage",
        "no_hallucination",
        "uncertainty_ack",
        "epistemic_clarity",
        "severity_consistency",
    ]:
        assert old_name not in names, f"Broken metric '{old_name}' still present"


# --- Test 10: Evaluator source_availability catches the S08 bug -------------


def test_source_availability_catches_missing_audio_bug():
    """If a report falsely claims audio is unavailable when audio is present,
    source_availability must penalize it."""

    from pathlib import Path

    from urbanintel.reporting.schema import IncidentReport

    evaluator_path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_reports.py"
    spec = __import__(
        "importlib.util", fromlist=["spec_from_file_location"]
    ).spec_from_file_location("evaluate_reports", evaluator_path)
    module = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    spec.loader.exec_module(module)

    # S08-style bundle: audio present (siren), video missing
    bundle = EvidenceBundle(
        incident_id="test",
        timestamp=_TS,
        evidence=[
            AudioEventEvidence(class_name="siren", confidence=0.88, timestamp=_TS),
        ],
    )

    # Bug report: says BOTH audio and video unavailable
    buggy_report = IncidentReport(
        incident_id="test",
        severity="review",
        sources_cited=["audio_event"],
        report_text="The audio event classifier detected a siren. Video and audio feeds are unavailable for this window.",
    )

    score = module._source_availability_correctness(bundle, buggy_report)
    assert score < 1.0, (
        f"source_availability should penalize false 'audio unavailable' claim, but scored {score}"
    )
