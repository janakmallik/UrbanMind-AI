"""Evaluate report generators against ground-truth evidence.

Week 8.1 — corrected metrics. See docs/week8_1_audit.md for the reasoning.

Metrics:
    source_availability_correctness — per-source presence/missing stated correctly
    forbidden_phrase_check          — no prohibited phrases in the report
    scenario_specific_checks        — scenario-level required/forbidden claims
    policy_compliance               — recommendation matches severity policy
    section_separation              — OBSERVED vs INTERPRETATION separation

Removed (broken in Week 8):
    evidence_fidelity / signal_coverage — duplicate of each other
    no_hallucination                    — tautological by construction
    uncertainty_ack                     — keyword-counting, not real
    epistemic_clarity                   — section-name counting only
    severity_consistency                — same rule generates and scores

No predicted scores. Only measured results.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow

from urbanintel.reporting.scenarios import all_scenarios
from urbanintel.reporting.schema import (
    # AudioEventEvidence,
    EvidenceBundle,
    IncidentReport,
    rule_based_severity,
)

# --- Setup helpers ----------------------------------------------------------


def _get_generator(approach: str):
    if approach == "template":
        from urbanintel.reporting.template_baseline import generate_template_report

        return generate_template_report
    if approach == "qwen":
        from urbanintel.reporting.qwen_generator import generate_qwen_report

        return generate_qwen_report
    raise ValueError(f"Unknown approach: {approach}")


def _evidence_class(bundle: EvidenceBundle, source: str) -> str | None:
    """Return the class_name for a given source in the bundle (or None)."""
    for ev in bundle.evidence:
        if ev.source != source:
            continue
        if hasattr(ev, "class_name"):
            return ev.class_name
        return "present"  # segmentation/traffic/complaint — non-class sources
    return None


def _strip_header(report_text: str) -> str:
    """Remove the evidence summary header (added by build_evidence_header).

    We only want to evaluate the narrative part of the report, not the
    deterministic header. The header is produced by code; the evaluator
    should score the LLM's prose, not the header.
    """
    marker = "FIELD REPORT\n" + "=" * 70
    if marker in report_text:
        return report_text.split(marker, 1)[1].strip()
    return report_text


# --- Metric 1: source availability correctness ------------------------------


def _source_availability_correctness(bundle: EvidenceBundle, report: IncidentReport) -> float:
    """For each of the 6 possible sources, does the report state its
    availability correctly?

    - Present source: mentioned AND not falsely claimed unavailable.
    - Missing critical source: mentioned AND claimed unavailable.
    - Absent non-critical source: no requirement.

    Returns fraction correct (0..1) among checked sources.
    """
    narrative = _strip_header(report.report_text).lower()
    checks: list[bool] = []

    source_keywords = {
        "video_anomaly": ["video"],
        "audio_event": ["audio", "sound"],
        "image_scene": ["image", "scene"],
        "segmentation": ["segment", "person", "vehicle"],
        "traffic_forecast": ["traffic", "speed"],
        "complaint": ["complaint"],
    }

    def _claims_unavailable(keywords: list[str]) -> bool:
        """Check if narrative claims this source is unavailable.

        Handles variants like "audio feed unavailable", "audio feeds are
        unavailable", "no audio feed", "audio is missing", etc.
        """
        for kw in keywords:
            patterns = [
                # "<kw> [feed[s]]? [is|are|was|were]? (unavailable|missing|not available)"
                rf"\b{kw}\b(\s+feeds?)?\s+(is|are|was|were|isn[''`]?t|aren[''`]?t)?\s*(unavailable|missing|not\s+available)",
                # "no <kw> [feed|data]"
                rf"\bno\s+{kw}\b(\s+feeds?|\s+data)?",
                # "<kw> [feed|data] [is|are|was|were]? (missing|unavailable)"
                rf"\b{kw}\b\s+(feeds?|data)\s+(is|are|was|were)?\s*(missing|unavailable|not\s+available)",
            ]
            for p in patterns:
                if re.search(p, narrative, re.IGNORECASE):
                    return True
        return False

    # Present sources: mentioned AND not falsely claimed unavailable
    for source in bundle.sources_present():
        keywords = source_keywords.get(source, [source])
        mentioned = any(k in narrative for k in keywords)
        falsely_unavailable = _claims_unavailable(keywords)
        checks.append(mentioned and not falsely_unavailable)

    # Missing critical sources: mentioned AND claimed unavailable
    for source in bundle.critical_sources_missing():
        keywords = source_keywords.get(source, [source])
        has_keyword = any(k in narrative for k in keywords)
        claims_unavail = _claims_unavailable(keywords)
        checks.append(has_keyword and claims_unavail)

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


# --- Metric 2: forbidden phrase check ----------------------------------------


FORBIDDEN_PHRASES_ALWAYS = [
    "potential criminal activity",
    "suggesting criminal activity",
    "identified a criminal activity",
    "criminal activity was detected",
    "suspicious activity",
    "consistent with a potential criminal act",
]

# Phrases forbidden ONLY on review-tier scenarios (severity != escalate)
FORBIDDEN_ON_REVIEW = [
    "dispatch officers",
    "dispatch police",
    "immediate police response",
    "escalate for immediate",
    "respond immediately",
    "send officers immediately",
    "send police immediately",
]


def _forbidden_phrase_check(bundle: EvidenceBundle, report: IncidentReport) -> float:
    """1.0 if no forbidden phrase; 0.0 if any is present.

    Context-aware: 'dispatch' is only forbidden on review-tier scenarios.
    On escalate-tier, the word may appear but only as part of "urgent
    human review" / "flag for review" phrasing — the policy change
    (Phase 3) will refine this. For Phase 1, escalate-tier reports may
    use dispatch language; only always-forbidden phrases apply.
    """
    narrative = _strip_header(report.report_text).lower()

    # Always-forbidden phrases
    for phrase in FORBIDDEN_PHRASES_ALWAYS:
        if phrase in narrative:
            return 0.0

    # Review-tier forbidden phrases
    severity = rule_based_severity(bundle)
    if severity != "escalate":
        for phrase in FORBIDDEN_ON_REVIEW:
            if phrase in narrative:
                return 0.0

    return 1.0


# --- Metric 3: scenario-specific checks -------------------------------------


def _scenario_specific_checks(bundle: EvidenceBundle, report: IncidentReport) -> float:
    """Per-scenario required/forbidden claims. Returns fraction passed (0..1).

    Each scenario has both REQUIRED and FORBIDDEN phrases. A scenario only
    passes if forbidden phrases are absent AND required phrases are present.
    """
    narrative = _strip_header(report.report_text).lower()
    checks: list[bool] = []

    incident_id = bundle.incident_id

    # S08 / S27: video missing, audio present. Must not say audio unavailable.
    # Must recommend review/verify (not dispatch).
    if incident_id in {"S08-siren-only", "S27-missing-two"}:
        checks.append(
            "audio feed unavailable" not in narrative
            and "audio is unavailable" not in narrative
            and "audio feed is missing" not in narrative
        )
        checks.append(any(w in narrative for w in ["verify", "review", "monitor", "investigate"]))
        checks.append(
            not any(
                w in narrative
                for w in ["dispatch officers", "dispatch police", "immediate police response"]
            )
        )

    # S31: missing video AND audio. Must NOT say "normal".
    # Must state "not evidence of normal conditions".
    if incident_id == "S31-missing-video-audio":
        checks.append(
            not any(
                c in narrative
                for c in [
                    "scene is normal",
                    "conditions are normal",
                    "no incident",
                    "no emergency",
                    "nothing to report",
                ]
            )
        )
        checks.append(
            any(
                c in narrative
                for c in [
                    "not evidence of normal",
                    "do not treat",
                    "missing evidence",
                    "cannot confirm",
                    "do not assume",
                ]
            )
        )

    # S17: criminal class at 66%, review severity. Must not say dispatch.
    if incident_id == "S17-conflict-4":
        checks.append(
            not any(
                w in narrative
                for w in [
                    "dispatch officers",
                    "dispatch police",
                    "immediate police response",
                    "escalate for immediate",
                ]
            )
        )
        checks.append(
            any(
                w in narrative
                for w in ["categor", "label", "classifier placed", "not a finding", "not proof"]
            )
        )

    # S13: pedestrian/vehicle count must not be used to infer criminality.
    if incident_id == "S13-full-stack":
        checks.append(
            not any(
                f in narrative
                for f in [
                    "implies potential criminal",
                    "suggesting criminal",
                    "indicates criminal",
                    "pedestrians and vehicles imply",
                    "criminal activity",
                ]
            )
        )
        checks.append(any(c in narrative for c in ["different datasets", "no shared metadata"]))

    # S01: normal scenario. Report must NOT over-conclude.
    if incident_id == "S01-normal-1":
        checks.append(
            not any(
                c in narrative
                for c in [
                    "no incident occurred",
                    "definitely no incident",
                    "confirms no incident",
                    "proves no incident",
                ]
            )
        )
        checks.append(
            any(
                c in narrative
                for c in [
                    "does not prove",
                    "does not establish",
                    "cannot rule out",
                    "not evidence of normal",
                    "not proof of safety",
                    "absence of signal",
                    "does not confirm",
                ]
            )
        )

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


def _section_separation(bundle: EvidenceBundle, report: IncidentReport) -> float:
    """Check that OBSERVED contains only observation and INTERPRETATION only inference.

    If any of the three sections (OBSERVED, INTERPRETATION, UNKNOWN) is
    missing, return 0.0 immediately — the report has no structure to check.
    """
    narrative = _strip_header(report.report_text)

    def extract(name: str, next_names: list[str]) -> str:
        pattern = rf"{name}\s*\n(.*?)(?=\n(?:{'|'.join(next_names)})|$)"
        m = re.search(pattern, narrative, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    observed = extract("OBSERVED", ["INTERPRETATION", "UNKNOWN", "RECOMMENDED"])
    interpretation = extract("INTERPRETATION", ["UNKNOWN", "RECOMMENDED"])
    unknown = extract("UNKNOWN", ["RECOMMENDED"])

    # If ANY section is missing, no structure to check — fail.
    if not observed or not interpretation or not unknown:
        return 0.0

    checks = [
        # OBSERVED must reference a classifier or confidence
        bool(re.search(r"classifier|confidence \d|\d+%", observed, re.IGNORECASE)),
        # INTERPRETATION must not contain bare confidence percentages
        not bool(re.search(r"confidence \d+\.\d+", interpretation, re.IGNORECASE)),
        # UNKNOWN must contain at least one bullet or "whether"
        "-" in unknown or "*" in unknown or "whether" in unknown.lower(),
    ]
    return sum(checks) / len(checks)


# --- Metric 4: policy compliance --------------------------------------------


def _policy_compliance(bundle: EvidenceBundle, report: IncidentReport) -> float:
    """Does the recommendation match the Week 8.1 policy?

    Week 8.1 policy:
        escalate     → "urgent human review" / "flag for review" / "review immediately"
                       (NOT "dispatch" — that's a human decision, not LLM)
        review       → "verify" / "review" / "investigate"
        insufficient → "verify feed" / "restore feed" / "recover"
        normal       → "log" / "monitor" / "summarize"
    """
    narrative = _strip_header(report.report_text).lower()

    # Remove the severity line (e.g., "severity: review.") so the word
    # "review" in the metadata doesn't falsely count as a recommendation.
    narrative = re.sub(r"severity:\s*\w+\.?", "", narrative)

    severity = rule_based_severity(bundle)

    if severity == "escalate":
        keywords = [
            "urgent human review",
            "flag for review",
            "review immediately",
            "urgent review",
            "escalate for human review",
            "human review",
        ]
        return 1.0 if any(k in narrative for k in keywords) else 0.0

    if severity == "review":
        keywords = ["verify", "review", "investigate", "follow up", "follow-up"]
        has_any = any(k in narrative for k in keywords)
        over = ["dispatch officers", "dispatch police", "immediate police response"]
        if any(o in narrative for o in over):
            return 0.0
        return 1.0 if has_any else 0.0

    if severity == "insufficient":
        keywords = [
            "verify feed",
            "recover feed",
            "restore feed",
            "feed unavailable",
            "missing evidence",
            "cannot confirm",
        ]
        return 1.0 if any(k in narrative for k in keywords) else 0.0

    keywords = ["log", "monitor", "summary", "no action"]
    return 1.0 if any(k in narrative for k in keywords) else 0.0


# --- Metric registry --------------------------------------------------------


METRICS = [
    ("source_availability", _source_availability_correctness, 0.30),
    ("forbidden_phrase", _forbidden_phrase_check, 0.20),
    ("scenario_specific", _scenario_specific_checks, 0.20),
    ("policy_compliance", _policy_compliance, 0.20),
    ("section_separation", _section_separation, 0.10),
]


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["template", "qwen"], default="template")
    args = parser.parse_args()

    generate = _get_generator(args.approach)
    scenarios = all_scenarios()

    mlflow.set_experiment("urbanintel-week8-1")

    with mlflow.start_run(run_name=f"{args.approach}-{len(scenarios)}-scenarios"):
        mlflow.log_param("approach", args.approach)
        mlflow.log_param("n_scenarios", len(scenarios))

        per_metric: dict[str, list[float]] = {m[0]: [] for m in METRICS}

        for bundle in scenarios:
            report = generate(bundle)
            row = {"scenario": bundle.incident_id}
            for name, fn, _ in METRICS:
                score = fn(bundle, report)
                per_metric[name].append(score)
                row[name] = score
            print(
                f"  {bundle.incident_id:26s} "
                + "  ".join(f"{k}={v:.2f}" for k, v in row.items() if k != "scenario")
            )

        overall = 0.0
        print()
        for name, _, weight in METRICS:
            mean = sum(per_metric[name]) / len(per_metric[name])
            overall += weight * mean
            mlflow.log_metric(f"mean_{name}", mean)
            print(f"{name:24s} = {mean:.3f}  (weight {weight})")

        mlflow.log_metric("overall", overall)
        print(f"\n{'=' * 70}")
        print(f"OVERALL SCORE: {overall:.3f}  (approach: {args.approach})")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
