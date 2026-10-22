"""Evaluate report generators against ground-truth evidence.

Usage:
    uv run python scripts/evaluate_reports.py --approach template
    uv run python scripts/evaluate_reports.py --approach qwen  (Session 1b)

Metrics (weighted):
    evidence_fidelity       0.30   — do cited sources match present sources?
    signal_coverage         0.15   — proportion of present sources cited
    no_hallucination        0.15   — no cited source absent from evidence
    uncertainty_ack         0.15   — acknowledges uncertainty when needed
    recommendation_quality  0.15   — recommended action matches severity
    epistemic_clarity       0.10   — distinguishes observed from inferred

No predicted scores. Only measured results are printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow

from urbanintel.reporting.scenarios import all_scenarios
from urbanintel.reporting.schema import rule_based_severity


def _get_generator(approach: str):
    if approach == "template":
        from urbanintel.reporting.template_baseline import generate_template_report

        return generate_template_report
    if approach == "qwen":
        from urbanintel.reporting.qwen_generator import generate_qwen_report

        return generate_qwen_report
    raise ValueError(f"Unknown approach: {approach}")


def _evidence_fidelity(bundle, report) -> float:
    present = bundle.sources_present()
    if not present:
        return 1.0
    return len(present & set(report.sources_cited)) / len(present)


def _signal_coverage(bundle, report) -> float:
    return _evidence_fidelity(bundle, report)


def _no_hallucination(bundle, report) -> float:
    cited = set(report.sources_cited)
    present = bundle.sources_present()
    if not cited:
        return 1.0
    return 1.0 - (len(cited - present) / len(cited))


def _uncertainty_ack(bundle, report) -> float:
    if not _needs_uncertainty_ack(bundle):
        return 1.0

    text = report.report_text.lower()
    markers = [
        "uncertain",
        "may or may not",
        "not established",
        "conflicting",
        "low confidence",
        "unclear",
        "unconfirmed",
        "unknown",
        "missing",
        "cannot be confirmed",
    ]
    hits = sum(1 for m in markers if m in text)
    # Require 2+ distinct markers for full credit
    return min(1.0, hits / 2.0)


def _recommendation_quality(bundle, report) -> float:
    severity = rule_based_severity(bundle)
    text = report.report_text.lower()

    # Exclude the severity-line keyword — the template contains "escalate"
    # in the severity field, which must NOT count as a recommendation.
    if severity == "escalate":
        keywords = [
            "dispatch",
            "immediate response",
            "urgent",
            "respond now",
            "priority response",
            "send officers",
            "request medical",
        ]
    elif severity == "review":
        keywords = ["review", "verify", "investigate", "follow up", "follow-up"]
    elif severity == "insufficient":
        keywords = [
            "verify feed",
            "cannot confirm",
            "re-check",
            "do not treat",
            "recover",
            "missing data",
        ]
    else:  # normal
        keywords = ["monitor", "routine", "no action", "log"]

    return 1.0 if any(k in text for k in keywords) else 0.0


def _epistemic_clarity(bundle, report) -> float:
    text = report.report_text.lower()
    # Report must have distinct sections for observed / inferred / unknown
    sections = ["observed", "interpretation", "unknown", "recommend"]
    hits = sum(1 for s in sections if s in text)
    return hits / len(sections)


METRICS = [
    ("evidence_fidelity", _evidence_fidelity, 0.30),
    ("signal_coverage", _signal_coverage, 0.15),
    ("no_hallucination", _no_hallucination, 0.15),
    # uncertainty_ack is handled specially: averaged only over scenarios
    # where uncertainty is actually needed (see main()).
    ("uncertainty_ack", _uncertainty_ack, 0.15),
    ("recommendation_quality", _recommendation_quality, 0.15),
    ("epistemic_clarity", _epistemic_clarity, 0.10),
]


def _needs_uncertainty_ack(bundle) -> bool:
    """Does this scenario demand explicit uncertainty acknowledgment?"""
    if bundle.critical_sources_missing():
        return True
    for ev in bundle.evidence:
        if hasattr(ev, "confidence") and hasattr(ev, "class_name"):
            if ev.confidence < 0.5 and ev.class_name not in ("normal",):
                return True
    from urbanintel.reporting.schema import AudioEventEvidence, VideoAnomalyEvidence

    video_normal = any(
        isinstance(ev, VideoAnomalyEvidence) and ev.class_name == "normal" for ev in bundle.evidence
    )
    audio_critical = any(
        isinstance(ev, AudioEventEvidence) and ev.class_name == "gun_shot" for ev in bundle.evidence
    )
    return video_normal and audio_critical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["template", "qwen"], default="template")
    args = parser.parse_args()

    generate = _get_generator(args.approach)
    scenarios = all_scenarios()

    mlflow.set_experiment("urbanintel-week8")

    with mlflow.start_run(run_name=f"{args.approach}-{len(scenarios)}-scenarios"):
        mlflow.log_param("approach", args.approach)
        mlflow.log_param("n_scenarios", len(scenarios))

        per_metric: dict[str, list[float]] = {m[0]: [] for m in METRICS}
        uncertainty_needed_scores: list[float] = []
        severity_matches = 0

        for bundle in scenarios:
            report = generate(bundle)
            expected = rule_based_severity(bundle)
            severity_ok = report.severity == expected
            severity_matches += int(severity_ok)

            row_scores = {}
            for name, fn, _ in METRICS:
                score = fn(bundle, report)
                per_metric[name].append(score)
                row_scores[name] = score

            # Track uncertainty_ack only where it's actually needed
            if _needs_uncertainty_ack(bundle):
                uncertainty_needed_scores.append(_uncertainty_ack(bundle, report))

            print(
                f"  {bundle.incident_id:26s} "
                f"sev={report.severity:12s} "
                + "  ".join(f"{k}={v:.2f}" for k, v in row_scores.items())
            )

        overall = 0.0
        print()
        for name, _, weight in METRICS:
            if name == "uncertainty_ack":
                # Average only over scenarios where uncertainty matters
                if uncertainty_needed_scores:
                    mean = sum(uncertainty_needed_scores) / len(uncertainty_needed_scores)
                    print(
                        f"{name:24s} = {mean:.3f}  (weight {weight}, "
                        f"averaged over {len(uncertainty_needed_scores)} "
                        f"scenarios where uncertainty needed)"
                    )
                else:
                    mean = 1.0
                    print(f"{name:24s} = {mean:.3f}  (no scenarios needed it)")
            else:
                mean = sum(per_metric[name]) / len(per_metric[name])
                print(f"{name:24s} = {mean:.3f}  (weight {weight})")

            overall += weight * mean
            mlflow.log_metric(f"mean_{name}", mean)

        severity_rate = severity_matches / len(scenarios)
        mlflow.log_metric("severity_consistency", severity_rate)
        mlflow.log_metric("overall", overall)

        print(f"{'severity_consistency':24s} = {severity_rate:.3f}")
        print(f"\n{'=' * 70}")
        print(f"OVERALL SCORE: {overall:.3f}  (approach: {args.approach})")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
