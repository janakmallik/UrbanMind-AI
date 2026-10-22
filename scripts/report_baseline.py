"""Generate one template report from a named scenario.

Usage:
    uv run python scripts/report_baseline.py --scenario S13-full-stack
    uv run python scripts/report_baseline.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.reporting.scenarios import all_scenarios
from urbanintel.reporting.template_baseline import generate_template_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    scenarios = all_scenarios()
    out_dir = Path(args.out) if args.out else _REPO_ROOT / "reports"

    if args.all:
        out_dir.mkdir(exist_ok=True)
        for bundle in scenarios:
            report = generate_template_report(bundle)
            (out_dir / f"{bundle.incident_id}.json").write_text(
                report.model_dump_json(indent=2), encoding="utf-8"
            )
            print(
                f"  {bundle.incident_id}: severity={report.severity}, "
                f"sources_cited={len(report.sources_cited)}"
            )
        print(f"\nWrote {len(scenarios)} reports to {out_dir}")
        return

    if not args.scenario:
        print("Specify --scenario <id> or --all")
        sys.exit(1)

    matches = [b for b in scenarios if b.incident_id == args.scenario]
    if not matches:
        print(f"No scenario with id {args.scenario}")
        sys.exit(1)

    bundle = matches[0]
    report = generate_template_report(bundle)

    print("=" * 70)
    print(f"Incident: {report.incident_id}")
    print(f"Severity: {report.severity}")
    print(f"Sources:  {', '.join(report.sources_cited)}")
    print("=" * 70)
    print(report.report_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
