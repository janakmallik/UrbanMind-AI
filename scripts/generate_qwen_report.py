"""Generate a Qwen report for one scenario, for inspection.

Usage:
    uv run python scripts/generate_qwen_report.py --list
    uv run python scripts/generate_qwen_report.py --scenario S13-full-stack
    uv run python scripts/generate_qwen_report.py --all --out reports_qwen/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.reporting.qwen_generator import generate_qwen_report
from urbanintel.reporting.scenarios import all_scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true", help="List all available scenario IDs")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    scenarios = all_scenarios()

    if args.list:
        print(f"{len(scenarios)} available scenarios:")
        for b in scenarios:
            print(f"  {b.incident_id}")
        return

    if args.all:
        out_dir = Path(args.out) if args.out else _REPO_ROOT / "reports_qwen"
        out_dir.mkdir(exist_ok=True)
        t0 = time.time()
        for bundle in scenarios:
            report = generate_qwen_report(bundle)
            (out_dir / f"{bundle.incident_id}.txt").write_text(report.report_text, encoding="utf-8")
            print(f"  {bundle.incident_id}: {len(report.report_text)} chars")
        print(f"\nTotal time: {time.time() - t0:.1f}s")
        return

    if not args.scenario:
        print("Specify --scenario <id>, --all, or --list")
        print("Run with --list to see valid IDs.")
        sys.exit(1)

    matches = [b for b in scenarios if b.incident_id == args.scenario]
    if not matches:
        print(f"No scenario with id '{args.scenario}'")
        print()
        print("Available IDs:")
        for b in scenarios:
            print(f"  {b.incident_id}")
        print()
        print("Tip: the only scenario with 'full-stack' is S13.")
        sys.exit(1)

    bundle = matches[0]
    print(f"Generating report for {bundle.incident_id}...")
    t0 = time.time()
    report = generate_qwen_report(bundle)
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"Incident: {report.incident_id}")
    print(f"Severity: {report.severity}")
    print(f"Sources:  {', '.join(report.sources_cited)}")
    print(f"Generated in {elapsed:.1f}s")
    print("=" * 70)
    print(report.report_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
