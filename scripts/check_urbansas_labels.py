"""Check whether MAVD and TAU use compatible `offscreen` semantics.

If the two subsets have similar offscreen prevalence and event-duration
distributions, the external TAU test is a clean generalization test.
If they differ significantly, the external test conflates domain shift
with label-definition shift — document accordingly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\Janak Mallik\ml-data\urbansas")


def main() -> None:
    audio_ann = pd.read_csv(ROOT / "annotations" / "audio_annotations.csv")
    video_ann = pd.read_csv(ROOT / "annotations" / "video_annotations.csv")

    # Per-clip subset
    clip_subset = video_ann.drop_duplicates("filename")[["filename", "subset"]]
    merged = audio_ann.merge(clip_subset, on="filename")

    print("=" * 70)
    print("LABEL SEMANTICS CHECK — MAVD vs. TAU")
    print("=" * 70)

    for subset in ["mavd", "tau"]:
        sub = merged[merged["subset"] == subset]
        offscreen = sub[sub["label"] == "offscreen"]
        clips_with_offscreen = offscreen["filename"].nunique()
        all_clips = sub["filename"].nunique()
        print(f"\n=== {subset.upper()} ===")
        print(f"  Total clips: {all_clips}")
        pct = 100 * clips_with_offscreen / max(all_clips, 1)
        print(f"  Clips with offscreen: {clips_with_offscreen} ({pct:.1f}%)")
        print(f"  Total offscreen annotations: {len(offscreen)}")

        if len(offscreen) > 0:
            durations = offscreen["end"] - offscreen["start"]
            print(
                f"  Offscreen event duration: "
                f"min={durations.min():.2f}s, "
                f"median={durations.median():.2f}s, "
                f"max={durations.max():.2f}s"
            )
            events_per_clip = len(offscreen) / max(clips_with_offscreen, 1)
            print(f"  Mean events per clip: {events_per_clip:.2f}")
        print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("If MAVD and TAU show similar:")
    print("  - % of clips with offscreen")
    print("  - offscreen event durations")
    print("  - events per clip")
    print("→ Label semantics are compatible.")
    print("→ TAU external test measures domain generalization cleanly.")
    print()
    print("If they differ significantly:")
    print("→ TAU external test conflates domain shift + label-definition shift.")
    print("→ Document as a limitation.")


if __name__ == "__main__":
    main()
