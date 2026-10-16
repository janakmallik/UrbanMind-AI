"""NYC 311 text data preparation for DistilBERT fine-tuning.

Task: classify complaint_type from the free-text `descriptor` field.

Split methodology matches Week 1 (stratified random 70/15/15,
random_state=42) but operates on a smaller population (rows with
non-null descriptors). The row sets are therefore not identical to
Week 1's; the *method* is.

Important caveat (documented in the model card):
    The `descriptor` field is assigned by the intake operator at the
    same time as `complaint_type`. This makes it a near-deterministic
    predictor of the label, not a realistic early-warning signal.
    The experiment is a valid text-classification study, but the task
    should be framed as "text-to-label mapping," not "forecast from
    pre-complaint information."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TOP_N_CLASSES = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class TextSplit:
    """Container for a text classification split."""

    train_texts: list[str]
    train_labels: list[int]
    val_texts: list[str]
    val_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]
    label_names: list[str]
    overlap_report: dict


def prepare_text_data(
    df: pd.DataFrame,
    top_n: int = TOP_N_CLASSES,
    test_size: float = 0.30,
    val_size: float = 0.50,
    random_state: int = RANDOM_STATE,
) -> TextSplit:
    """Filter to top-N classes (selected from full dataset, matching Week 1),
    then drop rows without descriptor text, then split."""
    # Step 1: Select top-N classes from the FULL dataset (same as Week 1)
    top = df["complaint_type"].value_counts().head(top_n).index.tolist()
    df = df[df["complaint_type"].isin(top)].copy()

    # Step 2: NOW filter rows missing descriptor
    df = df.dropna(subset=["descriptor"]).copy()
    df["descriptor"] = df["descriptor"].astype(str).str.strip()
    df = df[df["descriptor"].str.len() > 0]

    # Step 3: Encode labels
    label_names = sorted(top)
    label_to_id = {name: i for i, name in enumerate(label_names)}
    df["label_id"] = df["complaint_type"].map(label_to_id)

    # Step 4: Stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=temp_df["label_id"],
    )

    # Compute descriptor overlap between train and test
    train_set = set(train_df["descriptor"].unique())
    test_set = set(test_df["descriptor"].unique())
    overlap = train_set & test_set
    overlap_report = {
        "unique_train_descriptors": len(train_set),
        "unique_test_descriptors": len(test_set),
        "overlap_count": len(overlap),
        "test_descriptor_overlap_fraction": len(overlap) / max(len(test_set), 1),
        "test_row_overlap_fraction": test_df["descriptor"].isin(train_set).mean(),
    }

    return TextSplit(
        train_texts=train_df["descriptor"].tolist(),
        train_labels=train_df["label_id"].tolist(),
        val_texts=val_df["descriptor"].tolist(),
        val_labels=val_df["label_id"].tolist(),
        test_texts=test_df["descriptor"].tolist(),
        test_labels=test_df["label_id"].tolist(),
        label_names=label_names,
        overlap_report=overlap_report,
    )
