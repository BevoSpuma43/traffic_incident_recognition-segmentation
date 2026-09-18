import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from cctv_incident.classifier import FEATURE_NAMES

parser = argparse.ArgumentParser()
parser.add_argument(
    "features", nargs="+", help="Feature CSV files with explicit train/val split and group"
)
parser.add_argument("--output", default="models/event-classifier.joblib")
parser.add_argument("--window", type=float, default=3)
parser.add_argument("--model", choices=["logistic", "hist"], default="hist")
args = parser.parse_args()
rows = []
for path in args.features:
    with Path(path).open(encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
groups = {}
for row in rows:
    if row["split"] == "test":
        raise ValueError("Test samples must not be used during training or threshold selection")
    previous = groups.setdefault(row["group"], row["split"])
    if previous != row["split"]:
        raise ValueError("Camera/source group leaks across train and validation")
for field in ("camera_id", "clip_id"):
    split_by_value = {}
    for row in rows:
        if split_by_value.setdefault(row[field], row["split"]) != row["split"]:
            raise ValueError(f"{field} leaks across train and validation")


def subset(split):
    selection = [row for row in rows if row["split"] == split]
    x = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in selection])
    y = np.array([int(row["label"]) for row in selection])
    if set(y) != {0, 1}:
        raise ValueError(f"{split} must contain both positive and negative windows")
    return x, y


x_train, y_train = subset("train")
x_val, y_val = subset("val")
if args.model == "logistic":
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(x_train, y_train)
else:
    model = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=42
    )
    model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
scores = model.predict_proba(x_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, scores)
f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
threshold = float(thresholds[np.argmax(f1)])
importance = permutation_importance(
    model, x_val, y_val, n_repeats=5, random_state=42, scoring="average_precision"
)
report = {
    "validation_average_precision": float(average_precision_score(y_val, scores)),
    "suggested_candidate_threshold": threshold,
    "score_is_calibrated_probability": False,
    "feature_importance": dict(
        zip(FEATURE_NAMES, importance.importances_mean.tolist(), strict=True)
    ),
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "window_seconds": args.window,
        "report": report,
    },
    output,
)
output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
