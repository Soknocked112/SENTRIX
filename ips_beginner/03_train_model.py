"""
Step 3: train the model and measure how good it is on data it has never seen.
Run:  python 03_train_model.py
Slow computer?  python 03_train_model.py --trees 30
"""
import argparse
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--trees", type=int, default=60, help="number of decision trees (more = slower, slightly better)")
args = parser.parse_args()

if not os.path.exists("prepared/data.joblib"):
    raise SystemExit("prepared/data.joblib not found. Run step 2 first:  python 02_prepare_data.py")

data = joblib.load("prepared/data.joblib")
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

print(f"Training a Random Forest with {args.trees} trees on {len(y_train):,} flows ... (this can take several minutes)")
start = time.time()
model = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=2, n_jobs=-1,
                               class_weight="balanced_subsample", random_state=42)
model.fit(X_train, y_train)
print(f"Training finished in {time.time() - start:.0f} seconds.")

# ---- measure on the test data (never seen during training) ----
predicted = model.predict(X_test)
is_attack_true = (y_test.values != "BENIGN")
is_attack_pred = (predicted != "BENIGN")

tp = int(np.sum(is_attack_true & is_attack_pred))
tn = int(np.sum(~is_attack_true & ~is_attack_pred))
fp = int(np.sum(~is_attack_true & is_attack_pred))
fn = int(np.sum(is_attack_true & ~is_attack_pred))
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)
fpr = fp / max(fp + tn, 1)

report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
importances = sorted(zip(data["feature_names"], model.feature_importances_), key=lambda t: -t[1])[:15]

metrics = {
    "rows_train": int(len(y_train)), "rows_test": int(len(y_test)), "trees": args.trees,
    "attack_vs_normal": {
        "accuracy": float((is_attack_true == is_attack_pred).mean()), "precision": precision, "recall": recall,
        "f1": f1, "false_positive_rate": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "attack_type_accuracy": float(accuracy_score(y_test, predicted)),
    "per_class": {k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")},
    "top_features": [[n, float(v)] for n, v in importances],
}
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
json.dump(metrics, open("results/metrics.json", "w"), indent=2)
joblib.dump({"model": model, "feature_names": data["feature_names"]}, "models/ips_model.joblib", compress=3)

# ---- pictures for your report ----
fig, ax = plt.subplots(figsize=(4.5, 4))
cm = np.array([[tn, fp], [fn, tp]])
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticklabels(["Normal", "Attack"])
ax.set_xlabel("Model said"); ax.set_ylabel("Truth"); ax.set_title("Attack vs normal")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=150); plt.close(fig)

classes = [k for k in metrics["per_class"] if k != "BENIGN"]
recalls = [metrics["per_class"][k]["recall"] * 100 for k in classes]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(classes, recalls, color="#2563eb"); ax.set_xlim(0, 100)
ax.set_xlabel("Recall % (how many of this attack were caught)"); ax.set_title("Detection per attack type")
fig.tight_layout(); fig.savefig("results/per_attack_recall.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
names, values = zip(*importances[::-1])
ax.barh(names, values, color="#16a34a"); ax.set_title("Top 15 most useful features")
fig.tight_layout(); fig.savefig("results/feature_importance.png", dpi=150); plt.close(fig)

print("\n================ RESULTS (on data the model never saw) ================")
print(f"Attack vs normal  accuracy : {metrics['attack_vs_normal']['accuracy'] * 100:.2f}%")
print(f"                  precision: {precision * 100:.2f}%   (of flows called attack, how many really were)")
print(f"                  recall   : {recall * 100:.2f}%   (of real attacks, how many were caught)")
print(f"                  F1 score : {f1:.4f}")
print(f"                  false alarms on normal traffic: {fpr * 100:.2f}%")
print(f"Exact attack-type accuracy : {metrics['attack_type_accuracy'] * 100:.2f}%")
print("\nPer attack type (recall):")
for k in classes:
    r = metrics["per_class"][k]
    print(f"  {k:<30} {r['recall'] * 100:6.1f}%   (test flows: {int(r['support'])})")
print("\nSaved: models/ips_model.joblib, results/metrics.json and 3 pictures in results/")
print("Continue to step 4.")
