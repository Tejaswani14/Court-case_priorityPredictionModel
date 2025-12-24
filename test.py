import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sys

# ==============================
# ⚙️ USER INPUTS
# ==============================
if len(sys.argv) < 3:
    predicted_path = input("Enter path to predicted CSV: ").strip().strip('"').strip("'")
    actual_path = input("Enter path to actual CSV: ").strip().strip('"').strip("'")
else:
    predicted_path = sys.argv[1].strip('"').strip("'")
    actual_path = sys.argv[2].strip('"').strip("'")

print(f"\n🔍 Loading files:\n- Predictions: {predicted_path}\n- Actual: {actual_path}")

# ==============================
# 📂 LOAD CSV FILES
# ==============================
pred_df = pd.read_csv(predicted_path)
actual_df = pd.read_csv(actual_path)

# ==============================
# 🔀 MERGE ON CASE_ID
# ==============================
if "case_id" in pred_df.columns and "case_id" in actual_df.columns:
    merged = pd.merge(pred_df, actual_df, on="case_id", suffixes=("_pred", "_actual"))
else:
    raise ValueError("❌ 'case_id' column missing in one of the files.")

# ==============================
# 🔢 ENCODE PRIORITY
# ==============================
# Predicted: string or numeric class
def encode_pred(series):
    mapping = {
        "Very Low Priority": 0,
        "Low Priority": 1,
        "Medium Priority": 2,
        "High Priority": 3,
        "Critical Priority": 4
    }
    if series.dtype == object:
        return series.str.strip().map(mapping)
    else:
        return series.astype(int)

if "predicted_priority" in merged.columns:
    merged["pred_encoded"] = encode_pred(merged["predicted_priority"])
elif "priority_class" in merged.columns:
    merged["pred_encoded"] = merged["priority_class"].astype(int)
else:
    raise ValueError("❌ No predicted priority column found.")

# Actual: map 1-100 score to 0-4 classes
def map_actual_score(score):
    score = float(score)
    if score <= 20:
        return 0  # Very Low
    elif score <= 40:
        return 1  # Low
    elif score <= 60:
        return 2  # Medium
    elif score <= 80:
        return 3  # High
    else:
        return 4  # Critical

if "priority_score" in merged.columns:
    merged["actual_encoded"] = merged["priority_score"].apply(map_actual_score)
else:
    raise ValueError("❌ No actual priority column found.")

# Drop rows with missing or unrecognized labels
merged = merged.dropna(subset=["pred_encoded", "actual_encoded"])

# ==============================
# 📊 METRICS
# ==============================
y_true = merged["actual_encoded"]
y_pred = merged["pred_encoded"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n=======================")
print("📈 MODEL PERFORMANCE")
print("=======================")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"📊 Recall:    {recall:.4f}")
print(f"🏆 F1 Score:  {f1:.4f}")

# Classification report with human-readable class names
class_names = ["Very Low", "Low", "Medium", "High", "Critical"]
print("\n=======================")
print("🔍 CLASSIFICATION REPORT")
print("=======================")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

print("\n=======================")
print("🧮 CONFUSION MATRIX")
print("=======================")
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df)

print(f"\nTotal compared cases: {len(merged)}")
print("\n✅ Comparison complete!\n")
