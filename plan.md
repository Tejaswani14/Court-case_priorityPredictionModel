## 🏆 MVP Feature List (for <17 hours)

**Core Features (must-have)**

1. **Prioritization Model**

   * Train a model (RandomForest/XGBoost) to assign **priority scores** (0–100) to cases.
   * Accuracy and reliability are key: provide basic metrics (accuracy, RMSE, etc.) for demo.

2. **Filtering & Sorting**

   * Judge can **filter by district, case type, age of case**.
   * Sort by **priority score**, newest/oldest filing, severity.

3. **Dashboard**

   * Minimal Streamlit dashboard:

     * Top N urgent cases list.
     * Simple bar chart / table for backlog overview.

4. **Case Details**

   * Clicking a case shows: metadata + **priority score + key contributing factors** (like SHAP feature importance).

5. **Batch Upload**

   * Upload CSV → model predicts priority → downloadable CSV.

**Polish Features (MVP can demo them lightly)**

* Role-based login (basic admin/judge token).
* Minimal accessibility (high contrast toggle + alt text).
* Demo scripts for SHAP visualizations (static images if necessary).

**Future Features (can mention in pitch)**

* Full accessibility suite (screen reader + TTS).
* Fairness / bias checks.
* Audit log.
* Export PDF reports.
* Real-time backend queue (Celery).
* Multiple language support.

---

## ✅ Demo Strategy to Impress Judges

1. **Start with Dashboard**

   * “Here’s today’s backlog; top cases prioritized automatically.”
   * Click a case → “Priority score is 88 — reason: repeated adjournments, severity score high.”

2. **Filtering & Sorting**

   * “Judge can quickly filter by district, case type, age, and sort by priority — saves hours of manual review.”

3. **Batch Prediction Upload**

   * Show uploading a CSV → predictions → downloadable CSV.

4. **Explainability**

   * Static SHAP chart showing top features (can be placeholder if time-constrained).

5. **Future Features Slide**

   * “With more time, we’ll add: full accessibility, TTS, audit logs, multi-language support, fairness checks, report generation.”

---

## 🧠 1 — Model Plan (ML & Data)

**Features / Tasks / Functions**

* **Data Preparation**

  * Load CSV files containing case data.
  * Clean and normalize features:

    * `case_id`, `district`, `case_type`, `filed_date`, `last_hearing_date`, `severity_score`, `adjournments_count`, `petitioner_status`, etc.
  * Handle missing values and outliers.
  * Encode categorical features (One-Hot / Label Encoding).

* **Model Training**

  * Train **RandomForest/XGBoost** to predict **priority score (0–100)**.
  * Split data into train/test sets.
  * Calculate **accuracy, RMSE, MAE** for demo metrics.
  * Save trained model using `joblib`.

* **Explainability**

  * Generate **SHAP feature importance** for top cases.
  * Create visualizations:

    * Bar chart of top contributing features.
    * Force plots (optional static images for demo).

* **Prediction Functions**

  * `predict_case(case_data)` → returns priority score + feature importance.
  * `predict_batch(csv_file)` → returns predictions for multiple cases as dataframe.

* **Future Enhancements (mentioned in pitch)**

  * Fairness / bias check.
  * Multi-class priority or multi-criteria scoring.

---

## ⚙️ 2 — Backend Plan (FastAPI)

**Features / Tasks / Functions**

* **Endpoints**

  * `POST /predict` → single case prediction.
  * `POST /batch` → batch CSV upload → return predictions CSV.
  * `GET /cases` → return filtered/sorted list of top N cases.
  * `GET /case/{id}` → return case details + priority score + feature importance.

* **Data & Model Integration**

  * Load pre-trained model (`joblib`) on startup.
  * Preprocess incoming data using same pipeline as training.
  * Run prediction functions.

* **Filtering & Sorting**

  * Allow filtering by:

    * `district`, `case_type`, `age_of_case`.
  * Allow sorting by:

    * `priority_score`, `filed_date`, `severity_score`.

* **Authentication / Role**

  * Basic JWT token auth (admin/judge).
  * Role-based access for endpoints.

* **Utility Functions**

  * `validate_input(data)` → ensures correct data types and required fields.
  * `generate_csv(predictions)` → returns downloadable CSV.
  * `log_action(user, action)` → stores audit info (for demo placeholder).

---

## 🎨 3 — Frontend Plan (Streamlit Dashboard)

**Features / Tasks / Functions**

* **Dashboard Pages**

  * **Top N Urgent Cases**

    * Table view of cases with priority scores.
    * Filters: district, case type, age of case.
    * Sorting by priority score, filing date, severity.
  * **Case Details**

    * Show metadata.
    * Display **priority score**.
    * Show **SHAP feature importance** visualization (static or dynamic).
  * **Batch Upload**

    * File uploader for CSV.
    * Display progress / status.
    * Download processed predictions CSV.
  * **Polish Features**

    * High-contrast toggle for accessibility.
    * Alt text for images / charts.
    * Placeholder for TTS or screen-reader friendly elements.

* **Utility Functions**

  * `load_cases()` → fetch data from backend.
  * `display_case_table()` → renders table with filters and sorting.
  * `display_case_detail(case_id)` → shows case metadata + SHAP.
  * `upload_and_predict(csv_file)` → sends file to backend batch endpoint.
  * `download_csv(predictions)` → triggers CSV download.

* **Future Enhancements (for pitch)**

  * Full accessibility (screen-reader + TTS).
  * Audit log viewer.
  * Multi-language support.
  * PDF report generation.

---
