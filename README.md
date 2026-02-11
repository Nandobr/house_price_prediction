# Volusia County Home Price Prediction

A machine learning project to predict residential home prices in Volusia County using property sales data, building characteristics, and location features. This project implements a reproducible MLOps pipeline to ensure robust evaluation and prevent data leakage.

## 🚀 Overview

The goal is to predict home prices using historical sales data (2015-2019) while strictly avoiding "future leakage" from 2026 tax assessments. The model uses a 2-Stage XGBoost approach:
1.  **Stage 1**: Classify the price range (Binning Strategy).
2.  **Stage 2**: Regress the final price using physical traits (`SqFt`, `YearBuilt`) and location (`Neighborhood`).

**Performance (5-Fold CV on 2015-2019 data):**
-   **R2 Score**: `0.928` (+/- 0.038)
-   **RMSE**: `$303,041`

## 📂 Project Structure

The workflow is modularized into three sequential scripts:

| Script | Purpose | Output |
| :--- | :--- | :--- |
| **`01_preprocess_data.py`** | Load Raw CSVs, Merge Tables, **Remove Leakage** (2026 Tax Values). | `processed_data.csv` |
| **`02_feature_engineering.py`** | Generate Features (Ratios, Polynomials, Date Parts). | `engineered_features.csv` |
| **`03_train_model.py`** | Run **5-Fold Cross-Validation** using a 2-Stage Binning + Regression approach to evaluate theoretical maximum performance. | `experiments.csv` |
| **`04_export_model.py`** | Train a robust **Single-Stage XGBoost Regressor** on the full dataset and export artifacts for the Streamlit App. | `../volusia_property_app/model_artifacts.pkl` |

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd volusia_prediction
    ```

2.  **Create a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Mac/Linux
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

Run the pipeline in order:

1.  **Preprocess Data** (Clean & Drop Leakage):
    ```bash
    python 01_preprocess_data.py
    ```
    *Check `data_stats.md` for a log of the data health.*

2.  **Generate Features**:
    ```bash
    python 02_feature_engineering.py
    ```

3.  **Train & Evaluate**:
    ```bash
    python 03_train_model.py
    ```
    *Results will be printed to console and appended to `experiments.csv`.*

## 📊 Experiment Tracking
-   **`experiments.csv`**: Contains a history of all model runs, including hyperparameters, feature sets, and performance metrics.
-   **`data_stats.md`**: Tracks the shape and distribution of the dataset after every preprocessing or engineering step.

## 📝 License
[MIT](LICENSE)
