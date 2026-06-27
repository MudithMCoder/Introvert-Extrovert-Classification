#  Personality Type Classifier - Introvert vs. Extrovert

A complete end-to-end machine learning project that predicts whether a person is an **Introvert** or **Extrovert** based on their behavioral patterns. The project covers the full ML lifecycle: exploratory data analysis → preprocessing → model training → evaluation → REST API deployment.

---

##  Project Overview

| Item | Detail                                |
|---|---------------------------------------|
| **Task** | Binary Classification                 |
| **Target** | `Personality` — Introvert / Extrovert |
| **Model** | Random Forest Classifier              |
| **Dataset Size** | ~2,900 rows, 7 features               |
| **API Framework** | Flask                                 |
| **Deployment** | PythonAnywhere                        |

---

##  Project Structure

```
personality-classifier/
│
├── Data/
│   └── personality_dataset.csv       
│
├── artifacts/                       
│   ├── num_imputer.pkl
│   ├── cat_imputer.pkl
│   ├── feature_encoders.pkl
│   ├── target_encoder.pkl
│   └── model.pkl
│
├── templates/
│   └── index.html                    
│
├── pipeline.ipynb                    
├── app.py                           
├── requirements.txt
└── README.md
```

---

##  Dataset

**File:** `personality_dataset.csv`

| Feature | Type | Description |
|---|---|---|
| `Time_spent_Alone` | Float (0–10) | Hours per day spent alone |
| `Social_event_attendance` | Float (0–10) | Events attended per month |
| `Going_outside` | Float (0–10) | Days per week going outside |
| `Friends_circle_size` | Float (0–10) | Number of close friends |
| `Post_frequency` | Float (0–10) | Social media posts per week |
| `Stage_fear` | String (Yes/No) | Whether the person has stage fright |
| `Drained_after_socializing` | String (Yes/No) | Whether socializing is draining |
| `Personality` | String | **Target** — Introvert / Extrovert |

> The dataset contains ~2–3% missing values across the numerical columns, handled during preprocessing.

---

##  ML Pipeline (`pipeline.ipynb`)

### 1. Exploratory Data Analysis
- Missing value analysis with `missingno` (matrix, bar, heatmap)
- Class imbalance check on the target variable
- Histograms with mean/median lines for all 5 numerical features
- Boxplots for outlier detection
- Bivariate analysis — boxplots & violin plots grouped by personality type
- Correlation heatmap
- PCA scatter plot for 2D visualization

### 2. Preprocessing
- **Train/Test split** (80/20, stratified) performed **before** any fitting to prevent data leakage
- **Numerical imputation** — `SimpleImputer(strategy="median")` fit on `X_train` only
- **Categorical imputation** — `SimpleImputer(strategy="most_frequent")` fit on `X_train` only
- **Feature encoding** — `LabelEncoder` per categorical column; encoders stored in a dict for reuse
- **Target encoding** — `LabelEncoder` on `y_train`, then `transform` on `y_test`

### 3. Model Training
- Baseline `RandomForestClassifier` trained first
- **RandomizedSearchCV** (50 iterations, 5-fold CV, `f1_weighted`) for fast hyperparameter search
- **GridSearchCV** (5-fold CV) for exhaustive search around promising regions
- Final model configuration:

```python
RandomForestClassifier(
    class_weight='balanced',
    max_depth=10,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=300
)
```

### 4. Evaluation
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap
- ROC-AUC curve

### 5. Artifact Saving
All preprocessing objects and the trained model are serialized with `pickle` into the `artifacts/` folder.

---

##  REST API (`app.py`)

A Flask API that accepts behavioral feature data as JSON and returns a personality prediction.

### Endpoint

```
POST /predict
```

### Request Body

```json
{
  "Time_spent_Alone": 7.0,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2.0,
  "Going_outside": 2.0,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3.0,
  "Post_frequency": 1.0
}
```

### Response

```json
{
  "prediction": "Introvert",
  "confidence": 0.9267,
  "all_classes": ["Extrovert", "Introvert"]
}
```

### Error Responses

| Code | Reason |
|---|---|
| `400` | Request body is not valid JSON |
| `422` | Unseen/invalid value in a categorical field |
| `500` | Unexpected server error |

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.9+
- `pip`

### Installation

```bash
#  Clone the repository
git clone https://github.com/your-username/personality-classifier.git
cd personality-classifier

#  Create a virtual environment
python -m venv venv
source venv/bin/activate        

#  Install dependencies
pip install -r requirements.txt
```

### Run the API Locally

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

> **Note:** The `artifacts/` folder must exist and contain all five `.pkl` files before starting the server. Run `pipeline.ipynb` end-to-end to generate them.

---

## 📦 Requirements

```
flask
numpy
pandas
scikit-learn
```
---
##  Accepted Feature Values

| Feature | Accepted Values                        |
|---|----------------------------------------|
| `Time_spent_Alone` | `0.0` – `10.0` (or `null` for missing) |
| `Social_event_attendance` | `0.0` – `10.0` (or `null` for missing) |
| `Going_outside` | `0.0` – `10.0` (or `null` for missing) |
| `Friends_circle_size` | `0.0` – `10.0` (or `null` for missing) |
| `Post_frequency` | `0.0` – `10.0` (or `null` for missing) |
| `Stage_fear` | `"Yes"` or `"No"`                      |
| `Drained_after_socializing` | `"Yes"` or `"No"`                      |

Missing numerical values are automatically imputed using training-set medians. Unrecognized categorical values return a `422` error.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
