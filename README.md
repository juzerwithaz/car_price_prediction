# 🧭 AutoPrice Compass: Used Car Valuation & Market Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Cars24%20(8%2C015%20listings)-green.svg)](https://www.cars24.com)

**AutoPrice Compass** is an end-to-end Machine Learning intelligence and valuation system for the Indian pre-owned car market. Built on **8,015 real-world used car listings** scraped from Cars24 across major Indian RTO zones, it integrates all four core data science pillars into a unified, interactive **Streamlit web application**:

1. **Continuous Regression**: Accurate fair market price prediction in INR ($\pm$ expected valuation tolerance).
2. **Multi-Class Classification**: Automatic budget categorization (*Budget*, *Mid-range*, *Premium*) with zero target leakage.
3. **Unsupervised Market Clustering**: K-Means clustering identifying natural vehicle tiers.
4. **Cohort Hidden State Modeling**: Gaussian Hidden Markov Model (HMM) mapping unobserved valuation degradation regimes across vehicle-year cohorts.

---

## 🚀 Key Features

* **Instant Fair Market Appraisal**: Enter any car's specifications and get an immediate valuation backed by production-trained scikit-learn pipelines.
* **Interactive Deal Meter**: Input any seller's asking quote to test whether a car is a *Steal Deal*, *Fair Price*, or *Overpriced*.
* **Market Segment Context**: View how the car compares to its natural K-Means cluster (median prices, mileage benchmarks, typical age).
* **Aging & Value Regimes**: Examine the car's HMM value regime (*Premium-like*, *Mid-market*, *Budget-like*) as vehicle cohorts mature over time.
* **Full Analytics Dashboard**: Explore EDA distributions, regression error spreads, confusion matrices, PCA cluster plots, and transition heatmaps directly within the web interface.

---

## 📊 Dataset Profile

The underlying dataset contains **8,015 car listings** across **9 core operational attributes**:

| Attribute | Type | Description | Example Values |
| :--- | :--- | :--- | :--- |
| **Car Name** | Categorical | Specific brand and model (126 unique models) | *Maruti Swift*, *Hyundai Grand i10*, *Honda City* |
| **Year** | Numerical | Vehicle manufacturing year (2010 – 2023) | *2018*, *2021* |
| **Distance** | Numerical | Odometer distance travelled in kilometers | *42,000 km* |
| **Owner** | Numerical | Number of previous registered owners | *1, 2, 3, 4* |
| **Fuel** | Categorical | Engine fuel type | *PETROL, DIESEL, CNG, LPG* |
| **Location** | Categorical | Regional Transport Office (RTO) code | *MH-14 (Pune), DL-8C (Delhi), KA-03 (Bengaluru)* |
| **Drive** | Categorical | Transmission mechanism | *Manual, Automatic* |
| **Type** | Categorical | Vehicle body style | *HatchBack, Sedan, SUV, Lux_SUV, Lux_sedan* |
| **Price** | Numerical | Listed transaction price in INR (Target) | *₹1,19,000 – ₹33,00,000* |

---

## 🔬 Machine Learning Pillars & Verified Results

All numbers below reflect verified models evaluated on the held-out 20% test partition:

```
                                      AutoPrice Compass
                                              │
        ┌───────────────────┬─────────────────┴─────────────────┬───────────────────┐
        ▼                   ▼                                   ▼                   ▼
  1. REGRESSION       2. CLASSIFICATION                   3. CLUSTERING           4. HMM
 Linear / Random Forest  Logistic / Random Forest            K-Means (K=2)         3-State Gaussian
 RMSE: ₹1.05 Lakh       Weighted F1: 79.98%                 Silhouette: 0.2167    Log-Likelihood: -144.01
 R²: 0.8291             Accuracy: 79.91%                    Segments: 2 Tiers     Regimes: 3 Cohort States
```

### 1. Regression (Continuous Valuation)
* **Goal**: Predict continuous car price (`Price` in INR).
* **Winning Model**: **Linear Regression** (RMSE: **₹1,05,096.38**, MAE: **₹69,128.93**, $R^2$: **0.8291**).
* **Key Finding**: Linear Regression slightly outperformed Random Forest Regressor (RMSE: ₹1,05,298.58) because linear combinations over high-cardinality one-hot encoded variables (126 car models + RTO locations) generalize cleanly without decision-tree depth constraints.

### 2. Classification (Price Tiers)
* **Goal**: Categorize cars into balanced market segments:
  * **Budget**: $\le$ ₹4,43,000
  * **Mid-range**: ₹4,43,000 to ₹6,32,000
  * **Premium**: $>$ ₹6,32,000
* **Winning Model**: **Logistic Regression** (Weighted F1: **0.7998**, Accuracy: **79.91%**).
* **Methodological Safeguard**: `Price` was strictly omitted from feature matrix $X$ to eliminate target leakage.

### 3. K-Means Clustering (Market Segmentation)
* **Goal**: Discover organic car segments without ground-truth labels.
* **Optimal Selection**: Selected **$K = 2$** via Silhouette analysis (0.2167) and Elbow curve.
* **Cluster Profiles**:
  * **Higher-priced segment** (4,619 cars): Newer (avg year 2018.7), lower mileage (avg 42,875 km), median price **₹6,52,000**.
  * **Lower-priced segment** (3,396 cars): Older (avg year 2014.7), higher mileage (avg 65,878 km), median price **₹3,84,000**.
* **Design Prudence**: High-cardinality one-hot columns were omitted to prevent sparse binary flags from dominating Euclidean distances.

### 4. Gaussian Hidden Markov Model (Cohort Regimes)
* **Goal**: Model underlying valuation degradation regimes as vehicle cohorts mature.
* **Architecture**: 3-state Gaussian HMM trained on `(Body Type, Year)` cohorts tracking Log Median Price, Log Median Distance, and Mean Owners.
* **Inferred Regimes**:
  * **Premium-like regime**: Youngest cohorts, low mileage ($\approx$ 22,500 km), median price $\approx$ **₹14.19 Lakh**.
  * **Mid-market regime**: Intermediate age, moderate mileage ($\approx$ 54,000 km), median price $\approx$ **₹6.90 Lakh**.
  * **Budget-like regime**: Older cohorts, high mileage ($\approx$ 76,500 km), median price $\approx$ **₹5.87 Lakh**.
* **Methodological Note**: Year represents vehicle manufacturing date, not chronological transaction timestamps; the model captures cohort aging patterns rather than stock-like time-series forecasts.

---

## 📁 Repository Structure

```
AutoPrice_Compass/
│
├── app.py                             # Interactive Streamlit Web Application
├── autoprice_compass.py               # Complete ML training, evaluation & plotting pipeline
├── cars_24_combined.csv               # Raw dataset (8,015 listings from Cars24)
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Standard Python gitignore rules
├── README.md                          # Comprehensive documentation
├── README_FINAL_IMPLEMENTATION.md     # Academic review reference specification
├── DATASET_SOURCE_README.md          # Scraper background notes
│
├── .streamlit/
│   └── config.toml                    # Streamlit server & browser performance settings
│
├── src/                               # Modular application backend
│   ├── __init__.py
│   ├── inference.py                   # High-performance inference engine for all 4 models
│   └── utils.py                       # Currency formatters, deal evaluators & UI styling
│
└── outputs/                           # Generated evaluation artifacts & visualizations
    ├── *.joblib                       # Serialized scikit-learn model pipelines
    ├── *.csv                          # Performance comparisons & cluster metrics
    ├── *.png                          # High-resolution charts & evaluation plots
    └── *.txt                          # Classification reports & executive summaries
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/juzerwithaz/car_price_prediction.git
cd car_price_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 4. (Optional) Run the Full ML Training Script
To re-train the models and regenerate all figures in headless mode:
```bash
python autoprice_compass.py --no-gui
```

---

## 💡 Key Design Decisions to Defend in an Interview

1. **Why Linear / Logistic Regression beat Random Forest:**
   With one-hot encoding on high-cardinality features (126 models, dozens of RTOs), linear models leverage sparse linear combinations cleanly. Un-tuned decision tree ensembles can overfit rare leaf nodes.
2. **Preventing Target Leakage in Classification:**
   The classification target is derived from price quantiles. The raw `Price` variable is strictly barred from the feature set $X$.
3. **Log Transformations for Skewed Features:**
   Mileage and price have strong right tails. Log-transforming them (`np.log1p`) stabilizes variance and ensures stable K-Means clusters and HMM emission distributions.
4. **K-Means Feature Space Prudence:**
   High-cardinality categorical variables were excluded from K-Means to prevent the "curse of dimensionality" from destroying Euclidean distance metrics.

---

## 📄 License
This project is open-source under the MIT License.
