"""
AutoPrice Compass - Inference Engine
Loads trained models, preprocessing pipelines, and precomputed analytical artifacts
to provide unified predictions across all 4 Machine Learning pillars.
"""

from pathlib import Path
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

# Suppress sklearn unpickling warnings between minor versions
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_FILE = BASE_DIR / "cars_24_combined.csv"


class ModelStore:
    """Singleton cache for loaded models and lookup artifacts."""
    def __init__(self):
        self._linear_reg = None
        self._rf_reg = None
        self._logistic_clf = None
        self._rf_clf = None
        self._kmeans_bundle = None
        self._cluster_summary = None
        self._hmm_cohorts = None
        self._hmm_summary = None
        self._price_boundaries = None
        self._raw_df = None

    def get_dataset(self) -> pd.DataFrame:
        if self._raw_df is None:
            if not DATA_FILE.exists():
                raise FileNotFoundError(f"Dataset file missing at {DATA_FILE}")
            df = pd.read_csv(DATA_FILE)
            df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()
            self._raw_df = df
        return self._raw_df

    def get_model(self, model_key: str):
        if model_key == "linear_regression":
            if self._linear_reg is None:
                path = OUTPUTS_DIR / "linear_regression.joblib"
                self._linear_reg = joblib.load(path)
            return self._linear_reg

        elif model_key == "rf_regression":
            if self._rf_reg is None:
                path = OUTPUTS_DIR / "random_forest_regression.joblib"
                self._rf_reg = joblib.load(path)
            return self._rf_reg

        elif model_key == "logistic_classification":
            if self._logistic_clf is None:
                path = OUTPUTS_DIR / "logistic_regression.joblib"
                self._logistic_clf = joblib.load(path)
            return self._logistic_clf

        elif model_key == "rf_classification":
            if self._rf_clf is None:
                path = OUTPUTS_DIR / "random_forest_classification.joblib"
                self._rf_clf = joblib.load(path)
            return self._rf_clf

        elif model_key == "kmeans_bundle":
            if self._kmeans_bundle is None:
                path = OUTPUTS_DIR / "kmeans_bundle.joblib"
                if path.exists():
                    self._kmeans_bundle = joblib.load(path)
            return self._kmeans_bundle
        
        else:
            raise KeyError(f"Unknown model key: {model_key}")

    def get_cluster_summary(self) -> pd.DataFrame:
        if self._cluster_summary is None:
            path = OUTPUTS_DIR / "cluster_summary.csv"
            if path.exists():
                self._cluster_summary = pd.read_csv(path, index_col=0)
            else:
                self._cluster_summary = pd.DataFrame()
        return self._cluster_summary

    def get_hmm_cohorts(self) -> pd.DataFrame:
        if self._hmm_cohorts is None:
            path = OUTPUTS_DIR / "hmm_cohort_hidden_states.csv"
            if path.exists():
                self._hmm_cohorts = pd.read_csv(path)
            else:
                self._hmm_cohorts = pd.DataFrame()
        return self._hmm_cohorts

    def get_hmm_summary(self) -> pd.DataFrame:
        if self._hmm_summary is None:
            path = OUTPUTS_DIR / "hmm_state_summary.csv"
            if path.exists():
                self._hmm_summary = pd.read_csv(path)
            else:
                self._hmm_summary = pd.DataFrame()
        return self._hmm_summary

    def get_price_boundaries(self) -> pd.DataFrame:
        if self._price_boundaries is None:
            path = OUTPUTS_DIR / "price_category_boundaries.csv"
            if path.exists():
                self._price_boundaries = pd.read_csv(path)
            else:
                self._price_boundaries = pd.DataFrame()
        return self._price_boundaries


# Global singleton
store = ModelStore()


def get_feature_options() -> Dict[str, List[Any]]:
    """Return lists of available categorical and numeric options extracted from dataset."""
    df = store.get_dataset()
    
    # Sort popular cars first
    popular_cars = df["Car Name"].value_counts().index.tolist()
    
    locations = sorted(df["Location"].dropna().unique().tolist())
    fuels = sorted(df["Fuel"].dropna().unique().tolist())
    drives = sorted(df["Drive"].dropna().unique().tolist())
    types = sorted(df["Type"].dropna().unique().tolist())
    
    min_year = int(df["Year"].min())
    max_year = int(df["Year"].max())
    
    return {
        "car_names": popular_cars,
        "locations": locations,
        "fuels": fuels,
        "drives": drives,
        "types": types,
        "years": list(range(max_year, min_year - 1, -1)),
        "default_distance": 45000,
        "default_owner": 1
    }


def predict_car_valuation(
    car_name: str,
    year: int,
    distance: float,
    owner: int,
    fuel: str,
    location: str,
    drive: str,
    car_type: str,
    reg_model_name: str = "Linear Regression",
    clf_model_name: str = "Logistic Regression"
) -> Dict[str, Any]:
    """
    Unified predictor evaluating a car listing across all 4 ML tasks:
    1. Continuous Fair Price (Regression)
    2. Budget Segment / Tier (Classification)
    3. Natural Market Cluster (K-Means)
    4. Vehicle Cohort Aging Regime (Gaussian HMM)
    """
    input_df = pd.DataFrame([{
        "Car Name": car_name,
        "Year": float(year),
        "Distance": float(distance),
        "Owner": float(owner),
        "Fuel": fuel,
        "Location": location,
        "Drive": drive,
        "Type": car_type
    }])

    # -------------------------------------------------------------
    # Pillar 1: Regression (Continuous Price Prediction)
    # -------------------------------------------------------------
    if "Random Forest" in reg_model_name:
        reg_model = store.get_model("rf_regression")
        mae = 69082.0
    else:
        reg_model = store.get_model("linear_regression")
        mae = 69129.0

    raw_pred_price = float(reg_model.predict(input_df)[0])
    # Guard against negative price edge-cases in extreme linear extrapolations
    fair_price = max(50_000.0, raw_pred_price)
    price_low = max(30_000.0, fair_price - mae)
    price_high = fair_price + mae

    # -------------------------------------------------------------
    # Pillar 2: Classification (Price Category Tier)
    # -------------------------------------------------------------
    if "Random Forest" in clf_model_name:
        clf_model = store.get_model("rf_classification")
    else:
        clf_model = store.get_model("logistic_classification")

    predicted_tier = str(clf_model.predict(input_df)[0])
    
    # Class probabilities if available
    tier_probs = {}
    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(input_df)[0]
        classes = clf_model.classes_
        for cls_name, p in zip(classes, probs):
            tier_probs[str(cls_name)] = float(p)

    # -------------------------------------------------------------
    # Pillar 3: K-Means Market Segmentation
    # -------------------------------------------------------------
    kmeans_bundle = store.get_model("kmeans_bundle")
    cluster_summary_df = store.get_cluster_summary()
    
    cluster_id = 0
    cluster_name = "Market Segment"
    cluster_stats = {}

    if kmeans_bundle is not None:
        try:
            c_df = pd.DataFrame([{
                "Year": float(year),
                "Log_Distance": np.log1p(max(0, float(distance))),
                "Owner": float(owner),
                "Log_Price": np.log1p(fair_price),
                "Fuel": fuel,
                "Drive": drive,
                "Type": car_type
            }])
            X_clust = kmeans_bundle["preprocessor"].transform(c_df)
            if hasattr(X_clust, "toarray"):
                X_clust = X_clust.toarray()
            cluster_id = int(kmeans_bundle["model"].predict(X_clust)[0])
            cluster_name = kmeans_bundle["cluster_name_map"].get(cluster_id, f"Cluster {cluster_id}")
        except Exception:
            # Fallback based on median price distance
            if not cluster_summary_df.empty and "Median_Price" in cluster_summary_df.columns:
                diffs = (cluster_summary_df["Median_Price"] - fair_price).abs()
                cluster_id = int(diffs.idxmin())
                cluster_name = str(cluster_summary_df.loc[cluster_id, "Cluster_Name"])
    else:
        # Fallback to cluster summary nearest median
        if not cluster_summary_df.empty and "Median_Price" in cluster_summary_df.columns:
            diffs = (cluster_summary_df["Median_Price"] - fair_price).abs()
            cluster_id = int(diffs.idxmin())
            cluster_name = str(cluster_summary_df.loc[cluster_id, "Cluster_Name"])

    if not cluster_summary_df.empty and cluster_id in cluster_summary_df.index:
        row = cluster_summary_df.loc[cluster_id]
        cluster_stats = {
            "num_cars": int(row.get("Number_of_Cars", 0)),
            "median_price": float(row.get("Median_Price", 0.0)),
            "avg_year": float(row.get("Average_Year", 0.0)),
            "avg_distance": float(row.get("Average_Distance", 0.0))
        }

    # -------------------------------------------------------------
    # Pillar 4: Gaussian Hidden Markov Model (Cohort Regime)
    # -------------------------------------------------------------
    hmm_cohorts_df = store.get_hmm_cohorts()
    hmm_summary_df = store.get_hmm_summary()
    
    hmm_regime = "Mid-market regime"
    cohort_info = {}
    
    if not hmm_cohorts_df.empty:
        type_cohorts = hmm_cohorts_df[hmm_cohorts_df["Type"] == car_type]
        if type_cohorts.empty:
            type_cohorts = hmm_cohorts_df
            
        # Find exact year or nearest year in cohort observations
        type_cohorts = type_cohorts.copy()
        type_cohorts["year_diff"] = (type_cohorts["Year"] - year).abs()
        nearest_row = type_cohorts.sort_values("year_diff").iloc[0]
        
        hmm_regime = str(nearest_row.get("Regime_Name", "Mid-market regime"))
        cohort_info = {
            "observed_year": int(nearest_row["Year"]),
            "cohort_median_price": float(nearest_row.get("Median_Price", fair_price)),
            "cohort_median_distance": float(nearest_row.get("Median_Distance", distance)),
            "listing_count": int(nearest_row.get("Listing_Count", 1))
        }

    return {
        "fair_price": fair_price,
        "price_range_low": price_low,
        "price_range_high": price_high,
        "mae": mae,
        "tier": predicted_tier,
        "tier_probabilities": tier_probs,
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "cluster_stats": cluster_stats,
        "hmm_regime": hmm_regime,
        "cohort_info": cohort_info
    }
