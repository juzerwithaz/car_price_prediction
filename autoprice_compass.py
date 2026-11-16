"""
AUTOPRICE COMPASS - INTEGRATED PROJECT IMPLEMENTATION
Covers:
  REVIEW 1: Regression + Classification
  REVIEW 2: K-Means Clustering + Gaussian Hidden Markov Model

Dataset: cars_24_combined.csv
Run:
  pip install -r requirements.txt
  python autoprice_compass.py

All generated results are saved in ./outputs/
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise SystemExit(
        "\nMissing dependency: hmmlearn\n"
        "Install all dependencies using: pip install -r requirements.txt"
    )

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "cars_24_combined.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------
# 1. LOAD DATA AND BASIC DATA UNDERSTANDING
# ------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Keep cars_24_combined.csv in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")].copy()

required = [
    "Car Name", "Year", "Distance", "Owner", "Fuel",
    "Location", "Drive", "Type", "Price"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Required columns missing from dataset: {missing}")

for col in ["Car Name", "Fuel", "Location", "Drive", "Type"]:
    df[col] = df[col].astype("string").str.strip().astype(object)
    df[col] = df[col].where(pd.notna(df[col]), np.nan)

for col in ["Year", "Distance", "Owner", "Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Price is essential for all supervised tasks.
df = df.dropna(subset=["Price"])
df = df[df["Price"] > 0].copy()

# Remove impossible negative distance values; they are treated as missing.
df.loc[df["Distance"] < 0, "Distance"] = np.nan
df.loc[df["Owner"] < 0, "Owner"] = np.nan

# Feature engineering: reduce skew in distance for clustering.
df["Log_Distance"] = np.log1p(df["Distance"].clip(lower=0))

# Dataset profile.
profile = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Values": [int(df[c].isna().sum()) for c in df.columns],
    "Unique_Values": [int(df[c].nunique(dropna=True)) for c in df.columns]
})
profile.to_csv(OUT / "dataset_profile.csv", index=False)

with open(OUT / "dataset_overview.txt", "w", encoding="utf-8") as f:
    f.write("AUTOPRICE COMPASS - DATASET OVERVIEW\n")
    f.write("=" * 45 + "\n")
    f.write(f"Records after removing missing/invalid target prices: {len(df)}\n")
    f.write(f"Meaningful original features: {len(required)}\n\n")
    f.write("Missing values:\n")
    f.write(df[required].isna().sum().to_string())
    f.write("\n\nNumerical summary:\n")
    f.write(df[["Year", "Distance", "Owner", "Price"]].describe().to_string())
    f.write("\n")


# Exploratory Data Analysis (EDA) required for Review 1.
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=40, kde=True)
plt.xlabel("Listed Price (INR)")
plt.title("Distribution of Used-Car Prices")
plt.tight_layout()
plt.savefig(OUT / "eda_price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Type", order=df["Type"].value_counts().index)
plt.xlabel("Car Type")
plt.ylabel("Number of Listings")
plt.title("Number of Cars by Body Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT / "eda_cars_by_type.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Fuel", order=df["Fuel"].value_counts().index)
plt.xlabel("Fuel Type")
plt.ylabel("Number of Listings")
plt.title("Number of Cars by Fuel Type")
plt.tight_layout()
plt.savefig(OUT / "eda_cars_by_fuel.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.sample(min(2500, len(df)), random_state=RANDOM_STATE),
    x="Distance", y="Price", hue="Type", alpha=0.65
)
plt.xlabel("Distance Travelled (km)")
plt.ylabel("Listed Price (INR)")
plt.title("Distance vs Used-Car Price")
plt.tight_layout()
plt.savefig(OUT / "eda_distance_vs_price.png", dpi=300)
plt.close()

corr = df[["Year", "Distance", "Owner", "Price"]].corr(numeric_only=True)
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(OUT / "eda_correlation_heatmap.png", dpi=300)
plt.close()

print("\nDATASET")
print("Rows:", len(df))
print("Columns:", len(required))
print("Missing values:\n", df[required].isna().sum())

# Shared features used by supervised learning.
# Car Name and Location are retained here because they can carry useful
# predictive information. One-hot encoding handles categorical variables.
numeric_features = ["Year", "Distance", "Owner"]
categorical_features = ["Car Name", "Fuel", "Location", "Drive", "Type"]
supervised_features = numeric_features + categorical_features

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])

# ------------------------------------------------------------
# 2. REVIEW 1 - REGRESSION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REVIEW 1: REGRESSION")
print("=" * 60)

X_reg = df[supervised_features]
y_reg = df["Price"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=RANDOM_STATE
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE,
        min_samples_leaf=2, n_jobs=-1
    )
}

regression_rows = []
regression_predictions = {}

for name, model in regression_models.items():
    pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])
    pipe.fit(X_train_r, y_train_r)
    pred = pipe.predict(X_test_r)
    regression_predictions[name] = pred

    mae = mean_absolute_error(y_test_r, pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, pred))
    r2 = r2_score(y_test_r, pred)

    regression_rows.append({
        "Model": name,
        "MAE_Rupees": mae,
        "RMSE_Rupees": rmse,
        "R2_Score": r2
    })

    # Save actual vs predicted values for review/report.
    pd.DataFrame({
        "Actual_Price": y_test_r.to_numpy(),
        "Predicted_Price": pred
    }).to_csv(
        OUT / (name.lower().replace(" ", "_") + "_predictions.csv"),
        index=False
    )

    # Save the fitted pipeline for later reuse.
    import joblib
    joblib.dump(pipe, OUT / (name.lower().replace(" ", "_") + ".joblib"))

reg_results = pd.DataFrame(regression_rows).sort_values("RMSE_Rupees")
reg_results.to_csv(OUT / "regression_comparison.csv", index=False)
print(reg_results.round(3).to_string(index=False))

best_reg_name = reg_results.iloc[0]["Model"]
best_reg_pred = regression_predictions[best_reg_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, best_reg_pred, alpha=0.55)
low = min(float(y_test_r.min()), float(np.min(best_reg_pred)))
high = max(float(y_test_r.max()), float(np.max(best_reg_pred)))
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.title(f"Actual vs Predicted Prices - {best_reg_name}")
plt.tight_layout()
plt.savefig(OUT / "regression_actual_vs_predicted.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=reg_results, x="Model", y="RMSE_Rupees")
plt.xticks(rotation=12)
plt.ylabel("RMSE (INR) - lower is better")
plt.title("Regression Model Comparison")
plt.tight_layout()
plt.savefig(OUT / "regression_model_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3. REVIEW 1 - CLASSIFICATION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REVIEW 1: CLASSIFICATION")
print("=" * 60)

# Derive target classes from price tertiles.
# Price is NOT included in X_class, avoiding direct target leakage.
# Quantile categories create approximately balanced classes.
df["Price_Category"] = pd.qcut(
    df["Price"], q=3,
    labels=["Budget", "Mid-range", "Premium"],
    duplicates="drop"
)
df["Price_Category"] = df["Price_Category"].astype(str)

X_class = df[supervised_features]
y_class = df["Price_Category"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.20, random_state=RANDOM_STATE,
    stratify=y_class
)

classification_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2500, class_weight="balanced"
    ),
    "Random Forest Classification": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE,
        min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1
    )
}

classification_rows = []
classification_predictions = {}

for name, model in classification_models.items():
    pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])
    pipe.fit(X_train_c, y_train_c)
    pred = pipe.predict(X_test_c)
    classification_predictions[name] = pred

    # Save a separate confusion matrix for each classification model.
    model_labels = sorted(y_class.unique())
    model_cm = confusion_matrix(y_test_c, pred, labels=model_labels)
    safe_model_name = name.lower().replace(" ", "_")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        model_cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model_labels, yticklabels=model_labels
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUT / f"{safe_model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    classification_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_c, pred),
        "Precision_Weighted": precision_score(
            y_test_c, pred, average="weighted", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_test_c, pred, average="weighted", zero_division=0
        ),
        "F1_Weighted": f1_score(
            y_test_c, pred, average="weighted", zero_division=0
        )
    })

    report = classification_report(y_test_c, pred, zero_division=0)
    with open(
        OUT / (name.lower().replace(" ", "_") + "_report.txt"),
        "w", encoding="utf-8"
    ) as f:
        f.write(report)

    pd.DataFrame({
        "Actual_Category": y_test_c.to_numpy(),
        "Predicted_Category": pred
    }).to_csv(
        OUT / (name.lower().replace(" ", "_") + "_predictions.csv"),
        index=False
    )

    import joblib
    joblib.dump(pipe, OUT / (name.lower().replace(" ", "_") + ".joblib"))

class_results = pd.DataFrame(classification_rows).sort_values(
    "F1_Weighted", ascending=False
)
class_results.to_csv(OUT / "classification_comparison.csv", index=False)
print(class_results.round(4).to_string(index=False))

best_class_name = class_results.iloc[0]["Model"]
best_class_pred = classification_predictions[best_class_name]
class_labels = sorted(y_class.unique())
cm = confusion_matrix(y_test_c, best_class_pred, labels=class_labels)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_labels, yticklabels=class_labels
)
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title(f"Confusion Matrix - {best_class_name}")
plt.tight_layout()
plt.savefig(OUT / "classification_confusion_matrix.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=class_results, x="Model", y="F1_Weighted")
plt.xticks(rotation=12)
plt.ylim(0, 1)
plt.ylabel("Weighted F1 Score")
plt.title("Classification Model Comparison")
plt.tight_layout()
plt.savefig(OUT / "classification_model_comparison.png", dpi=300)
plt.close()

# Price boundaries used for the derived classification target.
price_boundaries = df["Price"].quantile([0, 1/3, 2/3, 1]).to_dict()
pd.DataFrame({
    "Quantile": list(price_boundaries.keys()),
    "Price_Boundary_INR": list(price_boundaries.values())
}).to_csv(OUT / "price_category_boundaries.csv", index=False)

# ------------------------------------------------------------
# 4. REVIEW 2 - K-MEANS CLUSTERING
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REVIEW 2: K-MEANS CLUSTERING")
print("=" * 60)

# Clustering uses price and car specifications to segment the listings.
# Car Name and Location are excluded here to avoid high-cardinality
# sparse dimensions dominating the distance calculation.
cluster_numeric = ["Year", "Log_Distance", "Owner", "Price"]
cluster_categorical = ["Fuel", "Drive", "Type"]

cluster_df = df.copy()
cluster_df["Log_Distance"] = np.log1p(
    cluster_df["Distance"].clip(lower=0)
)
cluster_df["Log_Price"] = np.log1p(
    cluster_df["Price"].clip(lower=0)
)
cluster_numeric = ["Year", "Log_Distance", "Owner", "Log_Price"]

cluster_numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cluster_categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
cluster_preprocessor = ColumnTransformer([
    ("numeric", cluster_numeric_pipe, cluster_numeric),
    ("categorical", cluster_categorical_pipe, cluster_categorical)
])

X_cluster = cluster_preprocessor.fit_transform(
    cluster_df[cluster_numeric + cluster_categorical]
)
if hasattr(X_cluster, "toarray"):
    X_cluster = X_cluster.toarray()

sample_size = min(2500, len(X_cluster))
sample_idx = np.random.RandomState(RANDOM_STATE).choice(
    len(X_cluster), size=sample_size, replace=False
)
X_cluster_sample = X_cluster[sample_idx]

cluster_metric_rows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    sample_labels = km.fit_predict(X_cluster_sample)
    cluster_metric_rows.append({
        "K": k,
        "Inertia": km.inertia_,
        "Silhouette": silhouette_score(
            X_cluster_sample, sample_labels
        ),
        "Davies_Bouldin": davies_bouldin_score(
            X_cluster_sample, sample_labels
        ),
        "Calinski_Harabasz": calinski_harabasz_score(
            X_cluster_sample, sample_labels
        )
    })

cluster_metrics = pd.DataFrame(cluster_metric_rows)
cluster_metrics.to_csv(OUT / "clustering_metrics.csv", index=False)
best_k = int(
    cluster_metrics.loc[cluster_metrics["Silhouette"].idxmax(), "K"]
)

kmeans = KMeans(
    n_clusters=best_k, random_state=RANDOM_STATE, n_init=10
)
df["Cluster"] = kmeans.fit_predict(X_cluster)

cluster_summary = df.groupby("Cluster").agg(
    Number_of_Cars=("Price", "size"),
    Average_Price=("Price", "mean"),
    Median_Price=("Price", "median"),
    Average_Year=("Year", "mean"),
    Average_Distance=("Distance", "mean"),
    Average_Owners=("Owner", "mean")
).round(2)

# Descriptive names are assigned after clustering by median price.
ordered_clusters = cluster_summary["Median_Price"].sort_values().index.tolist()
cluster_name_map = {}
for rank, cluster_id in enumerate(ordered_clusters):
    if best_k == 2:
        label = ["Lower-priced segment", "Higher-priced segment"][rank]
    elif best_k == 3:
        label = ["Budget-oriented", "Mid-range", "Premium-oriented"][rank]
    else:
        label = f"Price Segment {rank + 1}"
    cluster_name_map[cluster_id] = label

df["Cluster_Name"] = df["Cluster"].map(cluster_name_map)
cluster_summary["Cluster_Name"] = cluster_summary.index.map(cluster_name_map)
cluster_summary.to_csv(OUT / "cluster_summary.csv")
df.to_csv(OUT / "cars_with_all_predictions_and_clusters.csv", index=False)

# Save K-Means bundle for external inference and web UI
import joblib
joblib.dump({
    "preprocessor": cluster_preprocessor,
    "model": kmeans,
    "best_k": best_k,
    "cluster_name_map": cluster_name_map
}, OUT / "kmeans_bundle.joblib")

# Full-data clustering metrics (silhouette sampled to limit runtime).
cluster_silhouette = silhouette_score(
    X_cluster, df["Cluster"],
    sample_size=min(2500, len(X_cluster)),
    random_state=RANDOM_STATE
)
cluster_dbi = davies_bouldin_score(X_cluster, df["Cluster"])
cluster_ch = calinski_harabasz_score(X_cluster, df["Cluster"])

print("Selected K:", best_k)
print(cluster_metrics.round(4).to_string(index=False))
print("\nCluster summary:\n", cluster_summary)

plt.figure(figsize=(8, 5))
plt.plot(cluster_metrics["K"], cluster_metrics["Inertia"], marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means")
plt.xticks(cluster_metrics["K"])
plt.tight_layout()
plt.savefig(OUT / "kmeans_elbow.png", dpi=300)
plt.close()

pca = PCA(n_components=2, random_state=RANDOM_STATE)
cluster_2d = pca.fit_transform(X_cluster)
plot_df = pd.DataFrame({
    "PC1": cluster_2d[:, 0],
    "PC2": cluster_2d[:, 1],
    "Cluster": df["Cluster"].astype(str).values
})
plot_df = plot_df.sample(min(3500, len(plot_df)), random_state=RANDOM_STATE)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df, x="PC1", y="PC2", hue="Cluster",
    palette="tab10", alpha=0.65, s=25
)
plt.title("K-Means Clusters - PCA Visualization")
plt.tight_layout()
plt.savefig(OUT / "kmeans_clusters_pca.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
sns.barplot(
    data=cluster_summary.reset_index(),
    x="Cluster", y="Median_Price"
)
plt.ylabel("Median Price (INR)")
plt.title("Median Price by K-Means Cluster")
plt.tight_layout()
plt.savefig(OUT / "cluster_median_prices.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 5. REVIEW 2 - GAUSSIAN HIDDEN MARKOV MODEL
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REVIEW 2: HIDDEN MARKOV MODEL")
print("=" * 60)

# Methodological limitation:
# The Year column is the vehicle's model/manufacturing year, not the
# date at which a historical price was recorded. Therefore, aggregate
# by vehicle Type and Year and model ordered year-cohort patterns.
# Do NOT describe this as historical price forecasting.

hmm_source = df.dropna(
    subset=["Year", "Price", "Distance", "Owner", "Type"]
).copy()
hmm_source["Year"] = hmm_source["Year"].round().astype(int)

cohorts = hmm_source.groupby(["Type", "Year"]).agg(
    Median_Price=("Price", "median"),
    Median_Distance=("Distance", "median"),
    Mean_Owners=("Owner", "mean"),
    Listing_Count=("Price", "size")
).reset_index()

cohorts["Log_Median_Price"] = np.log1p(cohorts["Median_Price"])
cohorts["Log_Median_Distance"] = np.log1p(
    cohorts["Median_Distance"].clip(lower=0)
)

hmm_features = [
    "Log_Median_Price", "Log_Median_Distance", "Mean_Owners"
]
hmm_scaler = StandardScaler()
cohorts[hmm_features] = hmm_scaler.fit_transform(
    cohorts[hmm_features]
)

sequences = []
sequence_lengths = []
sequence_meta = []

for car_type, group in cohorts.groupby("Type"):
    group = group.sort_values("Year").copy()
    arr = group[hmm_features].to_numpy(dtype=float)
    if len(arr) >= 3:
        sequences.append(arr)
        sequence_lengths.append(len(arr))
        sequence_meta.append(group[["Type", "Year"]].copy())

if len(sequences) < 2:
    raise ValueError(
        "Insufficient vehicle-type/year cohorts for HMM. "
        "At least two car types with three or more year cohorts are required."
    )

X_hmm = np.vstack(sequences)

hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=500,
    random_state=RANDOM_STATE,
    min_covar=1e-2,
    tol=1e-3
)
hmm.fit(X_hmm, lengths=sequence_lengths)

# Decode each car-type sequence separately to preserve sequence boundaries.
decoded = []
offset = 0
for meta, length in zip(sequence_meta, sequence_lengths):
    seq = X_hmm[offset:offset + length]
    state_sequence = hmm.predict(seq)
    part = meta.copy()
    part["Hidden_State"] = state_sequence
    decoded.append(part)
    offset += length

hmm_results = pd.concat(decoded, ignore_index=True)
hmm_results = hmm_results.merge(
    cohorts[[
        "Type", "Year", "Median_Price", "Median_Distance",
        "Mean_Owners", "Listing_Count"
    ]],
    on=["Type", "Year"], how="left"
)

state_prices = hmm_results.groupby(
    "Hidden_State"
)["Median_Price"].mean().sort_values()

state_name_map = {}
state_names = [
    "Budget-like regime",
    "Mid-market regime",
    "Premium-like regime"
]
for rank, state in enumerate(state_prices.index):
    state_name_map[state] = state_names[rank]

hmm_results["Regime_Name"] = hmm_results["Hidden_State"].map(
    state_name_map
)

hmm_summary = hmm_results.groupby(
    ["Hidden_State", "Regime_Name"]
).agg(
    Cohort_Observations=("Year", "size"),
    Average_Median_Price=("Median_Price", "mean"),
    Average_Median_Distance=("Median_Distance", "mean"),
    Average_Owners=("Mean_Owners", "mean")
).round(2)

hmm_results.to_csv(OUT / "hmm_cohort_hidden_states.csv", index=False)
hmm_summary.to_csv(OUT / "hmm_state_summary.csv")

hmm_log_likelihood = hmm.score(X_hmm, lengths=sequence_lengths)
pd.DataFrame({
    "Metric": ["Number of hidden states", "Log likelihood", "Cohort observations"],
    "Value": [3, hmm_log_likelihood, len(X_hmm)]
}).to_csv(OUT / "hmm_evaluation.csv", index=False)

print("HMM log likelihood:", round(hmm_log_likelihood, 4))
print("\nHMM state summary:\n", hmm_summary)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=hmm_results,
    x="Year", y="Median_Price",
    hue="Regime_Name", style="Type", s=100
)
plt.xlabel("Vehicle Manufacturing Year / Cohort")
plt.ylabel("Median Listed Price (INR)")
plt.title("HMM-Inferred Vehicle-Value Regimes by Cohort")
plt.tight_layout()
plt.savefig(OUT / "hmm_vehicle_value_regimes.png", dpi=300)
plt.close()

# Transition matrix visualization.
transition = pd.DataFrame(
    hmm.transmat_,
    index=[f"State {i}" for i in range(3)],
    columns=[f"State {i}" for i in range(3)]
)
transition.to_csv(OUT / "hmm_transition_matrix.csv")

plt.figure(figsize=(6, 5))
sns.heatmap(transition, annot=True, fmt=".3f", cmap="Blues")
plt.title("HMM Hidden-State Transition Probabilities")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.tight_layout()
plt.savefig(OUT / "hmm_transition_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 6. CONNECTED COMPARATIVE ANALYSIS
# ------------------------------------------------------------
# These algorithms solve complementary parts of the same car-pricing
# problem. Their raw metrics are not directly comparable across tasks.
comparison = pd.DataFrame([
    {
        "Review": "Review 1",
        "Algorithm": "Linear Regression",
        "Task": "Predict continuous car price",
        "Primary_Metric": "RMSE",
        "Result": float(reg_results.loc[
            reg_results["Model"] == "Linear Regression", "RMSE_Rupees"
        ].iloc[0])
    },
    {
        "Review": "Review 1",
        "Algorithm": "Random Forest Regression",
        "Task": "Predict continuous car price",
        "Primary_Metric": "RMSE",
        "Result": float(reg_results.loc[
            reg_results["Model"] == "Random Forest Regression", "RMSE_Rupees"
        ].iloc[0])
    },
    {
        "Review": "Review 1",
        "Algorithm": "Logistic Regression",
        "Task": "Classify price category",
        "Primary_Metric": "Weighted F1",
        "Result": float(class_results.loc[
            class_results["Model"] == "Logistic Regression", "F1_Weighted"
        ].iloc[0])
    },
    {
        "Review": "Review 1",
        "Algorithm": "Random Forest Classification",
        "Task": "Classify price category",
        "Primary_Metric": "Weighted F1",
        "Result": float(class_results.loc[
            class_results["Model"] == "Random Forest Classification", "F1_Weighted"
        ].iloc[0])
    },
    {
        "Review": "Review 2",
        "Algorithm": "K-Means",
        "Task": "Group similar car listings",
        "Primary_Metric": "Silhouette",
        "Result": float(cluster_silhouette)
    },
    {
        "Review": "Review 2",
        "Algorithm": "Gaussian HMM",
        "Task": "Infer hidden vehicle-value cohort regimes",
        "Primary_Metric": "Log likelihood",
        "Result": float(hmm_log_likelihood)
    }
])
comparison.to_csv(OUT / "integrated_algorithm_comparison.csv", index=False)

# ------------------------------------------------------------
# 7. AUTOMATIC RESULT SUMMARY FOR REPORT/PRESENTATION
# ------------------------------------------------------------
best_reg_row = reg_results.iloc[0]
best_class_row = class_results.iloc[0]
best_cluster_row = cluster_summary.loc[
    cluster_summary["Median_Price"].idxmax()
]
lowest_cluster_row = cluster_summary.loc[
    cluster_summary["Median_Price"].idxmin()
]
highest_state = hmm_summary["Average_Median_Price"].idxmax()
lowest_state = hmm_summary["Average_Median_Price"].idxmin()

summary_text = f"""
AUTOPRICE COMPASS - INTEGRATED PROJECT RESULTS
================================================

DATASET
- Records used: {len(df)}
- Original meaningful features: 9
- Target variable for regression: Price (INR)
- Classification target: Price_Category (Budget / Mid-range / Premium)
- Price category boundaries are in price_category_boundaries.csv.

REVIEW 1 - REGRESSION
- Best model by RMSE: {best_reg_row['Model']}
- MAE: INR {best_reg_row['MAE_Rupees']:,.2f}
- RMSE: INR {best_reg_row['RMSE_Rupees']:,.2f}
- R2: {best_reg_row['R2_Score']:.4f}
See regression_comparison.csv and regression_actual_vs_predicted.png.

REVIEW 1 - CLASSIFICATION
- Best model by weighted F1: {best_class_row['Model']}
- Accuracy: {best_class_row['Accuracy']:.4f}
- Weighted precision: {best_class_row['Precision_Weighted']:.4f}
- Weighted recall: {best_class_row['Recall_Weighted']:.4f}
- Weighted F1: {best_class_row['F1_Weighted']:.4f}
See classification_comparison.csv and classification_confusion_matrix.png.

REVIEW 2 - K-MEANS
- Selected K: {best_k}
- Silhouette (sampled): {cluster_silhouette:.4f}
- Davies-Bouldin: {cluster_dbi:.4f}
- Calinski-Harabasz: {cluster_ch:.4f}
- Highest-median-price cluster: {best_cluster_row['Cluster_Name']}
- Lowest-median-price cluster: {lowest_cluster_row['Cluster_Name']}
See cluster_summary.csv, kmeans_elbow.png and kmeans_clusters_pca.png.

REVIEW 2 - HMM
- Model: 3-state Gaussian HMM
- Log likelihood: {hmm_log_likelihood:.4f}
- Highest-average-price inferred state: {highest_state}
- Lowest-average-price inferred state: {lowest_state}
See hmm_state_summary.csv, hmm_vehicle_value_regimes.png and
hmm_transition_matrix.png.

IMPORTANT METHODOLOGICAL LIMITATION
The Year column represents vehicle manufacturing year, not the date of
a historical transaction. HMM is therefore used as exploratory
vehicle-year cohort hidden-state analysis, not as a validated future
price forecasting model. The dataset does not contain repeated
historical prices for the same vehicle.

INTEGRATED PROJECT CONCLUSION
Regression estimates a numerical used-car price. Classification
assigns a price segment. K-Means groups similar listings, while HMM
explores hidden value regimes across vehicle-year cohorts. Together,
the four approaches provide complementary views of the same used-car
pricing problem. They should be evaluated using task-appropriate
metrics rather than comparing unlike metric values directly.
"""

(OUT / "final_results_interpretation.txt").write_text(
    summary_text, encoding="utf-8"
)

print("\n" + "=" * 60)
print("INTEGRATED PROJECT COMPLETED")
print("=" * 60)
print("Best regression:", best_reg_row["Model"])
print("Best classification:", best_class_row["Model"])
print("Selected K for K-Means:", best_k)
print("HMM hidden states: 3")
print("All result files saved in:", OUT)
print("Read outputs/final_results_interpretation.txt for a summary.")

# ------------------------------------------------------------
# 8. DISPLAY ALL GENERATED GRAPHS
# ------------------------------------------------------------
# The graphs above are saved as PNG files. This gallery also displays
# them on screen in groups of four after the ML pipeline finishes.
def show_graph_gallery():
    graph_files = sorted(OUT.glob("*.png"))

    if not graph_files:
        print("No graph images were found in the outputs folder.")
        return

    print(f"\nDisplaying {len(graph_files)} graphs.")
    print("Close each graph-gallery window to move to the next group.")

    for start in range(0, len(graph_files), 4):
        batch = graph_files[start:start + 4]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for ax, graph_path in zip(axes, batch):
            image_data = plt.imread(graph_path)
            ax.imshow(image_data)
            ax.set_title(graph_path.stem.replace("_", " ").title(), fontsize=10)
            ax.axis("off")

        for ax in axes[len(batch):]:
            ax.axis("off")

        fig.suptitle(
            f"AutoPrice Compass - Graphs {start + 1} to {start + len(batch)} "
            f"of {len(graph_files)}",
            fontsize=15
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--no-gui" in sys.argv or "--headless" in sys.argv:
        print("\nHeadless mode: All graph images are preserved in outputs/.")
    else:
        show_graph_gallery()

