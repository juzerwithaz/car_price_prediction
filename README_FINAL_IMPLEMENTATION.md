# AutoPrice Compass – Integrated Review 1 + Review 2

This project combines all four required ML components into one pipeline using the Cars24 used-car dataset.

## Covered requirements

### Review 1
1. Regression implementation:
   - Linear Regression
   - Random Forest Regression
   - Evaluation: MAE, RMSE, R²
2. Classification implementation:
   - Logistic Regression
   - Random Forest Classification
   - Evaluation: Accuracy, weighted Precision, weighted Recall, weighted F1, confusion matrix

### Review 2
3. K-Means clustering:
   - Preprocessing, feature scaling and categorical encoding
   - K selection by silhouette score (with elbow plot)
   - Evaluation: Silhouette, Davies-Bouldin, Calinski-Harabasz
   - Cluster profiles and PCA visualization
4. Gaussian Hidden Markov Model:
   - Three hidden states
   - Cohort observations grouped by vehicle Type and manufacturing Year
   - Observation features: median price, median distance, mean previous owners
   - Viterbi decoding and transition matrix
   - State interpretation based on median price

## Dataset facts
The supplied CSV contains 8,015 rows and 10 physical columns, including an unnamed CSV index. After excluding the index, it has nine meaningful features:
Car Name, Year, Distance, Owner, Fuel, Location, Drive, Type, Price.

The implementation removes the unnecessary unnamed index column.

## Important methodology notes

- Regression predicts continuous `Price` in INR.
- Classification predicts a price category derived from Price tertiles. Price itself is excluded from the classification input features to avoid target leakage.
- Supervised models use Car Name and Location as predictive categorical features. Missing categorical values are imputed and one-hot encoded.
- K-Means uses Year, log-transformed Distance, Owner, log-transformed Price, Fuel, Drive and Type. Car Name and Location are excluded from clustering to avoid high-cardinality sparse dimensions dominating distance calculations.
- The dataset's Year field is the vehicle manufacturing year, not the date of a price transaction. The HMM therefore performs exploratory hidden-state analysis over vehicle-year cohorts by car type. It must not be described as validated future-price forecasting or as a true chronological history of prices for individual cars.
- Metrics from different tasks are not directly comparable. Regression, classification, clustering and HMM each use task-appropriate evaluation measures.

## Setup

1. Install Python 3.10 or newer.
2. Keep `autoprice_compass.py` and `cars_24_combined.csv` in the same folder.
3. Open a terminal in that folder.
4. Install dependencies:

   `pip install -r requirements.txt`

5. Run:

   `python autoprice_compass.py`

## Generated outputs

All outputs are saved in the `outputs` folder, including:

### Dataset and Review 1 EDA
- `dataset_profile.csv`
- `dataset_overview.txt`
- `eda_price_distribution.png`
- `eda_cars_by_type.png`
- `eda_cars_by_fuel.png`
- `eda_distance_vs_price.png`
- `eda_correlation_heatmap.png`

### Review 1 – Regression
- `regression_comparison.csv`
- `regression_actual_vs_predicted.png`
- `regression_model_comparison.png`
- model prediction CSVs
- saved fitted pipelines (`.joblib`)

### Review 1 – Classification
- `classification_comparison.csv`
- `classification_confusion_matrix.png`
- `classification_model_comparison.png`
- classification reports and prediction CSVs
- `price_category_boundaries.csv`
- saved fitted pipelines (`.joblib`)

### Review 2 – Clustering
- `clustering_metrics.csv`
- `cluster_summary.csv`
- `cars_with_all_predictions_and_clusters.csv`
- `kmeans_elbow.png`
- `kmeans_clusters_pca.png`
- `cluster_median_prices.png`

### Review 2 – HMM
- `hmm_cohort_hidden_states.csv`
- `hmm_state_summary.csv`
- `hmm_evaluation.csv`
- `hmm_transition_matrix.csv`
- `hmm_vehicle_value_regimes.png`
- `hmm_transition_matrix.png`

### Integrated results
- `integrated_algorithm_comparison.csv`
- `final_results_interpretation.txt`

## Suggested report / PPT sequence

1. Title and project overview
2. Problem identification and motivation
3. Dataset source, attributes and dataset profile
4. Data preprocessing
5. Regression implementation and evaluation
6. Classification implementation and evaluation
7. K-Means algorithm and implementation
8. K-Means results and cluster interpretation
9. HMM theory and project-specific observation/state design
10. HMM implementation and results
11. Comparative analysis of all four approaches
12. Conclusion, applications and limitations

## Do not invent results
Run the program first. Use the actual values from `outputs` in the report and presentation. Do not copy example metrics or claim a model is best before checking the generated evaluation files.
