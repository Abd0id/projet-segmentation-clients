# Project Brief — Customer Segmentation & Classification (Credit Card Customer Data)

## Project context

You are a junior AI developer on the data team of a financial institution that wants to build an intelligent customer segmentation system based on credit card usage behavior (balance, purchases, cash advances, payments, etc.).

The goal is twofold:

1. Discover, using unsupervised techniques, homogeneous behavior profiles among customers (segmentation).
2. Build a supervised model able to automatically predict the segment of a new customer, then deploy that model with experiment tracking (MLflow), orchestration (Airflow), and a user interface (Streamlit).

The dataset contains roughly 9,000 active customers, described by 7 behavioral variables over 6 months.

## Data dictionary

| Variable | Description |
| --- | --- |
| CUST_ID | Unique customer identifier |
| BALANCE | Remaining account balance available for purchases |
| PURCHASES | Total amount of purchases made by the customer |
| ONEOFF_PURCHASES | Amount of the largest purchase made in one go (one-off purchase) |
| INSTALLMENTS_PURCHASES | Amount of purchases paid in installments |
| CASH_ADVANCE | Amount of cash advances (cash withdrawals on the card) |
| CREDIT_LIMIT | Credit limit granted to the customer |
| PAYMENTS | Total amount paid by the customer over the period |

## Feature Story 1 — Exploratory Data Analysis (EDA)

**Objective:** understand the structure and quality of the data before any processing.

Tasks:

- Import the data; check types, dimensions, and a general overview.
- Identify missing values and duplicates.
- Analyze the distribution of numeric variables: histograms + displayed skewness, to spot asymmetric variables.
- Study correlations between variables (correlation matrix heatmap).
- Detect outliers on each numeric variable.

Visuals to produce:

- Histograms (+ KDE density curve) for each numeric variable, with the skewness coefficient shown in the title, so asymmetric variables are visible at a glance.
- Correlation matrix heatmap, to visualize relationships between features.
- Boxplots for each numeric variable, to visually spot outliers (IQR bounds).
- Summary table of missing values (count + %) per column, sorted in descending order.

## Feature Story 2 — Data preprocessing

**Objective:** produce a clean reference dataset, then a separate branch dedicated to clustering — never mixing the two, so the classification step (Feature Story 5) is not polluted by clustering-specific transformations.

Tasks:

- **Minimal cleaning (df_clean):** remove duplicates, drop useless columns. Save df_clean before any transformation: it is the shared starting point for clustering and, later, classification.
- **Clustering preparation** (on a dedicated copy, e.g. df_prepare_clustering, never directly on df_clean):
    1. Log transformation to correct the skewness of the 7 variables (positive amounts, right-skewed distribution).
    2. Standardization (StandardScaler), always after the log — never the other way around.
    3. Dimensionality reduction (PCA): run a full PCA, analyze explained variance per component and cumulative variance, then find the minimum number of components that exceeds an 80% cumulative variance threshold — this number is a first estimate, refined jointly with k in Feature Story 3.

Visuals to produce:

- Histograms after log transformation, to visually confirm the skew correction (skew approaching 0).
- Before/after table comparing the skewness of each variable, to objectively show the improvement.
- Cumulative explained variance curve from the PCA, with the 80% threshold marked, to show the number of components retained as a first estimate.

## Feature Story 3 — Training clustering models (K-means and DBSCAN)

**Objective:** jointly determine the number of PCA components and the number of clusters, without fixing them a priori, via a grid search.

Tasks:

- **K-means grid search:** on the transformed, PCA-reduced data (df_prepare_clustering), test several numbers of components (e.g. 2 to 5) crossed with several values of k (e.g. 2 to 7), computing inertia and silhouette for each combination. Summarize in a heatmap (components × k, value = silhouette), complemented by an elbow curve and a silhouette curve to confirm the choice of k. Check cluster balance (no class < ~10%) before validating.
- Train the selected model on df_prepare_clustering, then attach the result to df_clean as a cluster_kmeans column.
- **DBSCAN grid search:** same logic, with several (eps, min_samples) pairs crossed with the component counts; compute silhouette, number of clusters, and noise proportion; summarize in a table sorted by descending silhouette. Train with the best combination, attach the result to df_clean as a cluster_dbscan column (-1 = noise), and add a boolean column est_atypique_dbscan.

Choosing between the two methods:

- Highest silhouette between the two methods → indicates the better cluster separation.
- Point coverage: DBSCAN labels some customers as noise (-1), so if a significant share of noise remains, it does not cover all customers — a problem, since target must be defined for every customer, with no missing values.
- Cluster balance: check that neither method produces an overwhelming cluster (> 90%) or an overly small one (< 10%).
- In practice, K-means is recommended by default when all points must be classified (required for target) — DBSCAN is chosen instead only if it achieves a clearly higher silhouette and a very low noise proportion (e.g. < 5%).
- Once the method is chosen, create a cluster_final column in df_clean as a copy of the selected method's column (e.g. df_clean["cluster_final"] = df_clean["cluster_kmeans"]): this single column is the one used in Feature Story 4, with no ambiguity about the method.

<aside>
⚠️

**Watch out:** training (fit) is always done on df_prepare_clustering — never on df_clean, which stays in original values. Only the label columns (cluster_kmeans, cluster_dbscan, est_atypique_dbscan, cluster_final) are brought back onto df_clean.

</aside>

Visuals to produce:

- Heatmap of PCA components × k, with silhouette as the value.
- Elbow curve and silhouette curve (vs k).
- Table of the best DBSCAN combinations, sorted by descending silhouette.
- Comparison table K-means vs DBSCAN (silhouette, Davies-Bouldin, Calinski-Harabasz, % noise).
- Final scatter plot (PCA=2, chosen k), colored by cluster_final.
- K-means centroids overlaid on that scatter plot, to visualize each cluster's center in PCA space.

## Feature Story 4 — Cluster analysis and interpretation

**Objective:** give meaning to the cluster numbers (0, 1, 2...) — understand who these customers really are, so you can describe them in plain words.

Tasks:

- **Compute per-cluster means (standardized view):** for each cluster, compute the mean of each variable on the transformed/standardized dataset. This makes clusters easy to compare on the same scale and reveals at a glance the most extreme variables (positive or negative) for each cluster — those are what tell the cluster's story.
- **Compute the same per-cluster means (real-value view):** repeat the calculation on df_cleaned (actual, non-standardized values, with the same cluster_kmeans column already attached). This second table gives real amounts, essential for writing business-readable thresholds.

Example conditions derived from the two tables (illustrative figures; they depend on your own results):

| Cluster | Condition retained |
| --- | --- |
| 0 | PURCHASES > 2000 and PAYMENTS > 3000 and CREDIT_LIMIT > 6000 |
| 1 | CASH_ADVANCE < 100 and CREDIT_LIMIT < 3000 and PAYMENTS < 700 |
| 2 | CASH_ADVANCE > 1500 and PURCHASES < 100 |
- Also check carefully the clusters that look alike on one variable but diverge on another.
- **Name each cluster:** from the conditions above, pick a clear and consistent name for each cluster (e.g. cluster 0 → "Premium Customers"; cluster 1 → "Low-Activity Customers"; cluster 2 → "At-Risk Customers"), explaining each name in 2–3 sentences describing the segment's typical behavior.
- **Create the final column:** map each cluster number to the chosen business name, then apply it to df_cleaned to create the target column, verifying that no cluster number was forgotten, that target has no missing values, and that its proportions match those of the original clusters.

<aside>
⚠️

**Watch out:** the number K-means assigns to each cluster (0, 1, 2...) has no fixed meaning — it can change between training runs, even with the same parameters. Always rebuild both mean tables (standardized + real) on your own result before assigning names and thresholds, rather than reusing those from another run.

</aside>

Visuals to produce:

- Heatmap of per-cluster means (standardized view), to see at a glance what differentiates each group.
- Bar chart of each cluster's size (number of customers and %).
- Small bar charts comparing clusters on key variables (CASH_ADVANCE, PURCHASES, PAYMENTS), in real values.

## Feature Story 5 — Supervised classification and model evaluation

**Objective:** automatically predict the segment (target) of a new customer.

Tasks:

- **Starting point:** df_clean + target — cleaned but not transformed/standardized/PCA-reduced (those steps are clustering-specific and would cause data leakage for classification).
- Define X (behavioral variables) and y (target), excluding target and any clustering column (cluster_kmeans, cluster_dbscan, etc.).
- Stratified train/test split, handling class imbalance if needed (over/under-sampling via imblearn).
- Build a single scikit-learn pipeline (Pipeline([("scaler", StandardScaler()), ("classifier", ...)])), grouping preprocessing and model into one object, to guarantee the same steps are applied at training and prediction time.
- Train several versions of this pipeline, changing only the classifier step: Random Forest, SVM, Decision Tree, Logistic Regression.
- Evaluate each pipeline (confusion matrix, precision, recall, F1-score), validate with cross-validation, optimize via GridSearchCV/RandomizedSearchCV applied directly to the full pipeline — preprocessing must be re-fitted on each fold, not once upfront.
- Compare the models and select the best one.
- Save the complete pipeline (preprocessing + model) with joblib: this single object is what Streamlit will reload.

<aside>
⚠️

**Critical watch-out:** preprocessing (standardization, etc.) must be learned only on X_train, inside the pipeline, then applied to X_test — never the other way around, or the evaluation will be biased.

</aside>

Visuals to produce:

- Confusion matrix (annotated heatmap) for each model tested, to immediately see well/poorly predicted classes.
- Model comparison table (accuracy, precision, recall, F1-score, training time), sorted by the metric most relevant to the use case.
- Bar chart comparing the F1-score (or accuracy) of all tested models, to visually identify the best one.
- Feature importance plot (feature_importances_ for Random Forest, or coefficients for Logistic Regression), to identify the variables that most determine the predicted segment.
- Cross-validation curves (boxplot of per-fold scores), to visualize the stability and robustness of the selected model.

## Feature Story 6 — Experiment tracking with MLflow

**Objective:** track each classification training run, to objectively compare tested configurations and keep a record of what was tried (introductory level, focused on classification only — clustering is not tracked at this stage).

Tasks:

- **Initialize the experiment:** create a dedicated tracking space with mlflow.set_experiment("classification_clients"), to group all runs of this project in one place.
- **Start one run per training:** wrap each model training in a with mlflow.start_run(): block so MLflow automatically captures everything that happens inside.
- **Log the hyperparameters tested** (mlflow.log_param): model type, tree depth, number of estimators — any parameter that varies between runs and explains a difference in results.
- **Log the metrics obtained** (mlflow.log_metric): accuracy, precision, recall, F1-score — one value per metric per run, so they can be sorted and compared in the UI.
- **Log the trained pipeline** (mlflow.sklearn.log_model): save the complete pipeline object (preprocessing + model) directly in MLflow, not just the metrics — this lets you retrieve and reuse any run later without retraining.
- **Log additional artifacts:** the confusion matrix (image) and/or the full classification report (text), via mlflow.log_artifact, to keep a visual trace of model behavior, not just a score.
- **Compare runs visually:** launch the UI (mlflow ui), which shows all runs as a sortable table — useful to spot the best configuration by the chosen metric (e.g. sort by descending F1-score).

Visuals to produce:

- Screenshot (or export) of the run comparison table in the mlflow ui, sorted by the chosen metric.
- Confusion matrix logged as an artifact for the best model's run.

## Feature Story 7 — Streamlit application (demo / delivery)

**Objective:** demonstrate the model in real conditions.

Tasks:

- Build an app that lets you enter or upload a customer profile, predict its segment (target), and display the result (segment name, per-class probabilities if available).
- Load exclusively the single scikit-learn pipeline saved in Feature Story 5 — never the model alone, and never re-code the preprocessing manually in Streamlit (a simple pipeline.predict(new_raw_customer) is enough).

## Feature Story 8 — Documentation and reproducibility

**Objective:** ensure the project is understandable and reproducible by a third party.

Tasks:

- Comment the code (Markdown cells if using a notebook), especially non-trivial decisions (choice of k, eps, input dataset for classification, business thresholds).
- Split the code into reusable modules ([extraction.py](http://extraction.py), [preprocessing.py](http://preprocessing.py), [clustering.py](http://clustering.py), [classification.py](http://classification.py), [pipeline.py](http://pipeline.py) (bonus)) rather than one monolithic notebook.
- Provide a [README.md](http://README.md): project description, repository structure, installation, pipeline execution, launching MLflow/Streamlit.
- Plan the work in Jira (Epics + detailed tickets, effort estimates).

## Project structure

```
projet-segmentation-clients/
├── data/
│   ├── raw/          # original dataset
│   └── processed/    # df_clean, df_prepare_clustering
├── notebooks/        # exploration, prototyping
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── pipeline.py
├── models/           # serialized pipelines (.joblib)
├── app/
│   └── app.py        # Streamlit application
├── dags/             # Airflow DAG (bonus)
├── mlruns/           # MLflow tracking
├── pyproject.toml    # project dependencies and config (uv)
├── uv.lock           # pinned versions, guarantees reproducibility
└── README.md
```

## Bonus (optional)

- **Pipeline orchestration with Airflow:** model the full pipeline (ingestion → preprocessing → clustering → labeling → classification → MLflow tracking) as a DAG, split into independent tasks with defined dependencies and execution frequency, error/retry handling, and monitoring through the Airflow UI.

[Concepts to Research — Detailed Documentation](Project%20Brief%20%E2%80%94%20Customer%20Segmentation%20&%20Classifica/Concepts%20to%20Research%20%E2%80%94%20Detailed%20Documentation%20598ebcf123c6472984b09d7b001ccefc.md)