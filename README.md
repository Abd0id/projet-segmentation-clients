# Customer Segmentation and Classification

## What this project does

This project segments credit card customers into behavior groups using unsupervised learning, then trains a supervised model to automatically predict the group of a new customer.

The dataset has around 9,000 customers with 7 behavioral variables over 6 months.

## Data dictionary

| Variable | Description |
| --- | --- |
| CUST_ID | Unique customer identifier |
| BALANCE | Remaining account balance |
| PURCHASES | Total purchases made |
| ONEOFF_PURCHASES | Largest single purchase |
| INSTALLMENTS_PURCHASES | Purchases paid in installments |
| CASH_ADVANCE | Cash withdrawals on the card |
| CREDIT_LIMIT | Credit limit of the customer |
| PAYMENTS | Total amount paid over the period |

## Project structure

```
projet-segmentation-clients/
├── data/
│   ├── raw/            # original dataset
│   ├── processed/      # cleaned data with cluster labels
│   └── PCA/            # PCA-reduced data for clustering
├── models/             # saved model (.joblib)
├── mlruns/             # MLflow tracking data
├── src/
│   └── projet_segmentation_clients/
│       ├── preprocessing.py   # data cleaning and PCA
│       ├── clustering.py      # KMeans and DBSCAN
│       ├── classification.py  # model training with MLflow
│       ├── pipeline.py        # runs all steps in order
│       └── app.py             # Streamlit app
└── README.md
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline manually:

```bash
python src/projet_segmentation_clients/pipeline.py
```

Launch the Streamlit app (runs the pipeline automatically if the model is missing):

```bash
streamlit run src/projet_segmentation_clients/app.py
```

View MLflow experiment results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## What each file does

- `preprocessing.py` reads the raw CSV, drops the one missing row, applies a log transform to normalize the skewed variables, standardizes them, and runs PCA. It saves a clean CSV and a PCA CSV.
- `clustering.py` runs a grid search over KMeans and DBSCAN with different numbers of components and parameters. It picks the best combination by silhouette score, adds cluster columns to the clean data, and maps clusters to human-readable names (Premium Customers, Low-Activity Customers, At-Risk Customers).
- `classification.py` trains four classifiers (Random Forest, SVM, Decision Tree, Logistic Regression) each inside a scikit-learn Pipeline with StandardScaler. Each run is logged to MLflow. The best Random Forest configuration is found with GridSearchCV and saved as a joblib file.
- `pipeline.py` calls preprocessing, clustering, and classification in order.
- `app.py` is a Streamlit app that loads the saved model and predicts the segment for a new customer. If the model file does not exist it runs the full pipeline first.