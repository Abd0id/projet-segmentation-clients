import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import ROOT

MODEL_PATH = ROOT / "models" / "pipeline_segmentation.joblib"
DROP_COLS = ["target", "cluster_kmeans", "cluster_dbscan",
             "est_atypique_dbscan", "cluster_final"]


def prepare_data(df_clean, test_size=0.2, random_state=42):
    X = df_clean.drop(columns=[c for c in DROP_COLS if c in df_clean.columns])
    X = X.select_dtypes("number")
    y = df_clean["target"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_pipelines():
    return [
        ("Random Forest", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
        ])),
        ("SVM", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(class_weight="balanced"))
        ])),
        ("Decision Tree", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=42))
        ])),
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))
        ])),
    ]


def run_mlflow_training(X_train, X_test, y_train, y_test, experiment="classification_clients"):
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment(experiment)

    tmp_dir = ROOT / "data" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name, pipe in build_pipelines():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted")

            mlflow.log_param("model", name)
            mlflow.log_params(pipe.named_steps["classifier"].get_params())
            mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)
            mlflow.sklearn.log_model(pipe, "model")

            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(name)
            img_path = tmp_dir / ("cm_" + name.replace(" ", "_") + ".png")
            fig.savefig(img_path)
            plt.close(fig)
            mlflow.log_artifact(str(img_path))

            report_path = tmp_dir / "report.txt"
            report_path.write_text(classification_report(y_test, pred))
            mlflow.log_artifact(str(report_path))

            print(name, "done, f1 =", round(f1, 3))


def tune_and_save(X_train, X_test, y_train, y_test, model_path=MODEL_PATH):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }
    grid = GridSearchCV(pipe_rf, param_grid, cv=5, scoring="f1_weighted")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 3))
    print("Test accuracy:", round(best_model.score(X_test, y_test), 3))

    joblib.dump(best_model, model_path)
    print("Model saved:", model_path)
    return best_model


def run(df_clean, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test = prepare_data(df_clean)
    run_mlflow_training(X_train, X_test, y_train, y_test)
    return tune_and_save(X_train, X_test, y_train, y_test, model_path)


if __name__ == "__main__":
    from preprocessing import PROCESSED_PATH
    df_c = pd.read_csv(PROCESSED_PATH)
    run(df_c)