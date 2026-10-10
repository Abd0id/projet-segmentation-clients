import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH       = ROOT / "data" / "raw" / "dataset-6ab989b678d90892980305.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "card_profiler_clean.csv"
PCA_PATH       = ROOT / "data" / "PCA" / "card_profiler_clustering.csv"

NUMERIC_COLS = [
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
]


def load_and_clean(raw_path=RAW_PATH, out_path=PROCESSED_PATH):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_path)
    df.dropna(subset=["CREDIT_LIMIT"], inplace=True)
    df.to_csv(out_path, index=False)
    print("clean data saved:", out_path)
    return df


def apply_pca(df, out_path=PCA_PATH, variance_threshold=0.80):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_log = pd.DataFrame()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df_log[col + "_log"] = np.log1p(df[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_log)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative >= variance_threshold) + 1)
    print("number of PCA components:", n_components)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    cols = ["PC" + str(i + 1) for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=cols)

    df_pca.to_csv(out_path, index=False)
    print("PCA data saved:", out_path)
    return df_pca


def run(raw_path=RAW_PATH, processed_path=PROCESSED_PATH, pca_path=PCA_PATH):
    df_clean = load_and_clean(raw_path, processed_path)
    df_pca = apply_pca(df_clean, pca_path)
    return df_clean, df_pca


if __name__ == "__main__":
    run()