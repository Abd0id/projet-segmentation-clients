import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from preprocessing import PROCESSED_PATH

CLUSTER_NAMES = {
    0: "Premium Customers",
    1: "Low-Activity Customers",
    2: "At-Risk Customers",
}


def search_kmeans(X_full):
    rows = []
    for n in range(2, 6):
        X = X_full[:, :n]
        for k in range(2, 8):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            min_share = np.bincount(labels).min() / len(labels)
            rows.append({"n": n, "k": k, "sil": sil, "min_share": min_share})

    res = pd.DataFrame(rows)
    good = res[res["min_share"] >= 0.10]
    best = good.sort_values("sil", ascending=False).iloc[0]
    n_best = int(best["n"])
    k_best = int(best["k"])
    print("KMeans best: n =", n_best, ", k =", k_best)

    X_best = X_full[:, :n_best]
    km = KMeans(n_clusters=k_best, n_init=10, random_state=42)
    km.fit(X_best)
    return km, km.labels_, n_best


def search_dbscan(X_full):
    rows = []
    for n in range(2, 6):
        X = X_full[:, :n]
        for eps in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for ms in [5, 10, 20, 30]:
                labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    continue
                mask = labels != -1
                sil = silhouette_score(X[mask], labels[mask])
                noise = (labels == -1).mean()
                rows.append({"n": n, "eps": eps, "ms": ms, "sil": sil,
                             "n_clusters": n_clusters, "noise": noise})

    if not rows:
        return None, np.array([]), 0

    res = pd.DataFrame(rows).dropna().sort_values("sil", ascending=False)
    best = res.iloc[0]
    n_best = int(best["n"])
    print("DBSCAN best: n =", n_best, ", eps =", best["eps"], ", min_samples =", int(best["ms"]))

    X_best = X_full[:, :n_best]
    db = DBSCAN(eps=best["eps"], min_samples=int(best["ms"]))
    labels = db.fit_predict(X_best)
    return db, labels, n_best


def find_best_clusters(df_clean, df_pca):
    X_full = df_pca.values
    df_clean = df_clean.copy()

    km, km_labels, km_n = search_kmeans(X_full)
    df_clean["cluster_kmeans"] = km_labels

    db, db_labels, db_n = search_dbscan(X_full)
    if db is not None:
        df_clean["cluster_dbscan"] = db_labels
        df_clean["est_atypique_dbscan"] = db_labels == -1

    X_km = X_full[:, :km_n]
    km_sil = silhouette_score(X_km, km_labels)

    use_dbscan = False
    if db is not None:
        X_db = X_full[:, :db_n]
        mask = db_labels != -1
        db_sil = silhouette_score(X_db[mask], db_labels[mask])
        db_noise = (db_labels == -1).mean()
        use_dbscan = (db_sil > km_sil + 0.05) and (db_noise < 0.05)

    if use_dbscan:
        df_clean["cluster_final"] = db_labels
        print("Chosen method: DBSCAN")
    else:
        df_clean["cluster_final"] = km_labels
        print("Chosen method: K-means")

    df_clean["target"] = df_clean["cluster_kmeans"].map(CLUSTER_NAMES)
    print(df_clean["target"].value_counts(normalize=True).round(3))

    return df_clean


def run(df_clean, df_pca, out_path=PROCESSED_PATH):
    df_clean = find_best_clusters(df_clean, df_pca)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)
    print("enriched data saved:", out_path)
    return df_clean


if __name__ == "__main__":
    from preprocessing import PROCESSED_PATH, PCA_PATH
    df_c = pd.read_csv(PROCESSED_PATH)
    df_p = pd.read_csv(PCA_PATH)
    run(df_c, df_p)