import preprocessing
import clustering
import classification
from preprocessing import RAW_PATH, PROCESSED_PATH, PCA_PATH
from classification import MODEL_PATH


def run_full_pipeline():
    print("Step 1 - Preprocessing")
    df_clean, df_pca = preprocessing.run(RAW_PATH, PROCESSED_PATH, PCA_PATH)

    print("Step 2 - Clustering")
    df_clean = clustering.run(df_clean, df_pca, PROCESSED_PATH)

    print("Step 3 - Classification")
    classification.run(df_clean, MODEL_PATH)

    print("Pipeline done. Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    run_full_pipeline()