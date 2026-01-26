import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# base do diretorio projeto fora da pasta models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#setta o caminho do modelo, e do dataset usado
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "librai_rf.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

def main():
   
#le e carrega o csv para um dataframe
    df = pd.read_csv(DATASET_PATH)

    # checagem de balanceamento das amostras
    counts = df["label"].value_counts()
    print("Amostras por classe:")
    print(counts)
    print()
#separa features de rotulos
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(str)
#splita os dados para teste e treino, e distribui de forma correta
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
#define o modelo de arvore de decisao randomforest
    model = RandomForestClassifier(
        n_estimators=400, #definicao de numero de "arvores" do random forest
        random_state=42,    #aleatoriedade interna do modelo em 42 para garantir reprodutibilidade e o mesmo modelo se rodar denovo
        n_jobs=-1, #usa todos os nucleos da cpu
        class_weight="balanced"
    )
#execução completa do random forest e print do report de classificação
    print("Treinando RandomForest...")
    model.fit(X_train, y_train)

    print("\nAvaliando...")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix (labels ordenados alfabeticamente):")
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("Labels:", labels_sorted)
    print(cm)

    joblib.dump(model, MODEL_PATH)
  
if __name__ == "__main__":
    main()