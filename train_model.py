import gc
import os

# Limitar el paralelismo de las librerías numéricas.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


DATASET_PATH = "Fraude.csv"
MODEL_PATH = "fraud_model.joblib"


def create_features(data):
    df = data.copy()

    # Eliminar índice exportado y variable prohibida.
    df = df.drop(
        columns=[
            "Unnamed: 0",
            "isFlaggedFraud"
        ],
        errors="ignore"
    )

    # Variables temporales.
    df["hourOfDay"] = (
        df["step"] - 1
    ) % 24

    df["day"] = (
        (df["step"] - 1) // 24
    ) + 1

    df["isNight"] = (
        df["hourOfDay"]
        .between(0, 5)
        .astype("int8")
    )

    # Transformación del monto.
    df["logAmount"] = np.log1p(
        df["amount"]
    )

    # Variaciones de saldo.
    df["origBalanceDelta"] = (
        df["newbalanceOrig"]
        - df["oldbalanceOrg"]
    )

    df["destBalanceDelta"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    # Saldos esperados.
    expected_origin = np.where(
        df["type"].eq("CASH_IN"),
        df["oldbalanceOrg"] + df["amount"],
        df["oldbalanceOrg"] - df["amount"]
    )

    expected_destination = np.where(
        df["type"].eq("CASH_IN"),
        df["oldbalanceDest"] - df["amount"],
        df["oldbalanceDest"] + df["amount"]
    )

    # Errores o inconsistencias contables.
    df["origBalanceError"] = (
        df["newbalanceOrig"]
        - expected_origin
    )

    df["destBalanceError"] = (
        df["newbalanceDest"]
        - expected_destination
    )

    # Indicadores de saldos iguales a cero.
    df["origZeroBefore"] = (
        df["oldbalanceOrg"]
        .eq(0)
        .astype("int8")
    )

    df["origZeroAfter"] = (
        df["newbalanceOrig"]
        .eq(0)
        .astype("int8")
    )

    df["destZeroBefore"] = (
        df["oldbalanceDest"]
        .eq(0)
        .astype("int8")
    )

    df["destZeroAfter"] = (
        df["newbalanceDest"]
        .eq(0)
        .astype("int8")
    )

    # Sólo se conserva el prefijo del destinatario.
    df["destPrefix"] = (
        df["nameDest"]
        .astype(str)
        .str[0]
        .fillna("U")
    )

    # Relaciones entre monto y saldo.
    df["amountToOrigBalance"] = (
        df["amount"]
        / (
            df["oldbalanceOrg"].abs()
            + 1.0
        )
    )

    df["amountToDestBalance"] = (
        df["amount"]
        / (
            df["oldbalanceDest"].abs()
            + 1.0
        )
    )

    # Estrategia liviana para el despliegue web.
    # No se guardan diccionarios de cuentas.
    df["origFrequency"] = 0.0
    df["destFrequency"] = 0.0

    # Eliminar identificadores crudos.
    return df.drop(
        columns=[
            "nameOrig",
            "nameDest"
        ],
        errors="ignore"
    )


def validate_dataset(data):
    required_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if data["isFraud"].nunique() < 2:
        raise ValueError(
            "isFraud contiene una sola clase. "
            "No se puede entrenar el modelo."
        )


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"No se encontró {DATASET_PATH} "
            "en la carpeta actual."
        )

    print("Cargando Fraude.csv...")

    raw = pd.read_csv(
        DATASET_PATH
    )

    validate_dataset(raw)

    print("Creando variables derivadas...")

    data = create_features(raw)

    del raw
    gc.collect()

    # División temporal.
    train = data[
        data["step"] <= 405
    ].copy()

    validation = data[
        data["step"].between(
            406,
            548
        )
    ].copy()

    del data
    gc.collect()

    if train.empty:
        raise ValueError(
            "El conjunto de entrenamiento quedó vacío."
        )

    if validation.empty:
        raise ValueError(
            "El conjunto de validación quedó vacío."
        )

    if train["isFraud"].nunique() < 2:
        raise ValueError(
            "Train no contiene ambas clases."
        )

    if validation["isFraud"].nunique() < 2:
        raise ValueError(
            "Validación no contiene ambas clases."
        )

    features = [
        column
        for column in train.columns
        if column != "isFraud"
    ]

    categorical_features = [
        "type",
        "destPrefix"
    ]

    numeric_features = [
        column
        for column in features
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                RobustScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_features
            )
        ],
        sparse_threshold=1.0
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=500,
                    random_state=42
                )
            )
        ]
    )

    # Utilizar todos los fraudes y una muestra de ocho
    # operaciones legítimas por cada fraude.
    positive_train = train[
        train["isFraud"].eq(1)
    ].copy()

    negative_train = train[
        train["isFraud"].eq(0)
    ]

    if positive_train.empty:
        raise ValueError(
            "Train no contiene casos de fraude."
        )

    negative_sample_size = min(
        len(negative_train),
        len(positive_train) * 8
    )

    negative_sample = negative_train.sample(
        n=negative_sample_size,
        random_state=42
    ).copy()

    fit_data = pd.concat(
        [
            positive_train,
            negative_sample
        ],
        ignore_index=True
    ).sample(
        frac=1.0,
        random_state=42
    )

    del positive_train
    del negative_train
    del negative_sample
    del train

    gc.collect()

    print(
        "Entrenando Regresión Logística con "
        f"{len(fit_data):,} registros..."
    )

    model.fit(
        fit_data[features],
        fit_data["isFraud"]
    )

    del fit_data
    gc.collect()

    print(
        "Seleccionando umbral con validación..."
    )

    validation_scores = model.predict_proba(
        validation[features]
    )[:, 1]

    precision, recall, thresholds = (
        precision_recall_curve(
            validation["isFraud"],
            validation_scores
        )
    )

    if len(thresholds) == 0:
        threshold = 0.5

    else:
        f1_values = (
            2
            * precision
            * recall
            / (
                precision
                + recall
                + 1e-12
            )
        )

        best_index = int(
            np.nanargmax(
                f1_values[:-1]
            )
        )

        threshold = float(
            thresholds[best_index]
        )

    del validation
    del validation_scores
    del precision
    del recall
    del thresholds

    gc.collect()

    # No guardar origin_frequency ni destination_frequency.
    artifact = {
        "model": model,
        "threshold": threshold,
        "features": features,
        "best_model": (
            "Regresion Logistica - version web liviana"
        ),
        "web_frequency_strategy": (
            "origFrequency y destFrequency "
            "se asignan a 0"
        )
    }

    print(
        "Guardando fraud_model.joblib..."
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3
    )

    model_size_mb = (
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024)
    )

    print(
        "Modelo generado correctamente."
    )

    print(
        f"Umbral: {threshold:.6f}"
    )

    print(
        f"Tamaño: {model_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    main()
