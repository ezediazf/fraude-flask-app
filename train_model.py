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

    # Columnas que no deben utilizarse
    df = df.drop(
        columns=["Unnamed: 0", "isFlaggedFraud"],
        errors="ignore"
    )

    # Variables temporales
    df["hourOfDay"] = (df["step"] - 1) % 24
    df["day"] = ((df["step"] - 1) // 24) + 1
    df["isNight"] = df["hourOfDay"].between(0, 5).astype("int8")

    # Transformación del monto
    df["logAmount"] = np.log1p(df["amount"])

    # Variaciones de saldos
    df["origBalanceDelta"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"]
    )

    df["destBalanceDelta"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # Saldos esperados según tipo de transacción
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

    # Errores o inconsistencias contables
    df["origBalanceError"] = (
        df["newbalanceOrig"] - expected_origin
    )

    df["destBalanceError"] = (
        df["newbalanceDest"] - expected_destination
    )

    # Indicadores de saldos iguales a cero
    zero_columns = {
        "origZeroBefore": "oldbalanceOrg",
        "origZeroAfter": "newbalanceOrig",
        "destZeroBefore": "oldbalanceDest",
        "destZeroAfter": "newbalanceDest"
    }

    for new_column, source_column in zero_columns.items():
        df[new_column] = df[source_column].eq(0).astype("int8")

    # Prefijo del destinatario
    df["destPrefix"] = (
        df["nameDest"]
        .astype(str)
        .str[0]
        .fillna("U")
    )

    # Ratios monto/saldo
    df["amountToOrigBalance"] = (
        df["amount"] / (df["oldbalanceOrg"].abs() + 1)
    )

    df["amountToDestBalance"] = (
        df["amount"] / (df["oldbalanceDest"].abs() + 1)
    )

    return df


def add_identifier_frequencies(
    data,
    origin_frequency,
    destination_frequency
):
    df = data.copy()

    df["origFrequency"] = (
        df["nameOrig"]
        .map(origin_frequency)
        .fillna(0)
    )

    df["destFrequency"] = (
        df["nameDest"]
        .map(destination_frequency)
        .fillna(0)
    )

    return df.drop(
        columns=["nameOrig", "nameDest"],
        errors="ignore"
    )


print("Cargando dataset...")

raw = pd.read_csv(DATASET_PATH)

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

missing_columns = required_columns.difference(raw.columns)

if missing_columns:
    raise ValueError(
        "Faltan columnas requeridas en Fraude.csv: "
        + ", ".join(sorted(missing_columns))
    )

if raw["isFraud"].nunique() < 2:
    raise ValueError(
        "La variable isFraud contiene una sola clase. "
        "No es posible entrenar un clasificador binario."
    )

print("Creando variables derivadas...")

data = create_features(raw)

# División temporal
train = data[data["step"] <= 405].copy()

validation = data[
    data["step"].between(406, 548)
].copy()

if train.empty:
    raise ValueError("El conjunto de entrenamiento quedó vacío.")

if validation.empty:
    raise ValueError("El conjunto de validación quedó vacío.")

if train["isFraud"].nunique() < 2:
    raise ValueError(
        "El bloque de entrenamiento no contiene ambas clases."
    )

if validation["isFraud"].nunique() < 2:
    raise ValueError(
        "El bloque de validación no contiene ambas clases."
    )

print("Calculando frecuencias de identificadores...")

origin_frequency = train["nameOrig"].value_counts(
    normalize=True
)

destination_frequency = train["nameDest"].value_counts(
    normalize=True
)

train = add_identifier_frequencies(
    train,
    origin_frequency,
    destination_frequency
)

validation = add_identifier_frequencies(
    validation,
    origin_frequency,
    destination_frequency
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

# Muestra de entrenamiento controlada para reducir memoria
positive_train = train[
    train["isFraud"].eq(1)
].copy()

negative_train = train[
    train["isFraud"].eq(0)
].copy()

negative_sample_size = min(
    len(negative_train),
    len(positive_train) * 8
)

negative_sample = negative_train.sample(
    n=negative_sample_size,
    random_state=42
)

fit_data = pd.concat(
    [positive_train, negative_sample],
    ignore_index=True
).sample(
    frac=1,
    random_state=42
)

print(
    "Entrenando Regresión Logística con "
    f"{len(fit_data):,} registros..."
)

model.fit(
    fit_data[features],
    fit_data["isFraud"]
)

print("Calculando umbral sobre validación...")

validation_scores = model.predict_proba(
    validation[features]
)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    validation["isFraud"],
    validation_scores
)

f1_values = (
    2 * precision * recall
    / (precision + recall + 1e-12)
)

if len(thresholds) == 0:
    threshold = 0.5
else:
    best_index = int(
        np.nanargmax(f1_values[:-1])
    )
    threshold = float(
        thresholds[best_index]
    )

artifact = {
    "model": model,
    "threshold": threshold,
    "features": features,
    "origin_frequency": origin_frequency.to_dict(),
    "destination_frequency": destination_frequency.to_dict(),
    "best_model": "Regresión Logística - versión web"
}

print("Guardando modelo liviano...")

joblib.dump(
    artifact,
    MODEL_PATH,
    compress=9
)

print("Modelo generado correctamente.")
print(f"Archivo: {MODEL_PATH}")
print(f"Umbral: {threshold:.6f}")
