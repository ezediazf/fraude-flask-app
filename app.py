import gc
import os

# Limitar memoria y paralelismo antes de importar librerías numéricas
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.joblib")

app = Flask(__name__)

MODEL_ARTIFACT = None
MODEL_ERROR = None


def load_model():
    global MODEL_ARTIFACT
    global MODEL_ERROR

    if not os.path.exists(MODEL_PATH):
        MODEL_ERROR = "No se encontró fraud_model.joblib."
        return

    try:
        MODEL_ARTIFACT = joblib.load(MODEL_PATH)

        required_keys = {
            "model",
            "threshold",
            "features",
            "best_model"
        }

        missing_keys = required_keys.difference(
            MODEL_ARTIFACT.keys()
        )

        if missing_keys:
            raise ValueError(
                "Faltan claves en el modelo: "
                + ", ".join(sorted(missing_keys))
            )

        try:
            classifier = (
                MODEL_ARTIFACT["model"]
                .named_steps["model"]
            )

            if hasattr(classifier, "n_jobs"):
                classifier.n_jobs = 1

        except (KeyError, AttributeError, TypeError):
            pass

        MODEL_ERROR = None
        gc.collect()

    except Exception as error:
        MODEL_ARTIFACT = None
        MODEL_ERROR = (
            f"No fue posible cargar el modelo: {error}"
        )


load_model()


VALID_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
}


def parse_payload(payload):
    if not payload:
        raise ValueError(
            "No se recibieron datos para evaluar."
        )

    required_fields = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in payload
        or str(payload[field]).strip() == ""
    ]

    if missing_fields:
        raise ValueError(
            "Faltan campos: "
            + ", ".join(missing_fields)
        )

    transaction_type = str(
        payload["type"]
    ).strip().upper()

    if transaction_type not in VALID_TYPES:
        raise ValueError(
            "Tipo de transacción inválido."
        )

    parsed = {
        "step": int(float(payload["step"])),
        "type": transaction_type,
        "amount": float(payload["amount"]),
        "nameOrig": str(
            payload["nameOrig"]
        ).strip(),
        "oldbalanceOrg": float(
            payload["oldbalanceOrg"]
        ),
        "newbalanceOrig": float(
            payload["newbalanceOrig"]
        ),
        "nameDest": str(
            payload["nameDest"]
        ).strip(),
        "oldbalanceDest": float(
            payload["oldbalanceDest"]
        ),
        "newbalanceDest": float(
            payload["newbalanceDest"]
        )
    }

    numeric_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    for field in numeric_fields:
        if not np.isfinite(parsed[field]):
            raise ValueError(
                f"El campo {field} debe ser numérico y finito."
            )

    if parsed["step"] < 1:
        raise ValueError(
            "Step debe ser mayor o igual a 1."
        )

    if parsed["amount"] < 0:
        raise ValueError(
            "El monto no puede ser negativo."
        )

    return parsed


def prepare_transaction(payload):
    data = pd.DataFrame([payload])

    data["hourOfDay"] = (
        data["step"] - 1
    ) % 24

    data["day"] = (
        (data["step"] - 1) // 24
    ) + 1

    data["isNight"] = (
        data["hourOfDay"]
        .between(0, 5)
        .astype("int8")
    )

    data["logAmount"] = np.log1p(
        data["amount"]
    )

    data["origBalanceDelta"] = (
        data["newbalanceOrig"]
        - data["oldbalanceOrg"]
    )

    data["destBalanceDelta"] = (
        data["newbalanceDest"]
        - data["oldbalanceDest"]
    )

    expected_origin = np.where(
        data["type"].eq("CASH_IN"),
        data["oldbalanceOrg"] + data["amount"],
        data["oldbalanceOrg"] - data["amount"]
    )

    expected_destination = np.where(
        data["type"].eq("CASH_IN"),
        data["oldbalanceDest"] - data["amount"],
        data["oldbalanceDest"] + data["amount"]
    )

    data["origBalanceError"] = (
        data["newbalanceOrig"]
        - expected_origin
    )

    data["destBalanceError"] = (
        data["newbalanceDest"]
        - expected_destination
    )

    data["origZeroBefore"] = (
        data["oldbalanceOrg"]
        .eq(0)
        .astype("int8")
    )

    data["origZeroAfter"] = (
        data["newbalanceOrig"]
        .eq(0)
        .astype("int8")
    )

    data["destZeroBefore"] = (
        data["oldbalanceDest"]
        .eq(0)
        .astype("int8")
    )

    data["destZeroAfter"] = (
        data["newbalanceDest"]
        .eq(0)
        .astype("int8")
    )

    data["destPrefix"] = (
        data["nameDest"]
        .astype(str)
        .str[0]
        .fillna("U")
    )

    data["amountToOrigBalance"] = (
        data["amount"]
        / (
            data["oldbalanceOrg"].abs()
            + 1.0
        )
    )

    data["amountToDestBalance"] = (
        data["amount"]
        / (
            data["oldbalanceDest"].abs()
            + 1.0
        )
    )

    # Versión liviana sin diccionarios de cuentas
    data["origFrequency"] = 0.0
    data["destFrequency"] = 0.0

    data = data.drop(
        columns=[
            "nameOrig",
            "nameDest"
        ],
        errors="ignore"
    )

    expected_features = MODEL_ARTIFACT["features"]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "No se pudieron crear estas variables: "
            + ", ".join(missing_features)
        )

    return data[expected_features]


def predict_transaction(payload):
    if MODEL_ARTIFACT is None:
        raise RuntimeError(
            MODEL_ERROR
            or "El modelo no está disponible."
        )

    parsed_payload = parse_payload(payload)

    prepared_data = prepare_transaction(
        parsed_payload
    )

    try:
        probability = float(
            MODEL_ARTIFACT["model"]
            .predict_proba(prepared_data)[0, 1]
        )

        threshold = float(
            MODEL_ARTIFACT["threshold"]
        )

        classification = (
            "ALERTA"
            if probability >= threshold
            else "SIN ALERTA"
        )

        return {
            "clasificacion": classification,
            "probabilidad_fraude": probability,
            "probabilidad_porcentaje": round(
                probability * 100,
                2
            ),
            "umbral": threshold,
            "modelo": MODEL_ARTIFACT["best_model"]
        }

    finally:
        del prepared_data
        gc.collect()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            result = predict_transaction(
                request.form.to_dict()
            )

        except Exception as error:
            result = {
                "error": str(error)
            }

    return render_template(
        "index.html",
        result=result,
        ready=MODEL_ARTIFACT is not None,
        model_error=MODEL_ERROR
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(
            force=True
        )

        result = predict_transaction(
            payload
        )

        return jsonify(result), 200

    except Exception as error:
        status_code = (
            503
            if MODEL_ARTIFACT is None
            else 400
        )

        return jsonify(
            {
                "error": str(error)
            }
        ), status_code


@app.route("/health", methods=["GET"])
def health():
    if MODEL_ARTIFACT is None:
        return jsonify(
            {
                "status": "model_missing",
                "error": MODEL_ERROR
            }
        ), 503

    return jsonify(
        {
            "status": "ok",
            "model": MODEL_ARTIFACT["best_model"],
            "frequency_strategy": (
                MODEL_ARTIFACT.get(
                    "web_frequency_strategy",
                    "origFrequency y destFrequency iguales a cero"
                )
            )
        }
    ), 200


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )