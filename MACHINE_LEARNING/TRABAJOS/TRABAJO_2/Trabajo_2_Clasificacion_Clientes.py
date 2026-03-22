"""
Tarea 2 - Clasificacion de clientes

Modelo de red neuronal para segmentar clientes usando la base
base_clientes.xlsx.

Idea del trabajo:
- Se ingieren datos del archivo Excel (ID_CLIENTE, TIPO_CLIENTE, ZONA,
    N_COMPRAS, MONTO_TOTAL, FECHA_ULTIMA_COMPRA).
- Se crean variables derivadas enriquecidas: DIAS_RECENCIA, MONTO_PROMEDIO
    y SCORE_COMERCIAL, que capturan patrones de comportamiento de compra.
- Se genera una etiqueta de negocio ("SEGMENTO") en 3 categorias (BAJO, MEDIO, ALTO)
    basada en el score comercial que pondera monto, frecuencia y recencia.
- Se entrena una red neuronal (MLPClassifier) para automatizar la clasificacion
    de clientes en estos segmentos, utilizando features numericas y categoricas.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


plt.style.use("seaborn-v0_8")
np.set_printoptions(precision=4, suppress=True)


def build_output_paths(base_dir: Path) -> dict:
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return {
        "output_dir": output_dir,
        "predicciones": output_dir / "predicciones_trabajo2.xlsx",
        "distribucion": output_dir / "01_distribucion_segmentos.png",
        "matriz_confusion": output_dir / "02_matriz_confusion.png",
        "perdida": output_dir / "03_perdida_entrenamiento.png",
        "resumen": output_dir / "resumen_trabajo2.txt",
    }


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Limpieza basica
    df["ID_CLIENTE"] = df["ID_CLIENTE"].fillna("").astype(str).str.strip()
    df["TIPO_CLIENTE"] = df["TIPO_CLIENTE"].fillna("DESCONOCIDO").astype(str).str.strip()
    df["ZONA"] = df["ZONA"].fillna("DESCONOCIDO").astype(str).str.strip()

    df["N_COMPRAS"] = pd.to_numeric(df["N_COMPRAS"], errors="coerce")
    df["MONTO_TOTAL"] = pd.to_numeric(df["MONTO_TOTAL"], errors="coerce")
    df["FECHA_ULTIMA_COMPRA"] = pd.to_datetime(df["FECHA_ULTIMA_COMPRA"], errors="coerce")

    # Fecha de referencia para calcular recencia
    fecha_referencia = df["FECHA_ULTIMA_COMPRA"].max() + pd.Timedelta(days=1)
    df["DIAS_RECENCIA"] = (fecha_referencia - df["FECHA_ULTIMA_COMPRA"]).dt.days

    # Variables derivadas simples
    df["MONTO_PROMEDIO"] = df["MONTO_TOTAL"] / df["N_COMPRAS"].replace(0, np.nan)

    # Si falta algun valor por division, lo dejamos en la mediana
    df["MONTO_PROMEDIO"] = df["MONTO_PROMEDIO"].fillna(df["MONTO_PROMEDIO"].median())

    # Puntaje comercial simple:
    # - mas compras y mas monto suman
    # - mas recencia resta
    score = (
        0.45 * np.log1p(df["MONTO_TOTAL"].fillna(df["MONTO_TOTAL"].median()))
        + 0.35 * df["N_COMPRAS"].fillna(df["N_COMPRAS"].median())
        - 0.20 * df["DIAS_RECENCIA"].fillna(df["DIAS_RECENCIA"].median())
    )
    df["SCORE_COMERCIAL"] = score

    # Etiqueta de negocio: bajo, medio, alto
    df["SEGMENTO"] = pd.qcut(
        df["SCORE_COMERCIAL"],
        q=3,
        labels=["BAJO", "MEDIO", "ALTO"],
    )

    return df


def make_model():
    numeric_features = ["N_COMPRAS", "MONTO_TOTAL", "DIAS_RECENCIA", "MONTO_PROMEDIO"]
    categorical_features = ["TIPO_CLIENTE", "ZONA"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def plot_class_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["SEGMENTO"].value_counts().reindex(["BAJO", "MEDIO", "ALTO"])
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Distribucion de segmentos")
    plt.xlabel("Segmento")
    plt.ylabel("Cantidad de clientes")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Prediccion",
        ylabel="Real",
        title="Matriz de confusion",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_training_loss(model: MLPClassifier, path: Path) -> None:
    if not hasattr(model, "loss_curve_"):
        return

    plt.figure(figsize=(7, 4))
    plt.plot(model.loss_curve_, color="#2c7fb8")
    plt.title("Perdida de entrenamiento")
    plt.xlabel("Iteracion")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "base_clientes.xlsx"
    paths = build_output_paths(base_dir)
    class_order = ["BAJO", "MEDIO", "ALTO"]

    df = load_data(input_file)
    df = prepare_data(df)

    feature_cols = ["N_COMPRAS", "MONTO_TOTAL", "DIAS_RECENCIA", "MONTO_PROMEDIO", "TIPO_CLIENTE", "ZONA"]
    X = df[feature_cols]
    y = df["SEGMENTO"]

    # Separacion simple para evaluacion
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipe = make_model()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_order,
        target_names=class_order,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_order)

    # Predicciones sobre toda la base
    df["SEGMENTO_PRED"] = pipe.predict(X)

    # Guardar resultado con probabilidad estimada de cada clase
    probas = pipe.predict_proba(X)
    classes = list(pipe.named_steps["model"].classes_)
    probas_df = pd.DataFrame(probas, columns=classes)
    probas_df = probas_df.reindex(columns=class_order)
    probas_df.columns = [f"PROB_{c}" for c in probas_df.columns]
    result = pd.concat([df, probas_df], axis=1)
    result.to_excel(paths["predicciones"], index=False)

    plot_class_distribution(df, paths["distribucion"])
    plot_confusion_matrix(y_test, y_pred, class_names=class_order, path=paths["matriz_confusion"])
    plot_training_loss(pipe.named_steps["model"], paths["perdida"])

    resumen = []
    resumen.append("TRABAJO 2 - CLASIFICACION DE CLIENTES")
    resumen.append("")
    resumen.append(f"Registros usados: {len(df)}")
    resumen.append(f"Variables de entrada: {', '.join(feature_cols)}")
    resumen.append("Etiqueta de negocio: SEGMENTO = BAJO / MEDIO / ALTO")
    resumen.append("")
    resumen.append(f"Accuracy test: {acc:.4f}")
    resumen.append("")
    resumen.append("Matriz de confusion:")
    resumen.append(str(cm))
    resumen.append("")
    resumen.append("Classification report:")
    resumen.append(report)

    summary_text = "\n".join(resumen)
    print(summary_text)

    with open(paths["resumen"], "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nArchivo guardado en: {paths['predicciones']}")
    print(f"Graficas guardadas en: {paths['output_dir']}")


if __name__ == "__main__":
    main()
