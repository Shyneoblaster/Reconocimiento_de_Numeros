import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def cargar_test_mnist():
    """Carga y preprocesa el set de prueba de MNIST igual que en entrenamiento."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
    return x_test, y_test


def calcular_matriz_confusion(y_real, y_pred):
    """Devuelve matriz de confusion absoluta y normalizada por fila."""
    matriz = tf.math.confusion_matrix(y_real, y_pred, num_classes=10).numpy()

    # Evita division por cero en filas sin elementos.
    sumas_fila = matriz.sum(axis=1, keepdims=True)
    matriz_norm = np.divide(
        matriz.astype(np.float32),
        sumas_fila,
        out=np.zeros_like(matriz, dtype=np.float32),
        where=sumas_fila != 0,
    )
    return matriz, matriz_norm


def guardar_csvs(matriz, matriz_norm, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_abs = os.path.join(carpeta_salida, "matriz_confusion.csv")
    ruta_norm = os.path.join(carpeta_salida, "matriz_confusion_normalizada.csv")

    np.savetxt(ruta_abs, matriz, fmt="%d", delimiter=",", header="0,1,2,3,4,5,6,7,8,9", comments="")
    np.savetxt(
        ruta_norm,
        matriz_norm,
        fmt="%.6f",
        delimiter=",",
        header="0,1,2,3,4,5,6,7,8,9",
        comments="",
    )

    return ruta_abs, ruta_norm


def guardar_imagen(matriz, carpeta_salida):
    """Genera una imagen PNG de la matriz si matplotlib esta disponible."""
    ruta_png = os.path.join(carpeta_salida, "matriz_confusion.png")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title("Matriz de confusion (MNIST)")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz[i, j]
            color = "white" if valor > matriz.max() * 0.5 else "black"
            ax.text(j, i, str(valor), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)
    return ruta_png


def main():
    parser = argparse.ArgumentParser(description="Genera la matriz de confusion para un modelo MNIST guardado.")
    parser.add_argument("--modelo", default="modelo_mnist.keras", help="Ruta al archivo del modelo .keras")
    parser.add_argument("--salida", default="resultados", help="Carpeta de salida para CSV/PNG")
    args = parser.parse_args()

    print(f"Cargando modelo desde: {args.modelo}")
    modelo = load_model(args.modelo)

    print("Cargando datos de prueba MNIST...")
    x_test, y_test = cargar_test_mnist()

    print("Calculando predicciones...")
    predicciones = modelo.predict(x_test, verbose=0)
    y_pred = np.argmax(predicciones, axis=1)

    matriz, matriz_norm = calcular_matriz_confusion(y_test, y_pred)
    precision = float(np.mean(y_pred == y_test))

    ruta_abs, ruta_norm = guardar_csvs(matriz, matriz_norm, args.salida)
    ruta_png = guardar_imagen(matriz, args.salida)

    print(f"Accuracy en test: {precision * 100:.2f}%")
    print(f"Matriz absoluta guardada en: {ruta_abs}")
    print(f"Matriz normalizada guardada en: {ruta_norm}")
    if ruta_png:
        print(f"Imagen guardada en: {ruta_png}")
    else:
        print("No se pudo generar la imagen (matplotlib no esta instalado).")


if __name__ == "__main__":
    main()

