import tensorflow as tf
from tensorflow.keras import layers, models

class EntrenadorCNN:
    def __init__(self, ruta_modelo="modelo_mnist.keras"):
        self.ruta_modelo = ruta_modelo
        self.modelo = None

    def entrenar_y_guardar(self):
        print("Cargando dataset MNIST...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocesamiento: MNIST tiene imágenes de 28x28.
        # Las redimensionamos para indicar que tienen 1 canal (escala de grises) y normalizamos (0 a 1).
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

        print("Construyendo el modelo CNN...")
        self.modelo = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax') # 10 clases (dígitos del 0 al 9)
        ])

        self.modelo.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        print("Entrenando el modelo...")
        self.modelo.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        print(f"Guardando el modelo en: {self.ruta_modelo}")
        self.modelo.save(self.ruta_modelo)
        print("¡Modelo guardado con éxito!")

if __name__ == "__main__":
    # Ejecuta el entrenamiento
    entrenador = EntrenadorCNN("modelo_mnist.keras")
    entrenador.entrenar_y_guardar()