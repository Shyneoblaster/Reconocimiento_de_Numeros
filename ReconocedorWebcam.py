import cv2
import numpy as np
from tensorflow.keras.models import load_model


class ReconocedorWebcam:
    def __init__(self, ruta_modelo="modelo_mnist.keras"):
        print(f"Cargando modelo desde {ruta_modelo}...")
        self.modelo = load_model(ruta_modelo)

    def iniciar_reconocimiento(self):
        cap = cv2.VideoCapture(0)
        print("Cámara iniciada. Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al acceder a la cámara.")
                break

            # Obtener las dimensiones de la pantalla
            alto, ancho = frame.shape[:2]

            # 1. DEFINIR LA CAJA DE ESCANEO EN EL CENTRO
            tamano_caja = 300
            x_caja = int((ancho - tamano_caja) / 2)
            y_caja = int((alto - tamano_caja) / 2)

            # Dibujar el marco de la caja en la pantalla principal (Color Azul)
            cv2.rectangle(frame, (x_caja, y_caja), (x_caja + tamano_caja, y_caja + tamano_caja), (255, 0, 0), 2)
            cv2.putText(frame, "", (x_caja, y_caja - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

            # 2. RECORTAR LA IMAGEN (Solo analizamos el interior de la caja)
            area_escaneo = frame[y_caja:y_caja + tamano_caja, x_caja:x_caja + tamano_caja]

            # Convertir a grises y desenfocar SOLO el área de escaneo
            gray = cv2.cvtColor(area_escaneo, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            # Filtro Otsu: Excelente para separar tinta negra de papel blanco
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Buscar contornos solo en el cuadro recortado
            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contorno in contornos:
                # Filtrar manchitas pequeñas
                if cv2.contourArea(contorno) > 500:
                    x, y, w, h = cv2.boundingRect(contorno)

                    # Añadir margen para que se parezca al dataset MNIST
                    margen = 15
                    y1 = max(0, y - margen)
                    y2 = min(thresh.shape[0], y + h + margen)
                    x1 = max(0, x - margen)
                    x2 = min(thresh.shape[1], x + w + margen)

                    roi = thresh[y1:y2, x1:x2]

                    if roi.shape[0] == 0 or roi.shape[1] == 0:
                        continue

                    try:
                        # Preparar la imagen para el modelo
                        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        roi_normalized = roi_resized / 255.0
                        roi_reshaped = np.reshape(roi_normalized, (1, 28, 28, 1))

                        # Predecir
                        prediccion = self.modelo.predict(roi_reshaped, verbose=0)
                        numero_detectado = np.argmax(prediccion)
                        confianza = np.max(prediccion)

                        # Si está muy seguro, lo mostramos
                        if confianza > 0.8:
                            # Dibujamos el cuadro verde, pero sumando las coordenadas de la caja principal
                            # para que aparezca en el lugar correcto de la pantalla completa
                            cv2.rectangle(frame, (x_caja + x1, y_caja + y1), (x_caja + x2, y_caja + y2), (0, 255, 0), 2)
                            texto = f"{numero_detectado} ({confianza * 100:.1f}%)"
                            cv2.putText(frame, texto, (x_caja + x1, y_caja + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)
                    except Exception as e:
                        pass

            # Mostrar las ventanas
            cv2.imshow("Camara Original", frame)
            # Descomenta la siguiente línea si quieres ver cómo la máquina ve el interior de la caja
            # cv2.imshow("Vision de la Maquina (Caja)", thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    reconocedor = ReconocedorWebcam("modelo_mnist.keras")
    reconocedor.iniciar_reconocimiento()