import cv2
import numpy as np
from tensorflow.keras.models import load_model


class ReconocedorWebcam:
    def __init__(self, ruta_modelo="modelo_mnist.keras"):
        print(f"Cargando modelo desde {ruta_modelo}...")
        self.modelo = load_model(ruta_modelo)

    def iniciar_reconocimiento(self):
        cap = cv2.VideoCapture(0)

        # OPTIMIZACIÓN 1: Reducir resolución para que la laptop no sufra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Cámara iniciada. Presiona 'z' para salir.")

        # Variables para OPTIMIZACIÓN 3 (Frame Skipping)
        contador_frames = 0
        frames_a_saltar = 3  # La IA solo analizará 1 de cada 3 cuadros
        ultimo_texto = ""
        ultima_caja_verde = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al acceder a la cámara.")
                break

            alto, ancho = frame.shape[:2]

            tamano_caja = 300
            x_caja = int((ancho - tamano_caja) / 2)
            y_caja = int((alto - tamano_caja) / 2)

            cv2.rectangle(frame, (x_caja, y_caja), (x_caja + tamano_caja, y_caja + tamano_caja), (255, 0, 0), 2)
            cv2.putText(frame, "", (x_caja, y_caja - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 0), 2)

            area_escaneo = frame[y_caja:y_caja + tamano_caja, x_caja:x_caja + tamano_caja]

            gray = cv2.cvtColor(area_escaneo, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # OPTIMIZACIÓN 2: Procesar SOLO el contorno más grande
            if contornos:
                contorno_principal = max(contornos, key=cv2.contourArea)

                if cv2.contourArea(contorno_principal) > 500:
                    x, y, w, h = cv2.boundingRect(contorno_principal)

                    # 1. Extraemos el recorte original exactamente del tamaño del número
                    roi_recorte = thresh[y:y + h, x:x + w]

                    # 2. Calculamos el lado más largo para crear un cuadrado perfecto
                    lado_mayor = max(w, h)

                    # 3. Creamos un fondo negro cuadrado (con 30px extra para simular el margen de 15px por lado)
                    fondo_cuadrado = np.zeros((lado_mayor + 30, lado_mayor + 30), dtype=np.uint8)

                    # 4. Calculamos exactamente dónde pegar el recorte para que quede centrado en el cuadrado
                    x_offset = (lado_mayor + 30 - w) // 2
                    y_offset = (lado_mayor + 30 - h) // 2

                    # 5. Pegamos el número en el centro del fondo negro cuadrado
                    fondo_cuadrado[y_offset:y_offset + h, x_offset:x_offset + w] = roi_recorte

                    # 6. Ahora sí, redimensionamos a 28x28 SIN DEFORMAR EL NÚMERO
                    try:
                        roi_resized = cv2.resize(fondo_cuadrado, (28, 28), interpolation=cv2.INTER_AREA)
                        roi_normalized = roi_resized / 255.0
                        roi_reshaped = np.reshape(roi_normalized, (1, 28, 28, 1))

                        # Ejecutar la predicción (con el frame skipping que ya teníamos)
                        if contador_frames % frames_a_saltar == 0:
                            prediccion = self.modelo.predict(roi_reshaped, verbose=0)
                            numero_detectado = np.argmax(prediccion)
                            confianza = np.max(prediccion)

                            if confianza > 0.8:
                                ultimo_texto = f"{numero_detectado} ({confianza * 100:.1f}%)"
                                # Mantenemos el cuadro verde dibujado sobre el número real en la pantalla
                                ultima_caja_verde = (x_caja + x, y_caja + y, x_caja + x + w, y_caja + y + h)
                            else:
                                ultimo_texto = ""
                                ultima_caja_verde = None
                    except Exception as e:
                        pass
            else:
                # Si sacas el papel, limpiamos los dibujos verdes
                ultimo_texto = ""
                ultima_caja_verde = None

            # DIBUJAR: Mostramos constantemente la última predicción conocida para que no parpadee
            if ultima_caja_verde is not None and ultimo_texto != "":
                cx1, cy1, cx2, cy2 = ultima_caja_verde
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                cv2.putText(frame, ultimo_texto, (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Camara Original", frame)

            # Aumentamos el reloj de frames
            contador_frames += 1

            if cv2.waitKey(1) & 0xFF == ord('z'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    reconocedor = ReconocedorWebcam("modelo_mnist.keras")
    reconocedor.iniciar_reconocimiento()