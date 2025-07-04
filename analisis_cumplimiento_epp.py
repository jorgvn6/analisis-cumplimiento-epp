import cv2
import json
import time
import torch
import os
from datetime import datetime
from ultralytics import YOLO

# Selección del dispositivo (GPU si está disponible, si no CPU)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelos YOLO: uno personalizado para cascos y otro general para personas
modelo_casco = YOLO("casco.pt").to(dispositivo)
modelo_persona = YOLO("yolov8x.pt").to(dispositivo)

# Obtener marca de tiempo única
marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
carpeta_salida = f"resultados_{marca_tiempo}"

# Crear la carpeta si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# Rutas de archivos de entrada y salida
ruta_video_entrada = "video.mp4"
ruta_video_salida = os.path.join(carpeta_salida, "video_anotado.mp4")
ruta_json_salida = os.path.join(carpeta_salida, "detecciones.json")

# Abrir el video de entrada
captura_video = cv2.VideoCapture(ruta_video_entrada)
ancho_frame = int(captura_video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_frame = int(captura_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(captura_video.get(cv2.CAP_PROP_FPS))

# Crear el archivo de video de salida con las anotaciones
escritor_video = cv2.VideoWriter(
    ruta_video_salida,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (ancho_frame, alto_frame)
)

# Lista para almacenar las detecciones por frame
detecciones_video = []
tiempo_inicio = time.time()
indice_frame = 0

# Diccionario para almacenar el estado anterior (con/sin casco) por persona
estado_personas_rastreadas = {}

# Bucle principal de procesamiento de frames
while True:
    ret, frame = captura_video.read()
    if not ret:
        break  # Salir si no hay más frames

    indice_frame += 1
    timestamp = indice_frame / fps

    # Detección de cascos (sin tracking)
    # El parámetro `conf` define el umbral mínimo de confianza para aceptar una detección.
    # Aumentar `conf` (ej. 0.5) mejora la precisión reduciendo falsos positivos, pero puede perder detecciones.
    # Disminuir `conf` (ej. 0.2) detecta más objetos, pero puede aumentar falsos positivos.
    resultados_casco = modelo_casco.predict(source=frame, conf=0.3, verbose=False)

    cascos_detectados = []
    for resultado in resultados_casco:
        for caja in resultado.boxes:
            nombre_clase = modelo_casco.names[int(caja.cls)]
            if nombre_clase.lower() in ["helmet", "hardhat", "casco"]:
                cascos_detectados.append(caja.xyxy[0].tolist())

    # Detección de personas con tracking
    resultados_personas = modelo_persona.track(source=frame, conf=0.3, persist=True, verbose=False)

    personas_detectadas = []
    for resultado in resultados_personas:
        for caja in resultado.boxes:
            nombre_clase = modelo_persona.names[int(caja.cls)]
            if nombre_clase.lower() == "person":
                bbox_persona = caja.xyxy[0].tolist()
                id_rastreo = int(caja.id) if caja.id is not None else None
                if id_rastreo is not None:
                    personas_detectadas.append((id_rastreo, bbox_persona))

    # Crear estructura para almacenar las detecciones de este frame
    detecciones_frame = {
        "frame": indice_frame,
        "timestamp": round(timestamp, 2),
        "detecciones": []
    }

    total_personas = len(personas_detectadas)
    personas_con_casco = 0
    personas_sin_casco = 0

    ids_ya_vistos = set()

    for id_rastreo, bbox_persona in personas_detectadas:
        if id_rastreo in ids_ya_vistos:
            continue  # Evitar procesar la misma persona más de una vez por frame
        ids_ya_vistos.add(id_rastreo)

        px1, py1, px2, py2 = map(int, bbox_persona)
        centro_x_persona = (px1 + px2) / 2
        cabeza_y_persona = py1 + (py2 - py1) * 0.25  # Región superior del cuerpo

        tiene_casco = False
        # Verificar si hay un casco en la región de la cabeza de la persona
        for bbox_casco in cascos_detectados:
            cx1, cy1, cx2, cy2 = map(int, bbox_casco)
            centro_x_casco = (cx1 + cx2) / 2
            centro_y_casco = (cy1 + cy2) / 2

            if (px1 < centro_x_casco < px2) and (py1 < centro_y_casco < cabeza_y_persona):
                tiene_casco = True
                break

        # Actualizar conteo
        if tiene_casco:
            personas_con_casco += 1
        else:
            personas_sin_casco += 1

        # Consultar estado previo
        estado_previo = estado_personas_rastreadas.get(id_rastreo)

        # Guardar en JSON si es un cambio o primera vez
        if estado_previo is None or estado_previo != tiene_casco:
            detecciones_frame["detecciones"].append({
                "id": id_rastreo,
                "bbox": {"x_min": px1, "y_min": py1, "x_max": px2, "y_max": py2},
                "casco": tiene_casco,
                "timestamp": timestamp,
                "frame": indice_frame
            })
            estado_personas_rastreadas[id_rastreo] = tiene_casco

        # Dibujar caja y texto en el frame
        color = (0, 255, 0) if tiene_casco else (0, 0, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        # Escribir estado (con/sin casco)
        estado_texto = "Con casco" if tiene_casco else "Sin casco"
        cv2.putText(frame, estado_texto, (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, estado_texto, (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Escribir ID
        texto_id = f"ID: {id_rastreo}"
        cv2.putText(frame, texto_id, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, texto_id, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Escribir resumen general en la parte superior del frame
    texto_resumen = f"Personas: {total_personas} | Con casco: {personas_con_casco} | Sin casco: {personas_sin_casco}"
    cv2.putText(frame, texto_resumen, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(frame, texto_resumen, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Guardar detecciones del frame si hay cambios relevantes
    if detecciones_frame["detecciones"]:
        detecciones_video.append(detecciones_frame)

    # Guardar frame anotado en el video de salida
    escritor_video.write(frame)

# Liberar recursos al finalizar
captura_video.release()
escritor_video.release()

# Guardar archivo JSON con todas las detecciones
with open(ruta_json_salida, "w") as archivo_json:
    json.dump(detecciones_video, archivo_json, indent=4)

# Mostrar estadísticas finales
tiempo_total = time.time() - tiempo_inicio
print(f"Procesamiento completado en {tiempo_total:.2f} segundos")
print(f"Detecciones guardadas en {ruta_json_salida}")
print(f"Video anotado guardado en {ruta_video_salida}")
