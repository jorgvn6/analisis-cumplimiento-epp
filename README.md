# Análisis de Cumplimiento de Equipo de Protección Personal

## Detección de Cascos de Seguridad con YOLOv8 y Seguimiento de Personas

Este proyecto demuestra cómo utilizar modelos de visión por computadora basados en YOLOv8 para detectar personas y verificar si están usando casco de seguridad. Se combinan dos modelos: uno general para detectar personas y otro personalizado entrenado para reconocer cascos. El sistema anota un video de entrada, guarda los resultados en un archivo JSON y marca visualmente el estado de cada persona (con o sin casco).

## Objetivos

- Detectar personas en video usando YOLOv8 con seguimiento de objetos.
- Detectar cascos de seguridad usando un modelo personalizado entrenado con un dataset específico.
- Asociar detecciones de personas y cascos para determinar si cada persona lleva casco.
- Registrar los resultados en video y en formato JSON para análisis posterior.

## Ejemplo Visual

A continuación se muestra un fragmento del video procesado, donde se visualiza el sistema en funcionamiento:

![Demo del sistema](media/demo.gif)

## Modelos Utilizados

- `yolov8x.pt`: Modelo general de YOLOv8 para detectar personas, descargado desde [Ultralytics](https://github.com/ultralytics/assets/releases).
- `casco.pt`: Modelo personalizado entrenado con el dataset v12 [Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers) de Roboflow. Detecta cascos de seguridad en imágenes.

## Cómo Entrenar el Modelo Personalizado (`casco.pt`)

Para entrenar tu propio modelo de detección de cascos basado en YOLOv8, puedes utilizar el siguiente script en Python. Este código entrena un modelo a partir del archivo de configuración `data.yaml`, el cual debe contener las rutas a tus imágenes y las clases del dataset.

### Script de entrenamiento

```python
from ultralytics import YOLO

def main():
    model = YOLO("yolov8x.pt")  # También se puede usar yolov8n.pt o yolov8s.pt
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=512,
        batch=32,
        device=0,       # GPU (usa "cpu" si no tienes GPU)
        cache=True      # Precarga los datos en RAM
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
```

### Consideraciones al entrenar

Versiones de YOLOv8 disponibles:

- `yolov8n.pt` (Nano): Muy rápido y ligero, ideal para dispositivos con recursos limitados, pero con menor precisión.

- `yolov8s.pt` (Small): Un equilibrio entre velocidad y precisión, recomendado para pruebas rápidas o entrenamiento en laptops.

- `yolov8x.pt` (Extra Large): Más lento y pesado, pero con mayor capacidad de detección. Requiere más memoria GPU.

En este proyecto se usó `yolov8x.pt` como base para entrenar `casco.pt`, ya que se priorizó la precisión sobre la velocidad.

La opción `cache=True` carga todo el dataset en memoria RAM antes de entrenar. Esto puede acelerar el entrenamiento si tienes suficiente RAM, pero también puede provocar:

- Errores de memoria si el dataset es muy grande.

- Resultados **no deterministas** si se reutiliza la caché sin reiniciar.

- Problemas si usas múltiples procesos de entrenamiento simultáneamente.

En el archivo `data.yaml` debe modificarse la ruta de las imágenes de entrenamiento para que esté acorde a la ruta real en la máquina donde se va a entrenar.

## Resultados Esperados

Al ejecutar el sistema de detección y seguimiento, se generan dos productos principales para su análisis y visualización:

1. **Video Anotado**  
   Un archivo de video donde cada persona detectada está rodeada por un recuadro de color: verde si lleva casco, rojo si no. Además, se muestra el ID único asignado a cada persona y un resumen en pantalla con el conteo de personas con y sin casco. Este video permite verificar visualmente el cumplimiento del uso de equipo de protección personal en las escenas procesadas.

2. **Archivo JSON con Detecciones**  
   Un archivo en formato JSON que contiene un registro detallado de las detecciones relevantes en cada frame del video. Por cada persona detectada y rastreada, se registra su ID, la caja delimitadora (bounding box), si lleva casco o no, el número de frame y el timestamp en segundos. Esto facilita el análisis posterior y la integración con otros sistemas o bases de datos.

### Ejemplo de Estructura JSON Generada

```json
[
    {
        "frame": 1,
        "timestamp": 0.03,
        "detecciones": [
            {
                "id": 1,
                "bbox": {
                    "x_min": 588,
                    "y_min": 187,
                    "x_max": 717,
                    "y_max": 594
                },
                "casco": true,
                "timestamp": 0.03333333333333333,
                "frame": 1
            },
            {
                "id": 4,
                "bbox": {
                    "x_min": 735,
                    "y_min": 202,
                    "x_max": 817,
                    "y_max": 545
                },
                "casco": false,
                "timestamp": 0.03333333333333333,
                "frame": 1
            }
        ]
    }
]
```
