# Procesamiento de Imagenes

Herramienta para aplicar tecnicas de mejora de imagenes usando OpenCV y scikit-image.

## Descripcion

Este proyecto procesa imagenes aplicando multiples tecnicas de mejora automaticamente:

| Tecnica | Descripcion |
|---------|-------------|
| **Correccion Gamma** | Ajusta la luminosidad de la imagen. Valores < 1 aclaran, > 1 oscurecen |
| **Estiramiento de Contraste** | Expande el rango dinamico usando percentiles (2-98%) |
| **Transformacion Logaritmica** | Expande detalles en zonas oscuras comprimiendo las claras |
| **CLAHE** | Ecualizacion adaptativa que mejora contraste local sin saturar |
| **Ecualizacion Global** | Redistribuye el histograma para usar todo el rango de intensidades |
| **Adicion de Brillo** | Suma un valor constante a todos los pixeles |

Ademas calcula metricas para cada imagen:
- **Brillo**: Valor medio de intensidad
- **Contraste**: Desviacion estandar de intensidad
- **Entropia**: Medida de informacion/complejidad de la imagen

## Estructura del Proyecto

```
procesamiento-imagenes/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── imagenes_originales/   # Coloca aqui tus imagenes
    └── resultados/            # Imagenes procesadas
```

## Instalacion

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

1. Coloca tus imagenes en `src/imagenes_originales/`
2. Ejecuta el script:

```bash
cd src
python main.py
```

3. Las imagenes procesadas apareceran en `src/resultados/` con prefijos:
   - `gamma_` - Correccion gamma
   - `estiramiento_` - Estiramiento de contraste
   - `log_` - Transformacion logaritmica
   - `clahe_` - CLAHE
   - `eq_global_` - Ecualizacion global
   - `adicion_` - Adicion de brillo

## Ejemplo de Salida

```
--- Metricas para: foto1.jpg ---
Resolucion: 1920x1080
Brillo: 127.45 | Contraste: 45.32 | Entropia: 7.21
------------------------------
Procesamiento finalizado. Revisa la carpeta 'resultados'.
```

## Funciones Disponibles

Las funciones pueden usarse de forma independiente importandolas:

```python
from main import correccion_gamma, aplicar_clahe
import cv2

img = cv2.imread("mi_imagen.jpg")
img_mejorada = aplicar_clahe(img, clip_limit=2.0)
cv2.imwrite("mejorada.jpg", img_mejorada)
```

## Dependencias

- Python 3.7+
- OpenCV
- NumPy
- scikit-image
