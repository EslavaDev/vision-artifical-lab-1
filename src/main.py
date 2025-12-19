import cv2
import numpy as np
import os
from skimage.measure import shannon_entropy


def calcular_metricas(img, nombre):
    """Calcula y muestra metricas de una imagen: resolucion, brillo, contraste y entropia."""
    if img is None:
        return None

    alto, ancho = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo = np.mean(img_gray)
    contraste = np.std(img_gray)
    entropia = shannon_entropy(img_gray)

    print(f"--- Metricas para: {nombre} ---")
    print(f"Resolucion: {ancho}x{alto}")
    print(f"Brillo: {brillo:.2f} | Contraste: {contraste:.2f} | Entropia: {entropia:.2f}")
    print("-" * 30)
    return {"brillo": brillo, "contraste": contraste, "entropia": entropia}


def correccion_gamma(img, gamma=0.5):
    """Aplica correccion gamma para ajustar la luminosidad."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def estiramiento_contraste(img):
    """Estira el contraste usando percentiles en el canal Y (luminancia)."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)
    p2, p98 = np.percentile(y, (2, 98))
    y_stretched = np.clip((y - p2) * (255.0 / (p98 - p2)), 0, 255).astype('uint8')
    img_final = cv2.merge((y_stretched, u, v))
    return cv2.cvtColor(img_final, cv2.COLOR_YUV2BGR)


def transformacion_log(img):
    """Aplica transformacion logaritmica para expandir valores oscuros."""
    img_float = np.float32(img)
    c = 255 / np.log(1 + np.max(img_float))
    log_image = c * np.log(img_float + 1)
    return np.array(log_image, dtype=np.uint8)


def ecualizacion_global(img):
    """Aplica ecualizacion de histograma global en el canal Y."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def aplicar_clahe(img, clip_limit=3.0):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) en espacio LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_final = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)


def adicionar_brillo(img, valor=60):
    """Aumenta el brillo sumando un valor constante a todos los pixeles."""
    matriz_valor = np.full(img.shape, valor, dtype='uint8')
    return cv2.add(img, matriz_valor)


def multiplicar_contraste(img, factor=1.5):
    """Aumenta el contraste multiplicando los valores de los pixeles."""
    matrix = np.ones(img.shape, dtype="float32") * factor
    return np.clip(cv2.multiply(img.astype("float32"), matrix), 0, 255).astype("uint8")


def procesar_imagenes(input_folder, output_folder):
    """Procesa todas las imagenes de una carpeta aplicando multiples tecnicas."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Carpeta '{input_folder}' creada. Agrega imagenes y vuelve a ejecutar.")
        return

    extensiones = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    nombres_imagenes = [f for f in os.listdir(input_folder) if f.lower().endswith(extensiones)]

    if not nombres_imagenes:
        print(f"No se encontraron imagenes en '{input_folder}'")
        return

    for nombre in nombres_imagenes:
        ruta_img = os.path.join(input_folder, nombre)
        img = cv2.imread(ruta_img)

        if img is None:
            print(f"Error cargando {nombre}")
            continue

        calcular_metricas(img, nombre)

        tecnicas = [
            ("gamma", correccion_gamma, {"gamma": 0.5}),
            ("estiramiento", estiramiento_contraste, {}),
            ("log", transformacion_log, {}),
            ("clahe", aplicar_clahe, {}),
            ("eq_global", ecualizacion_global, {}),
            ("adicion", adicionar_brillo, {}),
        ]

        for prefijo, funcion, params in tecnicas:
            resultado = funcion(img, **params) if params else funcion(img)
            cv2.imwrite(os.path.join(output_folder, f"{prefijo}_{nombre}"), resultado)

    print(f"Procesamiento finalizado. Revisa la carpeta '{output_folder}'.")


if __name__ == "__main__":
    input_folder = "imagenes_originales"
    output_folder = "resultados"
    procesar_imagenes(input_folder, output_folder)
