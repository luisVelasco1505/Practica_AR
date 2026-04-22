import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------
# BLOQUE 1: CARGA DE LA IMAGEN
# -----------------------------------------------------
# Se carga la imagen desde el archivo "ciudad.jpg".
# cv2.imread() devuelve la imagen en formato BGR,
# que es el formato por defecto de OpenCV.
img = cv2.imread("ciudad.jpg")

# Se convierte la imagen de BGR a RGB.
# Esto se hace porque matplotlib espera los colores en formato RGB.
# Si no se realiza esta conversión, la imagen se vería con colores alterados.
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------
# BLOQUE 2: PREPROCESAMIENTO CON FILTRO GAUSSIANO
# -----------------------------------------------------
# Se aplica un filtro gaussiano a la imagen original.
# Este filtro suaviza la imagen y reduce ruido visual,
# ayudando a que algunos algoritmos detecten características
# más estables y menos sensibles a pequeños detalles irrelevantes.
#
# Parámetros:
# - img: imagen de entrada
# - (15, 15): tamaño del kernel o ventana del filtro
# - 0: desviación estándar calculada automáticamente
img_blur = cv2.GaussianBlur(img, (15, 15), 0)

# Se convierte la imagen filtrada a RGB para poder visualizarla correctamente.
img_blur_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------
# BLOQUE 3: DETECCIÓN DE PUNTOS CLAVE CON SIFT
# -----------------------------------------------------
# Se crea el detector SIFT.
# SIFT permite encontrar puntos clave robustos ante cambios
# de escala, rotación e iluminación.
sift = cv2.SIFT_create()

# detectAndCompute() realiza dos tareas:
# 1. Detecta los puntos clave (kp_sift)
# 2. Calcula los descriptores asociados a cada punto (des_sift)
#
# Los puntos clave representan zonas importantes de la imagen,
# como esquinas, bordes y regiones con textura.
kp_sift, des_sift = sift.detectAndCompute(img_blur, None)

# Se dibujan sobre la imagen filtrada los puntos detectados por SIFT.
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS permite mostrar
# círculos con tamaño y orientación, haciendo la visualización más informativa.
img_sift = cv2.drawKeypoints(
    img_blur_rgb,
    kp_sift,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# -----------------------------------------------------
# BLOQUE 4: VISUALIZACIÓN DE LA IMAGEN ORIGINAL,
#           IMAGEN FILTRADA Y RESULTADO DE SIFT
# -----------------------------------------------------
# Se crea una figura grande para mostrar tres imágenes en una sola ventana.
plt.figure(figsize=(18, 6))

# Subgráfico 1: imagen original
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")

# Subgráfico 2: imagen después del filtro gaussiano
plt.subplot(1, 3, 2)
plt.imshow(img_blur_rgb)
plt.title("Filtro gaussiano")
plt.axis("off")

# Subgráfico 3: imagen con puntos clave detectados por SIFT
plt.subplot(1, 3, 3)
plt.imshow(img_sift)
plt.title(f"SIFT - Puntos detectados: {len(kp_sift)}")
plt.axis("off")

# Se muestra la figura completa
plt.show()

# Se imprime en consola la cantidad de puntos detectados por SIFT
print("Puntos detectados por SIFT:", len(kp_sift))

# -----------------------------------------------------
# BLOQUE 5: INTENTO DE EJECUCIÓN DE SURF
# -----------------------------------------------------
# Se intenta ejecutar SURF dentro de un bloque try/except,
# porque este algoritmo no siempre está disponible en todas
# las instalaciones de OpenCV por temas de licencia.
try:
    # Creación del detector SURF
    surf = cv2.xfeatures2d.SURF_create()

    # Detección de puntos clave y cálculo de descriptores
    kp_surf, des_surf = surf.detectAndCompute(img_blur, None)

    # Dibujo de puntos clave detectados por SURF
    img_surf = cv2.drawKeypoints(
        img_blur_rgb,
        kp_surf,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Mostrar resultado de SURF
    plt.figure(figsize=(12, 8))
    plt.imshow(img_surf)
    plt.title(f"SURF - Puntos detectados: {len(kp_surf)}")
    plt.axis("off")
    plt.show()

    # Imprimir cantidad de puntos detectados
    print("Puntos detectados por SURF:", len(kp_surf))

except Exception as e:
    # Si SURF no está disponible, se captura el error
    # y se informa en consola sin detener el resto del programa.
    print("SURF no se pudo ejecutar")
    print("Error:", e)

# -----------------------------------------------------
# BLOQUE 6: DETECCIÓN DE PUNTOS CLAVE CON ORB
# -----------------------------------------------------
# Se crea el detector ORB.
# ORB es más rápido que SIFT y está pensado para aplicaciones
# en tiempo real, aunque suele detectar menos puntos o menos detallados.
orb = cv2.ORB_create()

# Detección de puntos clave y cálculo de descriptores sobre la imagen filtrada
kp_orb, des_orb = orb.detectAndCompute(img_blur, None)

# Dibujo de los puntos clave encontrados por ORB
img_orb = cv2.drawKeypoints(
    img_blur_rgb,
    kp_orb,
    None,
    color=(255, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Mostrar resultado de ORB
plt.figure(figsize=(12, 8))
plt.imshow(img_orb)
plt.title(f"ORB - Puntos detectados: {len(kp_orb)}")
plt.axis("off")
plt.show()

# Imprimir cantidad de puntos detectados por ORB
print("Puntos detectados por ORB:", len(kp_orb))

# -----------------------------------------------------
# BLOQUE 7: DETECCIÓN DE PUNTOS CLAVE CON BRISK
# -----------------------------------------------------
# Se crea el detector BRISK.
# BRISK suele detectar una gran cantidad de puntos clave,
# por lo que puede producir una salida muy cargada visualmente.
brisk = cv2.BRISK_create()

# En este caso BRISK se aplica sobre la imagen original,
# no sobre la imagen filtrada.
kp_brisk, des_brisk = brisk.detectAndCompute(img, None)

# Se dibujan los puntos clave detectados por BRISK
img_brisk = cv2.drawKeypoints(
    img_rgb,
    kp_brisk,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Mostrar resultado de BRISK
plt.figure(figsize=(12, 8))
plt.imshow(img_brisk)
plt.title(f"BRISK - Puntos detectados: {len(kp_brisk)}")
plt.axis("off")
plt.show()

# Imprimir cantidad de puntos detectados por BRISK
print("Puntos detectados por BRISK:", len(kp_brisk))

# -----------------------------------------------------
# BLOQUE 8: DETECCIÓN DE PUNTOS CLAVE CON AKAZE
# -----------------------------------------------------
# Se crea el detector AKAZE.
# AKAZE busca un equilibrio entre precisión y rendimiento,
# detectando una cantidad intermedia de puntos clave.
akaze = cv2.AKAZE_create()

# AKAZE también se aplica sobre la imagen original
kp_akaze, des_akaze = akaze.detectAndCompute(img, None)

# Dibujo de los puntos clave detectados por AKAZE
img_akaze = cv2.drawKeypoints(
    img_rgb,
    kp_akaze,
    None,
    color=(0, 0, 255),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Mostrar resultado de AKAZE
plt.figure(figsize=(12, 8))
plt.imshow(img_akaze)
plt.title(f"AKAZE - Puntos detectados: {len(kp_akaze)}")
plt.axis("off")
plt.show()

# Imprimir cantidad de puntos detectados por AKAZE
print("Puntos detectados por AKAZE:", len(kp_akaze))