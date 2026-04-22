import cv2
import matplotlib.pyplot as plt
#-----------------------------------------------------
# Cargar imagen
img = cv2.imread("ciudad.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#-----------------------------------------------------
# Aplicar filtro gaussiano
img_blur = cv2.GaussianBlur(img, (15, 15), 0)
img_blur_rgb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)

#-----------------------------------------------------
# Aplicar SIFT
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img_blur, None)

img_sift = cv2.drawKeypoints(
    img_blur_rgb,
    kp_sift,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

#-----------------------------------------------------
# Mostrar todo
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_blur_rgb)
plt.title("Filtro gaussiano")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_sift)
plt.title(f"SIFT - Puntos detectados: {len(kp_sift)}")
plt.axis("off")

plt.show()

print("Puntos detectados por SIFT:", len(kp_sift))

#-----------------------------------------------------
# Probar SURF
try:
    surf = cv2.xfeatures2d.SURF_create()
    kp_surf, des_surf = surf.detectAndCompute(img_blur, None)

    img_surf = cv2.drawKeypoints(
        img_blur_rgb,
    kp_surf,
        None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(img_surf)
    plt.title(f"SURF - Puntos detectados: {len(kp_surf)}")
    plt.axis("off")
    plt.show()

    print("Puntos detectados por SURF:", len(kp_surf))

except Exception as e:
    print("SURF no se pudo ejecutar")
    print("Error:", e)

#-----------------------------------------------------
# ORB
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(img_blur, None)

img_orb = cv2.drawKeypoints(
    img_blur_rgb,
    kp_orb,
    None,
    color=(255, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12, 8))
plt.imshow(img_orb)
plt.title(f"ORB - Puntos detectados: {len(kp_orb)}")
plt.axis("off")
plt.show()

print("Puntos detectados por ORB:", len(kp_orb))

#-----------------------------------------------------
# BRISK
brisk = cv2.BRISK_create()
kp_brisk, des_brisk = brisk.detectAndCompute(img, None)

img_brisk = cv2.drawKeypoints(
    img_rgb,
    kp_brisk,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12,8))
plt.imshow(img_brisk)
plt.title(f"BRISK - Puntos detectados: {len(kp_brisk)}")
plt.axis("off")
plt.show()

print("Puntos detectados por BRISK:", len(kp_brisk))

#-----------------------------------------------------
# AKAZE
akaze = cv2.AKAZE_create()
kp_akaze, des_akaze = akaze.detectAndCompute(img, None)

img_akaze = cv2.drawKeypoints(
    img_rgb,
    kp_akaze,
    None,
    color=(0, 0, 255),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12,8))
plt.imshow(img_akaze)
plt.title(f"AKAZE - Puntos detectados: {len(kp_akaze)}")
plt.axis("off")
plt.show()

print("Puntos detectados por AKAZE:", len(kp_akaze))