import cv2
import matplotlib.pyplot as plt

img = cv2.imread("ciudad.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Imagen original")
plt.axis("off")
plt.show()


#--------------------------------------
#ALGORITMO ORB (Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)

img_kp = cv2.drawKeypoints(
    img_rgb,
    kp,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8,5))
plt.imshow(img_kp)
plt.title(f"ORB - Puntos detectados: {len(kp)}")
plt.axis("off")
plt.show()

print("Puntos detectados por ORB:", len(kp))



#-------------------------------------
#ALGORITMO BRISK (Binary Robust Invariant Scalable Keypoints)
brisk = cv2.BRISK_create()
kp_brisk, des_brisk = brisk.detectAndCompute(img, None)

img_brisk = cv2.drawKeypoints(
    img_rgb,
    kp_brisk,
    None,
    color=(255, 0, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(8,5))
plt.imshow(img_brisk)
plt.title(f"BRISK - Puntos detectados: {len(kp_brisk)}")
plt.axis("off")
plt.show()

print("Puntos detectados por BRISK:", len(kp_brisk))


#--------------------------------------
#ALGORITMO AKAZE (Accelerated-KAZE)
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
