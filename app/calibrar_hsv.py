import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

DATASET_DIR = "../dataset/train"
SAMPLES_PER_CLASS = 10

def get_image_paths(root):
    paths = []
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        images = images[:SAMPLES_PER_CLASS]

        for img_name in images:
            paths.append(os.path.join(cls_path, img_name))
    return paths

def calibrate_hsv():
    hsv_values = []

    paths = get_image_paths(DATASET_DIR)

    print(f"Analizando {len(paths)} imágenes del dataset...")

    for path in paths:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Aplanamos a lista de píxeles
        pixels = img_hsv.reshape(-1, 3)

        # KMeans para separar fondo vs hoja (2 clusters)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Cluster con más saturación → generalmente es la hoja
        sat_values = centers[:, 1]
        leaf_cluster = np.argmax(sat_values)

        leaf_pixels = pixels[labels == leaf_cluster]

        hsv_values.append(leaf_pixels)

    hsv_values = np.vstack(hsv_values)

    # Calcular límites HSV reales del dataset
    lower = np.percentile(hsv_values, 5, axis=0).astype(int)
    upper = np.percentile(hsv_values, 95, axis=0).astype(int)

    print("\n=== NUEVO RANGO HSV DETECTADO ===")
    print("lower =", lower)
    print("upper =", upper)

    print("\nCÓPIA ESTO EN TU train_model.py:")
    print(f"lower_green = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
    print(f"upper_green = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")


if __name__ == "__main__":
    calibrate_hsv()
