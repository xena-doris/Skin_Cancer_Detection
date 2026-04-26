import cv2
import numpy as np
import os
from tqdm import tqdm

input_folder = "data/final_subset_images"
output_folder = "data/preprocessed_images"

os.makedirs(output_folder, exist_ok=True)

IMG_SIZE = 224


def load_and_resize(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ✅ FIXED SHARPEN
def sharpen(img):
    return cv2.addWeighted(img, 1.1, img, 0, 0)


for img_name in tqdm(os.listdir(input_folder)):
    img_path = os.path.join(input_folder, img_name)

    try:
        img = load_and_resize(img_path)
        if img is None:
            continue

        img = apply_clahe(img)
        img = sharpen(img)

        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"Error: {img_name}, {e}")

print("✅ FINAL DONE")