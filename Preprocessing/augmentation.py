import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A

# ========== PATHS ==========
train_img_folder = "data/train/images"
train_csv = "data/train/train_labels.csv"

aug_img_folder = "data/augmented/images"
aug_csv_path = "data/augmented/augmented_labels.csv"

final_img_folder = "data/final_train/images"
final_csv_path = "data/final_train/final_train_labels.csv"

# create folders
os.makedirs(aug_img_folder, exist_ok=True)
os.makedirs(final_img_folder, exist_ok=True)

# ========== LOAD DATA ==========
df = pd.read_csv(train_csv)

# ========== SELECT MALIGNANT ==========
malignant_df = df[df['malignant'] == 1]

# randomly select 50
sample_df = malignant_df.sample(n=50, random_state=42)

# ========== AUGMENTATION PIPELINE ==========
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2)
])

augmented_rows = []

# ========== GENERATE AUGMENTED IMAGES ==========
for i, (_, row) in enumerate(tqdm(sample_df.iterrows(), total=50)):
    
    img_name = row['isic_id'] + ".jpg"
    img_path = os.path.join(train_img_folder, img_name)

    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping {img_name}")
        continue

    # apply augmentation
    augmented = transform(image=img)
    aug_img = augmented['image']

    # new image name
    new_id = row['isic_id'] + f"_aug_{i}"
    new_img_name = new_id + ".jpg"

    # save image
    cv2.imwrite(os.path.join(aug_img_folder, new_img_name), aug_img)

    # create new metadata row
    new_row = row.copy()
    new_row['isic_id'] = new_id

    augmented_rows.append(new_row)

# ========== SAVE AUGMENTED METADATA ==========
aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv(aug_csv_path, index=False)

print("✅ Augmentation done!")

# ========== CREATE FINAL TRAIN SET ==========

# copy original images
for img_name in os.listdir(train_img_folder):
    src = os.path.join(train_img_folder, img_name)
    dst = os.path.join(final_img_folder, img_name)
    if not os.path.exists(dst):
        cv2.imwrite(dst, cv2.imread(src))

# copy augmented images
for img_name in os.listdir(aug_img_folder):
    src = os.path.join(aug_img_folder, img_name)
    dst = os.path.join(final_img_folder, img_name)
    cv2.imwrite(dst, cv2.imread(src))

# merge metadata
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(final_csv_path, index=False)

print("✅ Final training dataset ready!")