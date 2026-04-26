import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
 
# ========== PATHS ==========
image_folder = "data/preprocessed_images"
metadata_path = "data/final_subset_metadata.csv"

train_img_folder = "data/train/images"
test_img_folder = "data/test/images"

os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(test_img_folder, exist_ok=True)

# ========== LOAD METADATA ==========
df = pd.read_csv(metadata_path)

# IMPORTANT: adjust column names if needed
# assuming:
# df['image'] -> image file name
# df['label'] -> 0 or 1

# ========== TRAIN-TEST SPLIT ==========
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['malignant'],   # keeps class balance
    random_state=42
)

# ========== FUNCTION TO COPY FILES ==========
def copy_images(dataframe, src_folder, dest_folder):
    for _, row in dataframe.iterrows():
        img_name = row['isic_id'] + ".jpg"   # change if needed
        src_path = os.path.join(src_folder, img_name)
        dest_path = os.path.join(dest_folder, img_name)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Missing: {img_name}")

# ========== COPY TRAIN IMAGES ==========
copy_images(train_df, image_folder, train_img_folder)

# ========== COPY TEST IMAGES ==========
copy_images(test_df, image_folder, test_img_folder)

# ========== SAVE LABEL FILES ==========
train_df.to_csv("data/train/train_labels.csv", index=False)
test_df.to_csv("data/test/test_labels.csv", index=False)

print("✅ Dataset split complete!")