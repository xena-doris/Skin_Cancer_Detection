import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report
from torchvision import transforms

# Import models
from src.ResNet.model import build_model as build_resnet
from src.DenseNet.model import build_model as build_densenet
from src.EfficentNet.model import build_model as build_efficientnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
IMAGE_DIR = "data/final_subset_images"
CSV_PATH = "data/test/test_labels.csv"

MODEL_PATHS = {
    "ResNet": "models/best_model.pth",
    "DenseNet": "models/best_DenseNet_model.pth",
    "EfficientNet": "models/best_EffecientNet_model.pth"
}

IMG_SIZE = 224

# Transform (NO augmentation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset
class UnprocessedDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['isic_id'] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"❌ Missing image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)

        label = torch.tensor(row['malignant'], dtype=torch.float32)

        return image, label


# Load dataset once
dataset = UnprocessedDataset(CSV_PATH, IMAGE_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# 🔥 Safe evaluation function
def evaluate_model(model_name, model_builder, model_path):
    print(f"\n================ {model_name} Evaluation =================")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return

    model = model_builder()

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print("👉 Likely wrong model weights file (architecture mismatch)")
        return

    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, np.array(all_preds)))


# 🚀 RUN ALL MODELS

evaluate_model("ResNet", build_resnet, MODEL_PATHS["ResNet"])
evaluate_model("DenseNet", build_densenet, MODEL_PATHS["DenseNet"])
evaluate_model("EfficientNet", build_efficientnet, MODEL_PATHS["EfficientNet"])