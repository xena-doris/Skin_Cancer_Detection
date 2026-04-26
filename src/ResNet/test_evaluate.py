import torch
from torch.utils.data import DataLoader
from src.ResNet.dataset import SkinDataset, val_transform
from src.ResNet.model import build_model
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "data/test/test_labels.csv"
IMG_DIR = "data/test/images"

dataset = SkinDataset(CSV_PATH, IMG_DIR, transform=val_transform)
loader = DataLoader(dataset, batch_size=32)

model = build_model().to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth"))
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

print('=============Test Evaluation================')
print(classification_report(all_labels, np.array(all_preds)))