import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.DenseNet.dataset import SkinDataset, train_transform, val_transform
from src.DenseNet.model import build_model
from sklearn.model_selection import train_test_split
import pandas as pd
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "data/final_train/final_train_labels.csv"
IMG_DIR = "data/final_train/images"

# Create models folder if missing
os.makedirs("models", exist_ok=True)

# Split dataset
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['malignant'], random_state=42)

train_df.to_csv("train_temp.csv", index=False)
val_df.to_csv("val_temp.csv", index=False)

# Datasets
train_dataset = SkinDataset("train_temp.csv", IMG_DIR, transform=train_transform)
val_dataset = SkinDataset("val_temp.csv", IMG_DIR, transform=val_transform)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = build_model().to(DEVICE)

# 🔥 Better loss
pos_weight = torch.tensor([2.0]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 🔥 Better optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

EPOCHS = 20

best_loss = float('inf')
patience = 5
counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 🔍 VALIDATION
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    acc = correct / total

    print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={acc:.4f}")

    # 🔥 EARLY STOPPING
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "models/best_DenseNet_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

print("Training complete.")