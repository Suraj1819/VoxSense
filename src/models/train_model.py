# src/models/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.models.model_architecture import AudioClassifier

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # Load features
    df = pd.read_csv("data/processed/features.csv")
    X = df.drop('label', axis=1).values
    y = df['label'].values

    print(f"Total samples: {len(X)}")
    print(f"Classes: {np.unique(y)}")
    print(f"Samples per class: {pd.Series(y).value_counts()}")

    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ================== FIX FOR SMALL DATASET ==================
    n_samples = len(X)
    
    if n_samples <= 3:
        print("⚠️ Very small dataset detected. Using all data for training (no split).")
        X_train = X
        y_train = y_encoded
    else:
        # Agar koi class mein sirf 1 sample hai toh stratify=False kar do
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        except ValueError:
            print("⚠️ Stratify failed. Using simple split without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
    # =========================================================

    # Dataset & DataLoader
    train_dataset = AudioDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=min(8, len(X_train)), shuffle=True)

    # Model
    input_size = X.shape[1]
    num_classes = len(le.classes_)

    model = AudioClassifier(input_size=input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print(f"\nTraining started on {len(X_train)} samples | {num_classes} classes...")
    model.train()

    for epoch in range(80):        # More epochs for small data
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/80, Loss: {loss.item():.4f}")

    # Save Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': le,
        'input_size': input_size,
        'classes': le.classes_.tolist(),
        'num_classes': num_classes
    }, "models/best_model.pth")

    print("\n✅ Model trained and saved successfully!")
    print(f"Classes: {le.classes_.tolist()}")

if __name__ == "__main__":
    main()