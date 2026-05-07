# reports/report.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.models.model_architecture import AudioClassifier
from src.features.extract_features import extract_features

def generate_report():
    print("📊 Generating Model Report...\n")
    
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print("❌ Model not found! Pehle model train karo.")
        return

    # Load model
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    num_classes = len(checkpoint['label_encoder'].classes_)
    input_size = checkpoint.get('input_size', 40)

    model = AudioClassifier(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    le = checkpoint['label_encoder']
    model.eval()

    print(f"✅ Model Loaded | Classes: {list(le.classes_)} | Input Size: {input_size}\n")

    # Load features
    features_path = Path("data/processed/features.csv")
    if not features_path.exists():
        print("❌ features.csv not found!")
        return

    df = pd.read_csv(features_path)
    X = df.drop('label', axis=1).values
    y_true = df['label'].values

    # Prediction
    print("🔮 Making predictions...")
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred = torch.argmax(outputs, dim=1).numpy()

    y_true_encoded = le.transform(y_true)

    # Metrics
    accuracy = accuracy_score(y_true_encoded, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    # Classification Report
    print("📋 Classification Report:")
    print(classification_report(y_true, le.inverse_transform(y_pred), zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()
    print("📊 Confusion Matrix saved as: reports/confusion_matrix.png")

    # Save Report as Text
    report_text = f"""
VoxSense - Model Performance Report
====================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Model Information:
-----------------
- Classes          : {list(le.classes_)}
- Total Samples    : {len(X)}
- Input Features   : {input_size}
- Accuracy         : {accuracy:.4f} ({accuracy*100:.2f}%)

Classification Report:
---------------------
{classification_report(y_true, le.inverse_transform(y_pred), zero_division=0)}

Note: This is a basic report. 
For better results, more audio samples per class are recommended.
"""

    with open("reports/model_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n✅ Report saved as: reports/model_report.txt")
    print("✅ Confusion Matrix saved as: reports/confusion_matrix.png")
    print("\n🎉 Report Generation Completed!")

def main():
    generate_report()

if __name__ == "__main__":
    main()