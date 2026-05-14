# # src/evaluation/evaluate_cnn.py
# import sys
# from pathlib import Path

# # Add project root to Python path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import LabelEncoder
# import warnings
# import json
# from datetime import datetime
# from tqdm import tqdm
# warnings.filterwarnings("ignore")

# from src.models.cnn_model import AudioCNN

# class SpectrogramTestDataset:
#     """Dataset for loading spectrogram images for testing"""
    
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = Path(root_dir)
#         self.transform = transform
#         self.images = []
#         self.labels = []
#         self.class_names = []
        
#         # Expected classes
#         expected_classes = ['dog_angry', 'dog_happy', 'dog_normal', 
#                            'cat_angry', 'cat_happy', 'cat_normal']
        
#         # Load images from folders
#         for class_name in expected_classes:
#             class_folder = self.root_dir / class_name
#             if class_folder.exists():
#                 for img_path in class_folder.glob("*.png"):
#                     self.images.append(img_path)
#                     self.labels.append(class_name)
        
#         # If no class folders, try to infer from filenames
#         if len(self.images) == 0:
#             for img_path in self.root_dir.glob("*.png"):
#                 stem = img_path.stem.lower()
#                 for class_name in expected_classes:
#                     if class_name in stem:
#                         self.images.append(img_path)
#                         self.labels.append(class_name)
#                         break
        
#         # Encode labels
#         self.label_encoder = LabelEncoder()
#         self.label_encoder.fit(expected_classes)
#         self.encoded_labels = self.label_encoder.transform(self.labels)
#         self.class_names = list(self.label_encoder.classes_)
        
#         print(f"Loaded {len(self.images)} spectrograms")
#         print(f"Classes: {self.class_names}")
        
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         label = self.encoded_labels[idx]
        
#         image = Image.open(img_path).convert('L')
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label

# def evaluate_cnn_model():
#     print_colored("\n" + "="*60, Colors.HEADER)
#     print_colored("🔍 VOXSENSE - CNN MODEL EVALUATION", Colors.HEADER)
#     print_colored("="*60, Colors.HEADER)
    
#     # Paths
#     model_path = Path("models/best_cnn_model.pth")
#     spectrogram_path = Path("data/spectrograms")
#     eval_dir = Path("src/evaluation")
#     eval_dir.mkdir(parents=True, exist_ok=True)
    
#     # Check if model exists
#     if not model_path.exists():
#         print_colored(f"❌ Model not found at {model_path}", Colors.RED)
#         print("\nPlease train the model first: python src/models/train_cnn.py")
#         return
    
#     # Check if spectrograms exist
#     if not spectrogram_path.exists():
#         print_colored(f"❌ Spectrograms not found at {spectrogram_path}", Colors.RED)
#         return
    
#     # Load model
#     print_colored("\n📦 Loading model...", Colors.BOLD)
#     checkpoint = torch.load(model_path, map_location='cpu')
    
#     # Get number of classes
#     if 'num_classes' in checkpoint:
#         num_classes = checkpoint['num_classes']
#     elif 'class_to_idx' in checkpoint:
#         num_classes = len(checkpoint['class_to_idx'])
#     else:
#         num_classes = 6
    
#     model = AudioCNN(num_classes=num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     print_colored("✅ Model loaded successfully!", Colors.GREEN)
    
#     # Load test dataset
#     print_colored("\n📁 Loading spectrograms...", Colors.BOLD)
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
    
#     test_dataset = SpectrogramTestDataset(spectrogram_path, transform=transform)
    
#     if len(test_dataset) == 0:
#         print_colored("❌ No test images found!", Colors.RED)
#         return
    
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
#     # Make predictions
#     print_colored("\n🔄 Making predictions...", Colors.BOLD)
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc="Evaluating"):
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.numpy())
#             all_labels.extend(labels.numpy())
    
#     # Calculate metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
#     class_names = test_dataset.class_names
    
#     print_colored("\n" + "="*60, Colors.GREEN)
#     print_colored("📊 EVALUATION RESULTS", Colors.GREEN)
#     print_colored("="*60, Colors.GREEN)
    
#     print(f"\n🎯 Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
#     print(f"📌 Precision : {precision:.4f}")
#     print(f"📌 Recall    : {recall:.4f}")
#     print(f"📌 F1-Score  : {f1:.4f}")
    
#     print(f"\n{classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)}")
    
#     # Confusion Matrix
#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names)
#     plt.title('Confusion Matrix - CNN Model', fontsize=14, fontweight='bold')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.tight_layout()
#     plt.savefig(eval_dir / "cnn_confusion_matrix.png", dpi=300)
#     plt.close()
    
#     print_colored(f"\n📊 Confusion matrix saved to: {eval_dir}/cnn_confusion_matrix.png", Colors.BLUE)
    
#     # Save report
#     report = f"""VOXSENSE - CNN MODEL EVALUATION REPORT
# ============================================
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# OVERALL METRICS:
# ----------------
# Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)
# Precision : {precision:.4f}
# Recall    : {recall:.4f}
# F1-Score  : {f1:.4f}

# CLASSES: {', '.join(class_names)}

# File saved: {eval_dir}/cnn_confusion_matrix.png
# """
#     with open(eval_dir / "cnn_evaluation_report.txt", 'w') as f:
#         f.write(report)
    
#     print_colored("\n✅ Evaluation completed!", Colors.GREEN)

# # Colors class
# class Colors:
#     HEADER = '\033[95m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     RED = '\033[91m'
#     END = '\033[0m'
#     BOLD = '\033[1m'

# def print_colored(text, color=Colors.GREEN):
#     print(f"{color}{text}{Colors.END}")

# if __name__ == "__main__":
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     evaluate_cnn_model()


# src/evaluation/evaluate_cnn.py
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
from tqdm import tqdm
from datetime import datetime

from src.models.cnn_model import AudioCNN

warnings.filterwarnings("ignore")

def evaluate_cnn_model():
    print("="*70)
    print("🔍 VOXSENSE - CNN MODEL EVALUATION")
    print("="*70)

    model_path = Path("models/best_cnn_model.pth")
    spectrogram_dir = Path("data/spectrograms")
    eval_dir = Path("src/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("❌ Model not found! Please train the model first.")
        print("Run: python -m src.models.train_cnn")
        return

    if not spectrogram_dir.exists() or len(list(spectrogram_dir.glob("*.png"))) == 0:
        print("❌ No spectrograms found!")
        print("Run: python -m src.preprocessing.generate_spectrograms")
        return

    # ================== Load Model Safely ==================
    print("\n📦 Loading Model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

    num_classes = checkpoint.get('num_classes', 3)
    model = AudioCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    le = checkpoint['label_encoder']

    print(f"✅ Model Loaded Successfully!")
    print(f"Classes: {le.classes_}")

    # ================== Load Spectrograms ==================
    print("\n📁 Loading Spectrograms...")
    image_paths = list(spectrogram_dir.glob("*.png"))
    print(f"Found {len(image_paths)} spectrogram images.")

    # Simple prediction on all images
    transform = torch.nn.Sequential(
        torch.nn.Resize((128, 128)),
        torch.nn.ToTensor(),
        torch.nn.Normalize([0.5], [0.5])
    )

    all_preds = []
    all_labels = []

    print("\n🔄 Running Evaluation...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # Try to infer label from filename
            stem = img_path.stem.lower()
            true_label = None
            for cls in le.classes_:
                if cls.lower() in stem:
                    true_label = cls
                    break
            
            if true_label is None:
                continue

            # Load and transform image
            img = Image.open(img_path).convert('L')
            img_tensor = transform(img).unsqueeze(0)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            
            all_preds.append(pred)
            all_labels.append(le.transform([true_label])[0])

    if len(all_labels) == 0:
        print("❌ Could not match any labels with images.")
        return

    # ================== Calculate Metrics ==================
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("\n" + "="*60)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"🎯 Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📌 Precision  : {precision:.4f}")
    print(f"📌 Recall     : {recall:.4f}")
    print(f"📌 F1-Score   : {f1:.4f}")
    print("="*60)

    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=le.classes_, 
                                zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix - VoxSense CNN Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(eval_dir / "cnn_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\n📊 Confusion Matrix saved: src/evaluation/cnn_confusion_matrix.png")

    # Save Report
    report = f"""VOXSENSE CNN EVALUATION REPORT
====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Model Performance:
------------------
Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-Score  : {f1:.4f}

Classes: {list(le.classes_)}

Confusion Matrix saved successfully.
"""

    with open(eval_dir / "cnn_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Full Report saved: src/evaluation/cnn_evaluation_report.txt")
    print("🎉 Evaluation Completed Successfully!")

if __name__ == "__main__":
    evaluate_cnn_model()