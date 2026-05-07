# src/models/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AudioCNN(nn.Module):
    """CNN Model for Spectrogram Classification"""
    
    def __init__(self, num_classes=6):
        super(AudioCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # Calculate flattened size (for 128x128 input)
        self.flattened_size = 256 * 8 * 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class SpectrogramDataset(Dataset):
    """Custom Dataset for loading spectrogram images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Check if root_dir exists
        if not self.root_dir.exists():
            raise ValueError(f"Directory {root_dir} does not exist!")
        
        # Get all class folders (looking for emotion subfolders)
        class_folders = []
        for animal in ['dog', 'cat']:
            animal_path = self.root_dir / animal
            if animal_path.exists():
                for emotion in ['angry', 'happy', 'normal']:
                    class_folder = animal_path / emotion
                    if class_folder.exists():
                        class_folders.append(class_folder)
        
        # If no emotion subfolders, look for direct class folders
        if not class_folders:
            class_folders = [f for f in self.root_dir.iterdir() if f.is_dir()]
        
        # Create class mapping
        for idx, folder in enumerate(sorted(class_folders)):
            class_name = folder.name
            # Handle different folder structures
            if 'dog' in str(folder):
                if 'angry' in class_name:
                    class_name = 'dog_angry'
                elif 'happy' in class_name:
                    class_name = 'dog_happy'
                elif 'normal' in class_name:
                    class_name = 'dog_normal'
            elif 'cat' in str(folder):
                if 'angry' in class_name:
                    class_name = 'cat_angry'
                elif 'happy' in class_name:
                    class_name = 'cat_happy'
                elif 'normal' in class_name:
                    class_name = 'cat_normal'
            
            self.class_to_idx[class_name] = idx
            
            # Get all images in class folder
            for img_path in folder.glob("*.png"):
                self.images.append(img_path)
                self.labels.append(idx)
        
        # Reverse mapping for predictions
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.images)} images from {len(class_folders)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path="models/best_cnn_model.pth", class_to_idx=None):
    """Train the CNN model"""
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss_avg)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': model.fc3.out_features,
                'class_to_idx': class_to_idx,
                'idx_to_class': {v: k for k, v in class_to_idx.items()} if class_to_idx else None,
                'val_acc': val_acc,
                'train_acc': train_acc
            }, save_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the trained model"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ F1 Score: {f1:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    
    return accuracy, f1, precision, recall, cm

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue', marker='o')
    ax1.plot(val_losses, label='Val Loss', color='red', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue', marker='o')
    ax2.plot(val_accuracies, label='Val Accuracy', color='red', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def predict_spectrogram(model, image_path, transform, device, class_names):
    """Predict emotion from a single spectrogram"""
    
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()

def main():
    print("="*60)
    print("🐕🐱 VOXSENSE - CNN MODEL TRAINING")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 8  # Reduced batch size for small dataset
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    IMG_SIZE = 128
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load dataset
    spectrogram_path = Path("data/spectrograms")
    
    if not spectrogram_path.exists():
        print(f"❌ Spectrogram directory not found: {spectrogram_path}")
        print("Please run generate_spectrograms.py first!")
        return
    
    # Create dataset
    full_dataset = SpectrogramDataset(spectrogram_path, transform=train_transform)
    
    # Store class mapping
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = full_dataset.idx_to_class
    
    # Split dataset (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms for validation and test
    # We need to modify the transform of the underlying dataset
    full_dataset.transform = val_test_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    print(f"   Testing: {len(test_dataset)} images")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Initialize model
    num_classes = len(class_to_idx)
    model = AudioCNN(num_classes=num_classes).to(device)
    
    print(f"\n📊 Model Architecture:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {list(class_to_idx.keys())}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("\n🚀 Starting Training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, 
        save_path="models/best_cnn_model.pth", class_to_idx=class_to_idx
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Load best model for evaluation
    best_model = AudioCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load("models/best_cnn_model.pth", map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    class_names = [idx_to_class[i] for i in range(num_classes)]
    accuracy, f1, precision, recall, cm = evaluate_model(best_model, test_loader, device, class_names)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy * 100,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'class_names': class_names,
        'num_classes': num_classes
    }
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'metrics': metrics
    }, "models/best_cnn_model.pth")
    
    print("\n🎉 Training Complete!")
    print(f"✅ Best model saved to: models/best_cnn_model.pth")
    print(f"✅ Training history saved to: models/training_history.png")
    print(f"✅ Confusion matrix saved to: models/confusion_matrix.png")

if __name__ == "__main__":
    main()