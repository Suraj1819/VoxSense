# src/models/train_cnn.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import librosa
import json
import warnings
warnings.filterwarnings('ignore')

from src.models.cnn_model import AudioCNN

class MultiSourceSpectrogramDataset(Dataset):
    """
    Dataset that loads spectrograms from multiple sources:
    1. Original dog/cat spectrograms
    2. Augmented dog/cat spectrograms
    3. Can also generate spectrograms on-the-fly from WAV files
    """
    
    def __init__(self, data_config, transform=None, use_spectrograms=True, use_wav=False):
        """
        data_config: Dictionary with paths to different data sources
        transform: Image transforms
        use_spectrograms: Load pre-generated spectrograms
        use_wav: Generate spectrograms from WAV files on-the-fly
        """
        self.transform = transform
        self.images = []
        self.labels = []
        self.wav_files = []  # For on-the-fly processing
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Expected classes
        self.expected_classes = ['dog_angry', 'dog_happy', 'dog_normal', 
                                 'cat_angry', 'cat_happy', 'cat_normal']
        
        # Create class mapping
        for idx, class_name in enumerate(self.expected_classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        print("\n" + "="*60)
        print("📁 LOADING DATA FROM MULTIPLE SOURCES")
        print("="*60)
        
        # Load spectrograms if enabled
        if use_spectrograms:
            self._load_spectrograms(data_config)
        
        # Load WAV files and generate spectrograms on-the-fly if enabled
        if use_wav:
            self._load_wav_files(data_config)
        
        if len(self.images) == 0 and len(self.wav_files) == 0:
            print("\n⚠️ WARNING: No images loaded! Check your data paths.")
        else:
            print(f"\n✅ Total spectrogram images: {len(self.images)}")
            print(f"✅ Total WAV files: {len(self.wav_files)}")
            print(f"✅ Total samples: {len(self)}")
            
            # Print class distribution for spectrograms
            if len(self.images) > 0:
                print("\n📊 Spectrogram Class Distribution:")
                unique, counts = np.unique(self.labels, return_counts=True)
                for label, count in zip(unique, counts):
                    class_name = self.idx_to_class[label]
                    print(f"   {class_name}: {count} images")
    
    def _load_spectrograms(self, data_config):
        """Load pre-generated spectrograms from folders"""
        
        print("\n🎨 Loading Spectrograms...")
        
        # Define spectrogram paths
        spec_paths = []
        
        # 1. Original spectrograms
        if 'original_spectrograms' in data_config:
            for animal in ['dog', 'cat']:
                path = Path(data_config['original_spectrograms']) / animal
                if path.exists():
                    spec_paths.append(path)
                    print(f"   ✓ Original {animal} spectrograms: {path}")
        
        # 2. Augmented spectrograms
        if 'augmented_spectrograms' in data_config:
            aug_path = Path(data_config['augmented_spectrograms'])
            for animal in ['dog', 'cat']:
                path = aug_path / animal
                if path.exists():
                    spec_paths.append(path)
                    print(f"   ✓ Augmented {animal} spectrograms: {path}")
        
        # Load images from each path
        total_loaded = 0
        for path in spec_paths:
            if path.exists():
                # Look for PNG files directly in folder
                for img_path in path.glob("*.png"):
                    label = self._get_label_from_path(img_path)
                    if label is not None and label in self.class_to_idx:
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[label])
                        total_loaded += 1
                
                # Also check for emotion subfolders
                for emotion_folder in path.glob("*"):
                    if emotion_folder.is_dir():
                        for img_path in emotion_folder.glob("*.png"):
                            label = self._get_label_from_path(img_path)
                            if label is not None and label in self.class_to_idx:
                                self.images.append(img_path)
                                self.labels.append(self.class_to_idx[label])
                                total_loaded += 1
        
        print(f"   Loaded {total_loaded} spectrograms")
    
    def _load_wav_files(self, data_config):
        """Load WAV files for on-the-fly spectrogram generation"""
        
        print("\n🎵 Loading WAV files for on-the-fly spectrogram generation...")
        
        wav_paths = []
        
        # Original WAV files
        if 'original_wav' in data_config:
            for animal in ['dog', 'cat']:
                path = Path(data_config['original_wav']) / animal
                if path.exists():
                    wav_paths.append(path)
                    print(f"   ✓ Original {animal} WAV: {path}")
        
        # Augmented WAV files
        if 'augmented_wav' in data_config:
            aug_path = Path(data_config['augmented_wav'])
            for animal in ['dog', 'cat']:
                path = aug_path / animal
                if path.exists():
                    wav_paths.append(path)
                    print(f"   ✓ Augmented {animal} WAV: {path}")
        
        # Store WAV paths for on-the-fly processing
        total_loaded = 0
        for path in wav_paths:
            if path.exists():
                for wav_path in path.glob("*.wav"):
                    label = self._get_label_from_path(wav_path)
                    if label is not None and label in self.class_to_idx:
                        self.wav_files.append((wav_path, self.class_to_idx[label]))
                        total_loaded += 1
        
        print(f"   Loaded {total_loaded} WAV files for on-the-fly processing")
    
    def _get_label_from_path(self, file_path):
        """Extract label from file path or filename"""
        file_stem = file_path.stem.lower()
        path_str = str(file_path).lower()
        
        # Check for dog/cat and emotion
        for class_name in self.expected_classes:
            animal, emotion = class_name.split('_')
            if animal in path_str or animal in file_stem:
                if emotion in path_str or emotion in file_stem:
                    return class_name
        
        # Try to extract from filename pattern
        if 'dog' in file_stem or 'dog' in path_str:
            if 'angry' in file_stem or 'bark' in file_stem or 'growl' in file_stem:
                return 'dog_angry'
            elif 'happy' in file_stem or 'joy' in file_stem or 'excited' in file_stem:
                return 'dog_happy'
            elif 'normal' in file_stem or 'calm' in file_stem:
                return 'dog_normal'
            else:
                return 'dog_normal'
        
        elif 'cat' in file_stem or 'cat' in path_str:
            if 'angry' in file_stem or 'hiss' in file_stem or 'growl' in file_stem:
                return 'cat_angry'
            elif 'happy' in file_stem or 'purr' in file_stem or 'meow' in file_stem:
                return 'cat_happy'
            elif 'normal' in file_stem or 'calm' in file_stem:
                return 'cat_normal'
            else:
                return 'cat_normal'
        
        return None
    
    def _generate_spectrogram(self, wav_path, target_size=(128, 128)):
        """Generate spectrogram from WAV file on-the-fly"""
        try:
            # Load audio
            y, sr = librosa.load(wav_path, sr=16000)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-255 range
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            mel_spec_db = (mel_spec_db * 255).astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(mel_spec_db, mode='L')
            img = img.resize(target_size)
            
            return img
        except Exception as e:
            print(f"Error generating spectrogram from {wav_path.name}: {e}")
            return None
    
    def __len__(self):
        return len(self.images) + len(self.wav_files)
    
    def __getitem__(self, idx):
        if idx < len(self.images):
            # Load pre-generated spectrogram
            img_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('L')
        else:
            # Generate spectrogram from WAV file on-the-fly
            wav_idx = idx - len(self.images)
            wav_path, label = self.wav_files[wav_idx]
            image = self._generate_spectrogram(wav_path)
            if image is None:
                # Fallback to a blank image
                image = Image.new('L', (128, 128), color=128)
        
        if self.transform:
            image = self.transform(image)
        
        # Return as (image, label) where label is a tensor
        return image, torch.tensor(label, dtype=torch.long)

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the trained model"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    
    # Get unique classes present in test data
    unique_labels = np.unique(all_labels)
    unique_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    if len(unique_labels) > 0:
        print("\n📈 Classification Report:")
        print(classification_report(all_labels, all_preds, 
                                    target_names=unique_class_names,
                                    labels=unique_labels))
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_class_names, 
                    yticklabels=unique_class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()
        print("📊 Confusion matrix saved to: models/confusion_matrix.png")
    
    return accuracy

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
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
    ax2.plot(train_accs, label='Train Accuracy', color='blue', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', color='red', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    print("📊 Training history saved to: models/training_history.png")

def main():
    print("="*60)
    print("🐕🐱 VOXSENSE - MULTI-SOURCE CNN TRAINING")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    IMG_SIZE = 128
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")
    
    # Create models directory
    Path("models").mkdir(parents=True, exist_ok=True)
    
    # Data configuration - ALL your data sources
    data_config = {
        'original_spectrograms': Path("data/spectrograms"),
        'augmented_spectrograms': Path("data/spectrograms/augmented"),
        'original_wav': Path("data/processed"),
        'augmented_wav': Path("data/processed_augmented"),
    }
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load dataset
    print("\n📁 Loading data from all sources...")
    full_dataset = MultiSourceSpectrogramDataset(
        data_config, 
        transform=train_transform,
        use_spectrograms=True,
        use_wav=True
    )
    
    if len(full_dataset) == 0:
        print("\n❌ No data found! Please check your folder structure.")
        return
    
    # Split dataset
    total_size = len(full_dataset)
    test_size = int(TEST_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    train_size = total_size - val_size - test_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total samples: {total_size}")
    print(f"   Training: {len(train_dataset)} images ({train_size/total_size*100:.1f}%)")
    print(f"   Validation: {len(val_dataset)} images ({val_size/total_size*100:.1f}%)")
    print(f"   Testing: {len(test_dataset)} images ({test_size/total_size*100:.1f}%)")
    
    # Initialize model
    num_classes = len(full_dataset.class_to_idx)
    model = AudioCNN(num_classes=num_classes).to(device)
    
    print(f"\n📊 Model Architecture:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes: {list(full_dataset.class_to_idx.keys())}")
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(full_dataset.class_to_idx.keys()))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("\n🚀 Starting Training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
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
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model WITH label encoder
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': full_dataset.class_to_idx,
                'idx_to_class': full_dataset.idx_to_class,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'label_encoder': label_encoder,  # Save label encoder
                'classes': list(full_dataset.class_to_idx.keys())  # Save class names
            }, "models/best_cnn_model.pth")
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for evaluation
    best_model = AudioCNN(num_classes=num_classes).to(device)
    if Path("models/best_cnn_model.pth").exists():
        checkpoint = torch.load("models/best_cnn_model.pth", map_location=device)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        class_names = [full_dataset.idx_to_class[i] for i in range(num_classes)]
        test_acc = evaluate_model(best_model, test_loader, device, class_names)
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE!")
        print("="*60)
        print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
        print(f"✅ Model saved to: models/best_cnn_model.pth")
        
        # Save class mapping
        class_mapping = {
            'class_to_idx': full_dataset.class_to_idx,
            'idx_to_class': full_dataset.idx_to_class,
            'classes': class_names,
            'label_encoder_classes': label_encoder.classes_.tolist()
        }
        with open('models/class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"✅ Class mapping saved to: models/class_mapping.json")

if __name__ == "__main__":
    main()