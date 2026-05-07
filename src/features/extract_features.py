# src/features/extract_features.py
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
from PIL import Image
import torchvision.transforms as transforms
warnings.filterwarnings('ignore')

def extract_audio_features(file_path):
    """Extract comprehensive audio features from audio file"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # MFCC Features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Additional Features
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_std = np.std(spectral_rolloff)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Pitch/Tempo features
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > np.max(magnitudes) * 0.1]
            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        except:
            tempo = 0
            pitch_mean = 0
            pitch_std = 0
        
        # Combine all features (75 features total)
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            [zcr_mean, zcr_std],
            [centroid_mean, centroid_std],
            [rolloff_mean, rolloff_std],
            [rms_mean, rms_std],
            [tempo, pitch_mean, pitch_std]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features from {file_path.name}: {e}")
        return None

def extract_spectrogram_features(image_path):
    """Extract features from spectrogram image (for CNN)"""
    try:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        img = Image.open(image_path).convert('L')
        img_tensor = transform(img)
        # Flatten the image to use as features (16384 features)
        features = img_tensor.numpy().flatten()
        
        return features
        
    except Exception as e:
        print(f"Error extracting spectrogram features from {image_path.name}: {e}")
        return None

def get_label_from_filename(filename):
    """Extract label from filename"""
    filename_lower = filename.lower()
    
    # Remove augmentation suffix if present
    if '_aug' in filename_lower:
        filename_lower = filename_lower.split('_aug')[0]
    
    # Check for dog files
    if 'dog' in filename_lower:
        if 'angry' in filename_lower:
            return 'dog_angry'
        elif 'happy' in filename_lower:
            return 'dog_happy'
        elif 'normal' in filename_lower:
            return 'dog_normal'
        else:
            return 'dog_normal'
    
    # Check for cat files
    elif 'cat' in filename_lower:
        if 'angry' in filename_lower:
            return 'cat_angry'
        elif 'happy' in filename_lower:
            return 'cat_happy'
        elif 'normal' in filename_lower:
            return 'cat_normal'
        else:
            return 'cat_normal'
    
    else:
        return 'unknown'

def process_audio_folder(folder_path, source_name, feature_type='audio'):
    """Process audio files from a folder"""
    
    if not folder_path.exists():
        print(f"⚠️ Folder not found: {folder_path}")
        return [], [], 0
    
    # Find all audio files
    audio_files = list(folder_path.glob("*.wav"))
    
    if not audio_files:
        print(f"⚠️ No WAV files found in {folder_path}")
        return [], [], 0
    
    print(f"\n📁 Processing {source_name}: {len(audio_files)} files")
    print(f"   Location: {folder_path}")
    
    features_list = []
    labels_list = []
    failed_files = []
    
    for audio_file in tqdm(audio_files, desc=f"Extracting {source_name}"):
        try:
            # Extract features
            features = extract_audio_features(audio_file)
            
            if features is not None:
                features_list.append(features)
                label = get_label_from_filename(audio_file.stem)
                labels_list.append(label)
            else:
                failed_files.append(audio_file.name)
                
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            failed_files.append(audio_file.name)
    
    if failed_files:
        print(f"⚠️ Failed: {len(failed_files)} files")
    
    return features_list, labels_list, len(audio_files)

def process_spectrogram_folder(folder_path, source_name):
    """Process spectrogram images from a folder"""
    
    if not folder_path.exists():
        print(f"⚠️ Folder not found: {folder_path}")
        return [], [], 0
    
    # Find all spectrogram images
    image_files = list(folder_path.glob("*.png"))
    
    if not image_files:
        print(f"⚠️ No PNG files found in {folder_path}")
        return [], [], 0
    
    print(f"\n📁 Processing {source_name}: {len(image_files)} spectrograms")
    print(f"   Location: {folder_path}")
    
    features_list = []
    labels_list = []
    failed_files = []
    
    for image_file in tqdm(image_files, desc=f"Extracting {source_name}"):
        try:
            # Extract features from spectrogram
            features = extract_spectrogram_features(image_file)
            
            if features is not None:
                features_list.append(features)
                label = get_label_from_filename(image_file.stem)
                labels_list.append(label)
            else:
                failed_files.append(image_file.name)
                
        except Exception as e:
            print(f"❌ Error processing {image_file.name}: {e}")
            failed_files.append(image_file.name)
    
    if failed_files:
        print(f"⚠️ Failed: {len(failed_files)} files")
    
    return features_list, labels_list, len(image_files)

def main():
    print("="*60)
    print("🐕🐱 VOXSENSE - FEATURE EXTRACTOR")
    print("="*60)
    
    print("\n📌 What data do you want to process?")
    print("1. 🎵 Original Audio only (data/processed/dog/ & cat/)")
    print("2. 🔄 Augmented Audio only (data/processed_augmented/dog/ & cat/)")
    print("3. 🎨 Spectrograms only (data/spectrograms/)")
    print("4. 🎵 + 🔄 (Original + Augmented Audio)")
    print("5. 🎵 + 🎨 (Original Audio + Spectrograms)")
    print("6. 🔄 + 🎨 (Augmented Audio + Spectrograms)")
    print("7. 🎯 ALL (Original + Augmented + Spectrograms) - Recommended")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    # Define paths
    processed_path = Path("data/processed")
    augmented_path = Path("data/processed_augmented")
    spectrogram_path = Path("data/spectrograms")
    
    all_features = []
    all_labels = []
    stats = {}
    
    # Process based on choice
    if choice in ['1', '4', '5', '7']:
        print("\n" + "="*60)
        print("🎵 PROCESSING ORIGINAL AUDIO FILES")
        print("="*60)
        
        # Original Dog
        dog_original = processed_path / "dog"
        f, l, c = process_audio_folder(dog_original, "Original Dog", 'audio')
        all_features.extend(f)
        all_labels.extend(l)
        stats['original_dog'] = c
        
        # Original Cat
        cat_original = processed_path / "cat"
        f, l, c = process_audio_folder(cat_original, "Original Cat", 'audio')
        all_features.extend(f)
        all_labels.extend(l)
        stats['original_cat'] = c
    
    if choice in ['2', '4', '6', '7']:
        print("\n" + "="*60)
        print("🔄 PROCESSING AUGMENTED AUDIO FILES")
        print("="*60)
        
        # Augmented Dog
        dog_augmented = augmented_path / "dog"
        if dog_augmented.exists():
            f, l, c = process_audio_folder(dog_augmented, "Augmented Dog", 'audio')
            all_features.extend(f)
            all_labels.extend(l)
            stats['augmented_dog'] = c
        else:
            print(f"⚠️ Augmented dog folder not found: {dog_augmented}")
        
        # Augmented Cat
        cat_augmented = augmented_path / "cat"
        if cat_augmented.exists():
            f, l, c = process_audio_folder(cat_augmented, "Augmented Cat", 'audio')
            all_features.extend(f)
            all_labels.extend(l)
            stats['augmented_cat'] = c
        else:
            print(f"⚠️ Augmented cat folder not found: {cat_augmented}")
    
    if choice in ['3', '5', '6', '7']:
        print("\n" + "="*60)
        print("🎨 PROCESSING SPECTROGRAMS")
        print("="*60)
        
        # Spectrogram Dog (original)
        spec_dog = spectrogram_path / "dog"
        if spec_dog.exists():
            f, l, c = process_spectrogram_folder(spec_dog, "Spectrogram Dog")
            all_features.extend(f)
            all_labels.extend(l)
            stats['spectrogram_dog'] = c
        else:
            print(f"⚠️ Spectrogram dog folder not found: {spec_dog}")
        
        # Spectrogram Cat (original)
        spec_cat = spectrogram_path / "cat"
        if spec_cat.exists():
            f, l, c = process_spectrogram_folder(spec_cat, "Spectrogram Cat")
            all_features.extend(f)
            all_labels.extend(l)
            stats['spectrogram_cat'] = c
        else:
            print(f"⚠️ Spectrogram cat folder not found: {spec_cat}")
        
        # Augmented Spectrograms
        spec_aug_dog = spectrogram_path / "augmented" / "dog"
        if spec_aug_dog.exists():
            f, l, c = process_spectrogram_folder(spec_aug_dog, "Augmented Spectrogram Dog")
            all_features.extend(f)
            all_labels.extend(l)
            stats['spectrogram_aug_dog'] = c
        
        spec_aug_cat = spectrogram_path / "augmented" / "cat"
        if spec_aug_cat.exists():
            f, l, c = process_spectrogram_folder(spec_aug_cat, "Augmented Spectrogram Cat")
            all_features.extend(f)
            all_labels.extend(l)
            stats['spectrogram_aug_cat'] = c
    
    if not all_features:
        print("\n❌ No features extracted! Please check your data folders.")
        print("\nExpected folder structure:")
        print("  data/processed/dog/*.wav")
        print("  data/processed/cat/*.wav")
        print("  data/processed_augmented/dog/*.wav")
        print("  data/processed_augmented/cat/*.wav")
        print("  data/spectrograms/dog/*.png")
        print("  data/spectrograms/cat/*.png")
        print("  data/spectrograms/augmented/dog/*.png")
        print("  data/spectrograms/augmented/cat/*.png")
        return
    
    # Convert to numpy array
    X = np.array(all_features)
    
    # Create DataFrame
    df = pd.DataFrame(X)
    df['label'] = all_labels
    
    # Save features
    features_csv_path = processed_path / "features.csv"
    df.to_csv(features_csv_path, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("📊 FEATURE EXTRACTION SUMMARY")
    print("="*60)
    
    print("\n📈 Data Sources Processed:")
    for key, count in stats.items():
        if count > 0:
            print(f"   {key.replace('_', ' ').title()}: {count} files")
    
    print(f"\n✅ Total samples: {len(df)}")
    print(f"✅ Feature dimensions: {X.shape[1]}")
    
    print("\n📈 Class Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✅ Features saved to: {features_csv_path}")
    
    # Save detailed stats
    stats_file = processed_path / "feature_extraction_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("VOXSENSE - FEATURE EXTRACTION STATISTICS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Feature dimensions: {X.shape[1]}\n\n")
        f.write("Data Sources:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nClass Distribution:\n")
        for label, count in class_counts.items():
            f.write(f"  {label}: {count}\n")
    
    print(f"✅ Statistics saved to: {stats_file}")
    
    # Show folder structure summary
    print("\n📂 Folder Structure Summary:")
    print("data/")
    print("  processed/")
    print("    dog/     ← Original dog audio")
    print("    cat/     ← Original cat audio")
    print("    features.csv  ← Extracted features")
    print("  processed_augmented/")
    print("    dog/     ← Augmented dog audio")
    print("    cat/     ← Augmented cat audio")
    print("  spectrograms/")
    print("    dog/     ← Dog spectrograms")
    print("    cat/     ← Cat spectrograms")
    print("    augmented/")
    print("      dog/   ← Augmented dog spectrograms")
    print("      cat/   ← Augmented cat spectrograms")
    
    print("\n🎉 Feature extraction complete!")

if __name__ == "__main__":
    main()