# src/preprocessing/augment_data.py
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def augment_audio(input_file, output_dir, num_aug=5):
    """
    Augment audio file with various techniques
    
    Techniques:
    - Time stretching
    - Noise addition
    - Pitch shifting
    - Speed change
    - Volume change
    """
    try:
        y, sr = librosa.load(input_file, sr=16000)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(input_file).stem
        augmented_files = []

        for i in range(num_aug):
            y_aug = y.copy()
            applied_techniques = []
            
            # Apply random augmentation techniques
            technique = random.choice([1, 2, 3, 4, 5])
            
            # 1. Time Stretch
            if technique == 1 or random.random() > 0.6:
                rate = random.uniform(0.7, 1.3)
                y_aug = librosa.effects.time_stretch(y_aug, rate=rate)
                applied_techniques.append(f"time_stretch({rate:.2f})")
            
            # 2. Add Noise
            if technique == 2 or random.random() > 0.6:
                noise_level = random.uniform(0.001, 0.01)
                noise = np.random.randn(len(y_aug)) * noise_level
                y_aug = y_aug + noise
                applied_techniques.append(f"noise({noise_level:.4f})")
            
            # 3. Pitch Shift
            if technique == 3 or random.random() > 0.6:
                steps = random.randint(-4, 4)
                if steps != 0:
                    y_aug = librosa.effects.pitch_shift(y_aug, sr=sr, n_steps=steps)
                    applied_techniques.append(f"pitch_shift({steps})")
            
            # 4. Speed Change (without pitch correction)
            if technique == 4 or random.random() > 0.7:
                speed = random.uniform(0.8, 1.2)
                y_aug = librosa.effects.time_stretch(y_aug, rate=speed)
                applied_techniques.append(f"speed({speed:.2f})")
            
            # 5. Volume Change
            if technique == 5 or random.random() > 0.6:
                volume = random.uniform(0.5, 1.5)
                y_aug = y_aug * volume
                applied_techniques.append(f"volume({volume:.2f})")
            
            # Ensure audio is within reasonable range
            y_aug = np.clip(y_aug, -1, 1)
            
            # Save augmented file
            output_path = output_dir / f"{stem}_aug{i+1}.wav"
            sf.write(output_path, y_aug, sr)
            augmented_files.append(output_path.name)
        
        return True, augmented_files, applied_techniques
    
    except Exception as e:
        print(f"❌ Error augmenting {input_file.name}: {e}")
        return False, [], []

def process_animal_augmentation(animal_type, processed_dir, aug_dir, num_aug=8):
    """Augment audio files for specific animal"""
    
    # Define paths
    animal_processed_dir = processed_dir / animal_type
    animal_aug_dir = aug_dir / animal_type
    
    if not animal_processed_dir.exists():
        print(f"⚠️ {animal_type} folder not found: {animal_processed_dir}")
        return 0, 0
    
    # Find all WAV files
    wav_files = list(animal_processed_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"⚠️ No WAV files found in {animal_processed_dir}")
        return 0, 0
    
    print(f"\n📁 Augmenting {animal_type.upper()} files: {len(wav_files)} original files")
    print(f"   Each file will generate {num_aug} augmented versions")
    print(f"   Output: {animal_aug_dir}")
    
    total_original = len(wav_files)
    total_augmented = 0
    failed_files = []
    
    for wav_file in tqdm(wav_files, desc=f"Augmenting {animal_type} files"):
        success, augmented, techniques = augment_audio(wav_file, animal_aug_dir, num_aug)
        if success:
            total_augmented += len(augmented)
        else:
            failed_files.append(wav_file.name)
    
    if failed_files:
        print(f"⚠️ Failed to augment {len(failed_files)} files")
    
    return total_original, total_augmented

def combine_original_and_augmented(processed_dir, aug_dir):
    """Combine original and augmented files for training"""
    
    print("\n" + "="*60)
    print("📊 AUGMENTATION SUMMARY")
    print("="*60)
    
    total_original = 0
    total_augmented = 0
    
    for animal in ['dog', 'cat']:
        animal_processed = processed_dir / animal
        animal_aug = aug_dir / animal
        
        original_count = len(list(animal_processed.glob("*.wav"))) if animal_processed.exists() else 0
        augmented_count = len(list(animal_aug.glob("*.wav"))) if animal_aug.exists() else 0
        
        total_original += original_count
        total_augmented += augmented_count
        
        print(f"\n🐕🐱 {animal.upper()}:")
        print(f"   Original files: {original_count}")
        print(f"   Augmented files: {augmented_count}")
        print(f"   Total available: {original_count + augmented_count}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total original: {total_original}")
    print(f"   Total augmented: {total_augmented}")
    print(f"   Total files for training: {total_original + total_augmented}")
    print(f"   Data increase: {total_original} → {total_original + total_augmented} ({((total_original + total_augmented)/total_original - 1)*100:.1f}% increase)")
    
    return total_original, total_augmented

def main():
    print("="*60)
    print("🎵 VOXSENSE - DATA AUGMENTATION")
    print("="*60)
    print("\nAugmentation Techniques:")
    print("  • Time Stretching (0.7x - 1.3x)")
    print("  • Noise Addition (0.001 - 0.01)")
    print("  • Pitch Shifting (-4 to +4 semitones)")
    print("  • Speed Change (0.8x - 1.2x)")
    print("  • Volume Change (0.5x - 1.5x)")
    
    # Define paths
    processed_dir = Path("data/processed")
    aug_dir = Path("data/processed_augmented")
    
    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"\n❌ Processed directory not found: {processed_dir}")
        print("Please run feature extraction first!")
        return
    
    # Ask for number of augmentations
    try:
        num_aug = int(input("\n📝 Number of augmentations per file (default 8): ") or 8)
        num_aug = max(1, min(20, num_aug))  # Limit between 1-20
    except:
        num_aug = 8
    
    print(f"\n✅ Augmenting each file {num_aug} times")
    
    # Process DOG files
    dog_original, dog_augmented = process_animal_augmentation("dog", processed_dir, aug_dir, num_aug)
    
    # Process CAT files
    cat_original, cat_augmented = process_animal_augmentation("cat", processed_dir, aug_dir, num_aug)
    
    # Show summary
    combine_original_and_augmented(processed_dir, aug_dir)
    
    # Optional: Create combined dataset for training
    print("\n" + "="*60)
    print("💾 SAVING AUGMENTED DATA")
    print("="*60)
    
    # Save list of all augmented files
    summary_file = aug_dir / "augmentation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("VOXSENSE - DATA AUGMENTATION SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Augmentations per file: {num_aug}\n\n")
        
        for animal in ['dog', 'cat']:
            animal_aug = aug_dir / animal
            if animal_aug.exists():
                aug_files = list(animal_aug.glob("*.wav"))
                f.write(f"\n{animal.upper()} Augmented Files:\n")
                for file in aug_files:
                    f.write(f"  - {file.name}\n")
    
    print(f"✅ Augmentation summary saved to: {summary_file}")
    
    # Show folder structure
    print("\n📂 FINAL FOLDER STRUCTURE:")
    print("data/")
    print("  processed/")
    print("    dog/     (original dog WAV files)")
    print("    cat/     (original cat WAV files)")
    print("  processed_augmented/")
    print("    dog/     (augmented dog WAV files)")
    print("    cat/     (augmented cat WAV files)")
    
    print("\n🎉 Data Augmentation Complete!")

def augment_specific_animal():
    """Augment only specific animal"""
    print("\n" + "="*60)
    print("Augment specific animal:")
    print("1. 🐕 Dog only")
    print("2. 🐱 Cat only")
    print("3. 🐕🐱 Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    processed_dir = Path("data/processed")
    aug_dir = Path("data/processed_augmented")
    
    try:
        num_aug = int(input("Number of augmentations per file (default 8): ") or 8)
    except:
        num_aug = 8
    
    if choice == "1":
        original, augmented = process_animal_augmentation("dog", processed_dir, aug_dir, num_aug)
        print(f"\n✅ Dog augmentation complete! {original} → {original + augmented}")
    elif choice == "2":
        original, augmented = process_animal_augmentation("cat", processed_dir, aug_dir, num_aug)
        print(f"\n✅ Cat augmentation complete! {original} → {original + augmented}")
    elif choice == "3":
        dog_orig, dog_aug = process_animal_augmentation("dog", processed_dir, aug_dir, num_aug)
        cat_orig, cat_aug = process_animal_augmentation("cat", processed_dir, aug_dir, num_aug)
        print(f"\n✅ Both augmentation complete!")
        print(f"   Dog: {dog_orig} → {dog_orig + dog_aug}")
        print(f"   Cat: {cat_orig} → {cat_orig + cat_aug}")
    else:
        print("❌ Invalid choice!")

def preview_augmentation():
    """Preview augmentation techniques on a sample file"""
    print("\n🔍 AUGMENTATION PREVIEW")
    print("="*40)
    
    # Find a sample file
    processed_dir = Path("data/processed")
    sample_file = None
    
    for animal in ['dog', 'cat']:
        animal_path = processed_dir / animal
        if animal_path.exists():
            files = list(animal_path.glob("*.wav"))
            if files:
                sample_file = files[0]
                break
    
    if not sample_file:
        print("No sample file found to preview!")
        return
    
    print(f"\nSample file: {sample_file.name}")
    print("\nPreviewing 3 augmented versions...")
    
    # Create temp preview directory
    preview_dir = Path("data/preview")
    preview_dir.mkdir(exist_ok=True)
    
    # Generate preview augmentations
    success, files, techniques = augment_audio(sample_file, preview_dir, num_aug=3)
    
    if success:
        print("\nGenerated preview files:")
        for i, (file, tech) in enumerate(zip(files, techniques), 1):
            print(f"  {i}. {file}")
            print(f"     Techniques: {tech}")
        print(f"\n✅ Preview files saved in: {preview_dir}")
        print("💡 You can listen to these files to understand augmentation effects")
    else:
        print("❌ Failed to generate preview")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Full Augmentation (Dog + Cat)")
    print("2. Specific Animal Only")
    print("3. Preview Augmentation Techniques")
    
    mode = input("\nEnter your choice (1-3): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        augment_specific_animal()
    elif mode == "3":
        preview_augmentation()
    else:
        print("Invalid choice! Running full augmentation...")
        main()