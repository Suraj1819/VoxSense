# src/preprocessing/generate_spectrograms.py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

def generate_spectrogram(audio_path, output_path, title_prefix=""):
    """Generate mel spectrogram from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        
        # Set title
        if title_prefix:
            plt.title(f'{title_prefix} - {audio_path.stem}', fontsize=12)
        else:
            plt.title(f'Spectrogram - {audio_path.stem}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Error {audio_path.name}: {e}")
        return False

def process_audio_folder(input_folder, output_folder, title_prefix=""):
    """Generate spectrograms for all audio files in a folder"""
    
    if not input_folder.exists():
        print(f"⚠️ Input folder not found: {input_folder}")
        return 0, 0
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(input_folder.glob("*.wav"))
    
    if not wav_files:
        print(f"⚠️ No WAV files found in {input_folder}")
        return 0, 0
    
    print(f"\n📁 Input: {input_folder}")
    print(f"   Output: {output_folder}")
    print(f"   Files: {len(wav_files)}")
    
    success_count = 0
    fail_count = 0
    
    for wav_file in tqdm(wav_files, desc=f"Generating spectrograms"):
        try:
            # Output filename
            output_filename = f"{wav_file.stem}.png"
            output_path = output_folder / output_filename
            
            # Generate spectrogram
            success = generate_spectrogram(wav_file, output_path, title_prefix)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {wav_file.name}: {e}")
            fail_count += 1
    
    return success_count, fail_count

def main():
    print("="*60)
    print("🎨 VOXSENSE - SPECTROGRAM GENERATOR")
    print("="*60)
    
    # Define paths
    processed_dir = Path("data/processed")
    augmented_dir = Path("data/processed_augmented")
    spectrogram_dir = Path("data/spectrograms")
    
    # Create base spectrogram directory
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_success = 0
    total_fail = 0
    
    print("\n" + "="*60)
    print("📊 STEP 1: Processing ORIGINAL Audio Files")
    print("="*60)
    
    # Process ORIGINAL DOG files
    dog_original_input = processed_dir / "dog"
    dog_original_output = spectrogram_dir / "dog"
    success, fail = process_audio_folder(dog_original_input, dog_original_output, "Original Dog")
    total_success += success
    total_fail += fail
    
    # Process ORIGINAL CAT files
    cat_original_input = processed_dir / "cat"
    cat_original_output = spectrogram_dir / "cat"
    success, fail = process_audio_folder(cat_original_input, cat_original_output, "Original Cat")
    total_success += success
    total_fail += fail
    
    print("\n" + "="*60)
    print("🔄 STEP 2: Processing AUGMENTED Audio Files")
    print("="*60)
    
    # Process AUGMENTED DOG files
    dog_augmented_input = augmented_dir / "dog"
    dog_augmented_output = spectrogram_dir / "augmented" / "dog"
    if dog_augmented_input.exists():
        success, fail = process_audio_folder(dog_augmented_input, dog_augmented_output, "Augmented Dog")
        total_success += success
        total_fail += fail
    else:
        print(f"\n⚠️ Augmented dog folder not found: {dog_augmented_input}")
    
    # Process AUGMENTED CAT files
    cat_augmented_input = augmented_dir / "cat"
    cat_augmented_output = spectrogram_dir / "augmented" / "cat"
    if cat_augmented_input.exists():
        success, fail = process_audio_folder(cat_augmented_input, cat_augmented_output, "Augmented Cat")
        total_success += success
        total_fail += fail
    else:
        print(f"\n⚠️ Augmented cat folder not found: {cat_augmented_input}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SPECTROGRAM GENERATION SUMMARY")
    print("="*60)
    
    print("\n📁 Output Structure:")
    print("data/spectrograms/")
    
    # Count files in each folder
    dog_original_count = len(list((spectrogram_dir / "dog").glob("*.png"))) if (spectrogram_dir / "dog").exists() else 0
    cat_original_count = len(list((spectrogram_dir / "cat").glob("*.png"))) if (spectrogram_dir / "cat").exists() else 0
    dog_augmented_count = len(list((spectrogram_dir / "augmented" / "dog").glob("*.png"))) if (spectrogram_dir / "augmented" / "dog").exists() else 0
    cat_augmented_count = len(list((spectrogram_dir / "augmented" / "cat").glob("*.png"))) if (spectrogram_dir / "augmented" / "cat").exists() else 0
    
    print(f"├── dog/               ({dog_original_count} spectrograms) ← From processed/dog/")
    print(f"├── cat/               ({cat_original_count} spectrograms) ← From processed/cat/")
    print(f"└── augmented/")
    print(f"    ├── dog/           ({dog_augmented_count} spectrograms) ← From processed_augmented/dog/")
    print(f"    └── cat/           ({cat_augmented_count} spectrograms) ← From processed_augmented/cat/")
    
    print(f"\n✅ Successfully generated: {total_success} spectrograms")
    print(f"❌ Failed: {total_fail}")
    print(f"📁 Total spectrograms: {dog_original_count + cat_original_count + dog_augmented_count + cat_augmented_count}")
    
    print("\n🎉 Spectrogram generation complete!")

def generate_for_specific_source():
    """Generate spectrograms for specific source only"""
    print("\n" + "="*60)
    print("Generate spectrograms for:")
    print("1. 🐕 Original Dog only (processed/dog/)")
    print("2. 🐱 Original Cat only (processed/cat/)")
    print("3. 🔄 Augmented Dog only (processed_augmented/dog/)")
    print("4. 🔄 Augmented Cat only (processed_augmented/cat/)")
    print("5. 📁 All Original (Dog + Cat)")
    print("6. 🔄 All Augmented (Dog + Cat)")
    print("7. 🎯 Everything (Original + Augmented)")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    processed_dir = Path("data/processed")
    augmented_dir = Path("data/processed_augmented")
    spectrogram_dir = Path("data/spectrograms")
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    
    total_success = 0
    total_fail = 0
    
    if choice == "1":
        dog_input = processed_dir / "dog"
        dog_output = spectrogram_dir / "dog"
        success, fail = process_audio_folder(dog_input, dog_output, "Original Dog")
        total_success, total_fail = success, fail
    
    elif choice == "2":
        cat_input = processed_dir / "cat"
        cat_output = spectrogram_dir / "cat"
        success, fail = process_audio_folder(cat_input, cat_output, "Original Cat")
        total_success, total_fail = success, fail
    
    elif choice == "3":
        dog_input = augmented_dir / "dog"
        dog_output = spectrogram_dir / "augmented" / "dog"
        success, fail = process_audio_folder(dog_input, dog_output, "Augmented Dog")
        total_success, total_fail = success, fail
    
    elif choice == "4":
        cat_input = augmented_dir / "cat"
        cat_output = spectrogram_dir / "augmented" / "cat"
        success, fail = process_audio_folder(cat_input, cat_output, "Augmented Cat")
        total_success, total_fail = success, fail
    
    elif choice == "5":
        # Original Dog
        dog_input = processed_dir / "dog"
        dog_output = spectrogram_dir / "dog"
        s1, f1 = process_audio_folder(dog_input, dog_output, "Original Dog")
        # Original Cat
        cat_input = processed_dir / "cat"
        cat_output = spectrogram_dir / "cat"
        s2, f2 = process_audio_folder(cat_input, cat_output, "Original Cat")
        total_success, total_fail = s1+s2, f1+f2
    
    elif choice == "6":
        # Augmented Dog
        dog_input = augmented_dir / "dog"
        dog_output = spectrogram_dir / "augmented" / "dog"
        s1, f1 = process_audio_folder(dog_input, dog_output, "Augmented Dog")
        # Augmented Cat
        cat_input = augmented_dir / "cat"
        cat_output = spectrogram_dir / "augmented" / "cat"
        s2, f2 = process_audio_folder(cat_input, cat_output, "Augmented Cat")
        total_success, total_fail = s1+s2, f1+f2
    
    elif choice == "7":
        # Everything
        main()
        return
    
    else:
        print("❌ Invalid choice!")
        return
    
    print(f"\n✅ Generated: {total_success} spectrograms")
    print(f"❌ Failed: {total_fail}")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. 🚀 Full Generation (Original + Augmented for both Dog & Cat)")
    print("2. 🎯 Select specific source")
    
    mode = input("\nEnter your choice (1-2): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        generate_for_specific_source()
    else:
        print("Invalid choice! Running full generation...")
        main()