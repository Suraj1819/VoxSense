# src/preprocessing/preprocess.py
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import numpy as np

def preprocess_audio(input_dir, output_dir, target_sr=16000, remove_silence=True):
    """
    Preprocess audio files for better model training
    
    Args:
        input_dir: Source directory containing WAV files
        output_dir: Output directory for cleaned files
        target_sr: Target sample rate (default 16000)
        remove_silence: Remove leading/trailing silence
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    wav_files = list(input_path.glob("*.wav"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {input_dir}")
        return
    
    print(f"📁 Found {len(wav_files)} WAV files")
    print(f"🎵 Target sample rate: {target_sr} Hz")
    print(f"🔇 Remove silence: {remove_silence}")
    
    processed_count = 0
    error_count = 0
    
    for audio_file in tqdm(wav_files, desc="Preprocessing audio"):
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=target_sr)
            
            # Remove silence from beginning and end
            if remove_silence:
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            else:
                y_trimmed = y
            
            # Optional: Remove very short audio (< 0.5 seconds)
            if len(y_trimmed) / target_sr < 0.5:
                print(f"⚠️ Skipping {audio_file.name} - too short ({len(y_trimmed)/target_sr:.2f}s)")
                continue
            
            # Optional: Normalize volume (peak normalization)
            max_val = np.max(np.abs(y_trimmed))
            if max_val > 0:
                y_trimmed = y_trimmed / max_val * 0.95  # 95% peak
            
            # Save processed file
            output_file = output_path / audio_file.name
            sf.write(output_file, y_trimmed, target_sr)
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            error_count += 1
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Errors: {error_count}")
    print(f"   Output directory: {output_path}")

def preprocess_all_animals():
    """Preprocess both dog and cat audio files"""
    
    # Process original audio
    print("\n" + "="*60)
    print("🐕🐱 PREPROCESSING ORIGINAL AUDIO")
    print("="*60)
    
    for animal in ['dog', 'cat']:
        input_dir = Path(f"data/processed/{animal}")
        output_dir = Path(f"data/processed_clean/{animal}")
        
        if input_dir.exists():
            print(f"\n📁 Processing {animal.upper()} files...")
            preprocess_audio(input_dir, output_dir)
        else:
            print(f"\n⚠️ {animal.upper()} folder not found: {input_dir}")
    
    # Process augmented audio (optional)
    print("\n" + "="*60)
    print("🔄 PREPROCESSING AUGMENTED AUDIO")
    print("="*60)
    
    for animal in ['dog', 'cat']:
        input_dir = Path(f"data/processed_augmented/{animal}")
        output_dir = Path(f"data/processed_augmented_clean/{animal}")
        
        if input_dir.exists():
            print(f"\n📁 Processing augmented {animal.upper()} files...")
            preprocess_audio(input_dir, output_dir)
        else:
            print(f"\n⚠️ Augmented {animal.upper()} folder not found: {input_dir}")

def main():
    print("="*60)
    print("🎵 VOXSENSE - AUDIO PREPROCESSING")
    print("="*60)
    print("\nWhat would you like to do?")
    print("1. Preprocess original audio only (data/processed/)")
    print("2. Preprocess augmented audio only (data/processed_augmented/)")
    print("3. Preprocess ALL audio (original + augmented)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Process original only
        for animal in ['dog', 'cat']:
            input_dir = Path(f"data/processed/{animal}")
            if input_dir.exists():
                print(f"\n📁 Processing {animal.upper()}...")
                preprocess_audio(input_dir, Path(f"data/processed_clean/{animal}"))
            else:
                print(f"⚠️ {animal.upper()} folder not found")
    
    elif choice == "2":
        # Process augmented only
        for animal in ['dog', 'cat']:
            input_dir = Path(f"data/processed_augmented/{animal}")
            if input_dir.exists():
                print(f"\n📁 Processing augmented {animal.upper()}...")
                preprocess_audio(input_dir, Path(f"data/processed_augmented_clean/{animal}"))
            else:
                print(f"⚠️ Augmented {animal.upper()} folder not found")
    
    elif choice == "3":
        preprocess_all_animals()
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()