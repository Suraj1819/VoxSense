# src/preprocessing/convert_mp3_to_wav.py
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import os
import shutil

def convert_mp3_to_wav(input_dir="data/raw", output_dir="data/processed", target_sr=16000):
    """
    Convert MP3 files to WAV format while maintaining folder structure
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return
    
    # Create output directories
    dog_output = output_path / "dog"
    cat_output = output_path / "cat"
    dog_output.mkdir(parents=True, exist_ok=True)
    cat_output.mkdir(parents=True, exist_ok=True)
    
    # Process folders
    dog_input = input_path / "DOG"
    cat_input = input_path / "CAT"
    
    total_converted = 0
    total_copied = 0
    total_errors = 0
    
    def process_folder(source_folder, dest_folder, animal_name):
        nonlocal total_converted, total_copied, total_errors
        
        if not source_folder.exists():
            print(f"⚠️ {animal_name} folder not found: {source_folder}")
            return
        
        # Case-insensitive search for MP3 and WAV files
        mp3_files = []
        wav_files = []
        
        # Find all MP3 files (case insensitive)
        for ext in ['*.mp3', '*.MP3', '*.Mp3', '*.mP3']:
            mp3_files.extend(list(source_folder.glob(ext)))
        
        # Find all WAV files (case insensitive)
        for ext in ['*.wav', '*.WAV', '*.Wav', '*.wAV']:
            wav_files.extend(list(source_folder.glob(ext)))
        
        # Remove duplicates (if any)
        mp3_files = list(set(mp3_files))
        wav_files = list(set(wav_files))
        
        # Filter: Only process if filename contains 'dog' or 'cat' (optional)
        if animal_name == "dog":
            mp3_files = [f for f in mp3_files if 'dog' in f.stem.lower()]
            wav_files = [f for f in wav_files if 'dog' in f.stem.lower()]
        else:  # cat
            mp3_files = [f for f in mp3_files if 'cat' in f.stem.lower()]
            wav_files = [f for f in wav_files if 'cat' in f.stem.lower()]
        
        if not mp3_files and not wav_files:
            print(f"⚠️ No {animal_name} audio files found in {source_folder}")
            print(f"   Supported: .mp3, .MP3, .wav, .WAV")
            return
        
        print(f"\n📁 Processing {animal_name.upper()} folder:")
        print(f"   Found {len(mp3_files)} MP3 files and {len(wav_files)} WAV files")
        
        # Convert MP3 files
        if mp3_files:
            print(f"\n🎵 Converting MP3 to WAV for {animal_name}...")
            for mp3_file in tqdm(mp3_files, desc=f"Converting {animal_name} MP3s"):
                try:
                    wav_filename = mp3_file.stem + ".wav"
                    output_file = dest_folder / wav_filename
                    
                    # Skip if already exists
                    if output_file.exists():
                        print(f"⏭️ Skipping (already exists): {wav_filename}")
                        continue
                    
                    # Load and convert MP3
                    audio = AudioSegment.from_mp3(str(mp3_file))
                    audio = audio.set_channels(1).set_frame_rate(target_sr)
                    
                    # Normalize volume
                    try:
                        from pydub.effects import normalize
                        audio = normalize(audio)
                    except:
                        pass
                    
                    # Save as WAV
                    audio.export(str(output_file), format="wav")
                    
                    print(f"✅ Converted: {mp3_file.name} → {animal_name}/{wav_filename}")
                    total_converted += 1
                    
                except Exception as e:
                    print(f"❌ Error converting {mp3_file.name}: {e}")
                    total_errors += 1
        
        # Copy existing WAV files
        if wav_files:
            print(f"\n📋 Copying existing WAV files for {animal_name}...")
            for wav_file in tqdm(wav_files, desc=f"Copying {animal_name} WAVs"):
                try:
                    output_file = dest_folder / wav_file.name
                    
                    if output_file.exists():
                        print(f"⏭️ Skipping (already exists): {wav_file.name}")
                        continue
                    
                    shutil.copy2(wav_file, output_file)
                    print(f"✅ Copied: {wav_file.name} → {animal_name}/{wav_file.name}")
                    total_copied += 1
                    
                except Exception as e:
                    print(f"❌ Error copying {wav_file.name}: {e}")
                    total_errors += 1
    
    print("="*60)
    print("🐕🐱 VOXSENSE - MP3 to WAV CONVERTER")
    print("="*60)
    
    process_folder(dog_input, dog_output, "dog")
    process_folder(cat_input, cat_output, "cat")
    
    # Summary
    print("\n" + "="*60)
    print("📊 CONVERSION SUMMARY")
    print("="*60)
    print(f"✅ MP3 files converted: {total_converted}")
    print(f"✅ WAV files copied: {total_copied}")
    print(f"❌ Errors encountered: {total_errors}")
    
    # Count actual files in output
    dog_wav_count = len(list(dog_output.glob("*.wav")))
    cat_wav_count = len(list(cat_output.glob("*.wav")))
    
    print(f"\n📁 Output Summary:")
    print(f"   processed/dog/ : {dog_wav_count} WAV files")
    print(f"   processed/cat/ : {cat_wav_count} WAV files")
    
    # List files in dog folder
    if dog_wav_count > 0:
        print(f"\n📋 Dog WAV files in processed/dog/:")
        for i, f in enumerate(list(dog_output.glob("*.wav"))[:10], 1):
            print(f"   {i}. {f.name}")
        if dog_wav_count > 10:
            print(f"   ... and {dog_wav_count-10} more")

def convert_single_animal(animal_type="dog"):
    """Convert files for specific animal only"""
    input_dir = "data/raw"
    output_dir = "data/processed"
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    animal_input = input_path / animal_type.upper()
    animal_output = output_path / animal_type.lower()
    
    if not animal_input.exists():
        print(f"❌ {animal_type.upper()} folder not found: {animal_input}")
        return
    
    animal_output.mkdir(parents=True, exist_ok=True)
    
    # Case-insensitive search
    mp3_files = []
    wav_files = []
    
    for ext in ['*.mp3', '*.MP3', '*.Mp3', '*.mP3']:
        mp3_files.extend(list(animal_input.glob(ext)))
    
    for ext in ['*.wav', '*.WAV', '*.Wav', '*.wAV']:
        wav_files.extend(list(animal_input.glob(ext)))
    
    mp3_files = list(set(mp3_files))
    wav_files = list(set(wav_files))
    
    # Filter by animal type in filename
    mp3_files = [f for f in mp3_files if animal_type in f.stem.lower()]
    wav_files = [f for f in wav_files if animal_type in f.stem.lower()]
    
    if not mp3_files and not wav_files:
        print(f"⚠️ No {animal_type} audio files found in {animal_input}")
        print(f"   Files should contain '{animal_type}' in their names")
        return
    
    print(f"\n📁 Found {len(mp3_files)} MP3 and {len(wav_files)} WAV files")
    
    # First, show what files will be processed
    print(f"\n📋 Files to process:")
    for f in mp3_files:
        print(f"   MP3: {f.name}")
    for f in wav_files:
        print(f"   WAV: {f.name}")
    
    converted = 0
    copied = 0
    
    # Convert MP3 to WAV
    if mp3_files:
        print(f"\n🎵 Converting MP3 to WAV...")
        for mp3_file in tqdm(mp3_files, desc="Converting"):
            try:
                output_file = animal_output / (mp3_file.stem + ".wav")
                if output_file.exists():
                    print(f"⏭️ Skip (exists): {mp3_file.name}")
                    continue
                    
                audio = AudioSegment.from_mp3(str(mp3_file))
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(str(output_file), format="wav")
                print(f"✅ Converted: {mp3_file.name}")
                converted += 1
            except Exception as e:
                print(f"❌ Error converting {mp3_file.name}: {e}")
    
    # Copy existing WAV files
    if wav_files:
        print(f"\n📋 Copying WAV files...")
        for wav_file in tqdm(wav_files, desc="Copying"):
            try:
                output_file = animal_output / wav_file.name
                if not output_file.exists():
                    shutil.copy2(wav_file, output_file)
                    print(f"✅ Copied: {wav_file.name}")
                    copied += 1
                else:
                    print(f"⏭️ Skip (exists): {wav_file.name}")
            except Exception as e:
                print(f"❌ Error copying {wav_file.name}: {e}")
    
    print(f"\n✅ {animal_type.upper()} conversion complete!")
    print(f"   Converted MP3s: {converted}")
    print(f"   Copied WAVs: {copied}")
    print(f"   Total files in {animal_output}: {len(list(animal_output.glob('*.wav')))}")

def debug_check_files():
    """Debug function to check what files exist"""
    print("\n🔍 DEBUG: Checking files in raw folders...")
    
    raw_path = Path("data/raw")
    
    for animal in ['DOG', 'CAT']:
        animal_path = raw_path / animal
        if animal_path.exists():
            print(f"\n📁 {animal}/")
            files = list(animal_path.glob("*"))
            for f in files:
                print(f"   - {f.name} (extension: {f.suffix})")
        else:
            print(f"\n📁 {animal}/ - Folder not found!")

def main():
    """Main function with user input"""
    print("="*60)
    print("🐕🐱 VOXSENSE - MP3 to WAV Converter")
    print("="*60)
    
    # First, debug to show what files exist
    debug_check_files()
    
    print("\n" + "="*60)
    print("What would you like to do?")
    print("1. 🐕🐱 Convert Both (Dog + Cat)")
    print("2. 🐕 Convert Dog only")
    print("3. 🐱 Convert Cat only")
    print("4. 🔍 Check files only (no conversion)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        convert_mp3_to_wav()
    elif choice == "2":
        convert_single_animal("dog")
    elif choice == "3":
        convert_single_animal("cat")
    elif choice == "4":
        debug_check_files()
    elif choice == "5":
        print("Exiting...")
        return
    else:
        print("❌ Invalid choice! Please run again.")
        return

if __name__ == "__main__":
    main()