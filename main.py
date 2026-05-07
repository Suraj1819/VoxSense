# main.py
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    print("🚀 VoxSense Project Started...")
    print("1. Preprocessing")
    print("2. Feature Extraction")
    print("3. Model Training")
    print("4. Evaluation")
    print("5. Run App")

    choice = input("\nEnter your choice (1-5): ")

    if choice == '1':
        from src.preprocessing.preprocess import main as preprocess_main
        preprocess_main()
    elif choice == '2':
        from src.features.extract_features import main as extract_main
        extract_main()
    elif choice == '3':
        from src.models.train_model import main as train_main
        train_main()
    elif choice == '4':
        from src.evaluation.evaluate import main as eval_main
        eval_main()
    elif choice == '5':
        from app.app import main as app_main
        app_main()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()