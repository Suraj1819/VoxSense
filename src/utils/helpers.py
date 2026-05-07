# src/utils/helpers.py
import sys
from pathlib import Path

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Works regardless of where the script is called from.
    """
    # Method 1: Using Path(__file__)
    return Path(__file__).parent.parent.parent

def setup_python_path():
    """
    Adds project root to Python path so imports work.
    Call this at the beginning of your scripts.
    """
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

def get_data_path(subfolder=""):
    """Get path to data folder"""
    base = get_project_root() / "data"
    if subfolder:
        return base / subfolder
    return base

def get_processed_path():
    """Get path to processed audio files"""
    return get_project_root() / "data" / "processed"

def get_spectrogram_path():
    """Get path to spectrogram images"""
    return get_project_root() / "data" / "spectrograms"

def get_augmented_path():
    """Get path to augmented data"""
    return get_project_root() / "data" / "processed_augmented"

def get_model_path(model_name="best_cnn_model.pth"):
    """Get path to saved model"""
    return get_project_root() / "models" / model_name

def get_raw_path():
    """Get path to raw audio files"""
    return get_project_root() / "data" / "raw"