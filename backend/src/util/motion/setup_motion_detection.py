"""
Quick start script to set up and test motion detection.
"""
import os
import sys

def check_dependencies():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'streamlit': 'streamlit',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_directory_structure():
    """Check if required directories exist."""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'util/models',
        'util/data',
        'util/data/motion_signs'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Creating {dir_path}...")
            os.makedirs(dir_path, exist_ok=True)
        else:
            print(f"✓ {dir_path}")
    
    return True

def print_usage_guide():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("MOTION DETECTION SETUP COMPLETE")
    print("="*60)
    print("\nQuick Start Guide:")
    print("\n1. Collect Training Data:")
    print("   cd util")
    print("   python create_dataset_motion.py --mode interactive --hands 1")
    print("\n   ⚠️  IMPORTANT - Data Requirements:")
    print("   - MINIMUM: 10 sequences per sign (50+ total)")
    print("   - RECOMMENDED: 20-50 sequences per sign for good accuracy")
    print("   - Each sequence = 30 frames (1 second of performing the sign)")
    print("\n   Tips:")
    print("   - Press 0-4 to record different signs")
    print("   - Perform each sign clearly and consistently")
    print("   - Vary your hand position slightly between recordings")
    print("   - Press 'q' when done")
    
    print("\n2. Train the Model:")
    print("   python train_motion_classifier.py --data motion_data.pickle --model-type lstm")
    print("\n   This will train an LSTM model and save it to util/models/")
    
    print("\n3. Test Detection:")
    print("   python inference_motion.py --model models/motion_model.h5 --metadata models/motion_model_metadata.pkl")
    print("\n   Controls: 'q' to quit, 'r' to reset buffer, 'c' to clear signs")
    
    print("\n4. Run Enhanced App:")
    print("   cd ..")
    print("   streamlit run app_enhanced.py")
    print("\n   Use the radio button to switch between Static and Motion modes")
    
    print("\n" + "="*60)
    print("\nAvailable Signs (default configuration):")
    print("  0: thank_you")
    print("  1: hello")
    print("  2: sorry")
    print("  3: please")
    print("  4: help")
    print("\nEdit create_dataset_motion.py to add more signs!")
    print("="*60 + "\n")

def main():
    print("="*60)
    print("MOTION-BASED SIGN LANGUAGE DETECTION")
    print("Setup and Configuration")
    print("="*60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check directory structure
    check_directory_structure()
    
    # Print usage guide
    print_usage_guide()
    
    print("Ready to start! Follow the guide above.\n")

if __name__ == "__main__":
    main()

