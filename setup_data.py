"""
Helper script to download and setup NIH Chest X-ray dataset
"""

import os
import sys
from src.config import Config

def download_from_kaggle():
    """Download dataset from Kaggle"""
    print("=" * 60)
    print("DOWNLOADING NIH CHEST X-RAY DATASET FROM KAGGLE")
    print("=" * 60)
    
    print("\n⚠️ IMPORTANT: You need to setup Kaggle API credentials first!")
    print("\nSteps:")
    print("1. Create a Kaggle account at https://www.kaggle.com")
    print("2. Go to Account Settings -> API -> Create New API Token")
    print("3. This downloads kaggle.json")
    print("4. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    print("5. Run this script again")
    
    input("\nPress Enter when you've completed the above steps...")
    
    try:
        import kaggle
        
        print("\n📥 Downloading dataset...")
        print("This may take a while (dataset is ~42GB)...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'nih-chest-xrays/data',
            path=Config.RAW_DATA_DIR,
            unzip=True
        )
        
        print("\n✅ Dataset downloaded successfully!")
        print(f"Location: {Config.RAW_DATA_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/nih-chest-xrays/data")
        sys.exit(1)

def verify_dataset():
    """Verify dataset structure"""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET")
    print("=" * 60)
    
    csv_path = os.path.join(Config.METADATA_DIR, 'Data_Entry_2017.csv')
    images_dir = os.path.join(Config.RAW_DATA_DIR, 'images')
    
    if not os.path.exists(csv_path):
        print(f"\n❌ CSV file not found: {csv_path}")
        print("\nPlease ensure Data_Entry_2017.csv is in the metadata folder.")
        return False
    
    if not os.path.exists(images_dir):
        print(f"\n❌ Images directory not found: {images_dir}")
        print("\nPlease ensure images are extracted to the raw data folder.")
        return False
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"\n✅ Found {len(image_files)} images")
    
    # Check CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"✅ Found {len(df)} records in CSV")
    
    print("\n✅ Dataset verification successful!")
    return True

def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("NIH CHEST X-RAY DATASET SETUP")
    print("=" * 60)
    
    print("\nChoose an option:")
    print("1. Download from Kaggle (Requires Kaggle API)")
    print("2. I've already downloaded the dataset manually")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        download_from_kaggle()
        verify_dataset()
    elif choice == '2':
        print("\nPlease ensure:")
        print(f"1. Images are in: {os.path.join(Config.RAW_DATA_DIR, 'images')}")
        print(f"2. Data_Entry_2017.csv is in: {Config.METADATA_DIR}")
        input("\nPress Enter when ready to verify...")
        verify_dataset()
    else:
        print("\nExiting...")
        sys.exit(0)
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run training: python -m src.train")
    print("2. After training, run app: streamlit run app.py")

if __name__ == "__main__":
    main()
