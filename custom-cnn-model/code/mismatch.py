import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from pathlib import Path

def analyze_dataset_comparison():
    # Paths to your datasets
    train_data_dir = ''
    eval_data_dir = ''
    
    # Analyze image characteristics
    def analyze_images_in_directory(directory, class_name, sample_size=5):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            return None
            
        image_files = list(Path(class_dir).glob('*.jpg')) + list(Path(class_dir).glob('*.png'))
        if not image_files:
            return None
            
        # Analyze sample images
        sample_images = []
        for img_path in image_files[:sample_size]:
            try:
                img = Image.open(img_path)
                sample_images.append({
                    'size': img.size,
                    'mode': img.mode,
                    'path': str(img_path)
                })
            except:
                continue
                
        return sample_images
    
    # Compare each class
    classes = ['good', 'moderate', 'unhealthy-for-sensitive-groups', 'unhealthy', 'very-unhealthy']
    
    print("=== DATASET COMPARISON ANALYSIS ===")
    for class_name in classes:
        print(f"\n--- {class_name.upper()} ---")
        
        train_samples = analyze_images_in_directory(train_data_dir, class_name)
        eval_samples = analyze_images_in_directory(eval_data_dir, class_name)
        
        if train_samples:
            print(f"Training data: {len(train_samples)} samples")
            print(f"  Sizes: {[s['size'] for s in train_samples]}")
        else:
            print("Training data: Not found")
            
        if eval_samples:
            print(f"Eval data: {len(eval_samples)} samples")
            print(f"  Sizes: {[s['size'] for s in eval_samples]}")
        else:
            print("Eval data: Not found")

# Run the analysis
analyze_dataset_comparison()