import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from PIL import Image
import pandas as pd
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
MODEL_PATH = ''
EVAL_DATA_DIR = ''
RESULTS_DIR = ''
TARGET_SIZE = (64, 64)  # Model input size

# Class names (matching the training)
class_names = [
    'good',
    'moderate', 
    'unhealthy-for-sensitive-groups',
    'unhealthy',
    'very-unhealthy',
    'hazardous'  # Note: evaluating dataset has 5 classes but model expects 6
]

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_image(image_path, target_size):
    """Load and preprocess an image for the model"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model input size
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0
        
        return img_array, True
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, False

def load_evaluation_dataset(data_dir, class_names, target_size):
    """Load all images from the evaluation dataset"""
    images = []
    true_labels = []
    image_paths = []
    class_counts = {class_name: 0 for class_name in class_names}
    
    print("Loading evaluation dataset...")
    
    # Only process the 5 classes that actually exist
    existing_classes = ['good', 'moderate', 'unhealthy-for-sensitive-groups', 'unhealthy', 'very-unhealthy']
    
    for class_idx, class_name in enumerate(existing_classes):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
            
        # Get all image files in the class directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(Path(class_dir).glob(ext)))
        
        print(f"Found {len(image_files)} images in class '{class_name}'")
        
        for image_path in image_files:
            # Load and preprocess image
            img_array, success = load_and_preprocess_image(image_path, target_size)
            
            if success:
                images.append(img_array)
                true_labels.append(class_idx)  # Use index from existing_classes
                image_paths.append(str(image_path))
                class_counts[class_name] += 1
    
    # Convert to numpy arrays
    if images:
        images = np.array(images)
        true_labels = np.array(true_labels)
        
    print(f"\nDataset summary:")
    for class_name in existing_classes:
        print(f"  {class_name}: {class_counts[class_name]} images")
    print(f"Total images: {len(images)}")
    
    return images, true_labels, image_paths, class_counts

def evaluate_model(tflite_model_path, images, true_labels):
    """Evaluate the TFLite model on the dataset"""
    print(f"\nLoading TFLite model from {tflite_model_path}...")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")
    
    # Check if model expects quantized input
    input_dtype = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']
    
    print(f"Input type: {input_dtype}")
    print(f"Output type: {output_dtype}")
    
    predictions = []
    confidence_scores = []
    
    print("\nRunning inference on evaluation dataset...")
    
    for i, image in enumerate(images):
        # Preprocess based on model requirements
        if input_dtype == np.uint8:
            # Quantized model expects uint8 input [0, 255]
            input_data = (image * 255).astype(np.uint8)
        else:
            # Float model expects float32 input [0, 1] or [-1, 1]
            input_data = (image * 2) - 1  # Scale to [-1, 1]
            input_data = input_data.astype(np.float32)
        
        # Add batch dimension if needed (from 3D to 4D)
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Handle quantized output if needed
        if output_dtype == np.uint8:
            # Dequantize the output
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)
        
        predictions.append(np.argmax(output_data))
        confidence_scores.append(np.max(output_data))
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(images)} images")
    
    return np.array(predictions), np.array(confidence_scores)

def analyze_results(true_labels, predictions, confidence_scores, class_names, image_paths, results_dir):
    """Analyze and save evaluation results"""
    print("\nAnalyzing results...")
    
    # Since we only have 5 classes but model outputs 6, we need to handle this
    # For evaluation purposes, we'll consider predictions >= 5 as errors
    valid_predictions = predictions < 5  # Only consider predictions for the first 5 classes
    
    # Calculate accuracy only on valid predictions
    accuracy = np.mean(true_labels[valid_predictions] == predictions[valid_predictions])
    print(f"Overall Accuracy (valid predictions only): {accuracy:.4f}")
    print(f"Invalid predictions (hazardous class): {np.sum(~valid_predictions)}")
    
    # Classification report (only for the 5 existing classes)
    report = classification_report(true_labels[valid_predictions], 
                                  predictions[valid_predictions], 
                                  target_names=class_names[:5], 
                                  output_dict=True)
    
    # Confusion matrix (only for the 5 existing classes)
    cm = confusion_matrix(true_labels[valid_predictions], predictions[valid_predictions])
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': [class_names[i] for i in true_labels],
        'predicted_label': [class_names[i] if i < 5 else 'hazardous' for i in predictions],
        'confidence': confidence_scores,
        'correct': (true_labels == predictions) & valid_predictions,
        'valid_prediction': valid_predictions
    })
    
    # Calculate class-wise accuracy (only for valid predictions)
    class_accuracy = {}
    for i, class_name in enumerate(class_names[:5]):  # Only first 5 classes
        class_mask = true_labels == i
        valid_class_mask = class_mask & valid_predictions
        if np.sum(valid_class_mask) > 0:
            class_acc = np.mean(predictions[valid_class_mask] == i)
            class_accuracy[class_name] = class_acc
    
    # Save results
    print("\nSaving results...")
    
    # 1. Confusion matrix plot (only 5x5)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:5], yticklabels=class_names[:5])
    plt.title('Confusion Matrix\nModel Evaluation on External Dataset', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'evaluation_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy by class plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_accuracy)), list(class_accuracy.values()))
    plt.title('Accuracy by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_accuracy)), list(class_accuracy.keys()), rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'evaluation_class_accuracy.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'evaluation_confidence_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save detailed results to CSV
    results_df.to_csv(os.path.join(results_dir, 'evaluation_detailed_results.csv'), index=False)
    
    # 5. Save summary report
    summary_report = {
        'overall_accuracy': float(accuracy),
        'invalid_predictions_count': int(np.sum(~valid_predictions)),
        'class_accuracy': class_accuracy,
        'class_distribution': {class_names[i]: int(np.sum(true_labels == i)) 
                              for i in range(5)},  # Only first 5 classes
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    with open(os.path.join(results_dir, 'evaluation_summary_report.json'), 'w') as f:
        json.dump(summary_report, f, indent=4)
    
    # 6. Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy (valid predictions): {accuracy:.4f}")
    print(f"Total Images: {len(true_labels)}")
    print(f"Invalid Predictions (classified as 'hazardous'): {np.sum(~valid_predictions)}")
    print("\nClass-wise Accuracy:")
    for class_name, acc in class_accuracy.items():
        print(f"  {class_name}: {acc:.4f}")
    
    print("\nClass Distribution:")
    for class_name, count in summary_report['class_distribution'].items():
        print(f"  {class_name}: {count} images")
    
    return summary_report

def main():
    """Main evaluation function"""
    print("Starting model evaluation...")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {EVAL_DATA_DIR}")
    
    # Load evaluation dataset
    images, true_labels, image_paths, class_counts = load_evaluation_dataset(
        EVAL_DATA_DIR, class_names, TARGET_SIZE
    )
    
    if len(images) == 0:
        print("No images found for evaluation!")
        return
    
    # Evaluate model
    predictions, confidence_scores = evaluate_model(MODEL_PATH, images, true_labels)
    
    # Analyze and save results
    summary_report = analyze_results(
        true_labels, predictions, confidence_scores, 
        class_names, image_paths, RESULTS_DIR
    )
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Confusion matrix: evaluation_confusion_matrix.png")
    print(f"Detailed results: evaluation_detailed_results.csv")

if __name__ == "__main__":
    main()