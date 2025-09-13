import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
TARGET_HEIGHT = 64  
TARGET_WIDTH = 64 
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 60
PATIENCE = 6

# Paths
PROJECT_DIR = ''
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
CODE_DIR = os.path.join(PROJECT_DIR, 'code')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories
for directory in [RESULTS_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Class names
class_names = [
    'good',
    'moderate', 
    'unhealthy-for-sensitive-groups',
    'unhealthy',
    'very-unhealthy',
    'hazardous'
]

# Load dataset with resizing to 64x64
print("Loading dataset with resizing to 64x64...")
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(TARGET_HEIGHT, TARGET_WIDTH),
    label_mode='int'
)

# Compute class weights
y_train = np.concatenate([y for x, y in temp_ds], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Load the actual datasets with resizing
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(TARGET_HEIGHT, TARGET_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(TARGET_HEIGHT, TARGET_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_size = int(0.6667 * len(val_test_ds))
test_ds = val_test_ds.skip(val_size)
val_ds = val_test_ds.take(val_size)

# Simple data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def augment_images(x, y):
    return data_augmentation(x, training=True), y

augmented_train_ds = train_ds.map(augment_images)

# Cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
augmented_train_ds = augmented_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create VERY tiny model
def create_tiny_model():
    inputs = keras.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 3))
    x = inputs
    
    # Tiny CNN architecture
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Reduced filters
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(16, activation='relu')(x)  # Reduced neurons
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = create_tiny_model()
print("Tiny model created successfully!")
print(f"Model input shape: {TARGET_HEIGHT}x{TARGET_WIDTH}")
model.summary()

# Calculate model size estimation
total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
print(f"Estimated model size (float32): ~{total_params * 4 / 1024:.2f} KB")
print(f"Estimated model size (int8): ~{total_params / 1024:.2f} KB")

# Save model architecture information
model_info = {
    'input_shape': [TARGET_HEIGHT, TARGET_WIDTH, 3],
    'total_parameters': total_params,
    'estimated_size_float32_kb': total_params * 4 / 1024,
    'estimated_size_int8_kb': total_params / 1024,
    'layers': len(model.layers),
    'architecture': 'tiny_cnn_64x64'
}

with open(os.path.join(MODEL_DIR, 'model_architecture.json'), 'w') as f:
    json.dump(model_info, f, indent=4)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=PATIENCE,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

# Single-phase training for tiny model
print("Training tiny model...")
history = model.fit(
    augmented_train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Load best model (remove final model saving as requested)
best_model = keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))

# Create all required plots
# 1. Training accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
plt.title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'training_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Training loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='red')
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'training_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Validation accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='green')
plt.title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'validation_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Validation loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'validation_loss.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Training and validation accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Evaluate
print("Evaluating on test set...")
test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
print(f"Best Model Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_true, y_pred = [], []
for images, labels in test_ds:
    y_true.extend(tf.argmax(labels, axis=1).numpy())
    predictions = best_model.predict(images, verbose=0)
    y_pred.extend(tf.argmax(predictions, axis=1).numpy())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save results
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
with open(os.path.join(RESULTS_DIR, 'classification_report.json'), 'w') as f:
    json.dump(report, f, indent=4)

test_results_dict = {
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'model_input_shape': f'{TARGET_HEIGHT}x{TARGET_WIDTH}x3',
    'model_size_parameters': best_model.count_params(),
    'model_size_float32_kb': best_model.count_params() * 4 / 1024,
    'model_size_int8_kb': best_model.count_params() / 1024,
    'optimized_for': 'PSRAM 512KB constraint'
}
with open(os.path.join(RESULTS_DIR, 'test_results.json'), 'w') as f:
    json.dump(test_results_dict, f, indent=4)

# Enhanced TFLite conversion with int8 quantization
def convert_to_tflite_micro(model_path, tflite_path, input_height=64, input_width=64):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create a representative dataset for quantization
    def representative_data_gen():
        for images, _ in test_ds.take(50):  # Reduced samples for speed
            for i in range(min(images.shape[0], 5)):  # Fewer samples per batch
                yield [tf.expand_dims(images[i], axis=0)]
    
    # Convert to TFLite with int8 quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Also save as a C array for microcontrollers
    c_model_path = tflite_path.replace('.tflite', '.h')
    with open(c_model_path, 'w') as f:
        f.write('// This file contains the TFLite model as a C array\n\n')
        f.write(f'const unsigned char {os.path.basename(c_model_path).replace(".", "_")}[] = {{\n')
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write('  ')
            f.write(f'0x{byte:02x},')
            if i % 12 == 11:
                f.write('\n')
        f.write('\n};\n')
        f.write(f'const unsigned int {os.path.basename(c_model_path).replace(".", "_")}_len = {len(tflite_model)};\n')
    
    # Also save float32 version for comparison
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    float_tflite_model = float_converter.convert()
    
    float_tflite_path = tflite_path.replace('.tflite', '_float32.tflite')
    with open(float_tflite_path, 'wb') as f:
        f.write(float_tflite_model)
    
    print(f"Model size: {len(tflite_model)/1024:.2f} KB (int8) vs {len(float_tflite_model)/1024:.2f} KB (float32)")
    
    return tflite_model

print("Converting best model to TFLite formats...")
convert_to_tflite_micro(os.path.join(MODEL_DIR, 'best_model.keras'), 
                       os.path.join(MODEL_DIR, 'best_model.tflite'),
                       input_height=TARGET_HEIGHT, input_width=TARGET_WIDTH)

# Rename the int8 .h file to match requested name
int8_h_path = os.path.join(MODEL_DIR, 'best_model.h')
if os.path.exists(int8_h_path):
    os.rename(int8_h_path, os.path.join(MODEL_DIR, 'best_model_int8.h'))

print(f"\nTraining completed successfully!")
print(f"Best Model Test Accuracy: {test_accuracy:.4f}")
print(f"Model input shape: {TARGET_HEIGHT}x{TARGET_WIDTH}")
print(f"Total parameters: {total_params:,}")
print(f"Estimated size (int8): ~{total_params / 1024:.2f} KB")
print(f"Optimized for SRAM 512KB constraint")
print(f"6 plots + confusion matrix saved in: {PLOT_DIR}")
print(f"Models saved: best_model.keras, best_model.tflite, best_model_int8.h")