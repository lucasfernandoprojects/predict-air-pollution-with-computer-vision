#include "unihiker_k10.h"
#include <SD.h>
#include <SPI.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include "best_model_int8.h" // The ultra-tiny INT8 model must be in the same folder of this sketch

UNIHIKER_K10 k10;
const int screen_direction = 2;
const unsigned long delay_welcome_message = 1000;
const unsigned long delay_photo_mode = 2000;
const unsigned long debounce_delay = 200;

bool cameraInitialized = false;

const char* originalPhotoPath = "S:photos/landscape.bmp";
const char* inputFileName  = "/photos/landscape.bmp";
const char* outputFileName = "/photos/landscape_64.bmp";

const int srcWidth  = 240;
const int srcHeight = 320;
const int dstWidth  = 64;
const int dstHeight = 64;

// Static buffers for image processing to avoid heap fragmentation
static uint8_t srcData[240 * 320 * 2]; // Maximum expected size for 240x320 16-bit
static uint8_t dstData[64 * 64 * 3 + 64]; // 64x64 RGB with padding

struct BMPHeader {
  uint16_t bfType;
  uint32_t bfSize;
  uint16_t bfReserved1;
  uint16_t bfReserved2;
  uint32_t bfOffBits;
  uint32_t biSize;
  int32_t  biWidth;
  int32_t  biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  int32_t  biXPelsPerMeter;
  int32_t  biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
};

// Buffer must be on global memory to avoid stack overflow
static uint8_t imgData[64 * 64 * 3];

// Classes
const char* class_names[] = {
  "good", "moderate", "unhealthy-for-sensitive-groups", 
  "unhealthy", "very-unhealthy", "hazardous"
};

#define NUM_CLASSES (sizeof(class_names)/sizeof(class_names[0]))

// TFLite Micro globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 50 * 1024; // 50KB for tiny 64x64 model
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroErrorReporter micro_error_reporter;
}

void initializeTFLite() {
  Serial.println("Initializing TFLite Micro for tiny 64x64 model...");

  // Safety check for model data
  if (best_model_h == nullptr) {
    Serial.println("❌ Model data is null!");
    while(1) { delay(1000); }
  }

  model = tflite::GetModel(best_model_h);
  Serial.print("Model size: "); Serial.print(best_model_h_len); Serial.println(" bytes");

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("⚠️ Model version mismatch: "); Serial.println(model->version());
    Serial.print("Expected: "); Serial.println(TFLITE_SCHEMA_VERSION);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, 
      resolver, 
      tensor_arena, 
      kTensorArenaSize,
      &micro_error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ AllocateTensors() failed");
    while (1) { delay(1000); }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verify tensor allocation
  if (input->data.int8 == nullptr || output->data.int8 == nullptr) {
    Serial.println("❌ Tensor data pointers are null!");
    while (1) { delay(1000); }
  }

  Serial.print("Input dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print("x");
  }
  Serial.println();
  
  Serial.print("Input type: "); Serial.println(input->type);
  Serial.print("Output type: "); Serial.println(output->type);

  Serial.println("✅ TFLite Micro initialized successfully!");
}

// Preprocess and run inference
int processImage(uint8_t* image_data) {
  // ✅ SAFETY CHECK - Ensure TFLite is initialized
  if (model == nullptr || interpreter == nullptr || input == nullptr || output == nullptr) {
    Serial.println("❌ TFLite not initialized! Call initializeTFLite() first!");
    return -1;
  }

  if (input->data.int8 == nullptr) {
    Serial.println("❌ Input tensor data is null!");
    return -1;
  }

  int8_t* input_data = input->data.int8;
  const int num_pixels = 64 * 64 * 3;

  // Preprocess: uint8 [0,255] → int8 [-128,127]
  for (int i = 0; i < num_pixels; i++) {
    input_data[i] = static_cast<int8_t>(image_data[i] - 128);
  }

  Serial.println("Running inference...");
  unsigned long start_time = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    return -1; // Return error code
  }
  unsigned long inference_time = millis() - start_time;

  int8_t* output_data = output->data.int8;
  
  // PROPER SOFTMAX CALCULATION FOR INT8
  float exp_vals[NUM_CLASSES];
  float sum_exp = 0.0f;
  
  // Dequantize and calculate softmax
  for (int i = 0; i < NUM_CLASSES; i++) {
    exp_vals[i] = exp(output_data[i] / 32.0f); // Scale factor for better numerical stability
    sum_exp += exp_vals[i];
  }
  
  // Find predicted class and calculate probabilities
  int predicted_class = 0;
  float max_probability = 0.0f;
  float probabilities[NUM_CLASSES];
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    probabilities[i] = exp_vals[i] / sum_exp;
    if (probabilities[i] > max_probability) {
      max_probability = probabilities[i];
      predicted_class = i;
    }
  }

  Serial.println("=== Inference Result ===");
  Serial.print("Predicted: "); Serial.println(class_names[predicted_class]);
  Serial.print("Confidence: "); Serial.print(max_probability * 100.0f, 1); Serial.println("%");
  Serial.print("Time: "); Serial.print(inference_time); Serial.println(" ms");
  Serial.println("All class probabilities:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print("  "); Serial.print(class_names[i]); 
    Serial.print(": "); Serial.print(probabilities[i] * 100.0f, 1); Serial.println("%");
  }
  Serial.println("========================");

  return predicted_class; // Return the class index with highest score
}

// Load 64x64 BMP image from SD
bool loadImageFromSD(const char* filename, uint8_t* buffer) {
  if (!SD.exists(filename)) {
    Serial.print("File not found: "); Serial.println(filename);
    return false;
  }

  File imgFile = SD.open(filename, FILE_READ);
  if (!imgFile) {
    Serial.println("Failed to open file");
    return false;
  }

  if (imgFile.size() < 54) { // BMP header
    Serial.println("File too small for BMP header");
    imgFile.close();
    return false;
  }

  uint8_t header[54];
  imgFile.read(header, 54);
  if (header[0] != 'B' || header[1] != 'M') {
    Serial.println("Not a valid BMP file");
    imgFile.close();
    return false;
  }

  imgFile.read(buffer, 64 * 64 * 3);
  imgFile.close();
  Serial.print("Loaded image: "); Serial.println(filename);
  return true;
}

void printSDCardInfo() {
    // Check if SD card is present
    Serial.println("Initializing SD card...");
    
    if (!SD.begin()) {
        Serial.println("SD card initialization failed! Please check the card.");
        return;
    }

    Serial.println("SD card initialized successfully.");

    // Print card type
    uint8_t cardType = SD.cardType();
    Serial.print("Card type: ");
    if (cardType == CARD_NONE) {
        Serial.println("No SD card detected!");
        return;
    } else if (cardType == CARD_MMC) {
        Serial.println("MMC");
    } else if (cardType == CARD_SD) {
        Serial.println("SDSC");
    } else if (cardType == CARD_SDHC) {
        Serial.println("SDHC");
    } else {
        Serial.println("Unknown");
    }

    // Print card size
    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    Serial.print("Card size: ");
    Serial.print(cardSize);
    Serial.println(" MB");

    // Print total and used space
    uint64_t totalBytes = SD.totalBytes() / (1024 * 1024);
    uint64_t usedBytes = SD.usedBytes() / (1024 * 1024);
    Serial.print("Total space: ");
    Serial.print(totalBytes);
    Serial.println(" MB");
    Serial.print("Used space: ");
    Serial.print(usedBytes);
    Serial.println(" MB");
    Serial.print("Free space: ");
    Serial.print(totalBytes - usedBytes);
    Serial.println(" MB");
}

void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    Serial.println("=== UNIHIKER K10 Air Quality Analyzer ===");
    Serial.println("Tiny 64x64 model version");

    // Initialize K10 board, screen, and SD card system
    k10.begin();
    k10.initScreen(screen_direction);
    k10.initSDFile();

    // Check SD card status and print information
    printSDCardInfo();

    // ✅ CRITICAL: Initialize TFLite Micro
    initializeTFLite();

    // Show welcome screen
    k10.creatCanvas();
    k10.canvas->canvasDrawImage(0, 0, "S:screens/welcome.png");
    k10.canvas->updateCanvas();
    delay(delay_welcome_message);

    Serial.println("UNIHIKER K10 initialized successfully!\n");
}

void waitForButtonRelease() {
    // Wait until all buttons are released
    while (k10.buttonA->isPressed() || k10.buttonB->isPressed()) {
        delay(10);
    }
    delay(debounce_delay);
}

void initializeCamera() {
    if (!cameraInitialized) {
        k10.initBgCamerImage();
        cameraInitialized = true;
    }
}

void showCameraPreview() {
    k10.setBgCamerImage(true);
}

void hideCameraPreview() {
    k10.canvas->canvasClear();
    k10.setBgCamerImage(false);
}

void showPhotoMode() {
    // Show instructions once per entry into photo mode
    k10.canvas->canvasDrawImage(0, 0, "S:screens/instructions.png");
    k10.canvas->updateCanvas();
    delay(delay_photo_mode);

    // Initialize camera if not already done
    initializeCamera();

    // Show camera preview
    showCameraPreview();

    bool inPhotoMode = true;
    bool showingPreview = true;

    while (inPhotoMode) {
        if (showingPreview) {
            // Camera preview mode
            if (k10.buttonA->isPressed()) {
                waitForButtonRelease();

                // Save photo first
                k10.photoSaveToTFCard(originalPhotoPath);

                // Hide camera preview
                hideCameraPreview();

                // Now draw the photo
                k10.canvas->canvasDrawImage(0, 0, originalPhotoPath);
                k10.canvas->updateCanvas();

                // Convert photo from 240x320 to 64x64
                if (resizeBMP16to24(inputFileName, outputFileName)) {
                    // Run inference on the photo
                    Serial.println("\n=== New Inference Cycle ===");

                    if (loadImageFromSD(outputFileName, imgData)) {
                        int classIndex = processImage(imgData);
                        if (classIndex >= 0 && classIndex < NUM_CLASSES) {
                            // Display result on screen
                            k10.canvas->canvasText(class_names[classIndex], 1, 0xFF0000); // Row 1, red color
                            k10.canvas->updateCanvas();
                        }
                    } else {
                        Serial.println("No image found on SD card. Try again.");
                    }
                } else {
                    Serial.println("Failed to resize image.");
                }

                showingPreview = false; // We're showing the captured photo now
            }

            // Exit photo mode completely
            if (k10.buttonB->isPressed()) {
                waitForButtonRelease();
                inPhotoMode = false;
            }
        } else {
            // Showing captured photo
            if (k10.buttonB->isPressed()) {
                waitForButtonRelease();
                // Return to camera preview
                k10.canvas->canvasClear();
                showCameraPreview();
                showingPreview = true;
            }

            // Exit photo mode from photo view
            if (k10.buttonA->isPressed()) {
                waitForButtonRelease();
                inPhotoMode = false;
            }
        }

        delay(10);
    }

    // Cleanup after exiting photo mode
    hideCameraPreview();
}

void showHowItWorks() {
    // Show the "How It Works" page
    k10.canvas->canvasDrawImage(0, 0, "S:screens/how_it_works.png");
    k10.canvas->updateCanvas();
    bool inHowItWorksPage = true;

    // Wait for button press to return to menu
    while (inHowItWorksPage) {
        if ((k10.buttonB->isPressed())) {
            waitForButtonRelease();
            inHowItWorksPage = false;
        }
        delay(50);
    }
}

// --- helpers for little-endian ---
uint16_t readLE16(File &f) {
  uint8_t b0 = f.read();
  uint8_t b1 = f.read();
  return b0 | (b1 << 8);
}

uint32_t readLE32(File &f) {
  uint32_t b0 = f.read();
  uint32_t b1 = f.read();
  uint32_t b2 = f.read();
  uint32_t b3 = f.read();
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

void writeLE16(File &f, uint16_t val) {
  f.write(val & 0xFF);
  f.write((val >> 8) & 0xFF);
}

void writeLE32(File &f, uint32_t val) {
  f.write(val & 0xFF);
  f.write((val >> 8) & 0xFF);
  f.write((val >> 16) & 0xFF);
  f.write((val >> 24) & 0xFF);
}

bool resizeBMP16to24(const char* inName, const char* outName) {
  File inFile = SD.open(inName, FILE_READ);
  if(!inFile) {
    Serial.println("Failed to open input file");
    return false;
  }
  
  File outFile = SD.open(outName, FILE_WRITE);
  if(!outFile) {
    Serial.println("Failed to open output file");
    inFile.close();
    return false;
  }

  // Read BMP header
  BMPHeader header;
  header.bfType      = readLE16(inFile);
  header.bfSize      = readLE32(inFile);
  header.bfReserved1 = readLE16(inFile);
  header.bfReserved2 = readLE16(inFile);
  header.bfOffBits   = readLE32(inFile);
  header.biSize      = readLE32(inFile);
  header.biWidth     = readLE32(inFile);
  header.biHeight    = readLE32(inFile);
  header.biPlanes    = readLE16(inFile);
  header.biBitCount  = readLE16(inFile);
  header.biCompression = readLE32(inFile);
  header.biSizeImage = readLE32(inFile);
  header.biXPelsPerMeter = readLE32(inFile);
  header.biYPelsPerMeter = readLE32(inFile);
  header.biClrUsed       = readLE32(inFile);
  header.biClrImportant  = readLE32(inFile);

  if(header.biBitCount != 16) {
    Serial.println("❌ Only 16-bit BMP supported");
    inFile.close();
    outFile.close();
    return false;
  }

  // Determine source orientation and use actual dimensions from header
  int actualSrcWidth  = header.biWidth;
  int actualSrcHeight = header.biHeight > 0 ? header.biHeight : -header.biHeight;
  bool srcIsBottomUp  = (header.biHeight > 0);

  int srcRowSize = ((actualSrcWidth*16 + 31)/32)*4;  // 16-bit row padded to 4 bytes
  int dstRowSize = ((dstWidth*3 + 3)/4)*4;           // 24-bit row padded

  // Check buffer sizes
  if (srcRowSize * actualSrcHeight > sizeof(srcData)) {
    Serial.println("❌ Source image too large for buffer!");
    inFile.close();
    outFile.close();
    return false;
  }
  if (dstRowSize * dstHeight > sizeof(dstData)) {
    Serial.println("❌ Destination buffer too small!");
    inFile.close();
    outFile.close();
    return false;
  }

  // Write new BMP header (24-bit)
  writeLE16(outFile, 0x4D42);
  writeLE32(outFile, 54 + dstRowSize*dstHeight);
  writeLE16(outFile, 0);
  writeLE16(outFile, 0);
  writeLE32(outFile, 54);
  writeLE32(outFile, 40);
  writeLE32(outFile, dstWidth);
  writeLE32(outFile, dstHeight);
  writeLE16(outFile, 1);
  writeLE16(outFile, 24);
  writeLE32(outFile, 0);
  writeLE32(outFile, dstRowSize*dstHeight);
  writeLE32(outFile, header.biXPelsPerMeter);
  writeLE32(outFile, header.biYPelsPerMeter);
  writeLE32(outFile, 0);
  writeLE32(outFile, 0);

  // Read full 16-bit BMP into memory
  inFile.seek(header.bfOffBits);
  size_t readBytes = inFile.read(srcData, srcRowSize * actualSrcHeight);
  if (readBytes != (size_t)(srcRowSize * actualSrcHeight)) {
    Serial.println("Warning: couldn't read full pixel data from source.");
  }

  // Clear destination (important for padding bytes)
  memset(dstData, 0, sizeof(dstData));

  // Resize with nearest-neighbor and convert RGB565 -> RGB888
  for(int y = 0; y < dstHeight; y++) {
    for(int x = 0; x < dstWidth; x++) {
      int srcX = x * actualSrcWidth / dstWidth;
      int srcY = y * actualSrcHeight / dstHeight;

      // Map row depending on BMP orientation:
      int mappedSrcY = srcIsBottomUp ? (actualSrcHeight - 1 - srcY) : srcY;

      int srcIndex = mappedSrcY * srcRowSize + srcX * 2; // 2 bytes per pixel (RGB565)

      // Safety check bounds
      if (srcIndex < 0) srcIndex = 0;
      if (srcIndex + 1 >= srcRowSize * actualSrcHeight) {
        srcIndex = (actualSrcHeight - 1) * srcRowSize; // clamp to last row start
      }

      uint16_t pixel = srcData[srcIndex] | (srcData[srcIndex+1] << 8);
      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5) & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      int dstIndex = y * dstRowSize + x * 3;
      dstData[dstIndex + 0] = b;
      dstData[dstIndex + 1] = g;
      dstData[dstIndex + 2] = r;
    }
  }

  // Write bottom-to-top as BMP expects (we created dstData top-to-bottom)
  for(int y = dstHeight - 1; y >= 0; y--) {
    outFile.write(&dstData[y * dstRowSize], dstRowSize);
  }

  inFile.close();
  outFile.close();
  Serial.println("✅ Image resized successfully");
  return true;
}

void loop() {
    // Show menu screen
    k10.canvas->canvasDrawImage(0, 0, "S:screens/menu.png");
    k10.canvas->updateCanvas();

    // Check for button presses
    if (k10.buttonA->isPressed()) {
        waitForButtonRelease();
        showPhotoMode();
    } else if (k10.buttonB->isPressed()) {
        waitForButtonRelease();
        showHowItWorks();
    }

    delay(50);
}