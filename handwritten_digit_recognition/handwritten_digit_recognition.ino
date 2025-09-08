#include <TensorFlowLite.h>
#include "mnist_model_quant.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <TinyMLShield.h>

const int inWidth = 176;
const int inHeight = 144;
const int outWidth = 28;
const int outHeight = 28;
const int blockX = inWidth / outWidth;   // = 6
const int blockY = inHeight / outHeight; // = 5
const int cropSize = 20;
const int outSize = 28;

byte rawImage[inWidth * inHeight];
byte downsampledImage[outWidth * outHeight];

constexpr int kArenaSize = 120 * 1024;
uint8_t tensor_arena[kArenaSize];

uint8_t cropped[cropSize * cropSize];        // Resized digit 
int8_t inputTensor[outSize * outSize];       // Quantized model input

// TFLite globals
const tflite::Model* model = tflite::GetModel(mnist_model_quant);
tflite::AllOpsResolver resolver;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize, error_reporter);
TfLiteTensor* input;
TfLiteTensor* output;

bool readShieldButton();
bool prevButtonState = false;

// Command control flags
bool commandRecv = false;
bool captureFlag = false;
String command;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("Serial initialized");

  initializeShield();

  if (!Camera.begin(QCIF, GRAYSCALE, 1, OV7675)) {
    Serial.println("Failed to initialize camera");
    while (1);
  }
  Serial.println("Camera initialized. Press button or type 'capture' to classify.");

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  input = interpreter.input(0);
  output = interpreter.output(0);
}

void loop() {
  // Read serial commands
  while (Serial.available()) {
    char c = Serial.read();
    if ((c != '\n') && (c != '\r')) {
      command.concat(c);
    } else if (c == '\r') {
      commandRecv = true;
      command.toLowerCase();
    }
  }

  if (commandRecv) {
    commandRecv = false;
    if (command == "capture") {
      captureFlag = true;
    }
    command = ""; // Clear after processing
  }

  bool currButtonState = readShieldButton();  // reads the TinyMLx button
  if (prevButtonState == false && currButtonState == true) {
    // falling edge: just pressed
    captureFlag = true;
    Serial.println("Button pressed. Capturing image...");
  }

  prevButtonState = currButtonState;  // update for next loop

  if (!captureFlag) return;
  captureFlag = false;

  Camera.readFrame(rawImage);  // Step 1: capture grayscale image of size inWidth * inHeight

  // Step 2: Find bounding box of the digit (invert assumption: dark digit on light background)
  int xmin = inWidth, xmax = 0, ymin = inHeight, ymax = 0;
  for (int y = 0; y < inHeight; y++) {
    for (int x = 0; x < inWidth; x++) {
      uint8_t v = rawImage[y * inWidth + x];
      if (v < 250) {  // digit pixel (not white)
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
    }
  }
  if (xmin >= xmax || ymin >= ymax) {
    // No digit found: fill with zeros
    memset(downsampledImage, 0, outSize * outSize);
    return;
  }

  // Step 3: Resize cropped digit box to cropSize, e.g., 20x20, using bilinear interpolation
  int boxW = xmax - xmin + 1;
  int boxH = ymax - ymin + 1;

  for (int y = 0; y < cropSize; y++) {
    for (int x = 0; x < cropSize; x++) {
      float gx = xmin + ((float)x / cropSize) * boxW;
      float gy = ymin + ((float)y / cropSize) * boxH;

      int gxi = (int)gx;
      int gyi = (int)gy;
      float cfx = gx - gxi;
      float cfy = gy - gyi;

      int idx00 = gyi * inWidth + gxi;
      int idx10 = gyi * inWidth + min(gxi + 1, inWidth - 1);
      int idx01 = min(gyi + 1, inHeight - 1) * inWidth + gxi;
      int idx11 = min(gyi + 1, inHeight - 1) * inWidth + min(gxi + 1, inWidth - 1);

      float v00 = rawImage[idx00];
      float v10 = rawImage[idx10];
      float v01 = rawImage[idx01];
      float v11 = rawImage[idx11];

      float top = v00 + (v10 - v00) * cfx;
      float bottom = v01 + (v11 - v01) * cfx;
      float value = top + (bottom - top) * cfy;

      // Invert (dark digit on white bg)
      cropped[y * cropSize + x] = 255 - (uint8_t)value;
    }
  }

  // Step 4: Paste cropped image into center of 28x28 and zero-pad background
  int x_offset = (outSize - cropSize) / 2;
  int y_offset = (outSize - cropSize) / 2;
  memset(downsampledImage, 0, outSize * outSize);  // Init to black

  for (int y = 0; y < cropSize; y++) {
    for (int x = 0; x < cropSize; x++) {
      downsampledImage[(y + y_offset) * outSize + (x + x_offset)] = cropped[y * cropSize + x];
    }
  }

  // Step 5: Contrast boosting (MNIST-style)
  uint8_t minPixel = 255, maxPixel = 0;
  for (int i = 0; i < outSize * outSize; i++) {
    if (downsampledImage[i] < minPixel) minPixel = downsampledImage[i];
    if (downsampledImage[i] > maxPixel) maxPixel = downsampledImage[i];
  }
  if (maxPixel == minPixel) maxPixel = minPixel + 1;

  for (int i = 0; i < outSize * outSize; i++) {
    downsampledImage[i] = 255 * (downsampledImage[i] - minPixel) / (maxPixel - minPixel);
    if (downsampledImage[i] < 150) downsampledImage[i] = 0;        // suppress background
    if (downsampledImage[i] > 200) downsampledImage[i] = 255;     // boost highlight
  }

  // Step 6: Quantize to int8 (scale = 1/255, zero_point = -128)
  for (int i = 0; i < outSize * outSize; i++) {
    float pixel = (float)downsampledImage[i]/ 255.0f;
    input->data.int8[i] = (int8_t)(round(pixel * 255.0f) - 128);  // scale = 1/255, zero_point = -128
  }

  // Step 7: (Optional) Visualize result over serial
  Serial.println("Final 28x28 Input:");
  for (int y = 0; y < outSize; y++) {
    for (int x = 0; x < outSize; x++) {
      uint8_t v = downsampledImage[y * outSize + x];
      Serial.print(v > 200 ? "#" : (v > 150 ? "+" : "."));
    }
    Serial.println();
  }

  // Step 8: (Optional) Print quantized values
  //Serial.println("Quantized input values:");
  //for (int i = 0; i < 28 * 28; i++) {
  //  Serial.print(input->data.int8[i]);
  //  Serial.print(" ");
  //  if (i % 28 == 27) Serial.println();
  //}

  // Run inference
  if (interpreter.Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  } else {
    Serial.println("Invoke succeeded");
  }

  // Parse output
  float out_scale = output->params.scale;
  int out_zero_point = output->params.zero_point;
  int8_t max_val = -128;
  int predicted = -1;
  for (int i = 0; i < 10; i++) {
    int8_t raw = output->data.int8[i];
    float score = out_scale * (raw - out_zero_point);;
  //  Serial.print("Digit "); Serial.print(i);
  //  Serial.print(": "); Serial.println(score, 4);

    if (raw > max_val) {
      max_val = raw;
      predicted = i;
    }
  }
  Serial.print("Predicted digit: ");
  Serial.println(predicted);

  // Blink LED 'bestDigit' times
  for (uint8_t i = 0; i < predicted; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(300);
    digitalWrite(LED_BUILTIN, LOW);
    delay(300);
  }

} 
