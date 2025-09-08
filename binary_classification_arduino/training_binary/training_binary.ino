#include <TinyMLShield.h>

const int inWidth = 176;
const int inHeight = 144;
uint8_t frame[inWidth * inHeight];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin(QCIF, GRAYSCALE, 1, OV7675)) {
    Serial.println("Camera init failed!");
    while (1);
  }

  // Discard warm-up frames
  for (int i = 0; i < 3; i++) {
    Camera.readFrame(frame);
    delay(100);
  }

  Serial.println("Camera ready.");
}

void loop() {
  if (Serial.available() > 0) {
    Serial.read();
    delay(1000);  // Ensure clean next frame

    Camera.readFrame(frame);
    Serial.write(frame, inWidth * inHeight);
    Serial.println("DONE");
  }
}