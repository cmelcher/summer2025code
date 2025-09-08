import serial
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
PORT = 'COM7'
BAUD = 115200
WIDTH, HEIGHT = 176, 144
CROP_W, CROP_H = 160, 140
CROP_X, CROP_Y = (WIDTH - CROP_W) // 2, (HEIGHT - CROP_H) // 2
DOWNSAMPLED_SIZE = 20
BLOCK_W = CROP_W // DOWNSAMPLED_SIZE
BLOCK_H = CROP_H // DOWNSAMPLED_SIZE

# --- Load trained weights (theta) ---
theta = np.load("theta.npy")  # shape: (400,)
theta = theta.flatten()
assert theta.shape == (400,), "Expected theta to be a (400,) vector"

# --- Sigmoid Function ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Image Processing ---
def capture_frame(ser):
    ser.write(b'x')
    buf = bytearray()
    while len(buf) < WIDTH * HEIGHT:
        buf.extend(ser.read(WIDTH * HEIGHT - len(buf)))
    return np.frombuffer(buf, dtype=np.uint8).reshape((HEIGHT, WIDTH))

def crop_and_downsample(img):
    crop = img[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    downsampled = np.zeros((DOWNSAMPLED_SIZE, DOWNSAMPLED_SIZE), dtype=np.uint8)
    for i in range(DOWNSAMPLED_SIZE):
        for j in range(DOWNSAMPLED_SIZE):
            block = crop[i*BLOCK_H:(i+1)*BLOCK_H, j*BLOCK_W:(j+1)*BLOCK_W]
            downsampled[i, j] = int(np.mean(block))
    return downsampled.flatten()

# --- Main ---
ser = serial.Serial(PORT, BAUD, timeout=3)
input("Press Enter to capture and classify the image...")
img_raw = capture_frame(ser)
ser.close()

x = crop_and_downsample(img_raw) / 255.0  # normalize
plt.imshow(crop_and_downsample(img_raw).reshape(20, 20), cmap='gray')
plt.show()
# --- Predict ---
logit = np.dot(theta, x)
pred = 1 if logit >= 0 else -1

label_str = "Circle" if pred == 1 else "Star"
print(f"🔍 Prediction: {label_str}  (Raw value: {pred})")