import serial
import numpy as np
import os

# --- Parameters ---
PORT = 'COM7'
BAUD = 115200
WIDTH, HEIGHT = 176, 144
CROP_W, CROP_H = 160, 140
CROP_X, CROP_Y = (WIDTH - CROP_W) // 2, (HEIGHT - CROP_H) // 2
DOWNSAMPLED_SIZE = 20
BLOCK_W = CROP_W // DOWNSAMPLED_SIZE
BLOCK_H = CROP_H // DOWNSAMPLED_SIZE
NUM_IMAGES = 1  # how many images to collect
LABEL = 1        # 0 for capturing stars...change to 1 if capturing circles

# --- Output CSVs ---
image_csv_path = "images.csv"
label_csv_path = "labels.csv"

# --- Helper functions ---
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

# --- Load existing dataset if available ---
if os.path.exists(image_csv_path):
    images = np.loadtxt(image_csv_path, delimiter=",", dtype=np.uint8)
    labels = np.loadtxt(label_csv_path, delimiter=",", dtype=np.uint8)
    if images.ndim == 1:
        images = images.reshape(1, -1)
        labels = np.array([labels])
else:
    images = np.empty((0, DOWNSAMPLED_SIZE * DOWNSAMPLED_SIZE), dtype=np.uint8)
    labels = np.empty((0,), dtype=np.uint8)

# --- Capture new images ---
ser = serial.Serial(PORT, BAUD, timeout=3)

for i in range(NUM_IMAGES):
    input(f"[{i+1}/{NUM_IMAGES}] Press Enter to capture...")
    img_raw = capture_frame(ser)
    img_down = crop_and_downsample(img_raw)
    images = np.vstack((images, img_down))
    labels = np.append(labels, LABEL)

ser.close()

# --- Save updated dataset ---
np.savetxt(image_csv_path, images, delimiter=",", fmt='%d')
np.savetxt(label_csv_path, labels, delimiter=",", fmt='%d')

print(f"✅ Appended {NUM_IMAGES} new images to '{image_csv_path}' and labels to '{label_csv_path}'")