# Tutorial: Optimization, Neural Networks, and TinyML (Summer 2025)

This repository provides a hands-on tutorial for learning about optimization, neural networks, and TinyML using the Arduino Nano 33 BLE Sense.  
It introduces key ideas like gradient descent, classification (binary and multi-class), and deploying neural networks on microcontrollers.

No prior background in calculus or coding is assumed — the activities are designed to be accessible to high school and undergraduate students.  
The tutorial can be used in classrooms, outreach events, or for self-study.

---

## Contents
- binary/ – Binary classification of shapes (stars vs. circles)  
  - Python scripts for dataset creation, training, and testing  
  - Arduino sketches (`.ino`, `.h`) for deployment  
- multi/ – Multi-class handwritten digit classification (MNIST)  
  - Python training script and quantized model export  
  - Arduino sketch for deployment  
- slides/ – Lecture slides introducing the concepts  
- handout/ – Short reference handout for participants  

---

## How to Use
1. Choose either the binary or multi activity.  
2. Follow the step-by-step instructions on the [tutorial page](https://cmelcher.github.io/tinyml/summer2025/).  
3. Run Python scripts with Python 3.x and upload Arduino sketches using the Arduino IDE.  

---

## Requirements
- [Arduino IDE](https://www.arduino.cc/en/software) (for `.ino` + `.h` files)  
- Python 3.x with packages: `numpy`, `opencv-python`, `tensorflow` (for training)  
- Hardware: Arduino Nano 33 BLE Sense  

---

## Notes
This tutorial was originally implemented as part of a University of Arizona outreach session, but it is designed to be reused anytime.  
Anyone can download the materials, follow the steps, and learn how optimization and machine learning come together in TinyML.

---

📖 Full tutorial with instructions and images:  
[Summer 2025 Project Page](https://cmelcher.github.io/tinyml/summer2025/)
