import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Data
# -----------------------------
X = np.loadtxt('/Users/codymelcher/Desktop/ArduinoProjects/SEA Arduino/binary_classification_arduino/images.csv', delimiter=",", dtype=np.float32)
y = np.loadtxt('/Users/codymelcher/Desktop/ArduinoProjects/SEA Arduino/binary_classification_arduino/labels.csv', delimiter=",", dtype=np.float32).reshape(-1, 1)

# Normalize pixels
X /= 255.0

# Convert labels from {0, 1} to {-1, +1}
y = 2 * y - 1

n_samples, n_features = X.shape
theta = np.zeros((n_features, 1))  # No bias term

# -----------------------------
# Logistic Loss & Gradient
# -----------------------------
def logistic_loss(X, y, theta):
    z = y * (X @ theta)
    return np.mean(np.log(1 + np.exp(-z)))

def logistic_grad(X, y, theta):
    z = y * (X @ theta)
    coeff = -y / (1 + np.exp(z))
    return (X.T @ coeff) / n_samples

# -----------------------------
# Gradient Descent
# -----------------------------
epochs = 1000
lr = 0.1
grad_norms = []

for epoch in range(epochs):
    loss = logistic_loss(X, y, theta)
    grad = logistic_grad(X, y, theta)
    theta -= lr * grad

    grad_norm = np.linalg.norm(grad)
    grad_norms.append(grad_norm)

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Grad Norm = {grad_norm:.4f}")

np.save("theta.npy", theta)
# -----------------------------
# Accuracy on Training Data
# -----------------------------
y_pred = np.sign(X @ theta)
acc = np.mean(y_pred == y)
print(f"\n✅ Final Training Accuracy: {acc * 100:.2f}%")

# -----------------------------
# Plot Gradient Norm
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(grad_norms)
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm (L2)")
plt.title("Gradient Norm vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()