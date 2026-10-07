import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# Paths
# ==========================================================

MODEL_PATH = "data/models/dynamic_sign_model.keras"
DATASET_PATH = "data/datasets/dynamic_dataset.npz"

LABELS = {
    0: "J",
    1: "Z"
}

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading LSTM Model")
print("=" * 60)

model = load_model(MODEL_PATH)

print("Model loaded successfully.\n")

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

data = np.load(DATASET_PATH)

X = data["X"]
y = data["y"]

print(f"Sequences : {len(X)}")
print(f"Shape     : {X.shape}\n")

# ==========================================================
# Testing
# ==========================================================

print("=" * 60)
print("Running Predictions")
print("=" * 60)

correct = 0

y_true = []
y_pred = []

for i in range(len(X)):

    sequence = np.expand_dims(X[i], axis=0)

    probabilities = model.predict(sequence, verbose=0)[0]

    predicted_class = np.argmax(probabilities)
    actual_class = int(y[i])

    predicted_label = LABELS[predicted_class]
    actual_label = LABELS[actual_class]

    confidence = probabilities[predicted_class] * 100

    status = "✓" if predicted_class == actual_class else "✗"

    if predicted_class == actual_class:
        correct += 1

    y_true.append(actual_class)
    y_pred.append(predicted_class)

    print(f"Sequence {i+1:02d}")
    print(f"Actual      : {actual_label}")
    print(f"Predicted   : {predicted_label}")
    print(f"Confidence  : {confidence:.2f}%")
    print(f"Probabilities:")
    print(f"   J : {probabilities[0]:.4f}")
    print(f"   Z : {probabilities[1]:.4f}")
    print(f"Result : {status}")
    print("-" * 50)

# ==========================================================
# Final Accuracy
# ==========================================================

accuracy = correct / len(X) * 100

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Correct Predictions : {correct}/{len(X)}")
print(f"Dataset Accuracy    : {accuracy:.2f}%")

# ==========================================================
# Confusion Matrix
# ==========================================================

print("\nConfusion Matrix")
print(confusion_matrix(y_true, y_pred))

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["J", "Z"],
        digits=4
    )
)

print("=" * 60)