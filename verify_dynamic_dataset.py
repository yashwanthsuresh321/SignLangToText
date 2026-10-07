import numpy as np
import os

DATASET_PATH = "data/datasets/dynamic_dataset.npz"

print("=" * 60)
print("        SANKET DYNAMIC DATASET VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Check file
# ------------------------------------------------------------

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset file not found!")
    print(DATASET_PATH)
    raise SystemExit

print("\nDataset file found:")
print(DATASET_PATH)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

try:
    data = np.load(DATASET_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]

except Exception as e:
    print("\nERROR: Could not load dataset.")
    print(e)
    raise SystemExit

# ------------------------------------------------------------
# 3. Basic information
# ------------------------------------------------------------

print("\n--- DATASET SHAPES ---")

print("X shape :", X.shape)
print("y shape :", y.shape)

print("X dtype :", X.dtype)
print("y dtype :", y.dtype)

# ------------------------------------------------------------
# 4. Expected dimensions
# ------------------------------------------------------------

expected_X_shape = (300, 20, 64)
expected_y_shape = (300,)

print("\n--- DIMENSION CHECK ---")

if X.shape == expected_X_shape:
    print("PASS: X has expected shape (300, 20, 64)")
else:
    print("FAIL: X shape is", X.shape)
    print("Expected:", expected_X_shape)

if y.shape == expected_y_shape:
    print("PASS: y has expected shape (300,)")
else:
    print("FAIL: y shape is", y.shape)
    print("Expected:", expected_y_shape)

# ------------------------------------------------------------
# 5. Label distribution
# ------------------------------------------------------------

print("\n--- LABEL DISTRIBUTION ---")

j_count = np.sum(y == 0)
z_count = np.sum(y == 1)

print("J sequences :", j_count)
print("Z sequences :", z_count)
print("Total       :", len(y))

if j_count == 150:
    print("PASS: 150 J sequences")
else:
    print("FAIL: Expected 150 J sequences")

if z_count == 150:
    print("PASS: 150 Z sequences")
else:
    print("FAIL: Expected 150 Z sequences")

# ------------------------------------------------------------
# 6. Check labels
# ------------------------------------------------------------

print("\n--- LABEL VALIDATION ---")

unique_labels = np.unique(y)

print("Unique labels:", unique_labels)

if np.all(np.isin(unique_labels, [0, 1])):
    print("PASS: Only valid labels 0 (J) and 1 (Z) found")
else:
    print("FAIL: Invalid label detected")

# ------------------------------------------------------------
# 7. Check NaN / Inf
# ------------------------------------------------------------

print("\n--- NUMERICAL VALIDATION ---")

if np.isnan(X).any():
    print("FAIL: NaN values detected")
else:
    print("PASS: No NaN values")

if np.isinf(X).any():
    print("FAIL: Infinite values detected")
else:
    print("PASS: No infinite values")

# ------------------------------------------------------------
# 8. Check each sequence
# ------------------------------------------------------------

print("\n--- SEQUENCE VALIDATION ---")

bad_sequences = []

for i, sequence in enumerate(X):

    if sequence.shape != (20, 64):
        bad_sequences.append(
            (i, sequence.shape)
        )

if len(bad_sequences) == 0:
    print("PASS: All 300 sequences are exactly (20, 64)")
else:
    print("FAIL:", len(bad_sequences), "invalid sequences")

    for item in bad_sequences[:10]:
        print("Sequence:", item[0], "Shape:", item[1])

# ------------------------------------------------------------
# 9. Check feature ranges
# ------------------------------------------------------------

print("\n--- FEATURE STATISTICS ---")

print("Minimum value :", np.min(X))
print("Maximum value :", np.max(X))
print("Mean          :", np.mean(X))
print("Standard dev  :", np.std(X))

# ------------------------------------------------------------
# 10. Per-class statistics
# ------------------------------------------------------------

print("\n--- CLASS STATISTICS ---")

J = X[y == 0]
Z = X[y == 1]

print("\nJ:")
print("Shape :", J.shape)
print("Mean  :", np.mean(J))
print("Std   :", np.std(J))
print("Min   :", np.min(J))
print("Max   :", np.max(J))

print("\nZ:")
print("Shape :", Z.shape)
print("Mean  :", np.mean(Z))
print("Std   :", np.std(Z))
print("Min   :", np.min(Z))
print("Max   :", np.max(Z))

# ------------------------------------------------------------
# 11. Final result
# ------------------------------------------------------------

valid = True

if X.shape != (300, 20, 64):
    valid = False

if y.shape != (300,):
    valid = False

if j_count != 150:
    valid = False

if z_count != 150:
    valid = False

if not np.all(np.isin(unique_labels, [0, 1])):
    valid = False

if np.isnan(X).any():
    valid = False

if np.isinf(X).any():
    valid = False

if len(bad_sequences) != 0:
    valid = False

print("\n" + "=" * 60)

if valid:
    print("          DATASET VERIFICATION PASSED")
    print("=" * 60)
    print("300 valid sequences")
    print("150 J + 150 Z")
    print("20 frames per sequence")
    print("64 features per frame")
    print("No NaN / Inf values")
    print("Dataset is ready for LSTM training.")
else:
    print("          DATASET VERIFICATION FAILED")
    print("=" * 60)
    print("Do NOT train yet.")
    print("Review the failed checks above.")

print("=" * 60)
