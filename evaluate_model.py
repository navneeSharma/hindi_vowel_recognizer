import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import tensorflow as tf

# ── CONFIG ──────────────────────────────────────────────
DATA_DIR   = "processed"
IMG_SIZE   = 64
WRITERS    = ["writer_navnee", "writer_mom"]
CHARACTERS = ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]
# ────────────────────────────────────────────────────────

# ── LOAD DATA (same as training script, same random_state) ──
X = []
y = []

for writer in WRITERS:
    for label_index, char in enumerate(CHARACTERS):
        folder = os.path.join(DATA_DIR, writer, char)
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                path = os.path.join(folder, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                X.append(img)
                y.append(label_index)

X = np.array(X).astype("float32") / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── LOAD TRAINED MODEL ──────────────────────────────────
model = tf.keras.models.load_model("hindi_vowel_model.keras")

# ── PREDICTIONS ──────────────────────────────────────────
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── CLASSIFICATION REPORT ───────────────────────────────
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=CHARACTERS))

# ── CONFUSION MATRIX ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CHARACTERS)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — Hindi Vowel Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")