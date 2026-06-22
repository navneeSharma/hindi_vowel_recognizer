import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────
DATA_DIR   = "processed"
IMG_SIZE   = 64
WRITERS    = ["writer_navnee", "writer_mom"]
CHARACTERS = ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]
# ────────────────────────────────────────────────────────

# ── LOAD DATA ───────────────────────────────────────────
X = []  # images
y = []  # labels

for writer in WRITERS:
    for label_index, char in enumerate(CHARACTERS):
        folder = os.path.join(DATA_DIR, writer, char)
        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                path = os.path.join(folder, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                X.append(img)
                y.append(label_index)

X = np.array(X)
y = np.array(y)

print(f"Total images loaded: {len(X)}")
print(f"Image shape: {X[0].shape}")

# Normalize pixel values to 0-1 and reshape for CNN input
X = X.astype("float32") / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # add channel dimension

# ── SPLIT DATA ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
"""
# ── DATA AUGMENTATION ────────────────────────────────────
datagen = ImageDataGenerator(
    rotation_range=10,        # rotate up to 10 degrees
    width_shift_range=0.1,    # shift horizontally up to 10%
    height_shift_range=0.1,   # shift vertically up to 10%
    zoom_range=0.1,           # zoom in/out up to 10%
    fill_mode="constant",
    cval=0                    # fill empty areas with black (matches your background)
)

datagen.fit(X_train)
"""

# ── BUILD MODEL ─────────────────────────────────────────
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CHARACTERS), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── TRAIN ───────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=16
)

# ── SAVE MODEL ──────────────────────────────────────────
model.save("hindi_vowel_model.keras")
print("\nModel saved as hindi_vowel_model.keras")

# ── PLOT TRAINING CURVES ────────────────────────────────
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")
print("Training curves saved as training_curves.png")