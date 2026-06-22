import cv2
import os
import numpy as np

# ── CONFIG ──────────────────────────────────────────────
INPUT_DIR  = "dataset"
OUTPUT_DIR = "processed"
IMG_SIZE   = 64  # resize all images to 64x64
# ────────────────────────────────────────────────────────

writers    = ["writer_navnee", "writer_mom"]
characters = ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]

# Create output folders
for writer in writers:
    for char in characters:
        os.makedirs(os.path.join(OUTPUT_DIR, writer, char), exist_ok=True)

total = 0

for writer in writers:
    for char in characters:
        input_folder  = os.path.join(INPUT_DIR, writer, char)
        output_folder = os.path.join(OUTPUT_DIR, writer, char)

        files = [f for f in os.listdir(input_folder) if f.endswith(".jpeg")]

        for filename in files:
            input_path  = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".jpeg", ".png"))

            # Read image
            img = cv2.imread(input_path)

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize to 64x64
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize to 0-1 and save back as 0-255 png
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            cv2.imwrite(output_path, img)
            total += 1

        print(f"Processed {char} ({writer}): {len(files)} images")

print(f"\nDone. Total images processed: {total}")