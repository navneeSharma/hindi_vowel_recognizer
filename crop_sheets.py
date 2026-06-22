import cv2
import numpy as np
import os

# ── CONFIG ──────────────────────────────────────────────
GRID_ROWS = 9
GRID_COLS = 7

Y_START = 158
Y_END   = 3346
X_START = 0
X_END   = 2476

WRITERS = {
    "writer_navnee": {
        "a":  "a_navnee_sheet.jpeg",
        "aa": "aa_navnee_sheet.jpeg",
        "i":  "i_navnee_sheet.jpeg",
        "ii": "ii_navnee_sheet.jpeg",
        "u":  "u_navnee_sheet.jpeg",
        "uu": "uu_navnee_sheet.jpeg",
        "e":  "e_navnee_sheet.jpeg",
        "ai": "ai_navnee_sheet.jpeg",
        "o":  "o_navnee_sheet.jpeg",
        "au": "au_navnee_sheet.jpeg",
    },
    "writer_mom": {
        "a":  "a_mom_sheet.jpeg",
        "aa": "aa_mom_sheet.jpeg",
        "i":  "i_mom_sheet.jpeg",
        "ii": "ii_mom_sheet.jpeg",
        "u":  "u_mom_sheet.jpeg",
        "uu": "uu_mom_sheet.jpeg",
        "e":  "e_mom_sheet.jpeg",
        "ai": "ai_mom_sheet.jpeg",
        "o":  "o_mom_sheet.jpeg",
        "au": "au_mom_sheet.jpeg",
    },
}
# ────────────────────────────────────────────────────────

def crop_sheet(image_path, output_folder, label, writer_name, rows=9, cols=7):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read {image_path}")
        return

    h, w = img.shape[:2]

    # Calculate grid boundaries
    y_start = Y_START
    y_end   = Y_END
    x_start = X_START
    x_end   = X_END

    grid_h = y_end - y_start
    grid_w = x_end - x_start

    cell_h = grid_h // rows
    cell_w = grid_w // cols

    count = 1
    for row in range(rows):
        for col in range(cols):
            y1 = y_start + row * cell_h
            y2 = y1 + cell_h
            x1 = x_start + col * cell_w
            x2 = x1 + cell_w

            cell = img[y1:y2, x1:x2]

            filename = f"{label}_{writer_name}_{count:03d}.jpeg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, cell)
            count += 1

    print(f"Done: {count - 1} cells saved for {label} ({writer_name})")


for writer, characters in WRITERS.items():
    for label, filename in characters.items():
        image_path = os.path.join("raw_scans", writer, filename)
        output_folder = os.path.join("dataset", writer, label)
        crop_sheet(image_path, output_folder, label, writer)