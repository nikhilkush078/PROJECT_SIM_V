import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6))

# --------------------------
# Load video
# --------------------------
cap = cv2.VideoCapture('./road.mp4')

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# -------------------------------------
# YOUR PYRAMID-SHAPE COORDINATES
# -------------------------------------
polygon = np.array([
    [350, 100],
    [450, 100],
    [1500, 700],
    [-700, 700]
], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for display
    frame = cv2.resize(frame, (800, 700))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    # --------------------------------------
    # Create and apply mask
    # --------------------------------------
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_frame = cv2.bitwise_and(rgb, rgb, mask=mask)

    # --------------------------------------
    # Gaussian blur + Edges
    # --------------------------------------
    blur = cv2.GaussianBlur(masked_frame, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    # --------------------------------------
    # REGION SPLITTING (Left / Right)
    # --------------------------------------
    split_x1 = int(w * 0.33)     # Left boundary
    split_x2 = int(w * 0.66)     # Right boundary

    left_region = edges[:, :split_x1]
    right_region = edges[:, split_x2:]

    left_pixels = cv2.countNonZero(left_region)
    right_pixels = cv2.countNonZero(right_region)

    # --------------------------------------
    # Prepare edge window
    # --------------------------------------
    edges_copy = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cv2.line(edges_copy, (split_x1, 0), (split_x1, h), (255, 0, 0), 2)
    cv2.line(edges_copy, (split_x2, 0), (split_x2, h), (255, 0, 0), 2)

    cv2.putText(edges_copy, f"L:{left_pixels}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(edges_copy, f"R:{right_pixels}", (w-200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # --------------------------------------
    # Prepare display window
    # --------------------------------------
    display = masked_frame.copy()
    cv2.line(display, (split_x1, 0), (split_x1, h), (255, 0, 0), 2)
    cv2.line(display, (split_x2, 0), (split_x2, h), (255, 0, 0), 2)

    cv2.putText(display, f"L:{left_pixels}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(display, f"R:{right_pixels}", (w-200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # --------------------------------------
    # Show both windows
    # --------------------------------------
    ax.clear()
    ax.imshow(display)
    ax.set_title("Masked Video")
    ax.set_xticks(np.arange(0, w, 50))
    ax.set_yticks(np.arange(0, h, 50))
    ax.grid(True, color="white", linewidth=0.3)

    bx.clear()
    bx.imshow(edges_copy)
    bx.set_title("Edges View")
    bx.set_xticks(np.arange(0, w, 50))
    bx.set_yticks(np.arange(0, h, 50))
    bx.grid(True, color="white", linewidth=0.3)

    plt.pause(0.001)

cap.release()
plt.ioff()
plt.show()
