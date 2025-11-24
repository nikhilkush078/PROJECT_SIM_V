import cv2
import numpy as np

# -------------------------
# Road Direction Detector for Video
# -------------------------

# 1) Open video or webcam
cap = cv2.VideoCapture('./road_turn_video.mp4')  # Or use 0 for webcam

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Video ended or cannot read frame.")
        break

    # 2) Resize for display
    img = cv2.resize(img, (800, 400))

    # 3) Convert to HSV to detect green regions
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 4) Define color range for green areas (tune if necessary)
    lower_green = np.array([35, 50, 50])  # Lower bound of green
    upper_green = np.array([85, 255, 255])  # Upper bound of green

    # 5) Create mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 6) Blur the green areas
    blurred_image = img.copy()
    blurred_image[green_mask == 255] = cv2.GaussianBlur(img, (15, 15), 0)[green_mask == 255]

    # 7) Convert the blurred image to HSV
    hsv_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # 8) Create mask for road (gray/dark region)
    lower = np.array([0, 0, 30])
    upper = np.array([180, 70, 210])
    mask = cv2.inRange(hsv_blurred, lower, upper)

    # 9) Split into left, center, right
    h, w = mask.shape
    lw = int(0.2 * w)
    cw = int(0.6 * w)
    left = mask[:, 0:lw]
    center = mask[:, lw:lw+cw]
    right = mask[:, lw+cw:w]

    # 10) Count white pixels
    left_count = cv2.countNonZero(left)
    center_count = cv2.countNonZero(center)
    right_count = cv2.countNonZero(right)
    total = left_count + center_count + right_count + 1

    # 11) Calculate percentages
    left_pct = (left_count / total) * 100
    center_pct = (center_count / total) * 100
    right_pct = (right_count / total) * 100

    # 12) Direction decision (10% rule)
    difference = abs(left_pct - right_pct)
    if left_pct > right_pct and difference > 5:
        direction = "LEFT"
    elif right_pct > left_pct and difference > 5:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    # 13) Visualization
    vis = blurred_image.copy()  # Use the blurred image for display
    cv2.line(vis, (lw, 0), (lw, h), (0,255,0), 2)
    cv2.line(vis, (lw+cw, 0), (lw+cw, h), (0,255,0), 2)

    cv2.putText(vis, f"L:{left_pct:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"C:{center_pct:.1f}%", (lw+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"R:{right_pct:.1f}%", (lw+cw+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"Direction: {direction}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    overlay = vis.copy()
    overlay[mask > 0] = (0,200,0)  # Highlight detected road in green
    result = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    # 14) Display
    cv2.imshow("Road Direction Detection", result)
    cv2.imshow("Road video", mask)

    # Press 'q' to quit
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
