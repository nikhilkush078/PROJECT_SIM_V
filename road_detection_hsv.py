import cv2
import numpy as np

# 1) Read image
img = cv2.imread('./vlc_original.png')
if img is None:
    print("Cannot read image - put the image in the same folder.")
    exit()

# 2) Resize for display
img = cv2.resize(img, (800, 400))

# 3) Define color range for green areas (assuming a typical green range in HSV)
# This range will capture green regions (you may adjust these values if necessary)
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# 4) Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 5) Create mask for green regions
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# 6) Blur the green areas by applying a Gaussian blur on the entire image
# but only in the green regions.
blurred_img = img.copy()
blurred_img[green_mask == 255] = cv2.GaussianBlur(img, (15, 15), 0)[green_mask == 255]

# 7) Now we work with the blurred image for further processing
# Convert the blurred image to HSV
hsv_blurred = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

# 8) Make mask for road (gray/dark region)
lower = np.array([0, 0, 30])
upper = np.array([180, 70, 210])
mask = cv2.inRange(hsv_blurred, lower, upper)

# 9) Split mask into 3 regions (left, center, right)
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

# 12) Direction decision (10% difference rule)
difference = abs(left_pct - right_pct)
if left_pct > right_pct and difference > 5:
    direction = "LEFT"
elif right_pct > left_pct and difference > 5:
    direction = "RIGHT"
else:
    direction = "STRAIGHT"

# 13) Visualization
vis = blurred_img.copy()  # Use the blurred image for display
cv2.line(vis, (lw, 0), (lw, h), (0,255,0), 2)
cv2.line(vis, (lw+cw, 0), (lw+cw, h), (0,255,0), 2)

cv2.putText(vis, f"L:{left_pct:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(vis, f"C:{center_pct:.1f}%", (lw+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(vis, f"R:{right_pct:.1f}%", (lw+cw+10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(vis, f"Direction: {direction}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

overlay = vis.copy()

result = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

# 14) Show results
cv2.imshow("Original + Regions + Direction", result)
cv2.imshow("original image", hsv_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
