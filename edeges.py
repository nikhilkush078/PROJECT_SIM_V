import cv2
import numpy as np
import serial
import time

#ser = serial.Serial('COM5', 9600, timeout=1)  # Change COM5 to your Arduino COM port

last_sent = None
# -------------------------
# Road Detection + Direction (Video Feed Version)
# -------------------------

# 1) Choose Input Source
# cap = cv2.VideoCapture(0)                 # <- webcam
cap = cv2.VideoCapture("new_video.mp4")   # <- video file

if not cap.isOpened():
    print("Camera/Video not found.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        break  # End of video or camera error
    
    # Resize for speed
    img = cv2.resize(img, (600, 400))
    
    # ------ Convert to HSV ------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7,7), 0)    # noise reduction

    # HSV Mask (detect road)
    lower = np.array([0, 0, 30])
    upper = np.array([180, 70, 210])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological filtering to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ------ Edge Detection ------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # ------ Region Splitting ------
    h, w = edges.shape
    lw = int(0.2 * w)
    cw = int(0.6 * w)

    left = edges[:, 0:lw]
    center = edges[:, lw:lw+cw]
    right = edges[:, lw+cw:w]

    left_count = cv2.countNonZero(left)
    center_count = cv2.countNonZero(center)
    right_count = cv2.countNonZero(right)
    total = left_count + center_count + right_count + 1

    if right_count > 8000 and left_count < 2000:
        direction = "LEFT"
        command = '2'
    elif left_count > 8000 and right_count < 2000:
        direction = "RIGHT"
        command = '3'
    else:
        direction = "STRAIGHT"
        command = '0'

    # ------ SEND TO ARDUINO SAFELY ------
    """if command != last_sent:   # send only when something changes
        ser.reset_output_buffer()  
        ser.write(command.encode())    # <-- no newline
        print(f"Sent Command: {command}")
        last_sent = command
        time.sleep(0.01)"""

        # ------ Visualization ------
    vis = img.copy()

    # dividing lines
    cv2.line(vis, (lw, 0), (lw, h), (0, 255, 0), 2)
    cv2.line(vis, (lw + cw, 0), (lw + cw, h), (0, 255, 0), 2)

    # put text
    cv2.putText(vis, f"L:{left_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"C:{center_count}", (lw+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"R:{right_count}", (lw+cw+10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(vis, f"Direction: {direction}", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # mask overlay
    overlay = vis.copy()
    overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    # combine edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(result, 0.8, edges_colored, 0.5, 0)

    # ------ Show ------
    cv2.imshow("Road Direction", vis)
    cv2.imshow("Edges", edges)

    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
    # ------ Decide direction ------
    if right_count > 8000 and left_count < 2000:
        direction = "LEFT"
        ser.write(('2' + "\n").encode())
        print(f"Sent: L [2]\n")
    elif left_count > 8000 and right_count < 2000:
        direction = "RIGHT"
        ser.write(('3' + "\n").encode())
        print(f"Sent: R [3]\n")
    else:
        direction = "STRAIGHT"
        ser.write(('0' + "\n").encode())
        print(f"Sent: S [0]\n")
"""