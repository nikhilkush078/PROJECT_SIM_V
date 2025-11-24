import cv2
import os

def extract_equal_frames(video_path, output_folder, total_frames=60):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        print("Error: Video has 0 frames! Maybe codec not supported.")
        return

    # Calculate step size
    step = frame_count / total_frames

    for i in range(total_frames):
        frame_number = int(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if ret:
            filename = f"{output_folder}/frame_{i+1:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        else:
            print("Failed to read frame:", frame_number)

    cap.release()
    print("\nDone! Extracted", total_frames, "frames.")

# 🔻 YOUR VIDEO FILE NAME HERE
extract_equal_frames("video.webm", "output_images", 60)
