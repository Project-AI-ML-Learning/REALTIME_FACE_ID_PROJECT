# ============================================================
# DAY 1 - Live Webcam with Face Detection
# Project: Real-Time Human Behavior Analysis System
# Concept: OpenCV basics, webcam feed, haar cascade face detection
# ============================================================

import cv2  # OpenCV - our main computer vision library

# ---------------------------------------------------------------
# CONCEPT 1: What is a Haar Cascade?
# It's a pre-trained file that knows what a face looks like.
# OpenCV comes with it built-in. We just load and use it.
# Think of it as a "face shaped stencil" the computer uses to search
# ---------------------------------------------------------------

# Load the face detector (pre-trained by OpenCV team)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------------------------------------------
# CONCEPT 2: VideoCapture
# 0 = your default webcam
# It opens a connection to your webcam like opening a file
# ---------------------------------------------------------------
cap = cv2.VideoCapture(0)

# Check if webcam opened correctly
if not cap.isOpened():
    print("ERROR: Could not open webcam. Check if it's connected.")
    exit()

print("Webcam opened successfully!")
print("Press 'Q' to quit the window")

# ---------------------------------------------------------------
# CONCEPT 3: The Video Loop
# A video is just many images shown fast (30 per second = 30 FPS)
# We read one image (frame) at a time, process it, show it
# ---------------------------------------------------------------

while True:
    # Read one frame from webcam
    # ret = True if frame was read successfully
    # frame = the actual image (numpy array of pixel values)
    ret, frame = cap.read()

    # If frame wasn't captured properly, skip
    if not ret:
        print("Failed to grab frame")
        break

    # ---------------------------------------------------------------
    # CONCEPT 4: Grayscale Conversion
    # Face detection works on grayscale images (simpler = faster)
    # Color image has 3 channels (R, G, B)
    # Grayscale has 1 channel (just brightness)
    # ---------------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------------
    # CONCEPT 5: detectMultiScale
    # This scans the image at multiple sizes to find faces
    # scaleFactor: how much to shrink image each scan (1.1 = 10% smaller each time)
    # minNeighbors: how many detections needed to confirm a face (higher = more accurate)
    # minSize: smallest face size to detect (in pixels)
    # Returns: list of rectangles (x, y, width, height) for each face found
    # ---------------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ---------------------------------------------------------------
    # CONCEPT 6: Drawing on Frames
    # We draw a rectangle around each detected face
    # (x, y) = top-left corner of the rectangle
    # (x+w, y+h) = bottom-right corner
    # (0, 255, 0) = Green color in BGR format (not RGB!)
    # 2 = thickness of the rectangle border
    # ---------------------------------------------------------------
    for (x, y, w, h) in faces:
        # Draw green rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add label above the rectangle
        cv2.putText(
            frame,              # image to draw on
            f"Face Detected",   # text to show
            (x, y - 10),        # position (above the rectangle)
            cv2.FONT_HERSHEY_SIMPLEX,  # font style
            0.7,                # font size
            (0, 255, 0),        # color (green)
            2                   # thickness
        )

    # ---------------------------------------------------------------
    # CONCEPT 7: Showing Info on Screen
    # We display face count and instructions on the video feed
    # ---------------------------------------------------------------
    face_count = len(faces)
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q to quit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show the frame in a window called "Human Behavior AI"
    cv2.imshow("Human Behavior AI - Day 1", frame)

    # ---------------------------------------------------------------
    # CONCEPT 8: waitKey
    # Wait 1 millisecond for a key press
    # ord('q') = the keyboard code for 'q'
    # & 0xFF is a bitwise operation for cross-platform compatibility
    # ---------------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# ---------------------------------------------------------------
# CONCEPT 9: Always Release Resources
# Like closing a file after reading it
# cap.release() = close webcam connection
# cv2.destroyAllWindows() = close all OpenCV windows
# ---------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("Done! Webcam released successfully.")

#We hired a robot friend and gave him a face picture book. We turned on the webcam and told him to watch it non-stop. Every photo he takes, he converts to black and white, searches for faces using his picture book, draws green boxes around them, writes labels, and shows us the result — 30 times every single second. When we press Q, he cleans up and goes home.