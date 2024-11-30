"""
Author: Nadav Kfir
Description: ...
"""

from PIL import Image

IMAGE_PATH = r"C:\Users\nadav\OneDrive\תמונות\צילומי מסך\WhatsApp Image 2024-10-30 at 13.34.48.jpeg"

from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("runs/detect/traffic_cone_model/weights/best.pt")  # Path to your trained weights

# Load the video
video_path = "path/to/input_video.mp4"  # Path to the input video
output_path = "path/to/output_video.mp4"  # Path to save the output video

def main():
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for detection
        results = model.predict(source=frame, conf=0.5, show=False, verbose=False)  # Run detection

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Optional: Show the frame in real-time
        cv2.imshow("Traffic Cone Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved at: {output_path}")    


if __name__ == "__main__":
    main()
