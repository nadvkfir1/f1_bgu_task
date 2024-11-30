"""
Author: Nadav Kfir
Description: ...
"""

from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("runs/detect/traffic_cone_model/weights/best.pt")  # Path to your trained weights

# Load the video
video_path = "fsd1.mp4"  # Path to the input video
output_path = "iut.mp4"  # Path to save the output video

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

        annotated_frame = frame.copy()
        for idx, result in enumerate(results[0].boxes):  # Iterate through detected boxes
            box = result.xyxy[0]  # Bounding box coordinates (xmin, ymin, xmax, ymax)
            confidence = result.conf[0]  # Confidence score
            class_id = int(result.cls[0])  # Class ID
            label = results[0].names[class_id]  # Class label
            x,y = ((box[0] + box[2]) // 2, (box[1] *6 + box[3]) // 7)
            x = int(x)
            y = int(y)
            a, b, c = tuple(annotated_frame[y][x])

            # Add a counter to the text
            label_text = f"{idx + 1}: {label} ({confidence:.2f})"  # Add counter to text

            # Draw the bounding box in red
            color = int(a), int(b), int(c)  # Red color in BGR
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

            # Add the label and counter above the bounding box
            cv2.putText(annotated_frame, label_text, (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
