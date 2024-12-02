"""
Author: Nadav Kfir
Description: ...
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# Load the model of the AI
model = YOLO(Path(__file__).parent.absolute()/"runs/detect/traffic_cone_model/weights/best.pt")  

# Load the video
video_path = "fsd1.mp4"  # Path to the input video
output_path = "iut.mp4"  # Path to save the output video

show = True

def main():
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # runs the video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to BRG 
        results = model.predict(source=frame, verbose=False)  # Run detection

        annotated_frame = frame.copy()
        for idx, result in enumerate(results[0].boxes):  # Iterate through detected boxes
            box = result.xyxy[0]  # Bounding box coordinates (xmin, ymin, xmax, ymax)
            confidence = result.conf[0]  # Confidence score
            label = "cone"
            blue = green = red = 0
            for x in range(int(box[0]), int(box[2])):
                for y in range((int(box[1])+int(box[3]))//2, int(box[3])): # Collect only bottom half for precision
                    x = int(x)
                    y = int(y)
                    blue += int(annotated_frame[y][x][0])
                    green += int(annotated_frame[y][x][1])
                    red += int(annotated_frame[y][x][2])
            number_of_pixels = (int(box[2])-int(box[0]))*(int(box[3])-((int(box[1])+int(box[3]))//2)) # Counting the pixels in the frame and dividing for each color to get the average color
            blue = blue//number_of_pixels
            green = green//number_of_pixels
            red = red//number_of_pixels 

            # Add a counter to the cone in the frame
            label_text = f"{idx + 1}: {label} ({confidence:.2f})"  

            # Draw the bounding box in color of the cone
            color = int(blue), int(green), int(red)  
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

            # Add the label and counter above the bounding box
            cv2.putText(annotated_frame, label_text, (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Optional: Show the frame in real-time
        if show:
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
