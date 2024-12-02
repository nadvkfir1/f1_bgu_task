# f1_bgu_task

## Nadav kfir's automobile -  task #3

At the beginning I understood that I have to train an AI model.
In the task I saw that i can use the YOLO model within roboflow. I downloaded a dataset locally (provided in the resources).

## Installation

You have to install this requirments:
```bash
pip install -r ./requirements.text 
```
** Note
you have to install Python UNDER 3.12 (I used 3.11.6)

## Training

In order to train the AI model I ran the following command:
```bash
py train.py
```

## Execution

In order to run the program you would have to use the following command:
```bash
py main.py
```

## About

Model Loading:

The script uses a pre-trained YOLO model (best.pt) to detect objects in video frames. The model is loaded from a specified directory.

Video Processing:

The script reads the input video (fsd1.mp4) frame by frame and processes each frame to detect traffic cones.
After processing, the annotated video is saved to iut.mp4.

Traffic Cone Detection:

YOLO's object detection is applied to identify traffic cones in each frame.

For each detected cone:

A bounding box is drawn around the cone.
A label is added with a unique identifier and confidence score.

Dynamic Bounding Box Coloring:

The color of the bounding box is calculated based on the average color of the bottom half of the cone, ensuring accurate representation of its dominant color while avoiding white or reflective regions.

Real-Time Display:

The annotated frames are displayed in a real-time preview window. This allows users to observe the detection process as it happens.

Output Video:

The processed video, with bounding boxes and labels, is saved in the same directory as the input video (default output: iut.mp4).








