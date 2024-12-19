# Real-Time Face Detection with SSD + MobileNet

## Overview

This project implements real-time face detection using the SSD (Single Shot MultiBox Detector) architecture with MobileNet as the base network. The system captures video from the webcam, detects faces, and overlays bounding boxes with timestamps. It also displays real-time statistics such as the number of faces detected and FPS.

## Features
-	Real-time face detection using a pre-trained deep learning model.
-	Displays the number of detected faces, FPS, and timestamps on each frame.
-	Saves the processed video as an output file (output_video.avi).
-	Easily configurable threshold values for confidence and performance tuning.

## Project Structure
vision-AI/
│
├── myenv/                     # Virtual environment for project dependencies
├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained model for face detection
├── deploy.prototxt            # Configuration file for the model
├── main_code.py               # Python script for face detection
├── requirements.txt           # Dependencies
├── output_video.avi           # Example output video (after script execution)
└── README.md                  # Documentation file

## Requirements

### Software & Libraries
-	Python 3.8+
-	OpenCV 4.x

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Hardware
	•	Webcam for live video feed.
	•	CPU/GPU with reasonable processing power for real-time performance.

## How to Run
	1.	Set up the environment:
Activate the virtual environment if you have created one:

```bash
source myenv/bin/activate  # For Linux/macOS
myenv\Scripts\activate     # For Windows
```

    2.	Run the Python Script:
Execute the main script to start face detection:

python main_code.py

    3.	Control:
	•	Press q to quit the application.

	4.	Output:
	•	Processed video with annotations will be saved as output_video.avi.

## Configuration

You can modify the following parameters in the script:
	•	Confidence Threshold:
The minimum confidence level for detecting a face (default: 0.5).

```python
if confidence > 0.5:
```

## Output Video Settings:
Adjust the FPS or resolution of the output video.

```python
fps = 20.0
output_filename = "output_video.avi"
```

## Known Issues
	1.	Webcam Not Detected:
Ensure your webcam is connected and not used by another application.
	2.	Low Performance (FPS):
Reduce video resolution or use hardware acceleration if available.

## Acknowledgements
	•	SSD + MobileNet model and configurations were sourced from OpenCV’s deep learning module.

## License

This project is licensed under the MIT License. Feel free to use, modify, and share this project.
