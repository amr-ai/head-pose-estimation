# Head Pose Tracker for Exam Surveillance

## Description
This project is a computer vision system designed to detect suspicious head movements during exams. It uses YOLOv8 pose estimation to track head orientation and identifies potential cheating behaviors such as:

- Looking left or right for extended periods
- Repeated direction changes (looking left/right multiple times)
- Sustained glances in one direction (more than 5 seconds)

The system assigns unique IDs to each detected person, tracks their head movements over time, and saves screenshots when cheating is detected. It includes performance optimizations like frame skipping and model input resizing to enable real-time processing.

## Features
- Real-time head pose estimation using YOLOv8-pose
- Object tracking with SORT algorithm
- Cheating behavior detection algorithms
- Automatic screenshot capture of suspicious activities
- Performance optimizations for real-time processing
- Visual indicators of head direction and cheating status

## Requirements
Create a `requirements.txt` file with the following content:

```
ultralytics==8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
filterpy==1.4.5
scikit-image==0.19.3
```

## Installation
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`

## Usage
1. Place your video file in the same directory 
2. Run the script: `python head_pose_tracker.py`
3. Press ESC to exit the program

Key controls:
- ESC: Exit the program

## Output
The program will:
1. Display real-time tracking with bounding boxes and head direction indicators
2. Save screenshots of detected cheating cases in a "screenshots" directory
3. Show a counter of total cheating cases detected

## Performance Notes
The system includes several optimizations:
- Processes every 2nd frame by default (configurable)
- Resizes input frames for faster inference
- Caches detection results between frames

For better accuracy on high-end hardware, you can reduce or remove these optimizations by adjusting `frame_skip` and `model_input_size` parameters.

## Limitations
- Performance depends on hardware capabilities
- Accuracy may vary with video quality and camera angle
- Works best with frontal face views

## Future Improvements
- Add support for camera input
- Implement database logging of cheating incidents
- Add audio alerts for proctors
- Improve detection accuracy with additional heuristics
