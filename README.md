# Distance Estimation using YOLOv8
# moch change
This project is designed to estimate the distance between a target object and a camera using the YOLOv8s model. It tracks the bounding box of a user-specified ID throughout a video and calculates the distance based on the height of the bounding box. 

Users need to provide the height of the object, the focal length of the camera, and the ID of the bounding box. The script generates a graph of the estimated distances and outputs a CSV file that records the corresponding video frames and distance data.

## Usage

### Installing required packagess
Required packages listed below should be installed before executing the script:
- numpy
- opencv-python
- matplotlib 
To install, execute command ``` pip install -r requirements.txt``` in directory where requirements.txt is located
yolov8 packages should also be presented in the same working directory (clone). 

### Executing Python script
To execute the script, simply type ```python3 distance.py```
Before executing the script, user should change the value of variable "KnownHeight", "focallength", "video_name" and "track_id" (line 45), motify of h (line 47).

### Output Result
After execution, a plot, followed by a csv file will be saved in current working directory.

## Work in process
- Functionality to configure custom variables in CLI without altering the code.
- OOP
- Process in patch