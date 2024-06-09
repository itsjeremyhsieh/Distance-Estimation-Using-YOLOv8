from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

def distance_to_camera(knownWidth, focalLength, perHeight):
    # compute and return the distance from the maker to the camera
    return float((knownWidth * focalLength) / perHeight)

#set parameters
KnownWidth = 150
focallength = 1300
video_name = "outdoor_4"
x_axis = []
y_axis = []
cnt = 0

model = YOLO("./yolov8-object-tracking/yolov8s.pt")
video = "20240519_video/" + video_name + ".mp4"
cap = cv2.VideoCapture(video)
# out = cv2.VideoWriter("Car_away_indoor_1_offset.mp4", -1, 30.0, (1920, 1080))

track_history = defaultdict(lambda: [])
f = open(video_name + "_0.7.csv", "w", newline='')
writer = csv.writer(f)
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        try:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box  
                if track_id == 9: #need to set target id manually
                    x_axis.append(cnt)
                    h *= 0.7 # *0.7, 0.8, 0.9, 1.0
                    dist = distance_to_camera(KnownWidth, focallength, h)
                    dist = round(dist, 2)
                    y_axis.append(dist)
                    # print(f"Distance: {dist}\nH: {h}")
                    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    to_print = f"Distance: {dist} cm"
                    cv2.putText(annotated_frame, to_print, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    data = [cnt, dist]
                    writer.writerow(data)
                    cnt += 1

                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)


            finished_tracks = track_history.keys() - track_ids
            for ft_id in finished_tracks:
                ft = track_history.pop(ft_id)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            pass
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
x_axis = np.array(x_axis)
y_axis = np.array(y_axis)
plt.plot(x_axis, y_axis, "r")
plt.xlabel("Time (Frame)")
plt.ylabel("Distance (cm)")
plt.savefig(video+".png")
plt.show()
f.close()

