import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to determine robot movement based on obstacle positions
def decide_movement(detections, frame_width):
    """
    Decide the movement direction of the robot based on obstacle positions.
    :param detections: List of bounding boxes of detected obstacles.
    :param frame_width: Width of the video frame.
    :return: Movement command ('move_left', 'move_right', 'move_forward', 'stop').
    """
    left_count, right_count, center_count = 0, 0, 0

    for detection in detections:
        x_center = (detection[0] + detection[2]) / 2  # X-center of the bounding box
        if x_center < frame_width / 3:
            left_count += 1
        elif x_center > 2 * frame_width / 3:
            right_count += 1
        else:
            center_count += 1

    if center_count > 0:
        return "stop"  # Stop if obstacles are in the center
    elif left_count < right_count:
        return "move_left"  # Move left if there are more obstacles on the right
    else:
        return "move_right"  # Move right otherwise

# Initialize camera
camera = cv2.VideoCapture(0)  # Use camera index 0 or replace with a video file path
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_height, frame_width, _ = frame.shape

    # Perform YOLOv5 detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes, confidences, and class indices

    # Filter relevant obstacle classes (e.g., person, car, bicycle, chair)
    obstacle_classes = ['person', 'car', 'bicycle', 'chair']
    obstacles = [
        det[:4] for det in detections if results.names[int(det[5])] in obstacle_classes
    ]

    # Draw detections on the frame
    for obstacle in obstacles:
        x1, y1, x2, y2 = map(int, obstacle[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(
            frame,
            "Obstacle",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Decide movement based on detected obstacles
    movement_command = decide_movement(obstacles, frame_width)
    print(f"Command: {movement_command}")

    # Display the frame with detections
    cv2.imshow("YOLOv5 Real-Time Obstacle Avoidance", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
