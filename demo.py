import cv2
import mediapipe as mp
import numpy as np
from itertools import combinations
from ultralytics import YOLO
import joblib
import time
# Load your YOLO and XGBoost models here
model = YOLO('yolov8m.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
loaded_model = joblib.load("C:/Users/PRIYANSHU/sem 5/ML and PR/xgboost_model12.pkl")

# Open the video file
video_path = "C:/Users/PRIYANSHU/sem 5/ML and PR/demo_vid.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = "C:/Users/PRIYANSHU/sem 5/ML and PR/output_video.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
start_time = time.time()
window_duration = 5  # Calculate average score over a 5-second window

# Lists to store scores for each frame
scores = []

# Iterate through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current elapsed time
    elapsed_time = time.time() - start_time

    def process_image(image): #ANGLES COBMINATIONS , LANDS
        
        lands=[]
        
        resized_image = cv2.resize(image, (1920, 1080))
        results = model(resized_image, show=False, save=False)
        result = results[0]

        # Extract bounding boxes for 'person' class
        boxes_with_people = [box for box in result.boxes if result.names[box.cls[0].item()] == 'person' and box.conf[0] > 0.85]

        for i, box in enumerate(boxes_with_people):
            cords = [round(x) for x in box.xyxy[0].tolist()]
            
            x1, y1, x2, y2 = cords
            
            cropped_image = resized_image[y1:y2, x1:x2]    
            land = get_pose_angles(cropped_image,(x1, y1, x2, y2))
            if type(land)!= float:    
                lands.append(land)
            else:
                continue
            #cv2.imshow(f"Cropped Image {i}", cropped_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        # Return the list of cropped images
        return lands


    def get_pose_angles(cropped_image, bounding_box): #LANDMARKS, ANGLE COMBINATIONS 
        landmark_indices = [0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22,23,24]
        image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Extract holistic landmarks
        if results.pose_landmarks:
            all_landmarks = np.array([(lmk.x, lmk.y, lmk.z) for lmk in results.pose_landmarks.landmark])       
            # Extract specific landmarks based on indices
            landmarks = all_landmarks[landmark_indices]
            if len(landmarks)<10:
                return 0.0
        else:
            return 0.0

        # Handling missing keypoints using holistic landmarks
        for i, landmark in enumerate(landmarks):
            if np.isnan(landmark).any():  # Check if the landmark is missing
                landmarks[i] = [lmk for lmk in results.pose_landmarks.landmark if not np.isnan(lmk.x) and not np.isnan(lmk.y) and not np.isnan(lmk.z)][i]

        # Calculating all possible angles between combinations of three points

        return landmarks

    def calculate_angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    
    # Process the frame to get angles and landmarks
    LAND = process_image(frame)

    # Perform predictions using your XGBoost model on LAND
    LAND = np.array(LAND)
    split_arrays = np.split(LAND, LAND.shape[0], axis=0)
    labels = []
    for i in split_arrays:
        reshaped_i = i.reshape(1, 75)
        classification = loaded_model.predict(reshaped_i)
        labels.append(classification)

    # Calculate the score based on label values
    label_values = np.array(labels).flatten()
    for i in range(len(label_values)):
        label_values[i]+=1
    score = (label_values.sum()/len(label_values) - 1 )/ 2
    scores.append(score)

    if len(scores) >= window_duration:
        avg_score = np.mean(scores[-int(window_duration * fps):])
        score_text = f"Avg Score: {avg_score:.2f}"
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame)
        out.write(frame)
        scores.pop(0)

    # Display the frame (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and close the input video
out.release()
cap.release()
cv2.destroyAllWindows()