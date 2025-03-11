from deepface import DeepFace
from ultralytics import YOLO

import cv2


# WEB cam
cap = cv2.VideoCapture(0)
model = YOLO("yolov8n-face.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        for box in result.boxes.xyxy:
            box = box.int().tolist()
            print(box)
            if len(box) == 4:
                x1, y1, x2, y2 = box
            else:
                print('No face detected in', box)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),2)
   
   
    cv2.imshow('Real-time face detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




