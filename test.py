import os
import cv2
import numpy as np
from deepface import DeepFace

test_image = "testing.jpg"  # Path to the test image
faces_file = "faces"  # Path to the folder containing the faces images

# Load the test image
image = cv2.imread(test_image)
if image is None:
    raise ValueError(f"Test image '{test_image}' could not be loaded.")

# Convert to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces in the test image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("No faces detected in the test image.")
else:
    print(f"{len(faces)} face(s) detected in the test image.")

# Iterate over each detected face
for idx, (x, y, w, h) in enumerate(faces):
    # Extract the face region
    face = image[y:y+h, x:x+w]
    face_path = f"temp_face_{idx + 1}.jpg"
    cv2.imwrite(face_path, face)  # Save the face temporarily for DeepFace

    # Perform face matching using DeepFace
    result = DeepFace.find(img_path=face_path, db_path=faces_file, model_name="VGG-Face", enforce_detection=False)

    # Display matches for the current face
    if not result.empty:
        print(f"Matches for Face {idx + 1}:")
        for _, row in result.iterrows():
            matched_identity = row["identity"]
            matched_name = os.path.splitext(os.path.basename(matched_identity))[0]
            print(f"Face {idx + 1} matched with: {matched_name}")
    else:
        print(f"Face {idx + 1} not matched with any image in the database.")

    # Draw a rectangle around the face and label it
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"Face {idx + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Remove the temporary face file
    os.remove(face_path)

# Display the annotated image
cv2.imshow("Test Image with Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()