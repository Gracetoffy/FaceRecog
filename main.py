import os
import cv2
import numpy as np
import albumentations as A
from deepface import DeepFace



test_image_path = "facees.jpg"  #  test image
faces_db_path = "augmented_faces" # folder containing known faces

#DATA AUGMENTATION

augmented_dir = "augmented_faces"

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.Blur(blur_limit=3, p=0.3)
])

if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

for filename in os.listdir(faces_db_path):
    if filename.endswith('.jpg')or filename.endswith('.png'):
        img_path = os.path.join(faces_db_path,filename)
        image = cv2.imread(img_path)

        if image is None:
            continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(3):
        augumented= augmentations(image=image_rgb)['image']
        augumented_bgr = cv2.cvtColor(augumented,cv2.COLOR_RGB2BGR)

        aug_img_name= f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(augmented_dir, aug_img_name), augumented_bgr)

    print('Data augmentation completed')

# Detect faces in the test image
faces = DeepFace.extract_faces(img_path=test_image_path, detector_backend="retinaface", enforce_detection=False)

if len(faces) == 0:
    print("No faces detected in the test image.")
else:
    print(f"{len(faces)} face(s) detected in the test image.")

# Iterate over each detected face
for idx, face_info in enumerate(faces):
    face = face_info["face"]
    
    # Get bounding box coordinates (optional)
    facial_area = face_info.get("facial_area", {})
    x, y, w, h = facial_area.get("x", 0), facial_area.get("y", 0), facial_area.get("w", 0), facial_area.get("h", 0)


    # Save each detected face temporarily
    face_path = f"temp_face_{idx + 1}.jpg"
    cv2.imwrite(face_path, face)

    # Perform face recognition for the detected face
    try:
        results = DeepFace.find(img_path=face_path, db_path=faces_db_path, model_name="ArcFace",distance_metric='cosine', enforce_detection=False)
    
        if results and not results[0].empty:
            print(f"Matches for Face {idx + 1}:")
            
            # Iterate through all matches
            for _, row in results[0].iterrows():
                matched_identity = row["identity"]
                matched_name = os.path.splitext(os.path.basename(matched_identity))[0]
                print(f"  - Matched with: {matched_name}")

                # Draw rectangle & label on original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, matched_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            print(f"Face {idx + 1} not matched with any image in the database.")

    except Exception as e:
        print(f"Error in face matching: {e}")

    # Remove temporary face images
    os.remove(face_path)

# Show the image with labeled faces
cv2.imshow("Identified Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
