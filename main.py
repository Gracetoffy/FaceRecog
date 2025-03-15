import os
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace


# Folder containing known faces
KNOWN_FACES_DIR = "faces"
test_img = "tests.jpg"

test_face = cv2.imread(test_img)
if test_face is None:
    print(f"Error: Could not load image {test_img}")
    exit(1)

test_rgb = cv2.cvtColor(test_face, cv2.COLOR_BGR2RGB)

#FACE ENCODDING 
face_encoding = face_recognition.face_encodings(test_rgb)[0]




# Loop through images in the folder
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg",".png")) :
        person_name = os.path.splitext(filename)[0]  # Extract name from filename
        img_path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample_encoding = face_recognition.face_encodings(img_rgb)[0]

        result =face_recognition.compare_faces([face_encoding], sample_encoding, tolerance=0.6)
        print(result)
            
            


"""          print(f"Match found: {person_name} with distance {distance}")

                known_face = cv2.imread(img_path)

                for(x,y,w,h) in faces:
                    face_region= test_img[y:y+h, x:x+w]
                    temp_face_path = "temp_face.jpg"
                    cv2.imwrite(temp_face_path, face_region)

                    face_match = DeepFace.verify(img1_path=img_path,img2_path=temp_face_path, enforce_detection=False)

                    if isinstance(face_match, dict) and "verified" in face_match and face_match["verified"]:

                        best_match = (x,y,w,h)
                        break
                if best_match:
                    x,y,w,h = best_match
                    cv2.rectangle(test_face, (x,y), (x+w, y+h), (0,255,0), 2)

          

                # Resize the images to 160x160  
                test_face = cv2.resize(test_face, (160, 160))
                known_face = cv2.resize(known_face, (160, 160))

                combined_faces= np.hstack((test_face, known_face))

                cv2.imshow("Face comparison", combined_faces)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                if distance<best_match_score:
                    best_match = person_name
                    best_match_score = distance
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print(f"Unexpected result format: {result}")


    if best_match:
        print(f"Best match: {best_match} with distance {best_match_score}")
    else:
        print("No match found")"""
