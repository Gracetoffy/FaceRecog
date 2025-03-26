import os
import cv2
import numpy as np
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
import pickle

# Use webcam to detect face

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.7 )
cap = cv2.VideoCapture(0)

# Use webcam to detect face and extract face embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval()

#COSINE SIMILRITY 
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2.T)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))

# to get embedding 
def get_embedding(face):
    try:
        embedding=DeepFace.represent(face, model_name='Facenet',enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    except Exception as e:
        print('Error extracting face embedding:', e)
        return None
    
#save known faces embeddings
def save_known_faces(known_faces, folder="known_faces.pkl"):
    with open(folder,"wb") as f:
        pickle.dump(known_faces, f)

#load known faces embeddings
def load_faces_from_file(folder="known_faces.pkl"):
    if not os.path.exists(folder):  
         print("⚠️ No known faces file found. Creating a new one.")
         return{}
  # Return an empty dictionary if the file doesn't exist

    with open(folder, "rb") as f:
        return pickle.load(f)

        known_faces = pickle.load(f)
        


def load_known_faces(folder="faces"):
    known_faces = load_faces_from_file()
    if known_faces:
        print("Known faces loaded from file")
        return known_faces
    face_counts={}
    known_faces={}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = "-".join(os.path.splitext(filename)[0].split("_")[:-1])
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results= face_detection.process(img_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC= detection.location_data.relative_bounding_box
                    h,w,_ = img.shape
                    x,y,w_box,h_box = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
                    face_crop = img[y:y+h_box, x:x+w_box] 
                    if face_crop.shape[0]>0 and face_crop.shape[1]>0:
                        try:
                         embedding = DeepFace.represent(face_crop, model_name='Facenet',enforce_detection=False)[0]['embedding']
                         embedding = np.array(embedding)
                         if name in known_faces:
                            known_faces[name] += embedding
                            face_counts[name] += 1
                         else:
                          known_faces[name] = embedding
                          face_counts[name] = 1
                          print(f"Embedding for {name} saved")
                        except Exception as e:
                            print('Error extracting face embedding:', e)
    for name in known_faces.keys():
        known_faces[name] /= face_counts[name]
    save_known_faces(known_faces)
    known_faces = load_known_faces()
                       

    return known_faces
known_faces = load_known_faces()
print(known_faces)


while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    # rectangles on face
    if results.detections:
        for detection in results.detections:
            bboxC= detection.location_data.relative_bounding_box
            h,w,_ = frame.shape
            x,y,w,h = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            if face.shape[0]>0 and face.shape[1]>0:
                embedding = get_embedding(face)

                best_match="Unknown"
                best_score=0.6
                for name, know_embedding in known_faces.items():
                    similarity= cosine_similarity(embedding, know_embedding)

                    if similarity>best_score:
                         best_score= similarity
                         best_match= name

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, best_match, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.imshow('Face Recognition', frame)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

