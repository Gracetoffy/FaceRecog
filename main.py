from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Load Image
img_path = "IMG_20241119_230540_488.jpg"  # Replace with your image path

# Detect Faces
detected_faces = DeepFace.extract_faces(img_path, detector_backend='retinaface',enforce_detection=False)
face = detected_faces[0]["face"]

# Convert back to RGB (since OpenCV loads images in BGR format)
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting the Detected Face
for i, face in enumerate(detected_faces):
    face_img = face["face"]  # Get face image
    plt.subplot(1, len(detected_faces), i + 1)  # Arrange multiple faces in one row
    plt.imshow(face_img)
    plt.axis("off")
    plt.title(f"Face {i + 1}")
plt.show()
