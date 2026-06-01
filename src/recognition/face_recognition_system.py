import os
import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np

# ==============================
# CONFIG
# ==============================

CAMERA_INDEX = 1   # ⚠️ change si besoin (IVCam souvent 1 ou 2)
THRESHOLD = 0.6    # 🔥 augmenté (plus tolérant)

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "..", "detection", "models", "yolov8n-face.pt")

# ==============================
# LOAD YOLO
# ==============================

model = YOLO(MODEL_PATH)

# ==============================
# LOAD DATASET
# ==============================

known_encodings = []
known_names = []

print("Chargement du dataset...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

print(f"Dataset chargé : {len(known_names)} visages")

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la caméra")
    exit()

# ==============================
# FULLSCREEN
# ==============================

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ==============================
# STABILISATION
# ==============================

last_name = "Unknown"
stable_count = 0
display_name = "..."

# ==============================
# LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:

            h, w, _ = frame.shape

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ==============================
            # 🔥 PADDING (TRÈS IMPORTANT)
            # ==============================

            padding = 30

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb_face)

            name = "Unknown"

            if len(encodings) > 0:

                face_distances = face_recognition.face_distance(
                    known_encodings, encodings[0]
                )

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    distance = face_distances[best_match_index]

                    # 🔍 DEBUG
                    print(f"Distance: {distance:.2f}")

                    if distance < THRESHOLD:
                        name = known_names[best_match_index]

            # ==============================
            # STABILISATION
            # ==============================

            if name == last_name:
                stable_count += 1
            else:
                stable_count = 0

            last_name = name

            if stable_count > 5:
                display_name = name
            else:
                display_name = "..."

            # ==============================
            # DRAW
            # ==============================

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(frame, display_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()