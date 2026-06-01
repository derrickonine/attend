from ultralytics import YOLO
import cv2
import os

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
RAW_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed")

model = YOLO(MODEL_PATH)

# parcourir chaque étudiant
for person in os.listdir(RAW_PATH):

    person_path = os.path.join(RAW_PATH, person)
    output_person_path = os.path.join(OUTPUT_PATH, person)

    os.makedirs(output_person_path, exist_ok=True)

    count = 0

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        results = model(image, conf=0.5)

        faces = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2))

        # prendre le visage principal (le plus grand)
        if len(faces) > 0:

            face_box = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

            x1, y1, x2, y2 = face_box

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            save_path = os.path.join(output_person_path, f"face_{count}.jpg")

            cv2.imwrite(save_path, face)

            count += 1

    print(f"{person} → {count} faces extraites")