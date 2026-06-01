import os

from ultralytics import YOLO
import cv2

# Charger modèle visage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")
model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(1)

cv2.namedWindow("YOLO Face Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Face Test", 1200, 800)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5, iou=0.4)

    face_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_count += 1

            # Dessiner rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Numéro du visage
            cv2.putText(
                frame,
                f"Face {face_count}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Afficher nombre de visages détectés
    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Cas 1 ou cas 2
    if face_count == 1:
        status = "Single Face Mode"
    elif face_count > 1:
        status = "Multi Face Mode"
    else:
        status = "No Face"

    cv2.putText(
        frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("YOLO Face Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.getWindowProperty("YOLO Face Test", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()