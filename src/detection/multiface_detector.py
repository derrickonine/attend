import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Erreur caméra")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cv2.namedWindow("Multi Face Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi Face Detection", 1200, 800)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(40, 40)
    )

    # Dessiner TOUS les visages
    for (x, y, w, h) in faces:

        # filtre taille (évite bruit)
        if w * h < 2000:
            continue

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Multi Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.getWindowProperty("Multi Face Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()