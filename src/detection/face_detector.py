import cv2

def main():
    # Charger le modèle Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erreur : caméra non accessible")
        return
    
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection", 1200, 800)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur lecture frame")
            break

        # Conversion en gris (obligatoire pour Haar)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Dessiner les rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Affichage
        cv2.imshow("Face Detection", frame)

        # Quitter avec ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Fermer avec la croix
        if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()