import cv2

print("Recherche des caméras disponibles...")

for i in range(5):  # tester les caméras 0 à 4
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Camera {i} détectée")

        cv2.namedWindow(f"Camera {i}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Camera {i}", 800, 600)

        ret, frame = cap.read()

        if ret:
            cv2.imshow(f"Camera {i}", frame)

        cap.release()

print("Appuie sur une touche pour fermer les fenêtres")
cv2.waitKey(0)
cv2.destroyAllWindows()