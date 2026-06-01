import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()


cv2.namedWindow("iVCam Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("iVCam Test", 1200, 800)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Erreur de lecture caméra")
        break

    cv2.imshow("iVCam Test", frame)

    # quittez la fenêtre en cliquant sur la croix ou en appuyant sur ESC
    if cv2.getWindowProperty("iVCam Test", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()