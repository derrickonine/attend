# Face Recognition Attendance System
Facial recognition project for student attendance management.

## Current phase
Testing and validation of the video acquisition pipeline via iVCam.

## Hardware pipeline
iPhone/Android Camera → iVCam → Virtual webcam → Python (OpenCV) → Video display

## Scripts
camera_selector.py  
Detects available cameras.

test_camera.py  
Tests the selected camera and displays the video stream.

## Camera Setup Requirements
This project uses a smartphone camera through **iVCam**.

### Network Requirements
The computer and the smartphone must be on the **same local network**.

⚠️Some institutional networks (schools or companies) enable client isolation, which prevents devices from communicating with each other.

In such cases the camera connection will fail.

### Recommended solutions to ensure connectivity
- Use a personal **WiFi network**
- Use a **mobile hotspot**
- Connect both devices to the **same private network**

### Best Practices
- Ensure that **iVCam is running before starting the Python script**
- Verify the available cameras using `camera_selector.py`
- If multiple cameras are detected, select the correct index in `test_camera.py`

### Tested configuration
iPhone → iVCam → Virtual Webcam → OpenCV (Python)

********************************************************************************************************************************************************************************************************************************************************************

# Système de reconnaissance faciale pour la gestion des présences
Projet de reconnaissance faciale pour la gestion des présences des étudiants.

## Phase actuelle
Test et validation du pipeline d'acquisition vidéo via iVCam.

## Pipeline matériel
Caméra iPhone/Android → iVCam → Webcam virtuelle → Python (OpenCV) → Affichage vidéo

## Scripts
camera_selector.py  
Détecte les caméras disponibles.

test_camera.py  
Teste la caméra sélectionnée et affiche le flux vidéo.

## Configuration requise pour la caméra
Ce projet utilise l'appareil photo d'un smartphone via **iVCam**.

### Exigence Configuration réseau requise
L'ordinateur et le smartphone doivent être connectés au **même réseau local**.

⚠️Certains réseaux institutionnels (écoles ou entreprises) activent l'isolation des clients, ce qui empêche les appareils de communiquer entre eux.
Dans ce cas, la connexion de la caméra échouera.

### Solutions recommandées
- Utilisez un **réseau WiFi personnel**.
- Utilisez un **point d'accès mobile**.
- Connectez les deux appareils au **même réseau privé**.

### Bonnes pratiques
- Assurez-vous que **iVCam est en cours d'exécution avant de démarrer le script Python**
- Vérifiez les caméras disponibles à l'aide de `camera_selector.py`
- Si plusieurs caméras sont détectées, sélectionnez l'index correct dans `test_camera.py`

### Configuration testée
iPhone → iVCam → Webcam virtuelle → OpenCV (Python)

Traduit avec DeepL.com (version gratuite)