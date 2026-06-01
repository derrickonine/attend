# Application de presence par reconnaissance faciale

Cette version transforme le prototype OpenCV en application de bureau pour une ecole privee.
Elle gere une connexion professeur, une classe, les etudiants ajoutes depuis la camera, la reconnaissance faciale pendant l'appel et les exports CSV, Excel et PDF.

## Fonctionnalites principales

- Nouvelle interface web moderne inspiree de `FaceAttend_App.html`.
- Connexion par ID professeur.
- Classe de depart: `4ISI`.
- Tableau de bord avec nom du professeur, cours, classe et horaire.
- Inscription d'un etudiant sans preregistrement.
- Capture de plusieurs images du visage avec iVCam ou une webcam.
- Installation simplifiee sans `dlib` ni `face-recognition`.
- Reconnaissance pendant l'appel avec nom affiche au-dessus du visage.
- Presence enregistree automatiquement dans SQLite.
- Exports par seance en CSV, Excel `.xlsx` et PDF.
- Base de donnees locale: `data/attendance.db`.
- Administration web: ajout de classes, professeurs, import CSV et sauvegarde JSON.

## Comptes professeur inclus

| ID professeur | Nom | Cours | Classe |
|---|---|---|---|
| `PROF-4ISI` | Professeur 4ISI | Programmation Python | 4ISI |
| `PROF-MATH` | Nadia El Amrani | Mathematiques appliquees | 4ISI |
| `PROF-RESEAU` | Karim Bennis | Reseaux informatiques | 4ISI |
| `PROF-BD` | Samira Alaoui | Base de donnees | 4ISI |

## Installation

Python 3.9 ou plus est recommande.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Cette version utilise OpenCV pour detecter et comparer les visages, donc elle evite le probleme Windows/Python 3.13 lie a `dlib`.

## Lancement

Version web moderne:

```bash
python web_app.py
```

Puis ouvrir:

```text
http://127.0.0.1:5000
```

Version bureau simple:

```bash
python app.py
```

## Utilisation

1. Lance iVCam sur le telephone et sur le PC.
2. Ouvre l'application avec `python app.py`.
3. Connecte-toi avec `PROF-4ISI`.
4. Clique sur `Ajouter un etudiant + visages`.
5. Saisis le nom de l'etudiant.
6. Choisis l'index camera iVCam, souvent `1` ou `2`.
7. Dans la fenetre camera, appuie sur `ESPACE` pour capturer plusieurs images du visage.
8. Clique sur `Demarrer l'appel iVCam`.
9. Quand un etudiant est reconnu plusieurs frames de suite, il est marque present.
10. Appuie sur `Q` ou `ESC` pour terminer l'appel.
11. Selectionne une seance et exporte en CSV, Excel ou PDF.

## Dossiers importants

- `app.py`: nouvelle application principale.
- `data/attendance.db`: base SQLite creee automatiquement.
- `data/faces/`: images capturees pour chaque etudiant.
- `exports/`: fichiers CSV, Excel et PDF generes.
- `src/`: anciens scripts de test camera/detection conserves.

## Ameliorations possibles ensuite

- Ajouter un compte administrateur pour creer plusieurs classes et professeurs depuis l'interface.
- Ajouter des mots de passe professeurs.
- Ajouter une page d'historique par etudiant.
- Ajouter une correction manuelle des absences.
- Passer a une application web Flask/Django si plusieurs professeurs doivent l'utiliser depuis plusieurs machines.
