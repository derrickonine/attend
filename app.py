import csv
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, Entry, Frame, Label, Listbox, StringVar, Tk, Toplevel, messagebox, simpledialog

import cv2
import numpy as np

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:
    canvas = None
    A4 = None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EXPORTS_DIR = APP_DIR / "exports"
DB_PATH = DATA_DIR / "attendance.db"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FACE_SIZE = (80, 80)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return sorted(faces, key=lambda item: item[2] * item[3], reverse=True)


def crop_face(frame, face_box, padding=30):
    x, y, w, h = face_box
    height, width = frame.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    return frame[y1:y2, x1:x2]


def face_descriptor(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FACE_SIZE)
    gray = cv2.equalizeHist(gray)
    descriptor = gray.astype("float32") / 255.0
    return descriptor.flatten()


def descriptor_distance(left, right):
    return float(np.linalg.norm(left - right) / np.sqrt(left.size))


class Database:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_schema()
        self.seed_data()

    def create_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                level TEXT,
                school_year TEXT
            );

            CREATE TABLE IF NOT EXISTS teachers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_code TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                teacher_id INTEGER NOT NULL,
                weekday TEXT,
                starts_at TEXT,
                ends_at TEXT,
                UNIQUE(name, class_id, teacher_id),
                FOREIGN KEY(class_id) REFERENCES classes(id),
                FOREIGN KEY(teacher_id) REFERENCES teachers(id)
            );

            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(full_name, class_id),
                FOREIGN KEY(class_id) REFERENCES classes(id)
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                encoding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER NOT NULL,
                teacher_id INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                notes TEXT,
                FOREIGN KEY(course_id) REFERENCES courses(id),
                FOREIGN KEY(teacher_id) REFERENCES teachers(id),
                FOREIGN KEY(class_id) REFERENCES classes(id)
            );

            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                recognized_at TEXT NOT NULL,
                confidence REAL,
                UNIQUE(session_id, student_id),
                FOREIGN KEY(session_id) REFERENCES attendance_sessions(id),
                FOREIGN KEY(student_id) REFERENCES students(id)
            );
            """
        )
        self.conn.commit()

    def seed_data(self):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO classes(name, level, school_year) VALUES (?, ?, ?)",
            ("4ISI", "4eme annee Ingenierie des Systemes Informatiques", "2025-2026"),
        )
        teachers = [
            ("PROF-4ISI", "Professeur 4ISI"),
            ("PROF-MATH", "Nadia El Amrani"),
            ("PROF-RESEAU", "Karim Bennis"),
            ("PROF-BD", "Samira Alaoui"),
        ]
        for code, name in teachers:
            cur.execute(
                "INSERT OR IGNORE INTO teachers(teacher_code, full_name) VALUES (?, ?)",
                (code, name),
            )

        class_id = self.get_class_by_name("4ISI")["id"]
        default_courses = [
            ("PROF-4ISI", "Programmation Python", "Lundi", "09:00", "11:00"),
            ("PROF-MATH", "Mathematiques appliquees", "Mardi", "10:00", "12:00"),
            ("PROF-RESEAU", "Reseaux informatiques", "Mercredi", "14:00", "16:00"),
            ("PROF-BD", "Base de donnees", "Jeudi", "08:30", "10:30"),
        ]
        for teacher_code, course, weekday, starts_at, ends_at in default_courses:
            teacher = self.get_teacher_by_code(teacher_code)
            cur.execute(
                """
                INSERT OR IGNORE INTO courses(name, class_id, teacher_id, weekday, starts_at, ends_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (course, class_id, teacher["id"], weekday, starts_at, ends_at),
            )
        self.conn.commit()

    def get_teacher_by_code(self, teacher_code):
        return self.conn.execute(
            "SELECT * FROM teachers WHERE teacher_code = ?", (teacher_code.strip(),)
        ).fetchone()

    def get_class_by_name(self, name):
        return self.conn.execute("SELECT * FROM classes WHERE name = ?", (name,)).fetchone()

    def get_teacher_courses(self, teacher_id):
        return self.conn.execute(
            """
            SELECT courses.*, classes.name AS class_name
            FROM courses
            JOIN classes ON classes.id = courses.class_id
            WHERE courses.teacher_id = ?
            ORDER BY courses.weekday, courses.starts_at
            """,
            (teacher_id,),
        ).fetchall()

    def get_student_or_create(self, full_name, class_id):
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT OR IGNORE INTO students(full_name, class_id, created_at) VALUES (?, ?, ?)",
            (full_name.strip(), class_id, now),
        )
        self.conn.commit()
        return self.conn.execute(
            "SELECT * FROM students WHERE full_name = ? AND class_id = ?",
            (full_name.strip(), class_id),
        ).fetchone()

    def add_face_sample(self, student_id, image_path, encoding):
        self.conn.execute(
            """
            INSERT INTO face_samples(student_id, image_path, encoding, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                student_id,
                str(image_path.relative_to(APP_DIR)),
                pickle.dumps(np.asarray(encoding)),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self.conn.commit()

    def load_known_faces(self, class_id):
        rows = self.conn.execute(
            """
            SELECT students.id AS student_id, students.full_name, face_samples.encoding
            FROM face_samples
            JOIN students ON students.id = face_samples.student_id
            WHERE students.class_id = ?
            """,
            (class_id,),
        ).fetchall()
        encodings = []
        metadata = []
        for row in rows:
            encodings.append(pickle.loads(row["encoding"]))
            metadata.append({"student_id": row["student_id"], "name": row["full_name"]})
        return encodings, metadata

    def create_session(self, course):
        started_at = datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            """
            INSERT INTO attendance_sessions(course_id, teacher_id, class_id, started_at)
            VALUES (?, ?, ?, ?)
            """,
            (course["id"], course["teacher_id"], course["class_id"], started_at),
        )
        self.conn.commit()
        return cur.lastrowid

    def close_session(self, session_id):
        self.conn.execute(
            "UPDATE attendance_sessions SET ended_at = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), session_id),
        )
        self.conn.commit()

    def mark_present(self, session_id, student_id, confidence):
        self.conn.execute(
            """
            INSERT OR IGNORE INTO attendance_records(session_id, student_id, status, recognized_at, confidence)
            VALUES (?, ?, 'present', ?, ?)
            """,
            (
                session_id,
                student_id,
                datetime.now().isoformat(timespec="seconds"),
                float(confidence) if confidence is not None else None,
            ),
        )
        self.conn.commit()

    def list_students(self, class_id):
        return self.conn.execute(
            """
            SELECT students.*,
                   COUNT(face_samples.id) AS samples_count
            FROM students
            LEFT JOIN face_samples ON face_samples.student_id = students.id
            WHERE students.class_id = ?
            GROUP BY students.id
            ORDER BY students.full_name
            """,
            (class_id,),
        ).fetchall()

    def list_sessions(self, class_id):
        return self.conn.execute(
            """
            SELECT attendance_sessions.*,
                   courses.name AS course_name,
                   teachers.full_name AS teacher_name,
                   COUNT(attendance_records.id) AS present_count
            FROM attendance_sessions
            JOIN courses ON courses.id = attendance_sessions.course_id
            JOIN teachers ON teachers.id = attendance_sessions.teacher_id
            LEFT JOIN attendance_records ON attendance_records.session_id = attendance_sessions.id
            WHERE attendance_sessions.class_id = ?
            GROUP BY attendance_sessions.id
            ORDER BY attendance_sessions.started_at DESC
            """,
            (class_id,),
        ).fetchall()

    def attendance_report_rows(self, session_id):
        return self.conn.execute(
            """
            SELECT students.full_name,
                   COALESCE(attendance_records.status, 'absent') AS status,
                   attendance_records.recognized_at,
                   attendance_records.confidence
            FROM attendance_sessions
            JOIN students ON students.class_id = attendance_sessions.class_id
            LEFT JOIN attendance_records
              ON attendance_records.session_id = attendance_sessions.id
             AND attendance_records.student_id = students.id
            WHERE attendance_sessions.id = ?
            ORDER BY students.full_name
            """,
            (session_id,),
        ).fetchall()


class FaceAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.db = Database()
        self.teacher = None
        self.course = None
        self.root.title("Presence par reconnaissance faciale - Ecole privee")
        self.root.geometry("980x640")
        self.show_login()

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_login(self):
        self.clear()
        box = Frame(self.root, padx=40, pady=40)
        box.pack(fill=BOTH, expand=True)

        Label(box, text="Connexion professeur", font=("Segoe UI", 24, "bold")).pack(anchor="w")
        Label(box, text="Exemple: PROF-4ISI, PROF-MATH, PROF-RESEAU, PROF-BD", font=("Segoe UI", 11)).pack(anchor="w", pady=(6, 24))

        self.teacher_code = StringVar(value="PROF-4ISI")
        Entry(box, textvariable=self.teacher_code, font=("Segoe UI", 14), width=28).pack(anchor="w", pady=(0, 14))
        Button(box, text="Se connecter", command=self.login, font=("Segoe UI", 12), width=18).pack(anchor="w")

    def login(self):
        teacher = self.db.get_teacher_by_code(self.teacher_code.get())
        if not teacher:
            messagebox.showerror("Connexion refusee", "ID professeur introuvable.")
            return
        courses = self.db.get_teacher_courses(teacher["id"])
        if not courses:
            messagebox.showerror("Aucun cours", "Ce professeur n'a pas encore de cours.")
            return
        self.teacher = teacher
        self.course = courses[0]
        self.show_dashboard()

    def show_dashboard(self):
        self.clear()
        header = Frame(self.root, padx=24, pady=18, bg="#f5f7fb")
        header.pack(fill="x")
        Label(header, text=f"Professeur: {self.teacher['full_name']}", bg="#f5f7fb", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        Label(
            header,
            text=f"Cours: {self.course['name']} | Classe: {self.course['class_name']} | Horaire: {self.course['weekday']} {self.course['starts_at']}-{self.course['ends_at']}",
            bg="#f5f7fb",
            font=("Segoe UI", 12),
        ).pack(anchor="w", pady=(4, 0))

        actions = Frame(self.root, padx=24, pady=16)
        actions.pack(fill="x")
        Button(actions, text="Ajouter un etudiant + visages", command=self.capture_student, width=26).pack(side=LEFT, padx=(0, 8))
        Button(actions, text="Demarrer l'appel iVCam", command=self.run_attendance, width=22).pack(side=LEFT, padx=8)
        Button(actions, text="Exporter CSV", command=lambda: self.export_selected("csv"), width=14).pack(side=LEFT, padx=8)
        Button(actions, text="Exporter Excel", command=lambda: self.export_selected("xlsx"), width=14).pack(side=LEFT, padx=8)
        Button(actions, text="Exporter PDF", command=lambda: self.export_selected("pdf"), width=14).pack(side=LEFT, padx=8)
        Button(actions, text="Actualiser", command=self.refresh_lists, width=12).pack(side=RIGHT)

        content = Frame(self.root, padx=24, pady=10)
        content.pack(fill=BOTH, expand=True)

        left = Frame(content)
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 12))
        Label(left, text="Etudiants inscrits", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.students_list = Listbox(left, font=("Segoe UI", 11), height=18)
        self.students_list.pack(fill=BOTH, expand=True, pady=(8, 0))

        right = Frame(content)
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))
        Label(right, text="Seances d'appel", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.sessions_list = Listbox(right, font=("Segoe UI", 11), height=18)
        self.sessions_list.pack(fill=BOTH, expand=True, pady=(8, 0))

        self.refresh_lists()

    def refresh_lists(self):
        self.students_list.delete(0, END)
        for student in self.db.list_students(self.course["class_id"]):
            self.students_list.insert(END, f"{student['full_name']} - {student['samples_count']} image(s)")

        self.sessions = self.db.list_sessions(self.course["class_id"])
        self.sessions_list.delete(0, END)
        for session in self.sessions:
            self.sessions_list.insert(
                END,
                f"#{session['id']} | {session['started_at']} | {session['course_name']} | Presents: {session['present_count']}",
            )

    def ask_camera_index(self):
        value = simpledialog.askinteger("Camera", "Index camera iVCam (souvent 0, 1 ou 2):", initialvalue=1, minvalue=0, maxvalue=10)
        return 1 if value is None else value

    def capture_student(self):
        student_name = simpledialog.askstring("Nouvel etudiant", "Nom complet de l'etudiant:")
        if not student_name:
            return
        target_count = simpledialog.askinteger("Images", "Nombre d'images visage a capturer:", initialvalue=8, minvalue=3, maxvalue=30)
        if not target_count:
            return
        camera_index = self.ask_camera_index()
        student = self.db.get_student_or_create(student_name, self.course["class_id"])
        student_dir = FACES_DIR / self.course["class_name"] / self.safe_name(student["full_name"])
        student_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Impossible d'ouvrir la camera. Lance iVCam puis reessaie.")
            return

        captured = 0
        stable_frames = 0
        last_capture_at = 0.0
        pending_manual_capture = False
        messagebox.showinfo(
            "Capture",
            "La camera va s'ouvrir. Garde un seul visage visible: la capture se fera automatiquement. "
            "Tu peux aussi appuyer sur ESPACE, C ou ENTREE. ESC termine.",
        )
        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 80), 2)

            manual_capture = pending_manual_capture
            pending_manual_capture = False
            now = time.time()
            status = "Place un seul visage devant la camera"

            if len(faces) == 1:
                stable_frames += 1
                status = "Visage detecte - capture auto en cours"
                can_auto_capture = stable_frames >= 12 and now - last_capture_at >= 1.2
                if manual_capture or can_auto_capture:
                    face_img = crop_face(frame, faces[0])
                    descriptor = face_descriptor(face_img)
                    image_path = student_dir / f"face_{int(time.time())}_{captured + 1}.jpg"
                    saved = cv2.imwrite(str(image_path), face_img)
                    if saved:
                        self.db.add_face_sample(student["id"], image_path, descriptor)
                        captured += 1
                        last_capture_at = now
                        stable_frames = 0
                        status = "Photo enregistree"
                    else:
                        status = "Erreur: photo non enregistree"
            else:
                stable_frames = 0
                if manual_capture:
                    status = "Capture ignoree: il faut exactement un visage"
                elif len(faces) > 1:
                    status = "Plusieurs visages detectes: garde seulement l'etudiant"

            cv2.putText(frame, f"{student['full_name']} | {captured}/{target_count}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 80), 2)
            cv2.putText(frame, status, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 80), 2)
            cv2.putText(frame, "ESC: finir | ESPACE/C/ENTREE: capturer", (20, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Inscription visage", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            pending_manual_capture = key in (13, 32, ord("c"), ord("C"))

            if captured >= target_count:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.refresh_lists()
        messagebox.showinfo("Inscription terminee", f"{captured} image(s) ajoutee(s) pour {student['full_name']}.")

    def run_attendance(self):
        known_encodings, metadata = self.db.load_known_faces(self.course["class_id"])
        if not known_encodings:
            messagebox.showwarning("Aucun visage", "Ajoute d'abord les visages des etudiants de cette classe.")
            return
        camera_index = self.ask_camera_index()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Impossible d'ouvrir la camera. Lance iVCam puis reessaie.")
            return

        session_id = self.db.create_session(self.course)
        marked = set()
        stable = {}
        threshold = 0.36
        messagebox.showinfo("Appel", "Appel demarre. Appuie sur Q ou ESC pour terminer.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces(frame)

            for face_box in faces:
                face_img = crop_face(frame, face_box)
                descriptor = face_descriptor(face_img)
                distances = [descriptor_distance(known, descriptor) for known in known_encodings]
                name = "Inconnu"
                student_id = None
                confidence = None
                color = (0, 80, 255)
                if len(distances):
                    best_index = int(np.argmin(distances))
                    distance = float(distances[best_index])
                    if distance <= threshold:
                        student_id = metadata[best_index]["student_id"]
                        name = metadata[best_index]["name"]
                        confidence = max(0.0, 1.0 - distance)
                        color = (0, 180, 80)

                if student_id is not None:
                    stable[student_id] = stable.get(student_id, 0) + 1
                    if stable[student_id] >= 3 and student_id not in marked:
                        self.db.mark_present(session_id, student_id, confidence)
                        marked.add(student_id)
                left, top, width, height = face_box
                right = left + width
                bottom = top + height
                label = f"{name}"
                if student_id in marked:
                    label += " - present"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(frame, f"Classe {self.course['class_name']} | Presents: {len(marked)} | Q pour finir", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.imshow("Appel par reconnaissance faciale", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.db.close_session(session_id)
        self.refresh_lists()
        messagebox.showinfo("Appel termine", f"Seance #{session_id} enregistree avec {len(marked)} present(s).")

    def export_selected(self, export_type):
        selection = self.sessions_list.curselection()
        if not selection:
            messagebox.showwarning("Export", "Selectionne une seance dans la liste.")
            return
        session = self.sessions[selection[0]]
        rows = self.db.attendance_report_rows(session["id"])
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EXPORTS_DIR / f"presence_{self.course['class_name']}_session_{session['id']}_{stamp}.{export_type}"

        if export_type == "csv":
            self.export_csv(path, rows, session)
        elif export_type == "xlsx":
            if Workbook is None:
                messagebox.showerror("Excel", "Installe openpyxl: pip install openpyxl")
                return
            self.export_xlsx(path, rows, session)
        elif export_type == "pdf":
            if canvas is None:
                messagebox.showerror("PDF", "Installe reportlab: pip install reportlab")
                return
            self.export_pdf(path, rows, session)
        messagebox.showinfo("Export termine", f"Fichier cree:\n{path}")

    def export_csv(self, path, rows, session):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Classe", self.course["class_name"], "Cours", session["course_name"], "Professeur", session["teacher_name"]])
            writer.writerow(["Etudiant", "Statut", "Heure reconnaissance", "Confiance"])
            for row in rows:
                writer.writerow([row["full_name"], row["status"], row["recognized_at"] or "", row["confidence"] or ""])

    def export_xlsx(self, path, rows, session):
        wb = Workbook()
        ws = wb.active
        ws.title = "Presence"
        ws.append(["Classe", self.course["class_name"], "Cours", session["course_name"], "Professeur", session["teacher_name"]])
        ws.append(["Etudiant", "Statut", "Heure reconnaissance", "Confiance"])
        for row in rows:
            ws.append([row["full_name"], row["status"], row["recognized_at"] or "", row["confidence"] or ""])
        wb.save(path)

    def export_pdf(self, path, rows, session):
        pdf = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 50
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(40, y, "Feuille de presence")
        y -= 28
        pdf.setFont("Helvetica", 10)
        pdf.drawString(40, y, f"Classe: {self.course['class_name']} | Cours: {session['course_name']} | Professeur: {session['teacher_name']}")
        y -= 28
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(40, y, "Etudiant")
        pdf.drawString(260, y, "Statut")
        pdf.drawString(360, y, "Heure")
        y -= 16
        pdf.setFont("Helvetica", 10)
        for row in rows:
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)
            pdf.drawString(40, y, row["full_name"][:35])
            pdf.drawString(260, y, row["status"])
            pdf.drawString(360, y, (row["recognized_at"] or "")[:19])
            y -= 16
        pdf.save()

    @staticmethod
    def safe_name(value):
        return "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in value).strip()


def main():
    for folder in (DATA_DIR, FACES_DIR, EXPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    root = Tk()
    app = FaceAttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
