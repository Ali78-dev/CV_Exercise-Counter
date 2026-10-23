import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os

# --- Helper Functions ---
def calculate_angle(a, b, c):
    """Calculates the 2D angle (in degrees) between three points (a, b, c) at vertex b."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def get_landmark_coords(landmarks, landmark_id):
    """Returns (x, y) coordinates for a given landmark."""
    lm = landmarks[landmark_id]
    return [lm.x, lm.y]

def is_landmark_visible(landmarks, landmark_ids, min_visibility=0.5):
    """Checks landmark visibility."""
    for lid in landmark_ids:
        vis = getattr(landmarks[lid], 'presence', getattr(landmarks[lid], 'visibility', 1.0))
        if vis < min_visibility:
            return False
    return True

# Simple Skeleton Drawing Helper
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28)
]

def draw_skeleton(frame, landmarks, h, w):
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, (245, 117, 66), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (245, 66, 230), -1)

# --- Main Application ---
class ExerciseCounterApp:
    def __init__(self, window):
        self.window = window
        self.window.title(" CV Exercise Counter / عداد التمارين بالرؤية الحاسوبية")
        self.window.geometry("1100x700")
        self.window.configure(bg="#222831")

        # MediaPipe Pose Landmarker Tasks API Setup
        model_path = "pose_landmarker.task"
        if not os.path.exists(model_path):
            messagebox.showerror("Model Missing / النموذج غير موجود", 
                                 f"Please download '{model_path}' and place it in the project folder.\nيرجى تحميل ملف النموذج ووضعه في مجلد المشروع.")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Variables
        self.is_running = False
        self.cap = None
        self.rep_count = 0
        self.stage = "START"
        self.feedback = "Ready / جاهز"

        self.exercise_list = [
            "Bicep Curl / بايسبس", "Hammer Curl / هامر كيرل", "Shoulder Press / ضغط الكتف",
            "Front Raise / رفع أمامي", "Lateral Raise / رفع جانبي", "Squat / قرفصاء",
            "Lunge / اندفاع", "Push-up / ضغط", "Sit-up / بطن",
            "Jumping Jack / قفز جانبي", "High Knees / رفع الركبتين", "Calf Raise / رفع الساق"
        ]

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#222831", foreground="#EEEEEE", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#00ADB5")

        control_panel = tk.Frame(self.window, bg="#393E46", padx=15, pady=15, width=320)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False)

        video_panel = tk.Frame(self.window, bg="#222831")
        video_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Controls
        ttk.Label(control_panel, text="Control Panel / لوحة التحكم", style="Header.TLabel").pack(pady=(0, 15))

        ttk.Label(control_panel, text="Exercise / التمرين:").pack(anchor=tk.W, pady=(5, 0))
        self.combo_exercise = ttk.Combobox(control_panel, values=self.exercise_list, state="readonly")
        self.combo_exercise.current(0)
        self.combo_exercise.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_panel, text="Side / الجهة:").pack(anchor=tk.W, pady=(5, 0))
        self.combo_side = ttk.Combobox(control_panel, values=["Both / الاثنتان", "Left / اليسرى", "Right / اليمنى"], state="readonly")
        self.combo_side.current(0)
        self.combo_side.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_panel, text="Target Reps / الهدف:").pack(anchor=tk.W, pady=(5, 0))
        self.entry_target = ttk.Entry(control_panel)
        self.entry_target.insert(0, "10")
        self.entry_target.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_panel, text="Camera ID / الكاميرا:").pack(anchor=tk.W, pady=(5, 0))
        self.combo_cam = ttk.Combobox(control_panel, values=["0", "1", "2"], state="readonly")
        self.combo_cam.current(0)
        self.combo_cam.pack(fill=tk.X, pady=(0, 15))

        # Buttons
        btn_frame = tk.Frame(control_panel, bg="#393E46")
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_start = tk.Button(btn_frame, text="Start / بدء", bg="#00ADB5", fg="white", font=("Helvetica", 10, "bold"), command=self.start_video)
        self.btn_start.pack(fill=tk.X, pady=3)

        self.btn_stop = tk.Button(btn_frame, text="Stop / إيقاف", bg="#D9534F", fg="white", font=("Helvetica", 10, "bold"), command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=3)

        self.btn_reset = tk.Button(btn_frame, text="Reset / إعادة ضبط", bg="#F0AD4E", fg="white", font=("Helvetica", 10, "bold"), command=self.reset_counter)
        self.btn_reset.pack(fill=tk.X, pady=3)

        ttk.Separator(control_panel, orient='horizontal').pack(fill='x', pady=15)

        # Status
        ttk.Label(control_panel, text="Status / الحالة الحالية", style="Header.TLabel").pack(anchor=tk.W)
        self.lbl_reps = tk.Label(control_panel, text="Reps / التكرارات: 0 / 10", font=("Helvetica", 12, "bold"), bg="#393E46", fg="#00ADB5")
        self.lbl_reps.pack(anchor=tk.W, pady=5)

        self.lbl_state = tk.Label(control_panel, text="State / الحالة: START", font=("Helvetica", 11), bg="#393E46", fg="#EEEEEE")
        self.lbl_state.pack(anchor=tk.W, pady=2)

        self.lbl_feedback = tk.Label(control_panel, text="Feedback / الملاحظات:\nReady / جاهز", font=("Helvetica", 10, "bold"), bg="#393E46", fg="#28A745", justify=tk.LEFT)
        self.lbl_feedback.pack(anchor=tk.W, pady=5)

        self.lbl_video = tk.Label(video_panel, bg="#222831")
        self.lbl_video.pack(expand=True, fill=tk.BOTH)

    def start_video(self):
        if self.is_running: return
        try:
            cam_idx = int(self.combo_cam.get())
            self.cap = cv2.VideoCapture(cam_idx)
            if not self.cap.isOpened(): raise Exception("Camera open failed")
        except Exception:
            messagebox.showerror("Camera Error / خطأ في الكاميرا", "Could not connect to camera.\nتعذر الاتصال بالكاميرا.")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self.video_loop, daemon=True).start()

    def stop_video(self):
        self.is_running = False
        if self.cap and self.cap.isOpened(): self.cap.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def reset_counter(self):
        self.rep_count = 0
        self.stage = "START"
        self.feedback = "Reset Done / تم إعادة الضبط"
        self.update_dashboard()

    def update_dashboard(self):
        target = self.entry_target.get() or "10"
        self.lbl_reps.config(text=f"Reps / التكرارات: {self.rep_count} / {target}")
        self.lbl_state.config(text=f"State / الحالة: {self.stage}")
        self.lbl_feedback.config(text=f"Feedback / الملاحظات:\n{self.feedback}")

    def video_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.feedback = "Camera Error / خطأ في الكاميرا"
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Tasks API detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                draw_skeleton(frame, landmarks, h, w)
                self.analyze_exercise(landmarks)
            else:
                self.feedback = "No person detected / لم يتم اكتشاف شخص"

            self.update_dashboard()

            try: target = int(self.entry_target.get())
            except: target = 10

            if self.rep_count >= target:
                self.feedback = "Target Completed! / تم تحقيق الهدف الكلي!"
                cv2.putText(frame, "TARGET COMPLETED!", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)

        self.lbl_video.configure(image='')

    def analyze_exercise(self, landmarks):
        ex = self.combo_exercise.get().split(" / ")[0]
        side = self.combo_side.get().split(" / ")[0]

        try:
            if ex == "Bicep Curl":
                req = [11, 13, 15] if side == "Left" else ([12, 14, 16] if side == "Right" else [11, 13, 15, 12, 14, 16])
                if not is_landmark_visible(landmarks, req):
                    self.feedback = "Please adjust position / يرجى تعديل وضعيتك"
                    return

                l_angle = calculate_angle(get_landmark_coords(landmarks, 11), get_landmark_coords(landmarks, 13), get_landmark_coords(landmarks, 15))
                r_angle = calculate_angle(get_landmark_coords(landmarks, 12), get_landmark_coords(landmarks, 14), get_landmark_coords(landmarks, 16))
                angle = l_angle if side == "Left" else (r_angle if side == "Right" else (l_angle + r_angle) / 2)

                if angle > 150:
                    self.stage = "DOWN / أسفل"
                    self.feedback = "Good, now curl up / جيد، ارفع الآن"
                if angle < 35 and self.stage == "DOWN / أسفل":
                    self.stage = "UP / أعلى"
                    self.rep_count += 1
                    self.feedback = "Rep Counted! / تم التكرار!"

            elif ex == "Squat":
                req = [23, 25, 27, 24, 26, 28]
                if not is_landmark_visible(landmarks, req):
                    self.feedback = "Please adjust position / يرجى تعديل وضعيتك"
                    return

                l_knee = calculate_angle(get_landmark_coords(landmarks, 23), get_landmark_coords(landmarks, 25), get_landmark_coords(landmarks, 27))
                r_knee = calculate_angle(get_landmark_coords(landmarks, 24), get_landmark_coords(landmarks, 26), get_landmark_coords(landmarks, 28))
                knee_angle = (l_knee + r_knee) / 2

                if knee_angle > 160:
                    self.stage = "UP / أعلى"
                    self.feedback = "Squat down / انزل لأسفل"
                if knee_angle < 95 and self.stage == "UP / أعلى":
                    self.stage = "DOWN / أسفل"
                    self.rep_count += 1
                    self.feedback = "Deep Squat! / قرفصاء جيدة"
            else:
                self.feedback = f"{ex} Tracking / جاري التتبع"
        except Exception:
            self.feedback = "Error processing pose / خطأ أثناء معالجة الحركة"

if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_video(), root.destroy()))
    root.mainloop()