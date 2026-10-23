#!/usr/bin/env python3
"""
CV Exercise Counter / عداد التمارين بالرؤية الحاسوبية
A bilingual (English/Arabic) desktop application that uses MediaPipe Pose
to count exercise repetitions automatically from a front-facing webcam.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import sys
import os
import urllib.request

# MediaPipe (new Tasks API)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    """Calculate the angle (in degrees) at point b formed by segments b-a and b-c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_landmark_point(landmarks, idx, visibility_threshold=0.5):
    """Return [x, y] if landmark is visible enough, else None.
    Works with both legacy NormalizedLandmark and Tasks API NormalizedLandmark."""
    lm = landmarks[idx]
    vis = getattr(lm, 'visibility', None)
    if vis is not None and hasattr(vis, 'value'):
        vis = vis.value  # some protobuf wrappers
    if vis is not None and isinstance(vis, (int, float)) and vis < visibility_threshold:
        return None
    # Tasks API landmarks don't always have visibility; treat as visible if present
    return [float(lm.x), float(lm.y)]


def distance(p1, p2):
    if p1 is None or p2 is None:
        return 0
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def vertical_distance(p1, p2):
    if p1 is None or p2 is None:
        return 0
    return abs(p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Exercise counter classes
# ---------------------------------------------------------------------------

class ExerciseCounter:
    def __init__(self, side="right"):
        self.side = side
        self.count = 0
        self.stage = None
        self.feedback = "Ready / جاهز"
        self.ready = False

    def reset(self):
        self.count = 0
        self.stage = None
        self.feedback = "Ready / جاهز"
        self.ready = False

    def process(self, landmarks):
        raise NotImplementedError

    def _sides(self, left_indices, right_indices):
        if self.side == "left":
            return [left_indices]
        elif self.side == "right":
            return [right_indices]
        else:
            return [left_indices, right_indices]


class ElbowFlexionCounter(ExerciseCounter):
    UP_ANGLE = 45
    DOWN_ANGLE = 160

    def process(self, landmarks):
        sides = self._sides((11, 13, 15), (12, 14, 16))
        angles = []
        for s, e, w in sides:
            sh = get_landmark_point(landmarks, s)
            el = get_landmark_point(landmarks, e)
            wr = get_landmark_point(landmarks, w)
            if sh and el and wr:
                angles.append(calculate_angle(sh, el, wr))
        if not angles:
            self.feedback = "Please adjust your position / يرجى تعديل وضعيتك"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.side == "both" and len(angles) < 2:
            self.feedback = "Please show both arms / يرجى إظهار كلا الذراعين"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = min(angles) if self.side == "both" else angles[0]
        if angle > self.DOWN_ANGLE:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Curl up / ارفع"
        elif angle < self.UP_ANGLE and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower slowly / جيد! اخفض ببطء"
        elif angle < 90 and self.ready:
            self.feedback = "Keep going / استمر"
        elif angle > self.UP_ANGLE and self.stage == "up / أعلى":
            self.feedback = "Lower slowly / اخفض ببطء"
        return {"stage": self.stage, "feedback": self.feedback}


class BicepCurlCounter(ElbowFlexionCounter):
    pass


class HammerCurlCounter(ElbowFlexionCounter):
    pass


class ShoulderPressCounter(ExerciseCounter):
    def process(self, landmarks):
        sides = self._sides((11, 13, 15), (12, 14, 16))
        angles, wrist_above = [], []
        for s, e, w in sides:
            sh = get_landmark_point(landmarks, s)
            el = get_landmark_point(landmarks, e)
            wr = get_landmark_point(landmarks, w)
            if sh and el and wr:
                angles.append(calculate_angle(sh, el, wr))
                wrist_above.append(wr[1] < sh[1])
        if not angles:
            self.feedback = "Please adjust your position / يرجى تعديل وضعيتك"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.side == "both" and len(angles) < 2:
            self.feedback = "Please show both arms / يرجى إظهار كلا الذراعين"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = min(angles) if self.side == "both" else angles[0]
        all_up = all(wrist_above)
        if angle < 110 and not all_up:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Press up / اضغط للأعلى"
        elif angle > 150 and all_up and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower slowly / جيد! اخفض ببطء"
        elif angle > 130 and self.stage == "up / أعلى":
            self.feedback = "Lower / اخفض"
        return {"stage": self.stage, "feedback": self.feedback}


class FrontRaiseCounter(ExerciseCounter):
    def process(self, landmarks):
        sides = self._sides((11, 23, 15), (12, 24, 16))
        wrists_up, wrists_down = [], []
        for s, h, w in sides:
            sh = get_landmark_point(landmarks, s)
            hip = get_landmark_point(landmarks, h)
            wr = get_landmark_point(landmarks, w)
            if sh and wr and hip:
                ref = vertical_distance(sh, hip)
                rel = (sh[1] - wr[1]) / (ref + 1e-6)
                wrists_up.append(rel > -0.05)
                wrists_down.append(rel < -0.4)
        if not wrists_up:
            self.feedback = "Please adjust your position / يرجى تعديل وضعيتك"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.side == "both" and len(wrists_up) < 2:
            self.feedback = "Please show both arms / يرجى إظهار كلا الذراعين"
            return {"stage": self.stage, "feedback": self.feedback}
        all_up, all_down = all(wrists_up), all(wrists_down)
        if all_down:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Raise forward / ارفع للأمام"
        elif all_up and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower slowly / جيد! اخفض ببطء"
        elif self.stage == "up / أعلى" and not all_up:
            self.feedback = "Lower / اخفض"
        return {"stage": self.stage, "feedback": self.feedback}


class LateralRaiseCounter(ExerciseCounter):
    def process(self, landmarks):
        sides = self._sides((12, 11, 23, 15), (11, 12, 24, 16))
        wrists_up, wrists_down, wrists_out = [], [], []
        for opp, s, h, w in sides:
            sh = get_landmark_point(landmarks, s)
            opp_sh = get_landmark_point(landmarks, opp)
            hip = get_landmark_point(landmarks, h)
            wr = get_landmark_point(landmarks, w)
            if sh and wr and hip:
                ref = vertical_distance(sh, hip)
                rel = (sh[1] - wr[1]) / (ref + 1e-6)
                wrists_up.append(rel > -0.05)
                wrists_down.append(rel < -0.4)
                if opp_sh:
                    sw = abs(sh[0] - opp_sh[0])
                    wrists_out.append(abs(wr[0] - sh[0]) / (sw + 1e-6) > 0.9)
                else:
                    wrists_out.append(True)
        if not wrists_up:
            self.feedback = "Please adjust your position / يرجى تعديل وضعيتك"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.side == "both" and len(wrists_up) < 2:
            self.feedback = "Please show both arms / يرجى إظهار كلا الذراعين"
            return {"stage": self.stage, "feedback": self.feedback}
        all_up = all(wrists_up) and all(wrists_out)
        all_down = all(wrists_down)
        if all_down:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Raise to sides / ارفع للجانب"
        elif all_up and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower slowly / جيد! اخفض ببطء"
        elif self.stage == "up / أعلى" and not all_up:
            self.feedback = "Lower / اخفض"
        elif self.ready:
            self.feedback = "Raise arms to sides / ارفع الذراعين للجانب"
        return {"stage": self.stage, "feedback": self.feedback}


class SquatCounter(ExerciseCounter):
    def process(self, landmarks):
        sides = self._sides((23, 25, 27), (24, 26, 28))
        angles = []
        for h, k, a in sides:
            hip = get_landmark_point(landmarks, h)
            knee = get_landmark_point(landmarks, k)
            ankle = get_landmark_point(landmarks, a)
            if hip and knee and ankle:
                angles.append(calculate_angle(hip, knee, ankle))
        if not angles:
            self.feedback = "Please show your legs / يرجى إظهار الساقين"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = sum(angles) / len(angles)
        if angle > 160:
            self.stage = "up / أعلى"
            self.ready = True
            self.feedback = "Squat down / انزل للأسفل"
        elif angle < 135 and self.ready and self.stage == "up / أعلى":
            self.count += 1
            self.stage = "down / أسفل"
            self.feedback = "Good! Stand up / جيد! قف"
        elif angle < 150 and self.stage != "down / أسفل":
            self.feedback = "Deeper / انزل أكثر"
        elif self.stage == "down / أسفل" and angle < 150:
            self.feedback = "Stand up / قف"
        return {"stage": self.stage, "feedback": self.feedback}


class LungeCounter(ExerciseCounter):
    def process(self, landmarks):
        legs = []
        for name, (h, k, a) in [("left", (23, 25, 27)), ("right", (24, 26, 28))]:
            hip = get_landmark_point(landmarks, h)
            knee = get_landmark_point(landmarks, k)
            ankle = get_landmark_point(landmarks, a)
            if hip and knee and ankle:
                legs.append((name, calculate_angle(hip, knee, ankle)))
        if not legs:
            self.feedback = "Please show your legs / يرجى إظهار الساقين"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.side == "left":
            angles = [a for n, a in legs if n == "left"]
        elif self.side == "right":
            angles = [a for n, a in legs if n == "right"]
        else:
            angles = [min(a for _, a in legs)]
        if not angles:
            self.feedback = "Please adjust your position / يرجى تعديل وضعيتك"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = angles[0]
        if angle > 150:
            self.stage = "up / أعلى"
            self.ready = True
            self.feedback = "Lunge down / انزل"
        elif angle < 130 and self.ready and self.stage == "up / أعلى":
            self.count += 1
            self.stage = "down / أسفل"
            self.feedback = "Good! Push up / جيد! ادفع للأعلى"
        elif angle < 145 and self.stage != "down / أسفل":
            self.feedback = "Keep going / استمر"
        elif self.stage == "down / أسفل" and angle < 150:
            self.feedback = "Push up / ادفع للأعلى"
        return {"stage": self.stage, "feedback": self.feedback}


class PushupCounter(ExerciseCounter):
    def process(self, landmarks):
        angles = []
        for s, e, w in [(11, 13, 15), (12, 14, 16)]:
            sh = get_landmark_point(landmarks, s)
            el = get_landmark_point(landmarks, e)
            wr = get_landmark_point(landmarks, w)
            if sh and el and wr:
                angles.append(calculate_angle(sh, el, wr))
        if not angles:
            self.feedback = "Please show upper body / يرجى إظهار الجزء العلوي"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = sum(angles) / len(angles)
        if angle > 150:
            self.stage = "up / أعلى"
            self.ready = True
            self.feedback = "Lower down / انزل"
        elif angle < 100 and self.ready and self.stage == "up / أعلى":
            self.count += 1
            self.stage = "down / أسفل"
            self.feedback = "Good! Push up / جيد! ادفع للأعلى"
        elif angle < 140 and self.stage != "down / أسفل":
            self.feedback = "Keep going / استمر"
        elif self.stage == "down / أسفل" and angle < 150:
            self.feedback = "Push up / ادفع للأعلى"
        return {"stage": self.stage, "feedback": self.feedback}


class SitupCounter(ExerciseCounter):
    def process(self, landmarks):
        angles = []
        for s, h, k in [(11, 23, 25), (12, 24, 26)]:
            sh = get_landmark_point(landmarks, s)
            hip = get_landmark_point(landmarks, h)
            knee = get_landmark_point(landmarks, k)
            if sh and hip and knee:
                angles.append(calculate_angle(sh, hip, knee))
        if not angles:
            self.feedback = "Please show full body / يرجى إظهار الجسم كاملاً"
            return {"stage": self.stage, "feedback": self.feedback}
        angle = sum(angles) / len(angles)
        if angle > 150:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Sit up / اجلس"
        elif angle < 100 and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower down / جيد! انزل"
        elif angle < 130 and self.stage != "up / أعلى":
            self.feedback = "Keep going / استمر"
        elif self.stage == "up / أعلى" and angle < 140:
            self.feedback = "Lower down / انزل"
        return {"stage": self.stage, "feedback": self.feedback}


class JumpingJackCounter(ExerciseCounter):
    def process(self, landmarks):
        lw = get_landmark_point(landmarks, 15)
        rw = get_landmark_point(landmarks, 16)
        ls = get_landmark_point(landmarks, 11)
        rs = get_landmark_point(landmarks, 12)
        la = get_landmark_point(landmarks, 27)
        ra = get_landmark_point(landmarks, 28)
        if not all([lw, rw, ls, rs]):
            self.feedback = "Please show full body / يرجى إظهار الجسم كاملاً"
            return {"stage": self.stage, "feedback": self.feedback}
        wrists_high = (lw[1] < ls[1]) and (rw[1] < rs[1])
        legs_apart = False
        if la and ra and ls and rs:
            sw = abs(ls[0] - rs[0])
            legs_apart = abs(la[0] - ra[0]) > sw * 1.4 if sw > 0 else False
        is_out = wrists_high and legs_apart
        is_in = (not wrists_high) and (not legs_apart if la and ra else True)
        if is_out:
            if self.stage == "down / أسفل" and self.ready:
                self.count += 1
                self.stage = "up / أعلى"
                self.feedback = "Good! / جيد!"
            elif self.stage is None:
                self.stage = "up / أعلى"
                self.ready = True
                self.feedback = "Jump in / اقفز للداخل"
            else:
                self.feedback = "Jump in / اقفز للداخل"
        elif is_in:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Jump out / اقفز للخارج"
        else:
            self.feedback = "Keep jumping / استمر في القفز"
        return {"stage": self.stage, "feedback": self.feedback}


class HighKneesCounter(ExerciseCounter):
    def __init__(self, side="right"):
        super().__init__(side)
        self.left_up = False
        self.right_up = False
        self.baseline = None
        self._calib_count = 0

    def reset(self):
        super().reset()
        self.left_up = False
        self.right_up = False
        self.baseline = None
        self._calib_count = 0

    def process(self, landmarks):
        lk = get_landmark_point(landmarks, 25)
        rk = get_landmark_point(landmarks, 26)
        lh = get_landmark_point(landmarks, 23)
        rh = get_landmark_point(landmarks, 24)
        if not all([lk, rk, lh, rh]):
            self.feedback = "Please show full body / يرجى إظهار الجسم كاملاً"
            return {"stage": self.stage, "feedback": self.feedback}
        left_drop = lk[1] - lh[1]
        right_drop = rk[1] - rh[1]
        if self.baseline is None:
            avg_drop = (left_drop + right_drop) / 2
            if avg_drop > 0.05:
                self._calib_count += 1
                if self._calib_count >= 15:
                    self.baseline = avg_drop
                self.feedback = "Stand still... / قف ساكناً..."
            else:
                self.feedback = "Stand straight / قف باستقامة"
            self.stage = "down / أسفل"
            return {"stage": self.stage, "feedback": self.feedback}
        ref = self.baseline
        left_up = left_drop < ref * 0.4
        right_up = right_drop < ref * 0.4
        counted = False
        if self.side in ("left", "both") and left_up and not self.left_up:
            self.count += 1; counted = True
        if self.side in ("right", "both") and right_up and not self.right_up:
            self.count += 1; counted = True
        self.left_up, self.right_up = left_up, right_up
        self.ready = True
        if left_up or right_up:
            self.stage = "up / أعلى"
            self.feedback = "Good! / جيد!" if counted else "Keep going / استمر"
        else:
            self.stage = "down / أسفل"
            self.feedback = "Lift knees higher / ارفع الركبتين أعلى"
        return {"stage": self.stage, "feedback": self.feedback}


class CalfRaiseCounter(ExerciseCounter):
    def __init__(self, side="right"):
        super().__init__(side)
        self.baseline = None
        self.frames_calib = 0

    def reset(self):
        super().reset()
        self.baseline = None
        self.frames_calib = 0

    def process(self, landmarks):
        la = get_landmark_point(landmarks, 27)
        ra = get_landmark_point(landmarks, 28)
        lk = get_landmark_point(landmarks, 25)
        rk = get_landmark_point(landmarks, 26)
        lh = get_landmark_point(landmarks, 23)
        rh = get_landmark_point(landmarks, 24)
        if not all([la, ra, lk, rk, lh, rh]):
            self.feedback = "Please show full body / يرجى إظهار الجسم كاملاً"
            return {"stage": self.stage, "feedback": self.feedback}
        ref = (vertical_distance(lh, lk) + vertical_distance(rh, rk)) / 2
        if ref < 0.01:
            self.feedback = "Please stand straight / يرجى الوقوف باستقامة"
            return {"stage": self.stage, "feedback": self.feedback}
        ankle_drop = ((la[1] - lk[1]) + (ra[1] - rk[1])) / 2
        ratio = ankle_drop / ref
        if self.baseline is None:
            self.frames_calib += 1
            self.baseline = ratio
            if self.frames_calib > 15:
                self.ready = True
                self.feedback = "Ready - rise up / جاهز - ارفع"
            else:
                self.feedback = "Stand still... / قف ساكناً..."
            self.stage = "down / أسفل"
            return {"stage": self.stage, "feedback": self.feedback}
        if self.stage in (None, "down / أسفل"):
            self.baseline = 0.98 * self.baseline + 0.02 * ratio
        up_thr = self.baseline * 0.72
        dn_thr = self.baseline * 0.90
        if ratio > dn_thr:
            self.stage = "down / أسفل"
            self.ready = True
            self.feedback = "Rise on toes / ارفع على أصابع القدم"
        elif ratio < up_thr and self.ready and self.stage == "down / أسفل":
            self.count += 1
            self.stage = "up / أعلى"
            self.feedback = "Good! Lower slowly / جيد! اخفض ببطء"
        elif self.stage == "up / أعلى":
            self.feedback = "Lower slowly / اخفض ببطء"
        return {"stage": self.stage, "feedback": self.feedback}


# ---------------------------------------------------------------------------
# Exercise registry
# ---------------------------------------------------------------------------

EXERCISES = [
    ("Bicep Curl / بايسبس", BicepCurlCounter),
    ("Hammer Curl / هامر كيرل", HammerCurlCounter),
    ("Shoulder Press / ضغط الكتف", ShoulderPressCounter),
    ("Front Raise / رفع أمامي", FrontRaiseCounter),
    ("Lateral Raise / رفع جانبي", LateralRaiseCounter),
    ("Squat / قرفصاء", SquatCounter),
    ("Lunge / اندفاع", LungeCounter),
    ("Push-up / ضغط", PushupCounter),
    ("Sit-up / بطن", SitupCounter),
    ("Jumping Jack / قفز جانبي", JumpingJackCounter),
    ("High Knees / رفع الركبتين", HighKneesCounter),
    ("Calf Raise / رفع الساق", CalfRaiseCounter),
]

SIDES = [
    ("Left / اليسرى", "left"),
    ("Right / اليمنى", "right"),
    ("Both / الاثنتان", "both"),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = "pose_landmarker.task"


# ---------------------------------------------------------------------------
# Tkinter Application
# ---------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CV Exercise Counter by Computer Vision / عداد التمارين بالرؤية الحاسوبية")
        self.root.geometry("1150x750")
        self.root.minsize(900, 600)
        self.root.configure(bg="#f0f0f0")

        self.font_family = "TkDefaultFont"
        self._setup_fonts()

        self.pose = None
        self.available_cameras = []
        self.running = False
        self.cap = None
        self.camera_index = 0
        self.counter = None
        self.target_count = 10
        self.completed = False

        self._ensure_pose_model()
        self._build_gui()
        self._enumerate_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_fonts(self):
        try:
            from tkinter import font
            available = list(font.families())
            for f in ["DejaVu Sans", "Noto Sans Arabic", "Noto Sans",
                      "Arial", "Tahoma", "Segoe UI", "Ubuntu",
                      "Liberation Sans", "FreeSans"]:
                if f in available:
                    self.font_family = f
                    break
        except Exception:
            pass

    def _ensure_pose_model(self):
        if not os.path.exists(POSE_MODEL_PATH):
            try:
                urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
            except Exception as e:
                # Non-fatal; will surface as error when Start is pressed
                print(f"Warning: could not download pose model: {e}", file=sys.stderr)

    def _build_gui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bf = (self.font_family, 11)
        style.configure("TLabel", font=bf, background="#f0f0f0")
        style.configure("Header.TLabel", font=(self.font_family, 16, "bold"), background="#f0f0f0")
        style.configure("TButton", font=bf, padding=8)
        style.configure("Start.TButton", font=(self.font_family, 13, "bold"), padding=12)
        style.configure("TCombobox", font=bf)
        style.configure("TEntry", font=bf)

        control = tk.Frame(self.root, bg="#e8e8e8", width=350, padx=18, pady=18)
        control.pack(side=tk.LEFT, fill=tk.Y)
        control.pack_propagate(False)
        video = tk.Frame(self.root, bg="#111111")
        video.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(control, text="CV Exercise Counter\nعداد التمارين بالرؤية الحاسوبية",
                  style="Header.TLabel", justify="center", anchor="center").pack(pady=(0, 10))
        ttk.Separator(control, orient="horizontal").pack(fill="x", pady=5)

        def lbl(text):
            ttk.Label(control, text=text).pack(anchor="w", pady=(10, 2))

        lbl("Exercise / التمرين")
        self.exercise_combo = ttk.Combobox(control, values=[e[0] for e in EXERCISES],
                                           state="readonly", width=32)
        self.exercise_combo.current(0); self.exercise_combo.pack(fill="x")

        lbl("Side / الجهة")
        self.side_combo = ttk.Combobox(control, values=[s[0] for s in SIDES],
                                       state="readonly", width=32)
        self.side_combo.current(1); self.side_combo.pack(fill="x")

        lbl("Target Repetitions / عدد التكرارات المطلوب")
        self.target_var = tk.StringVar(value="10")
        self.target_entry = ttk.Entry(control, textvariable=self.target_var)
        self.target_entry.pack(fill="x")

        lbl("Camera / الكاميرا")
        self.camera_combo = ttk.Combobox(control, state="readonly", width=32)
        self.camera_combo.pack(fill="x")

        ttk.Separator(control, orient="horizontal").pack(fill="x", pady=12)

        self.start_btn = ttk.Button(control, text="Start / بدء", style="Start.TButton", command=self.start)
        self.start_btn.pack(fill="x", pady=3)
        self.stop_btn = ttk.Button(control, text="Stop / إيقاف", command=self.stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=3)
        self.reset_btn = ttk.Button(control, text="Reset / إعادة ضبط", command=self.reset)
        self.reset_btn.pack(fill="x", pady=3)

        ttk.Separator(control, orient="horizontal").pack(fill="x", pady=12)

        card = tk.Frame(control, bg="#ffffff", bd=1, relief="solid")
        card.pack(fill="x", pady=5)
        tk.Label(card, text="Repetitions / التكرارات", bg="#ffffff",
                 font=(self.font_family, 10)).pack(pady=(10, 0))
        self.rep_label = tk.Label(card, text="0 / 10", bg="#ffffff", fg="#2c3e50",
                                  font=(self.font_family, 28, "bold"))
        self.rep_label.pack(pady=(2, 8))
        tk.Label(card, text="State / الحالة", bg="#ffffff",
                 font=(self.font_family, 10)).pack()
        self.state_label = tk.Label(card, text="--", bg="#ffffff", fg="#3498db",
                                    font=(self.font_family, 15, "bold"))
        self.state_label.pack(pady=(2, 8))
        tk.Label(card, text="Feedback / الملاحظات", bg="#ffffff",
                 font=(self.font_family, 10)).pack()
        self.feedback_label = tk.Label(card, text="Ready / جاهز", bg="#ffffff", fg="#27ae60",
                                       font=(self.font_family, 11), wraplength=300, justify="center")
        self.feedback_label.pack(pady=(2, 12))

        self.canvas = tk.Canvas(video, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._show_idle_message()

    def _show_idle_message(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 640
        h = self.canvas.winfo_height() or 480
        self.canvas.create_text(w // 2, h // 2,
                                text="Press Start to begin\nاضغط بدء للبدء",
                                fill="#ffffff", font=(self.font_family, 22), justify="center")

    def _on_canvas_resize(self, event=None):
        if not self.running:
            self._show_idle_message()

    def _enumerate_cameras(self):
        self.available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.available_cameras.append(i)
        if not self.available_cameras:
            self.available_cameras = [0]
        labels = [f"Camera {i} / الكاميرا {i}" for i in self.available_cameras]
        self.camera_combo["values"] = labels
        if labels:
            self.camera_combo.current(0)

    def _cam_idx(self):
        i = self.camera_combo.current()
        return self.available_cameras[i] if 0 <= i < len(self.available_cameras) else 0

    def _exercise(self):
        i = self.exercise_combo.current()
        return EXERCISES[i] if 0 <= i < len(EXERCISES) else EXERCISES[0]

    def _side(self):
        i = self.side_combo.current()
        return SIDES[i][1] if 0 <= i < len(SIDES) else "right"

    def start(self):
        if self.running:
            return
        try:
            target = int(self.target_var.get())
            if target < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error / خطأ",
                                 "Enter a valid positive integer.\nأدخل عدداً صحيحاً موجباً.")
            return

        if not os.path.exists(POSE_MODEL_PATH):
            messagebox.showerror("Model error / خطأ في النموذج",
                                 "Pose model not found. Please ensure you are connected to the internet\n"
                                 "and restart the application to download it.\n"
                                 "لم يتم العثور على نموذج التعرف. تأكد من الاتصال بالإنترنت وأعد التشغيل.")
            return

        self.camera_index = self._cam_idx()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error / خطأ في الكاميرا",
                                 f"Could not open camera {self.camera_index}.\nتعذر فتح الكاميرا.")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.pose = mp_vision.PoseLandmarker.create_from_options(opts)
        except Exception as e:
            messagebox.showerror("Model error / خطأ في النموذج",
                                 f"Could not load pose model:\n{e}")
            self.cap.release()
            self.cap = None
            return

        _, cls = self._exercise()
        self.counter = cls(side=self._side())
        self.target_count = target
        self.completed = False

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.reset_btn.config(state="disabled")
        for w in (self.exercise_combo, self.side_combo, self.camera_combo):
            w.config(state="disabled")
        self.target_entry.config(state="disabled")
        self._update_rep_display()
        threading.Thread(target=self._video_loop, daemon=True).start()

    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap is not None:
            self.cap.release(); self.cap = None
        if self.pose is not None:
            self.pose.close(); self.pose = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.reset_btn.config(state="normal")
        for w in (self.exercise_combo, self.side_combo, self.camera_combo):
            w.config(state="readonly")
        self.target_entry.config(state="normal")
        self.feedback_label.config(text="Stopped / متوقف", fg="#7f8c8d")
        self._show_idle_message()

    def reset(self):
        if self.counter:
            self.counter.reset()
        self.completed = False
        try:
            target = int(self.target_var.get())
            if target < 1: target = 10
        except ValueError:
            target = 10
        self.target_count = target
        self._update_rep_display()
        self.state_label.config(text="--")
        self.feedback_label.config(text="Ready / جاهز", fg="#27ae60")

    def _update_rep_display(self):
        count = self.counter.count if self.counter else 0
        self.rep_label.config(text=f"{count} / {self.target_count}")
        self.root.update_idletasks()

    def _draw_pose(self, frame, landmarks_norm):
        h, w = frame.shape[:2]
        pts = []
        for lm in landmarks_norm:
            pts.append((int(lm.x * w), int(lm.y * h)))
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 0, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

    def _video_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._set_fb("Camera error / خطأ في الكاميرا", "#e74c3c"); break
            ret, frame = self.cap.read()
            if not ret:
                self._set_fb("Camera error / خطأ في الكاميرا", "#e74c3c")
                time.sleep(0.1); continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            try:
                result = self.pose.detect(mp_image)
            except Exception as e:
                self._set_fb(f"Detection error / خطأ في الكشف", "#e74c3c")
                time.sleep(0.05); continue

            landmarks_list = result.pose_landmarks if hasattr(result, 'pose_landmarks') else []

            if landmarks_list and len(landmarks_list) > 0 and len(landmarks_list[0]) >= 33:
                landmarks = landmarks_list[0]
                self._draw_pose(frame, landmarks)
                info = self.counter.process(landmarks)
                if self.counter.count >= self.target_count and not self.completed:
                    self.completed = True
                    self.root.after(0, self._on_completion)
                self.root.after(0, lambda s=info.get("stage", "--"), f=info.get("feedback", ""),
                                       c=self.counter.count: self._update_status(s, f, c))
            else:
                self._set_fb("No person detected / لم يتم اكتشاف شخص", "#e67e22")

            self._draw_overlay(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                img = img.resize((cw, ch), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            def _draw(it=imgtk):
                if self.running:
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=it)
                    self.canvas._ref = it
            self.root.after(0, _draw)
            time.sleep(0.025)

    def _draw_overlay(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        count = self.counter.count if self.counter else 0
        cv2.putText(frame, f"REPS: {count}/{self.target_count}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        stage = self.counter.stage if self.counter and self.counter.stage else "--"
        stage_en = stage.split(" / ")[0] if " / " in stage else stage
        cv2.putText(frame, f"STATE: {stage_en}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        prog = min(count / max(self.target_count, 1), 1.0)
        bx, by, bw, bh = 10, 115, 240, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * prog), by + bh), (0, 255, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)

    def _set_fb(self, text, color):
        self.root.after(0, lambda: self.feedback_label.config(text=text, fg=color))

    def _update_status(self, stage, feedback, count):
        self.state_label.config(text=stage or "--")
        self.rep_label.config(text=f"{count} / {self.target_count}")
        fl = feedback.lower()
        if "good" in fl or "جيد" in feedback or "complete" in fl or "مكتمل" in feedback:
            color = "#27ae60"
        elif ("error" in fl or "خطأ" in feedback or "adjust" in fl or
              "تعديل" in feedback or "no person" in fl):
            color = "#e74c3c"
        else:
            color = "#2980b9"
        self.feedback_label.config(text=feedback, fg=color)

    def _on_completion(self):
        self.feedback_label.config(text="Complete! / مكتمل! 🎉", fg="#27ae60")
        messagebox.showinfo(
            "Complete! / مكتمل!",
            f"Congratulations! You completed {self.target_count} repetitions!\n"
            f"تهانينا! لقد أكملت {self.target_count} تكراراً!"
        )

    def _on_close(self):
        self.running = False
        time.sleep(0.2)
        if self.cap is not None:
            self.cap.release()
        if self.pose is not None:
            self.pose.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
