from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from deepface import DeepFace
import base64
import io
from PIL import Image

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ── Config ──────────────────────────────────────────────
NORMAL_THRESHOLD  = 0.60
BEARD_THRESHOLD   = 0.38
BEARD_LBP_MIN     = 0.25
BEARD_EDGE_MIN    = 0.08
BEARD_NONSKIN_MIN = 0.30
BEARD_VOTE_MIN    = 2
TEXTURE_THRESHOLD = 10.0

SKIN_RANGES = [
    (np.array([0, 15, 100], dtype=np.uint8), np.array([20, 180, 255], dtype=np.uint8)),
    (np.array([0, 20,  60], dtype=np.uint8), np.array([20, 200, 220], dtype=np.uint8)),
    (np.array([0, 10,  30], dtype=np.uint8), np.array([20, 160, 150], dtype=np.uint8)),
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Beard Detector ───────────────────────────────────────
class BeardDetector:
    CHIN_IDS = [152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,365,397,288,361,323,454,356,389,251,284,332,297,338]
    MUSTACHE_IDS = [0,17,18,200,199,175,152,57,43,106,182,83,84,85,287,273,335,406,313,314]

    def __init__(self):
        self.mesh = None
        if not MEDIAPIPE_AVAILABLE:
            return
        mp_fm = mp.solutions.face_mesh
        self.mesh = mp_fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def _get_mask(self, landmarks, ids, h, w, y_frac_min=0.0):
        pts = []
        for idx in ids:
            lm = landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            if py >= h * y_frac_min:
                pts.append([px, py])
        if len(pts) < 3:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        return mask

    def _lbp_score(self, gray, mask):
        if SKIMAGE_AVAILABLE:
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            roi = lbp[mask > 0]
            if roi.size == 0:
                return 0.0
            return min(float(np.std(roi)) / 60.0, 1.0)
        else:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            roi = lap[mask > 0]
            if roi.size == 0:
                return 0.0
            return min(float(np.std(roi)) / 50.0, 1.0)

    def _edge_score(self, gray, mask):
        edges = cv2.Canny(gray, 40, 100)
        roi_e = cv2.bitwise_and(edges, edges, mask=mask)
        total = max(cv2.countNonZero(mask), 1)
        return cv2.countNonZero(roi_e) / total

    def _nonskin_score(self, hsv, mask):
        skin = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in SKIN_RANGES:
            skin = cv2.bitwise_or(skin, cv2.inRange(hsv, lo, hi))
        total = max(cv2.countNonZero(mask), 1)
        skin_px = cv2.countNonZero(cv2.bitwise_and(skin, skin, mask=mask))
        return max(total - skin_px, 0) / total

    def has_beard(self, frame, face_region):
        debug = {"lbp": 0.0, "edge": 0.0, "nonskin": 0.0, "votes": 0}
        if self.mesh is None:
            return False, 0.0, debug

        x  = face_region.get("x", 0)
        y  = face_region.get("y", 0)
        fw = face_region.get("w", 0)
        fh = face_region.get("h", 0)
        if fw == 0 or fh == 0:
            return False, 0.0, debug

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + fw), min(frame.shape[0], y + fh)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0, debug

        results = self.mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False, 0.0, debug

        lm = results.multi_face_landmarks[0].landmark
        h_c, w_c = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        chin_mask = self._get_mask(lm, self.CHIN_IDS, h_c, w_c, y_frac_min=0.45)
        mustache_mask = self._get_mask(lm, self.MUSTACHE_IDS, h_c, w_c, y_frac_min=0.40)

        if chin_mask is None:
            return False, 0.0, debug

        full_mask = chin_mask.copy()
        if mustache_mask is not None:
            full_mask = cv2.bitwise_or(full_mask, mustache_mask)

        lbp     = self._lbp_score(gray, full_mask)
        edge    = self._edge_score(gray, full_mask)
        nonskin = self._nonskin_score(hsv, full_mask)

        votes = sum([
            lbp     >= BEARD_LBP_MIN,
            edge    >= BEARD_EDGE_MIN,
            nonskin >= BEARD_NONSKIN_MIN,
        ])

        debug = {"lbp": round(lbp, 3), "edge": round(edge, 3),
                 "nonskin": round(nonskin, 3), "votes": votes}

        beard_score = lbp * 0.40 + edge * 0.35 + nonskin * 0.25
        return votes >= BEARD_VOTE_MIN, round(beard_score, 3), debug


beard_detector = BeardDetector()


def decode_image(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ── API Endpoint ─────────────────────────────────────────
@app.post("/check")
async def check_liveness(file: UploadFile = File(...)):
    contents = await file.read()
    frame = decode_image(contents)

    if frame is None:
        return {"error": "Invalid image"}

    try:
        res = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            anti_spoofing=True,
            enforce_detection=False
        )
    except Exception as e:
        return {"error": str(e)}

    if not res:
        return {"label": "NO_FACE", "is_real": False, "score": 0.0, "beard": False}

    face   = res[0]
    region = face.get("facial_area", {})
    score  = face.get("antispoof_score", 0.0)

    beard_found, beard_score, dbg = beard_detector.has_beard(frame, region)
    threshold = BEARD_THRESHOLD if beard_found else NORMAL_THRESHOLD

    # Texture check
    x, y   = region.get("x", 0), region.get("y", 0)
    fw, fh = region.get("w", 0), region.get("h", 0)
    crop   = frame[max(0,y):y+fh, max(0,x):x+fw]
    tex    = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size > 0 else 0.0
    texture_ok = tex >= TEXTURE_THRESHOLD

    is_real = (score >= threshold) and texture_ok

    return {
        "label":       "REAL" if is_real else "FAKE",
        "is_real":     is_real,
        "score":       round(score, 3),
        "threshold":   threshold,
        "beard":       beard_found,
        "beard_score": beard_score,
        "beard_debug": dbg,
        "texture":     round(tex, 1),
    }


@app.get("/")
def root():
    return {"status": "Liveness API running"}
