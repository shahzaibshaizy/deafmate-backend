"""
FastAPI Production Backend Server for Sign Language Translation App.

Provides:
1. REST API endpoints for diagnostics, single-frame prediction, landmark scoring,
   ASL sign dictionary catalog, and speech synthesis.
2. Low-latency bidirectional WebSocket endpoint (`/ws/stream`) for real-time video
   streaming from the Flutter mobile application.
3. Automatic OpenAPI/Swagger documentation at `/docs`.

Design Principles:
- Separation of Concerns: Uses modular services (Normalizer, Debouncer, Assembler, TTS).
- Thread Safety: Dedicated stateful session per WebSocket client.
- Cross-Origin Resource Sharing (CORS): Fully enabled for mobile emulators and web clients.
"""

from contextlib import asynccontextmanager
import base64
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from auth_service import (
    authenticate_user,
    create_access_token,
    create_user,
    decode_access_token,
    get_user_by_email,
    get_user_by_id,
    mark_user_verified,
)
from database import init_db
from debounce_engine import TemporalDebounceEngine
from email_service import generate_otp, send_otp_email, verify_otp
from history_service import (
    clear_user_history,
    delete_history_item,
    get_user_history,
    save_history,
)
from normalization import CoordinateNormalizer
from sentence_builder import SentenceAssembler
from tts_service import TTSService


# Paths configuration: checks local folder first, then parent directory
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent

MODEL_TASK_PATH = (
    CURRENT_DIR / "hand_landmarker.task"
    if (CURRENT_DIR / "hand_landmarker.task").exists()
    else PARENT_DIR / "hand_landmarker.task"
)
TRAINED_MODEL_PATH = (
    CURRENT_DIR / "models" / "asl_model.pkl"
    if (CURRENT_DIR / "models" / "asl_model.pkl").exists()
    else PARENT_DIR / "backend" / "models" / "asl_model.pkl"
)
LABEL_CLASSES_PATH = (
    CURRENT_DIR / "models" / "label_classes.npy"
    if (CURRENT_DIR / "models" / "label_classes.npy").exists()
    else PARENT_DIR / "backend" / "models" / "label_classes.npy"
)


# Global model singletons
classifier = None
classes = None
tts_service = None
landmarker_bytes = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes heavy machine learning models on server startup."""
    global classifier, classes, tts_service, landmarker_bytes

    print("\n=======================================================")
    print("  Initializing Sign Language AI Intelligence Backend...")
    print("=======================================================")

    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at: {TRAINED_MODEL_PATH}")
    if not LABEL_CLASSES_PATH.exists():
        raise FileNotFoundError(f"Class labels not found at: {LABEL_CLASSES_PATH}")
    if not MODEL_TASK_PATH.exists():
        raise FileNotFoundError(f"MediaPipe task model not found at: {MODEL_TASK_PATH}")

    # Load ML Classifier & Classes
    classifier = joblib.load(TRAINED_MODEL_PATH)
    classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
    print(f"Loaded classifier: {type(classifier).__name__} with {len(classes)} classes.")

    # Load MediaPipe model bytes into raw memory buffer
    with open(MODEL_TASK_PATH, "rb") as f:
        landmarker_bytes = f.read()

    # Initialize non-blocking TTS service
    tts_service = TTSService(speech_rate=160, volume=1.0)
    print("TTS Speech Service ready.")

    # Initialize SQLite Database & Tables
    init_db()

    print("Server ready at http://127.0.0.1:8000 (Swagger docs: http://127.0.0.1:8000/docs)\n")

    yield

    # Clean shutdown
    if tts_service:
        tts_service.stop()
    print("Backend shutdown completed cleanly.")


app = FastAPI(
    title="Sign Language Translator API",
    description="Real-Time Computer Vision and AI Backend for ASL Sign-to-Speech Translation",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Flutter mobile apps and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper: Create per-request/session MediaPipe detector
def create_image_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=landmarker_bytes),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    return HandLandmarker.create_from_options(options)


# -------------------------------------------------------------
# Request & Response Pydantic Schemas
# -------------------------------------------------------------
class SystemStatusResponse(BaseModel):
    status: str
    version: str
    model_type: str
    model_accuracy: str
    total_classes: int
    classes: List[str]


class Base64FrameRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image frame")


class LandmarkItem(BaseModel):
    x: float
    y: float
    z: float


class PredictionResponse(BaseModel):
    hand_detected: bool
    sign: Optional[str] = None
    confidence: float = 0.0
    landmarks: Optional[List[Dict[str, float]]] = None


class SpeakRequest(BaseModel):
    text: str


class DictionaryItem(BaseModel):
    sign: str
    category: str
    description: str


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=6, max_length=100)


class VerifyOtpRequest(BaseModel):
    email: str
    code: str = Field(..., min_length=6, max_length=6)


class ResendOtpRequest(BaseModel):
    email: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class HistoryCreateRequest(BaseModel):
    sentence: str
    mode: str = "Sign to Voice"


class HistoryItemResponse(BaseModel):
    id: int
    user_id: int
    sentence: str
    mode: str
    created_at: str


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.get("/", response_model=SystemStatusResponse, tags=["Diagnostics"])
def get_system_status():
    """Returns server diagnostics, active classes, and model health."""
    return SystemStatusResponse(
        status="ONLINE",
        version="1.0.0",
        model_type=type(classifier).__name__,
        model_accuracy="99.52%",
        total_classes=len(classes),
        classes=list(classes)
    )


@app.get("/api/dictionary", response_model=List[DictionaryItem], tags=["Dictionary"])
def get_sign_dictionary():
    """Returns catalog of all supported ASL letters and control gestures."""
    dictionary = []
    for c in sorted(classes):
        if c.lower() == "space":
            cat = "Control Gesture"
            desc = "Open flat hand moving horizontally. Inserts a space between words."
        elif c.lower() in ("del", "delete"):
            cat = "Control Gesture"
            desc = "Index finger pointing down or flicking. Deletes the last typed letter."
        else:
            cat = "Alphabet"
            desc = f"American Sign Language fingerspelling for letter '{c}'."
        dictionary.append(DictionaryItem(sign=c, category=cat, description=desc))
    return dictionary


@app.post("/api/predict/frame", response_model=PredictionResponse, tags=["Inference"])
async def predict_image_frame(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, extracts landmarks, and returns predicted sign.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detector = create_image_landmarker()
    try:
        results = detector.detect(mp_image)
        if not results.hand_landmarks:
            return PredictionResponse(hand_detected=False)

        hand_lms = results.hand_landmarks[0]
        feature_vector = CoordinateNormalizer.normalize_landmarks(hand_lms).reshape(1, -1)

        probs = classifier.predict_proba(feature_vector)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        predicted_sign = str(classes[best_idx])

        landmarks_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms]

        return PredictionResponse(
            hand_detected=True,
            sign=predicted_sign,
            confidence=round(confidence, 4),
            landmarks=landmarks_list
        )
    finally:
        detector.close()


@app.post("/api/tts/speak", tags=["Accessibility"])
def speak_text(req: SpeakRequest):
    """Asynchronously speaks text aloud via offline Text-to-Speech."""
    if tts_service:
        tts_service.speak(req.text)
        return {"status": "SUCCESS", "message": f"Speaking: '{req.text}'"}
    raise HTTPException(status_code=500, detail="TTS service is unavailable.")


# -------------------------------------------------------------
# Security & Current User Dependency
# -------------------------------------------------------------
async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Extracts and validates the Bearer JWT token from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token."
        )
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token."
        )
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with this token no longer exists."
        )
    return user


# -------------------------------------------------------------
# Authentication Endpoints
# -------------------------------------------------------------
@app.post("/api/auth/register", tags=["Authentication"])
def register_user_endpoint(req: RegisterRequest):
    """Registers a new user and sends a 6-digit OTP verification email."""
    clean_email = req.email.strip().lower()
    existing = get_user_by_email(clean_email)
    if existing:
        if existing["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists. Please sign in."
            )
        # Resend OTP to existing unverified user
        code = generate_otp(clean_email)
        send_otp_email(clean_email, code, existing["name"])
        return {
            "status": "OTP_SENT",
            "message": "Verification code sent to your email.",
            "email": clean_email
        }

    user = create_user(req.name, clean_email, req.password)
    code = generate_otp(clean_email)
    send_otp_email(clean_email, code, req.name)

    return {
        "status": "OTP_SENT",
        "message": "Account created! Verification code sent to your email.",
        "email": clean_email
    }


@app.post("/api/auth/verify-otp", response_model=AuthResponse, tags=["Authentication"])
def verify_otp_endpoint(req: VerifyOtpRequest):
    """Verifies the 6-digit OTP code and issues a 30-day JWT session token."""
    clean_email = req.email.strip().lower()
    is_valid, msg = verify_otp(clean_email, req.code)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    mark_user_verified(clean_email)
    user = get_user_by_email(clean_email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    token = create_access_token(user["id"], user["email"], user["name"])
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user={
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "is_verified": True
        }
    )


@app.post("/api/auth/resend-otp", tags=["Authentication"])
def resend_otp_endpoint(req: ResendOtpRequest):
    """Resends a fresh 6-digit verification code to the user."""
    clean_email = req.email.strip().lower()
    user = get_user_by_email(clean_email)
    name = user["name"] if user else "User"
    code = generate_otp(clean_email)
    send_otp_email(clean_email, code, name)
    return {"status": "SUCCESS", "message": "A new verification code has been dispatched."}


@app.post("/api/auth/login", response_model=AuthResponse, tags=["Authentication"])
def login_endpoint(req: LoginRequest):
    """Authenticates user credentials and returns a JWT token."""
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
    if not user["is_verified"]:
        code = generate_otp(user["email"])
        send_otp_email(user["email"], code, user["name"])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account not verified. A new verification code has been sent to your email."
        )

    token = create_access_token(user["id"], user["email"], user["name"])
    return AuthResponse(
        access_token=token,
        token_type="bearer",
        user=user
    )


@app.get("/api/auth/me", tags=["Authentication"])
def get_my_profile(current_user: Dict = Depends(get_current_user)):
    """Returns profile information for the authenticated user."""
    return current_user


# -------------------------------------------------------------
# Translation History Endpoints
# -------------------------------------------------------------
@app.get("/api/history", response_model=List[HistoryItemResponse], tags=["History"])
def list_history(current_user: Dict = Depends(get_current_user)):
    """Fetches translation history records for the authenticated user."""
    history_list = get_user_history(current_user["id"])
    return history_list


@app.post("/api/history", response_model=HistoryItemResponse, tags=["History"])
def create_history_record(
    req: HistoryCreateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Saves a new translated sentence to the authenticated user's history."""
    item = save_history(current_user["id"], req.sentence, req.mode)
    return item


@app.delete("/api/history/{history_id}", tags=["History"])
def delete_history_record(
    history_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Deletes a specific history record owned by the authenticated user."""
    deleted = delete_history_item(current_user["id"], history_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="History record not found."
        )
    return {"status": "SUCCESS", "message": "History item deleted."}


@app.delete("/api/history", tags=["History"])
def clear_all_history(current_user: Dict = Depends(get_current_user)):
    """Clears all translation history records for the authenticated user."""
    count = clear_user_history(current_user["id"])
    return {"status": "SUCCESS", "message": f"Cleared {count} history records."}


# -------------------------------------------------------------
# Real-Time WebSocket Streaming Endpoint
# -------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Bidirectional streaming WebSocket endpoint.
    Accepts video frames from Flutter mobile camera at 15-30 FPS.
    Returns real-time predictions, stabilized text, subtitles, and landmarks.
    """
    await websocket.accept()

    # Per-connection session state
    detector = create_image_landmarker()
    debounce_engine = TemporalDebounceEngine(
        window_size=10,
        confidence_threshold=0.80,
        consistency_ratio=0.80,
        absence_timeout_frames=45
    )
    sentence_assembler = SentenceAssembler()

    try:
        while True:
            # Receive frame data (text JSON or binary bytes)
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                img_array = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue

                # Handle client command actions
                action = payload.get("action")
                if action == "clear":
                    sentence_assembler.clear()
                    await websocket.send_json({
                        "action": "clear",
                        "display_text": ""
                    })
                    continue
                elif action == "speak":
                    text_to_speak = sentence_assembler.get_display_text()
                    if text_to_speak and tts_service:
                        tts_service.speak(text_to_speak)
                    await websocket.send_json({
                        "action": "speak",
                        "spoken_text": text_to_speak,
                        "display_text": text_to_speak
                    })
                    continue

                # Frame base64 payload
                b64_data = payload.get("frame")
                if not b64_data:
                    continue

                if "," in b64_data:
                    b64_data = b64_data.split(",")[1]
                img_bytes = base64.b64decode(b64_data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                continue

            if img is None:
                continue

            # Run MediaPipe Landmark Localization
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = detector.detect(mp_image)

            predicted_label = None
            confidence = 0.0
            hand_detected = False
            landmarks_list = []

            if results.hand_landmarks:
                hand_detected = True
                hand_lms = results.hand_landmarks[0]
                landmarks_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms]

                # 1. Coordinate Normalization
                feature_vector = CoordinateNormalizer.normalize_landmarks(hand_lms).reshape(1, -1)

                # 2. ML Inference
                probs = classifier.predict_proba(feature_vector)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
                predicted_label = str(classes[best_idx])

            # 3. Temporal Debounce & Stability Window
            emitted_symbol, is_boundary = debounce_engine.process_frame(
                predicted_label=predicted_label,
                confidence=confidence,
                hand_detected=hand_detected
            )

            word_completed = False
            if emitted_symbol:
                _, word_completed = sentence_assembler.ingest_symbol(emitted_symbol)

            auto_spoken = False
            # Buffer remains active under user control until cleared manually via the clear button

            # 4. Stream response back to mobile client
            response_payload = {
                "hand_detected": hand_detected,
                "current_sign": predicted_label,
                "confidence": round(confidence, 3),
                "emitted_symbol": emitted_symbol,
                "display_text": sentence_assembler.get_display_text(),
                "word_completed": word_completed,
                "auto_spoken": auto_spoken,
                "landmarks": landmarks_list
            }
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error]: {e}")
    finally:
        detector.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
