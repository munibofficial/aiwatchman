import os
import re
import threading
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import db, User, FaceEmbedding, DetectedFace, OtpCode
from email_utils import init_mail, send_otp_email, verify_otp


# -----------------------------
# Flask app & basic config
# -----------------------------
app = Flask(__name__)
CORS(app)

# Storage dirs (Heroku FS is ephemeral but writable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
QUERY_DIR = os.path.join(BASE_DIR, "queries")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(QUERY_DIR, exist_ok=True)


# -----------------------------
# Database (use Heroku DATABASE_URL if present)
# -----------------------------
def ensure_sslmode_require(url: str) -> str:
    """Force sslmode=require for Postgres connections unless explicitly set."""
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query))
    if parsed.scheme.startswith("postgres") and q.get("sslmode") is None:
        q["sslmode"] = "require"
        parsed = parsed._replace(query=urlencode(q))
        return urlunparse(parsed)
    return url


db_url = os.getenv("DATABASE_URL")
if db_url:
    # Heroku can provide postgres:// – SQLAlchemy prefers postgresql://
    db_url = db_url.replace("postgres://", "postgresql://", 1)
else:
    # Optional local fallback via individual envs
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"

db_url = ensure_sslmode_require(db_url)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# Keep connections fresh across dyno sleeps
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

# Mail (reads env: MAIL_SERVER, MAIL_PORT, MAIL_USE_TLS/SSL, MAIL_USERNAME, MAIL_PASSWORD, MAIL_DEFAULT_SENDER)
init_mail(app)

# Bind db and create tables
db.init_app(app)
with app.app_context():
    db.create_all()


# -----------------------------
# InsightFace lazy loader (avoids Heroku boot timeouts)
# -----------------------------
# Cache models in a writable path; re-downloaded per dyno restart
os.environ.setdefault("INSIGHTFACE_HOME", "/tmp/insightface")

_face_app = None
_face_lock = threading.Lock()


def get_face_app():
    """Initialize InsightFace on first use (thread-safe)."""
    global _face_app
    if _face_app is None:
        with _face_lock:
            if _face_app is None:
                from insightface.app import FaceAnalysis  # heavy import, do it lazily
                fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                # CPU, detector size; adjust if you need speed vs. accuracy
                fa.prepare(ctx_id=0, det_size=(640, 640))
                _face_app = fa
    return _face_app


def get_embeddings_from_image(img_path: str):
    """Return a list of L2-normalized face embeddings (float32 lists)."""
    img = cv2.imread(img_path)
    if img is None:
        return []
    faces = get_face_app().get(img)
    return [f.normed_embedding.astype("float32").tolist() for f in faces]


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"message": "Hello, brother! Your API is working."}), 200


@app.route("/warmup", methods=["GET"])
def warmup():
    """Trigger model init so subsequent requests are fast."""
    _ = get_face_app()
    return jsonify({"status": "ok", "message": "Model warmed"}), 200


@app.route("/upload-references", methods=["POST"])
def upload_references():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    added = 0

    for file in files:
        filename = file.filename
        if not filename:
            continue

        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Extract only the starting letters before any numbers/underscores
        base_name = os.path.splitext(filename)[0]
        m = re.match(r"[A-Za-z]+", base_name)
        if not m:
            # Skip files without a clean person prefix
            continue
        person = m.group(0).lower()

        embeddings = get_embeddings_from_image(save_path)
        for emb in embeddings:
            db.session.add(FaceEmbedding(person=person, embedding=emb))
            added += 1

    db.session.commit()
    return jsonify({"message": f"Added {added} embeddings to DB"}), 200


@app.route("/identify-image", methods=["POST"])
def identify_image():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "query.jpg"
    save_path = os.path.join(QUERY_DIR, filename)
    file.save(save_path)

    # Optional GPS
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    try:
        latitude = float(latitude) if latitude else None
        longitude = float(longitude) if longitude else None
    except ValueError:
        latitude = longitude = None

    query_embeddings = get_embeddings_from_image(save_path)
    if not query_embeddings:
        return jsonify({"error": "No face detected in the image"}), 400

    # Load all known embeddings
    all_embeddings = FaceEmbedding.query.all()
    results = []

    for emb in query_embeddings:
        emb_np = np.array(emb, dtype=np.float32)
        best_name, best_score = "unknown", -1.0

        for entry in all_embeddings:
            db_emb = np.array(entry.embedding, dtype=np.float32)
            score = float(np.dot(emb_np, db_emb))  # cosine since embeddings are normalized
            if score > best_score:
                best_score = score
                best_name = entry.person

        # Threshold for recognition
        if best_score < 0.6:
            best_name = "unknown"

        detected = DetectedFace(
            person=best_name if best_name != "unknown" else None,
            image_path=save_path,
            recognized=(best_name != "unknown"),
            latitude=latitude,
            longitude=longitude,
        )
        db.session.add(detected)
        db.session.commit()

        results.append({"person": best_name, "similarity": best_score})

    return jsonify(
        {
            "filename": filename,
            "results": results,
            "latitude": latitude,
            "longitude": longitude,
            "saved_id": detected.id,
        }
    ), 200


@app.route("/queries/<filename>")
def serve_queries(filename):
    return send_from_directory(QUERY_DIR, filename)


@app.route("/known-faces", methods=["GET"])
def get_known_faces():
    faces = DetectedFace.query.filter_by(recognized=True).all()
    faces_data = [
        {
            "person": face.person,
            "latitude": face.latitude,
            "longitude": face.longitude,
            "filename": os.path.basename(face.image_path),
        }
        for face in faces
    ]
    return jsonify({"known_faces": faces_data}), 200


@app.route("/unknown-faces", methods=["GET"])
def get_unknown_faces():
    faces = DetectedFace.query.filter_by(recognized=False).all()
    faces_data = [{"filename": os.path.basename(face.image_path)} for face in faces]
    return jsonify({"unknown_faces": faces_data}), 200


# --------- Auth + OTP ----------
@app.route("/send-otp/", methods=["POST"])
def route_send_otp():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    ok, msg = send_otp_email(email)
    return (jsonify({"message": msg}), 200) if ok else (jsonify({"error": msg}), 500)


@app.route("/verify-otp/", methods=["POST"])
def route_verify_otp():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    otp_input = data.get("otp")
    if not email or not otp_input:
        return jsonify({"error": "Email and OTP are required"}), 400

    if verify_otp(email, otp_input):
        return jsonify({"message": "OTP verified successfully!"}), 200
    return jsonify({"error": "Invalid or expired OTP"}), 400


@app.route("/signup/", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"error": "Email already registered"}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Signup successful!"}), 200


@app.route("/login/", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return jsonify({"message": "Login successful!"}), 200
    return jsonify({"error": "Invalid email or password"}), 401


# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    # Heroku sets PORT; local default is 5000
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=True
    )
