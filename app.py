from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *
import os, cv2, numpy as np
from insightface.app import FaceAnalysis
from email_utils import *
from flask import send_from_directory

# ================================
# STEP 1: Init Flask + DB
# ================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# PostgreSQL connection

# Replace with your PostgreSQL credentials
DB_USER = 'postgres'
DB_PASSWORD = 'Arqum%40123'
DB_HOST = 'localhost'   # or your VPS IP
DB_PORT = '5432'
DB_NAME = 'face_recognition'

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
# 🔑 Initialize Flask-Mail
init_mail(app)
# Create tables
with app.app_context():
    db.create_all()

# ================================
# STEP 2: Face Model
# ================================
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_embeddings_from_image(img_path):
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    return [f.normed_embedding.astype('float32') for f in faces]

# ================================
# STEP 3: Routes
# ================================
import re
@app.route('/', methods=['GET'])
def hello():
    return jsonify({
        "message": "Hello, brother! Your API is working."
    })

@app.route("/upload-references", methods=["POST"])
def upload_references():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    added = 0
    for file in files:
        filename = file.filename
        save_path = os.path.join(upload_dir, filename)
        file.save(save_path)

        # Extract only letters before numbers
        base_name = os.path.splitext(filename)[0]  # remove .jpg/.png
        person = re.match(r"[A-Za-z]+", base_name).group(0).lower()

        embeddings = get_embeddings_from_image(save_path)

        for emb in embeddings:
            db.session.add(FaceEmbedding(person=person, embedding=emb.tolist()))
            added += 1

    db.session.commit()
    return jsonify({"message": f"Added {added} embeddings to DB"})
@app.route("/identify-image", methods=["POST"])
def identify_image():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    save_dir = "queries"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    # Get latitude and longitude
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    print(f"Received image before: {filename}, Location: {latitude}, {longitude}")
    try:
        latitude = float(latitude) if latitude else None
        longitude = float(longitude) if longitude else None
    except ValueError:
        latitude = longitude = None

    print(f"Received image: {filename}, Location: {latitude}, {longitude}")

    # Get face embeddings
    query_embeddings = get_embeddings_from_image(save_path)

    if not query_embeddings or len(query_embeddings) == 0:
        return jsonify({"error": "No face detected in the image"}), 400

    results = []

    # Load all embeddings from DB
    all_embeddings = FaceEmbedding.query.all()

    for emb in query_embeddings:
        best_name, best_score = "unknown", -1
        for entry in all_embeddings:
            db_emb = np.array(entry.embedding)
            score = float(np.dot(emb, db_emb))  # cosine similarity
            if score > best_score:
                best_score, best_name = score, entry.person

        # Apply threshold (e.g. 0.6 = 60%)
        if best_score < 0.6:
            best_name = "unknown"

        # Save detected face with location
        detected = DetectedFace(
            person=best_name if best_name != "unknown" else None,
            image_path=save_path,
            recognized=(best_name != "unknown"),
            latitude=latitude,
            longitude=longitude
        )
        db.session.add(detected)
        db.session.commit()

        results.append({"person": best_name, "similarity": best_score})

    return jsonify({
        "filename": filename,
        "results": results,
        "latitude": latitude,
        "longitude": longitude,
        "saved_id": detected.id
    })

@app.route("/load-folder", methods=["POST"])
def load_folder():
    folder_path = r"C:\Users\Arham Ali\arqum model\images_matching"

    if not os.path.exists(folder_path):
        return jsonify({"error": f"Folder {folder_path} not found"}), 404

    count_added = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            person = filename.split("_")[0].lower()

            embeddings = get_embeddings_from_image(path)  # should return numpy array(s)
            if embeddings is not None:
                for emb in embeddings:
                    # Directly store NumPy array (PickleType handles it)
                    db.session.add(FaceEmbedding(person=person, embedding=emb))
                    count_added += 1

    db.session.commit()

    return jsonify({"message": f"Loaded {count_added} embeddings from {folder_path}"})

@app.route("/send-otp/", methods=["POST"])
def route_send_otp():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    success, msg = send_otp_email(email)
    if success:
        return jsonify({"message": msg}), 200
    else:
        return jsonify({"error": msg}), 500

@app.route("/verify-otp/", methods=["POST"])
def route_verify_otp():
    data = request.get_json()
    email = data.get("email")
    otp_input = data.get("otp")

    if not email or not otp_input:
        return jsonify({"error": "Email and OTP are required"}), 400

    if verify_otp(email, otp_input):
        return jsonify({"message": "OTP verified successfully!"}), 200
    else:
        return jsonify({"error": "Invalid or expired OTP"}), 400
@app.route("/signup/", methods=["POST"])
def signup():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "Email already registered"}), 400

    new_user = User(username=username, email=email)
    new_user.set_password(password)  # Hash the password
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Signup successful!"}), 200
@app.route("/login/", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        # Login successful, optionally return token
        return jsonify({"message": "Login successful!"}), 200
    return jsonify({"error": "Invalid email or password"}), 401
@app.route("/known-faces", methods=["GET"])
def get_known_faces():
    try:
        known_faces = DetectedFace.query.filter_by(recognized=True).all()
        faces_data = [
            {
                "person": face.person,
                "latitude": face.latitude,                  # add latitude
                "longitude": face.longitude,
                "filename": face.image_path.split("\\")[-1]  # just the filename
            }
            for face in known_faces
        ]
        return jsonify({"known_faces": faces_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

QUERIES_FOLDER = "queries"  # folder where identified images are saved
@app.route("/queries/<filename>")
def serve_queries(filename):
    return send_from_directory(QUERIES_FOLDER, filename)
@app.route("/unknown-faces", methods=["GET"])
def get_unknown_faces():
    try:
        unknown_faces = DetectedFace.query.filter_by(recognized=False).all()
        faces_data = [
            {
                "filename": face.image_path.split("\\")[-1]  # just the filename
            }
            for face in unknown_faces
        ]
        return jsonify({"unknown_faces": faces_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================
# STEP 4: Run
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
