from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import BYTEA
from datetime import datetime, timedelta
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    person = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)  # store NumPy as binary
class OTP(db.Model):
    __tablename__ = "otps"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    otp = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, default=lambda: datetime.utcnow() + timedelta(minutes=5))  # OTP valid 5 mins


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")  # default role = "user"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Setter for password
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Verifier for password
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
class DetectedFace(db.Model):
    __tablename__ = "detected_faces"
    id = db.Column(db.Integer, primary_key=True)
    person = db.Column(db.String(100), nullable=True)  # Name if known, None if unknown
    image_path = db.Column(db.String(255), nullable=False)  # Path to saved image
    recognized = db.Column(db.Boolean, default=False)  # True if known
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float, nullable=True)      # Latitude of capture
    longitude = db.Column(db.Float, nullable=True)     # Longitude of capture
    def __repr__(self):
        return f"<DetectedFace {self.person} recognized={self.recognized}>"
