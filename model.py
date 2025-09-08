from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, raw):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw):
        return check_password_hash(self.password_hash, raw)


class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    person = db.Column(db.String(120), index=True, nullable=False)
    # Store embedding as JSON array of floats to keep it simple/portable
    embedding = db.Column(JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DetectedFace(db.Model):
    __tablename__ = "detected_faces"

    id = db.Column(db.Integer, primary_key=True)
    person = db.Column(db.String(120), index=True, nullable=True)  # None if unknown
    image_path = db.Column(db.String(512), nullable=False)
    recognized = db.Column(db.Boolean, default=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class OtpCode(db.Model):
    __tablename__ = "otp_codes"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), index=True, nullable=False)
    code = db.Column(db.String(10), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)

    @staticmethod
    def not_expired_query(email, code):
        now = datetime.utcnow()
        return OtpCode.query.filter(
            OtpCode.email == email,
            OtpCode.code == code,
            OtpCode.expires_at > now,
        )

    @staticmethod
    def create(email, code, ttl_seconds=300):
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        return OtpCode(email=email, code=code, expires_at=expires_at)
