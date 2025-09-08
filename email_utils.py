from flask_mail import Mail, Message
from random import randint
from datetime import datetime
from model import db, OTP, User  # Make sure you have a User model
import os
from dotenv import load_dotenv

load_dotenv()

mail = Mail()

def init_mail(app):
    app.config['MAIL_SERVER'] = "smtp.gmail.com"
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = "arhumdoger@gmail.com"  # your Gmail address
    app.config['MAIL_PASSWORD'] = "ovqz vtsx mnkn vlpt"     # your Gmail App Password

    mail.init_app(app)

def send_otp_email(to_email):
    """Generate OTP, save to DB, and send email"""
    # Check if email already exists in the User table
    existing_user = User.query.filter_by(email=to_email).first()
    if existing_user:
        return False, "Email is already registered!"

    # Generate OTP
    otp_code = str(randint(100000, 999999))

    # Save OTP to DB
    otp_entry = OTP(email=to_email, otp=otp_code)
    db.session.add(otp_entry)
    db.session.commit()

    msg = Message(
        subject="Your OTP Code",
        sender=os.environ.get("MAIL_USERNAME"),
        recipients=[to_email],
        body=f"Your OTP code is: {otp_code}. It expires in 5 minutes."
    )

    try:
        mail.send(msg)
        return True, "OTP sent successfully!"
    except Exception as e:
        return False, str(e)

def verify_otp(to_email, otp_input):
    """Verify OTP from DB"""
    otp_entry = OTP.query.filter_by(email=to_email, otp=otp_input).order_by(OTP.created_at.desc()).first()
    if otp_entry and otp_entry.expires_at > datetime.utcnow():
        # OTP is valid, delete after verification
        db.session.delete(otp_entry)
        db.session.commit()
        return True
    return False
