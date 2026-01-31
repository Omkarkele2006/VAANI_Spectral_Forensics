from datetime import datetime
from database.db import db

class User(db.Model):
    """
    FR-01: Stores user credentials.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='User') # 'User' or 'Admin'
    
    # Relationship: One user has many logs
    logs = db.relationship('AuditLog', backref='owner', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'

class AuditLog(db.Model):
    """
    FR-05: The Forensic Chain of Custody.
    Stores metadata about every analysis performed.
    """
    __tablename__ = 'audit_logs'
    
    log_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False) # SHA-256 Hash
    prediction = db.Column(db.String(50)) # "Real" or "Fake"
    confidence_score = db.Column(db.Float) # e.g., 98.5
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Log {self.filename} - {self.prediction}>'