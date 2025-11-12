# Configuration settings
import os

class Config:
    """Konfiguracja aplikacji"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # Parametry analizy technicznej
    SUPPORT_RESISTANCE_THRESHOLD = 0.02
    PATTERN_LOOKBACK_PERIOD = 100
