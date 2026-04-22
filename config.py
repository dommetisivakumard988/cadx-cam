"""
Configuration for CadX CAM Microservice.
All settings are read from environment variables with sensible defaults.
"""
import os
from typing import Optional

class Settings:
    # Core settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    port: int = int(os.getenv("PORT", "5001"))
    workers: int = int(os.getenv("WORKERS", "1"))
    
    # External services
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_service_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    next_app_url: str = os.getenv("NEXT_PUBLIC_APP_URL", "http://localhost:3000")
    
    # Limits
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

settings = Settings()
