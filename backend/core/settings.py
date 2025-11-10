from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # App info
    APP_NAME: str = "Chatbot Agent"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Chatbot Agent for Insurance POC"

    # Google AI / API Keys
    GOOGLE_API_KEY: str  # will be read from environment
    GOOGLE_GENAI_USE_VERTEXAI: bool = False
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]

    # Logging
    LOG_LEVEL: str = "DEBUG"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
