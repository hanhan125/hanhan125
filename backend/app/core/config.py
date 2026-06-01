from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    app_name: str = "TeachingAssistanceSystem"
    sqlite_path: str = "tas.db"
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174"
    )
    # Allow phone / other PC browser on LAN to open Vite page and call API (scheme A).
    # Set to empty in .env to disable: CORS_ORIGIN_REGEX=
    cors_origin_regex: str = (
        r"https?://("
        r"localhost|127\.0\.0\.1|"
        r"192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"(?:[a-z0-9-]+\.)?ngrok-free\.(?:dev|app)|"
        r"(?:[a-z0-9-]+\.)?ngrok\.(?:io|app)"
        r")(:\d+)?"
    )


settings = Settings()

