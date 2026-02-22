from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # GreenPull/
    CLONE_DIR: Path = Path("/tmp/greenpull_repos")
    DATABASE_URL: str = f"sqlite:///{Path(__file__).resolve().parents[2] / 'greenpull.db'}"

    REDIS_URL: str = "redis://localhost:6379/0"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    GITHUB_TOKEN: str = ""

    ELECTRICITYMAPS_API_TOKEN: str = ""
    ELECTRICITYMAPS_BASE_URL: str = "https://api.electricitymaps.com/v3"
    DEFAULT_WUE: float = 1.8  # Water Usage Effectiveness (L/kWh), industry average

    DEBUG: bool = True

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"


settings = Settings()
