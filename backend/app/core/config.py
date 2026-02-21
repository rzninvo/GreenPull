from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # GreenPull/
    CLONE_DIR: Path = Path("/tmp/greenpull_repos")
    RESULTS_DIR: Path = Path("/tmp/greenpull_results")
    DATABASE_URL: str = f"sqlite:///{Path(__file__).resolve().parents[2] / 'greenpull.db'}"

    REDIS_URL: str = "redis://localhost:6379/0"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    COUNTRY_ISO_CODE: str = "USA"
    DEBUG: bool = True

    class Config:
        env_file = Path(__file__).resolve().parents[2] / ".env"


settings = Settings()
