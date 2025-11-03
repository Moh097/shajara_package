import os
from dotenv import load_dotenv

# Load .env if present (optional)
load_dotenv(override=True)

def env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default

# SQLite DB path (file on disk)
DB_PATH = env("SHAJARA_DB_PATH", "shajara.db")

# OpenAI (direct) or Azure OpenAI
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL   = env("OPENAI_MODEL", "gpt-4o-mini")

AZURE_OPENAI_ENDPOINT   = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY    = env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = env("AZURE_OPENAI_DEPLOYMENT")

def using_azure() -> bool:
    return bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT)
