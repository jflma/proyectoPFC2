import os
from dotenv import load_dotenv

# Carga .env desde la raíz del proyecto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── Ollama ───────────────────────────────────────────────
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
PERPLEXITY_MODEL     = os.getenv("OLLAMA_PERPLEXITY_MODEL","codellama:7b-code-q4_K_M")
REFACTOR_MODEL       = os.getenv("OLLAMA_REFACTOR_MODEL",  "llama3:8b-instruct-q4_K_M")

# ─── Thresholds ───────────────────────────────────────────
PERPLEXITY_THRESHOLD = float(os.getenv("PERPLEXITY_THRESHOLD", "25.0"))
K_MAX                = int(os.getenv("MAX_MITIGATION_RETRIES", "2"))

# ─── Herramientas externas ────────────────────────────────
CODEQL_PATH          = os.getenv("CODEQL_PATH", "codeql")
GITHUB_TOKEN         = os.getenv("GITHUB_TOKEN", "")

# ─── Rutas del pipeline (relativas a la raíz del proyecto) ─
ROOT_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEVGPT_DATA_PATH  = os.getenv("DEVGPT_DATA_PATH",
                               os.path.join(ROOT_DIR, "dataset", "data", "raw", "devgpt_snapshots"))

DATA_CORPUS_DIR   = os.path.join(ROOT_DIR, "data", "corpus")
DATA_FILTERED_DIR = os.path.join(ROOT_DIR, "data", "filtered")
DATA_REPORTS_DIR  = os.path.join(ROOT_DIR, "data", "reports")
DATA_PATCHED_DIR  = os.path.join(ROOT_DIR, "data", "patched")
DATA_VALIDATED_DIR= os.path.join(ROOT_DIR, "data", "validated")
METADATA_DIR      = os.path.join(ROOT_DIR, "metadata")
METADATA_CSV      = os.path.join(METADATA_DIR, "registro_metadatos.csv")
METRICS_JSON      = os.path.join(METADATA_DIR, "metricas_finales.json")

# ─── Extensiones de lenguaje ──────────────────────────────
EXTENSION_MAP = {"python": ".py", "c": ".c", "cpp": ".cpp"}

# Lenguajes válidos para el pipeline
VALID_LANGUAGES = frozenset(EXTENSION_MAP.keys())

# ─── CWEs en scope (Regla R-04) ───────────────────────────
SECURITY_CWES = frozenset({
    "CWE-89",   # SQL Injection
    "CWE-78",   # OS Command Injection
    "CWE-79",   # XSS
    "CWE-120",  # Buffer Copy without Checking Size
    "CWE-125",  # Out-of-bounds Read
    "CWE-787",  # Out-of-bounds Write
    "CWE-476",  # NULL Pointer Dereference
    "CWE-862",  # Missing Authorization
    "CWE-306",  # Missing Authentication for Critical Function
    "CWE-22",   # Path Traversal
    "CWE-434",  # Unrestricted Upload of Dangerous File Type
})
