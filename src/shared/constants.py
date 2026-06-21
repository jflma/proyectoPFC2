import os
from dotenv import load_dotenv

# Carga .env desde la raíz del proyecto
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── Ollama ───────────────────────────────────────────────
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",         "http://localhost:11434")
PERPLEXITY_MODEL     = os.getenv("OLLAMA_PERPLEXITY_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")
REFACTOR_MODEL       = os.getenv("OLLAMA_REFACTOR_MODEL",   "qwen2.5-coder:7b-instruct-q4_K_M")
VERIFIER_MODEL       = PERPLEXITY_MODEL  # Retrocompatibilidad

# ─── Thresholds ───────────────────────────────────────────
PERPLEXITY_THRESHOLD = float(os.getenv("PERPLEXITY_THRESHOLD",           "25.0"))
K_BUFFER             = int(os.getenv("MAX_MITIGATION_RETRIES_BUFFER",    "2"))
K_MAX_FALLBACK       = K_BUFFER + 1

# ─── Herramientas externas ────────────────────────────────
CODEQL_PATH          = os.getenv("CODEQL_PATH", "codeql")
GITHUB_TOKEN         = os.getenv("GITHUB_TOKEN", "")
NVD_API_KEY          = os.getenv("NVD_API_KEY", "")  # opcional, solo para rate-limit NVD

# ─── Rutas del pipeline (relativas a la raíz del proyecto) ─
ROOT_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_CORPUS_DIR    = os.path.join(ROOT_DIR, "data", "corpus")
DATA_FILTERED_DIR  = os.path.join(ROOT_DIR, "data", "filtered")
DATA_REPORTS_DIR   = os.path.join(ROOT_DIR, "data", "reports")
DATA_PATCHED_DIR   = os.path.join(ROOT_DIR, "data", "patched")
DATA_VALIDATED_DIR = os.path.join(ROOT_DIR, "data", "validated")
DATA_GRAFICOS_DIR  = os.path.join(ROOT_DIR, "data", "graficos")
METADATA_DIR       = os.path.join(ROOT_DIR, "metadata")
METADATA_CSV       = os.path.join(METADATA_DIR, "registro_metadatos.csv")
METRICS_JSON       = os.path.join(METADATA_DIR, "metricas_finales.json")

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
