import os
import sys
import json
import logging
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_FILTERED_DIR,
    DATA_PATCHED_DIR,
    DATA_REPORTS_DIR,
    OLLAMA_BASE_URL,
    REFACTOR_MODEL,
    EXTENSION_MAP,
)
from src.shared.code_extractor import extract_code_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-4] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 180

PROMPT_TEMPLATE = """\
SYSTEM:
Eres un experto en seguridad de software. Tu unica tarea es corregir vulnerabilidades de seguridad en codigo. \
Responde UNICAMENTE con el codigo corregido dentro de bloques de codigo markdown (```{language} ... ```). \
No incluyas explicaciones, comentarios adicionales ni texto fuera del bloque.

USER:
Lenguaje: {language}
Vulnerabilidad: {cwe_id}
Descripcion: {description}
Linea aproximada: {line}

REGLAS:
1. Corrige UNICAMENTE la vulnerabilidad indicada.
2. NO cambies la logica de negocio ni el comportamiento observable.
3. Usa parametrizacion, validacion de entrada o control de acceso segun corresponda.
4. El codigo resultante debe ejecutarse/compilarse sin errores de sintaxis.

CODIGO A CORREGIR:
```{language}
{code_content}
```"""


def request_patch(code: str, vuln: dict, language: str) -> str | None:
    prompt = PROMPT_TEMPLATE.format(
        language=language,
        cwe_id=vuln.get("cwe_id", "UNKNOWN"),
        description=vuln.get("description", "N/A"),
        line=vuln.get("line", "?"),
        code_content=code,
    )
    payload = {
        "model": REFACTOR_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        code_out = extract_code_block(raw, language)
        if not code_out:
            log.warning("LLM no retorno bloque de codigo valido.")
        return code_out
    except requests.exceptions.Timeout:
        log.warning("Timeout esperando respuesta de Ollama.")
    except requests.exceptions.ConnectionError:
        log.warning("No se pudo conectar a Ollama en %s.", OLLAMA_BASE_URL)
    except Exception as e:
        log.warning("Error en request_patch: %s", e)
    return None


def run_phase4_single(file_id: str, iteracion: int) -> bool:
    report_path = os.path.join(DATA_REPORTS_DIR, f"{file_id}.json")
    if not os.path.exists(report_path):
        log.warning("Reporte no encontrado: %s", report_path)
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    vulns = report.get("vulnerabilities", [])
    if not vulns:
        log.info("Archivo %s sin vulnerabilidades, nada que parchear.", file_id)
        return False

    language = report.get("language", "python")
    ext = EXTENSION_MAP.get(language, ".py")

    # Fuente: iteracion 1 → filtered con nombre real, iteraciones siguientes → patched anterior
    if iteracion == 1:
        # El reporte guarda el filename original (ej. syn_sqli_001.py)
        original_filename = report.get("filename", "")
        if original_filename:
            src = os.path.join(DATA_FILTERED_DIR, original_filename)
        else:
            src = os.path.join(DATA_FILTERED_DIR, f"{file_id}{ext}")
    else:
        src = os.path.join(DATA_PATCHED_DIR, f"{file_id}_v{iteracion - 1}{ext}")

    if not os.path.exists(src):
        log.warning("Archivo fuente no encontrado: %s", src)
        return False

    with open(src, "r", encoding="utf-8", errors="replace") as f:
        code = f.read()

    # Parchear la vulnerabilidad de mayor CVSS primero
    vuln = sorted(vulns, key=lambda v: v.get("cvss_score", 0), reverse=True)[0]
    log.info("Parcheando %s iter %d — %s (CVSS %.1f)", file_id, iteracion,
             vuln.get("cwe_id"), vuln.get("cvss_score", 0))

    patched = request_patch(code, vuln, language)
    if patched is None:
        return False

    os.makedirs(DATA_PATCHED_DIR, exist_ok=True)
    dest = os.path.join(DATA_PATCHED_DIR, f"{file_id}_v{iteracion}{ext}")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(patched)

    log.info("Parche guardado en %s", dest)
    return True
