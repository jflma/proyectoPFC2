import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_FILTERED_DIR,
    DATA_PATCHED_DIR,
    DATA_REPORTS_DIR,
    EXTENSION_MAP,
    OLLAMA_BASE_URL,
    REFACTOR_MODEL,
)
from src.shared.code_extractor import extract_code_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-4] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 240  


PROMPT_TEMPLATE = """\
SYSTEM:
You are a software security expert. Your ONLY task is to fix ALL security \
vulnerabilities listed below in the provided code. \
Respond ONLY with the corrected code inside a markdown code block \
(```{language} ... ```). Do NOT include explanations, additional comments, \
or any text outside the code block.

USER:
Language: {language}

VULNERABILITIES TO FIX (fix ALL of them):
{vulns_str}

RULES:
1. Fix ALL vulnerabilities listed above in a SINGLE response.
2. Do NOT change business logic or observable behavior.
3. Use parameterization, input validation, or access control as appropriate.
4. The resulting code MUST compile/run without syntax errors.
5. Apply the minimal changes needed — do not refactor unrelated code.

CODE TO FIX:
```{language}
{code_content}
```"""


def _formatear_todas_las_vulnerabilidades(vulns: list[dict]) -> str:

    lines = []
    for i, v in enumerate(vulns, 1):
        lines.append(
            f"[{i}] {v.get('cwe_id', 'UNKNOWN')} — {v.get('description', 'N/A')}\n"
            f"    Line ≈ {v.get('line', '?')} | "
            f"CVSS: {v.get('cvss_score', 'N/A')} | "
            f"Source: {v.get('source', 'N/A')}"
        )
    return "\n".join(lines)

def _llamar_ollama(prompt: str, language: str) -> Optional[str]:
    """Envía el prompt a Ollama y retorna el código extraído, o None si falla."""
    payload = {
        "model": REFACTOR_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        },
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
            log.warning("LLM no retorno bloque de código valido.")
        return code_out
    except requests.exceptions.Timeout:
        log.warning("Timeout esperando respuesta de Ollama (model=%s).", REFACTOR_MODEL)
    except requests.exceptions.ConnectionError:
        log.warning("No se pudo conectar a Ollama en %s.", OLLAMA_BASE_URL)
    except Exception as e:
        log.warning("Error en llamada a Ollama: %s", e)
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

    if iteracion == 1:
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

    vulns_str = _formatear_todas_las_vulnerabilidades(vulns)

    log.info(
        "Parcheando %s (iter %d) — %d vulnerabilidades: %s",
        file_id, iteracion,
        len(vulns),
        ", ".join(v.get("cwe_id", "?") for v in vulns),
    )

    prompt = PROMPT_TEMPLATE.format(
        language=language,
        vulns_str=vulns_str,
        code_content=code,
    )

    codigo_parcheado = _llamar_ollama(prompt, language)

    if codigo_parcheado is None:
        log.warning(
            "Ollama no pudo generar parche para %s (iter %d). "
            "Considera configurar una API externa como fallback.",
            file_id, iteracion,
        )
        return False

    os.makedirs(DATA_PATCHED_DIR, exist_ok=True)
    dest = os.path.join(DATA_PATCHED_DIR, f"{file_id}_v{iteracion}{ext}")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(codigo_parcheado)

    log.info("Parche guardado en %s", dest)
    return True
