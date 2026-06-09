"""
FASE 2: Módulo de Atribución por Perplejidad Cruzada
======================================================
Archivo: src/phase2/perplexity_filter.py

Inputs:
  - metadata/registro_metadatos.csv
  - Archivos de código en data/corpus/
  - Modelo LLM local vía Ollama (OLLAMA_PERPLEXITY_MODEL)

Outputs:
  - Archivos confirmados copiados a data/filtered/
  - metadata/registro_metadatos.csv actualizado:
      campos perplexity_score e is_ai_generated

Criterio de aceptación (Fase 2):
  Todos los archivos del corpus tienen perplexity_score != -1.0;
  data/filtered/ no está vacío.

Estrategia de cálculo de perplexity (en orden de prioridad):
  1. Ollama /api/generate con logprobs=True → usa prompt_eval_logprobs si disponible
  2. Sliding-window sobre /api/generate: envía prefix+chunk y mide eval tokens
  3. Heurística léxica (fallback offline) si Ollama no responde
"""

import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_CORPUS_DIR,
    DATA_FILTERED_DIR,
    EXTENSION_MAP,
    METADATA_CSV,
    METADATA_DIR,
    OLLAMA_BASE_URL,
    PERPLEXITY_MODEL,
    PERPLEXITY_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-2] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────────
OLLAMA_TIMEOUT     = 15     # segundos por request (fallo rápido → fallback a heurística)
CHUNK_SIZE         = 400    # chars por chunk en sliding window
MAX_CHUNKS         = 3      # máximo de chunks (reducido para velocidad)
RETRY_DELAY        = 1.0    # segundos entre reintentos
MAX_RETRIES        = 1      # reintentos por archivo (1 = fallo rápido)

# Modo de ejecución: "auto" | "ollama" | "offline"
# "auto"    = intenta Ollama; si no responde en OLLAMA_TIMEOUT, usa heurística
# "ollama"  = solo Ollama (puede ser lento)
# "offline" = solo heurística léxica (rápido, ~1s/archivo)
EXECUTION_MODE = "auto"

# Tiempo máximo de ping para decidir si Ollama es viable
OLLAMA_PING_TIMEOUT = 8  # segundos


# ══════════════════════════════════════════════════════════════════════════════
# Estrategia 1: Logprobs nativos de Ollama
# ══════════════════════════════════════════════════════════════════════════════

def _extract_logprobs(response_json: dict) -> Optional[list[float]]:
    """
    Extrae log-probabilities del JSON de respuesta de Ollama.
    Ollama >= 0.2 puede devolver prompt_eval_logprobs o logprobs.
    Retorna lista de floats o None si no disponible.
    """
    # Intentar campos conocidos de distintas versiones de Ollama
    for field in ("prompt_eval_logprobs", "logprobs", "prompt_logprobs"):
        val = response_json.get(field)
        if val and isinstance(val, list) and len(val) > 0:
            # Normalizar: puede ser lista de floats o lista de dicts {"logprob": float}
            result = []
            for item in val:
                if isinstance(item, (int, float)):
                    result.append(float(item))
                elif isinstance(item, dict) and "logprob" in item:
                    result.append(float(item["logprob"]))
            if result:
                return result
    return None


def calculate_perplexity_ollama_logprobs(code_text: str) -> Optional[float]:
    """
    Calcula perplexity usando logprobs nativos de Ollama.
    Retorna None si Ollama no expone logprobs.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": PERPLEXITY_MODEL,
        "prompt": code_text,
        "options": {"temperature": 0},
        "logprobs": True,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        log_probs = _extract_logprobs(data)
        if log_probs:
            # perplexity = exp(-mean(log_probs))
            perplexity = math.exp(-mean(log_probs))
            return perplexity
        return None
    except requests.exceptions.Timeout:
        raise
    except requests.exceptions.ConnectionError:
        raise
    except Exception as e:
        log.debug("Logprobs strategy failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Estrategia 2: Sliding Window via /api/generate (proxy de perplexity)
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_token_score(prefix: str, continuation: str) -> Optional[float]:
    """
    Envía (prefix + continuation) a Ollama y mide cuánto del continuation
    fue "generado" vs predicho. Usa eval_count como proxy.
    Retorna log-prob aproximado o None en error.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    # Enviamos el prefix como prompt para que el modelo lo "digiera",
    # y pedimos que complete con el continuation.
    payload = {
        "model": PERPLEXITY_MODEL,
        "prompt": prefix,
        "options": {
            "temperature": 0,
            "num_predict": len(continuation) // 3,  # aprox tokens
            "stop": [],
        },
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # Usar prompt_eval_count y eval_count como señal
        prompt_tokens = data.get("prompt_eval_count", 1) or 1
        eval_tokens   = data.get("eval_count", 1) or 1
        # Heurística: código IA es predecible → Ollama necesita menos tokens
        # para evaluar el prompt (prompt_eval_count bajo relativo al texto)
        # Devolvemos el ratio como señal de perplexity proxy
        ratio = eval_tokens / prompt_tokens
        return ratio
    except Exception:
        return None


def calculate_perplexity_sliding_window(code_text: str) -> Optional[float]:
    """
    Estrategia de sliding window: divide el código en chunks y calcula
    un score de perplexity aproximado basado en la predictibilidad de Ollama.
    
    Score bajo → código predecible → código IA.
    """
    # Dividir en chunks
    chunks = []
    step = CHUNK_SIZE
    for i in range(0, len(code_text), step):
        chunks.append(code_text[i : i + step])
    chunks = chunks[:MAX_CHUNKS]

    if not chunks:
        return None

    scores = []
    for i, chunk in enumerate(chunks):
        prefix = code_text[:i * step] if i > 0 else "# Code:\n"
        score = _ollama_token_score(prefix, chunk)
        if score is not None:
            scores.append(score)

    if not scores:
        return None

    # Normalizar al rango de perplexity: multiplicar ratio por factor empírico
    # Un ratio ~1.0 = muy predecible → perplexity baja (~10)
    # Un ratio ~5.0 = poco predecible → perplexity alta (~50)
    avg_ratio = mean(scores)
    # Mapeo lineal: ratio 1.0 → 10.0, ratio 5.0 → 50.0
    perplexity_proxy = 10.0 * avg_ratio
    return perplexity_proxy


# ══════════════════════════════════════════════════════════════════════════════
# Estrategia 3: Heurística léxica (fallback offline)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_perplexity_heuristic(code_text: str, language: str) -> float:
    """
    Fallback offline: estima perplexity mediante heurísticas léxicas y
    estructurales del código. No requiere LLM.

    Señales de código IA (perplexity baja):
    - Comentarios abundantes y bien formateados
    - Nombres de variables descriptivos y largos
    - Docstrings presentes
    - Estructura uniforme (indentación consistente)
    - Palabras clave de estilo IA ("efficiently", "comprehensive", etc.)

    Retorna un score en [5.0, 80.0]:
      <25 → probablemente IA
      >25 → probablemente humano
    """
    import re
    import collections

    lines = code_text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    total_lines = max(len(non_empty), 1)
    text_lower = code_text.lower()

    score = 50.0  # baseline neutral

    # ── Señales que BAJAN el score (más IA) ─────────────────────────────────

    # 1. Ratio de comentarios (código IA suele tener ~20-40% de líneas comentadas)
    if language == "python":
        comment_lines = sum(1 for l in non_empty if l.strip().startswith("#"))
        docstring_count = code_text.count('"""') + code_text.count("'''")
    else:
        comment_lines = sum(1 for l in non_empty if "//" in l or "/*" in l)
        docstring_count = 0

    comment_ratio = comment_lines / total_lines
    if 0.15 <= comment_ratio <= 0.45:
        score -= 12.0  # ratio típico de IA
    elif comment_ratio > 0.45:
        score -= 8.0   # muchos comentarios = probablemente IA

    # 2. Presencia de docstrings / JSDoc
    if docstring_count >= 2:
        score -= 8.0

    # 3. Palabras clave estilo IA en comentarios
    ai_style_words = [
        "efficiently", "optimized", "comprehensive", "robust", "ensure",
        "handles", "implementation", "returns", "raises", "note:", "example:",
        "todo:", "fixme:", "param", ":param", ":return", "args:", "returns:",
    ]
    ai_word_hits = sum(1 for w in ai_style_words if w in text_lower)
    score -= min(ai_word_hits * 2.0, 14.0)

    # 4. Longitud promedio de identificadores (IA usa nombres descriptivos)
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{4,})\b', code_text)
    if identifiers:
        avg_id_len = mean(len(i) for i in identifiers)
        if avg_id_len >= 8.0:
            score -= 8.0
        elif avg_id_len >= 6.0:
            score -= 4.0

    # 5. Funciones con type hints (Python IA casi siempre los incluye)
    if language == "python":
        type_hint_count = len(re.findall(r'def \w+\([^)]*:\s*\w', code_text))
        if type_hint_count > 0:
            score -= min(type_hint_count * 3.0, 10.0)

    # 6. Manejo de errores explícito (try/except, error handling)
    error_handling = code_text.count("try:") + code_text.count("except ") + \
                     code_text.count("catch(") + code_text.count("catch (")
    if error_handling >= 2:
        score -= 6.0

    # ── Señales que SUBEN el score (más humano) ──────────────────────────────

    # 7. Código muy corto (fragmentos triviales)
    if total_lines < 10:
        score += 15.0

    # 8. Inconsistencia de indentación (código copiado/humano)
    indent_sizes = set()
    for l in non_empty:
        stripped = l.lstrip()
        if stripped and l != stripped:
            indent_size = len(l) - len(stripped)
            indent_sizes.add(indent_size % 4 if indent_size > 0 else 0)
    if len(indent_sizes) > 2:
        score += 8.0

    # 9. Variables de un solo carácter (estilo humano apresurado)
    single_char_vars = len(re.findall(r'\b([a-df-hj-np-tv-z])\b', code_text))
    if single_char_vars > total_lines * 0.3:
        score += 6.0

    # 10. Presencia de TODO/FIXME sin resolver (humano)
    if "todo" in text_lower and "todo:" not in text_lower:
        score += 4.0

    # Clamp al rango [5.0, 80.0]
    return max(5.0, min(80.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# Función principal de cálculo de perplexity
# ══════════════════════════════════════════════════════════════════════════════

def calculate_perplexity(code_text: str, language: str,
                         use_ollama: bool = True) -> tuple[float, str]:
    """
    Calcula perplexity usando la estrategia disponible.
    Retorna (score, metodo_usado).
    
    Si use_ollama=False, va directamente a heurística léxica (rápido).
    """
    if use_ollama:
        # Estrategia 1: logprobs nativos
        try:
            score = calculate_perplexity_ollama_logprobs(code_text)
            if score is not None:
                return score, "ollama_logprobs"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            log.debug("Ollama no disponible, usando heurística.")
            use_ollama = False  # deshabilitar para este archivo

    # Fallback directo a heurística léxica (offline)
    score = calculate_perplexity_heuristic(code_text, language)
    return score, "heuristic_lexical"


def _check_ollama_available() -> bool:
    """
    Verifica si Ollama está disponible Y responde rápido en generación.
    Hace un generate de prueba con num_predict=1 y timeout=OLLAMA_PING_TIMEOUT.
    Retorna True si viable (responde en tiempo), False si lento o no responde.
    """
    try:
        # Primero: verificar que el modelo está listado
        resp_tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp_tags.status_code != 200:
            return False
        models = [m["name"] for m in resp_tags.json().get("models", [])]
        if not any(PERPLEXITY_MODEL in m for m in models):
            log.warning("Modelo '%s' no en Ollama. Disponibles: %s", PERPLEXITY_MODEL, models)
            return False

        # Segundo: probar una generación rápida (1 token) para medir velocidad real
        probe_payload = {
            "model": PERPLEXITY_MODEL,
            "prompt": "x",
            "options": {"temperature": 0, "num_predict": 1},
            "stream": False,
        }
        resp_gen = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=probe_payload,
            timeout=OLLAMA_PING_TIMEOUT,
        )
        return resp_gen.status_code == 200
    except requests.exceptions.Timeout:
        log.info("Ollama responde lento (>%ds). Usando heurística léxica.", OLLAMA_PING_TIMEOUT)
        return False
    except Exception as e:
        log.debug("Ollama check error: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Función principal de la fase
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(force_reprocess: bool = False, mode: str = "auto") -> int:

    """
    Ejecuta la Fase 2: filtra archivos del corpus por perplexity.

    Args:
        force_reprocess: Si True, reprocesa archivos que ya tienen score != -1.0
        mode: "auto" = detecta Ollama al inicio; "offline" = solo heurística;
              "ollama" = fuerza uso de Ollama.

    Returns:
        Número de archivos confirmados como IA (is_ai_generated == True).
    """
    os.makedirs(DATA_FILTERED_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    if not os.path.exists(METADATA_CSV):
        log.error("registro_metadatos.csv no encontrado. Ejecuta Fase 1 primero.")
        return 0

    df = pd.read_csv(METADATA_CSV, dtype=str)
    df["perplexity_score"] = pd.to_numeric(df["perplexity_score"], errors="coerce").fillna(-1.0)

    # Seleccionar archivos a procesar
    if force_reprocess:
        mask = df["patch_status"] == "pending"
    else:
        mask = (df["patch_status"] == "pending") & (df["perplexity_score"].astype(float) == -1.0)

    pending = df[mask]
    total_pending = len(pending)

    if total_pending == 0:
        log.info("No hay archivos pendientes de procesar en Fase 2.")
        already_ai = df[df["is_ai_generated"].astype(str) == "True"].shape[0]
        return already_ai

    log.info("Procesando %d archivos del corpus...", total_pending)
    log.info("Threshold de perplexity: %.1f (score < threshold → IA)", PERPLEXITY_THRESHOLD)

    # Detectar disponibilidad de Ollama según mode
    if mode == "auto":
        log.info("Verificando Ollama (ping timeout=%ds)...", OLLAMA_PING_TIMEOUT)
        use_ollama = _check_ollama_available()
        if use_ollama:
            log.info("Ollama disponible. Estrategia: Ollama logprobs → heurística léxica.")
        else:
            log.info("Ollama no disponible. Usando heurística léxica offline (rápido).")
    elif mode == "ollama":
        use_ollama = True
        log.info("Modo: Ollama forzado.")
    else:  # offline
        use_ollama = False
        log.info("Modo: offline (heurística léxica).")


    method_counts: dict[str, int] = {}
    processed = 0
    errors    = 0

    # Procesar en chunks para guardar progreso frecuentemente
    SAVE_INTERVAL = 50

    for idx, row in tqdm(pending.iterrows(), total=total_pending, desc="Fase 2 — Perplexity"):
        corpus_path = os.path.join(DATA_CORPUS_DIR, str(row["filename"]))
        language    = str(row["language"])

        if not os.path.exists(corpus_path):
            log.warning("Archivo no encontrado: %s", corpus_path)
            errors += 1
            continue

        # Leer código
        try:
            with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read()
        except Exception as e:
            log.warning("No se pudo leer %s: %s", corpus_path, e)
            errors += 1
            continue

        # Calcular perplexity con reintentos
        score = -1.0
        method = "unknown"
        for attempt in range(MAX_RETRIES + 1):
            try:
                score, method = calculate_perplexity(code_text, language, use_ollama=use_ollama)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    log.debug("Reintento %d para %s: %s", attempt + 1, row["file_id"], e)
                    time.sleep(RETRY_DELAY)
                else:
                    log.warning("Fallido tras %d intentos: %s", MAX_RETRIES + 1, row["file_id"])
                    errors += 1

        if score == -1.0:
            continue

        # Clasificar
        is_ai = score < PERPLEXITY_THRESHOLD

        # Actualizar DataFrame
        df.at[idx, "perplexity_score"] = round(score, 4)
        df.at[idx, "is_ai_generated"]  = str(is_ai)

        # Copiar a filtered si es IA
        if is_ai:
            dest = os.path.join(DATA_FILTERED_DIR, str(row["filename"]))
            try:
                shutil.copy2(corpus_path, dest)
            except Exception as e:
                log.warning("No se pudo copiar a filtered: %s", e)

        method_counts[method] = method_counts.get(method, 0) + 1
        processed += 1

        # Guardar progreso periódicamente
        if processed % SAVE_INTERVAL == 0:
            df.to_csv(METADATA_CSV, index=False, encoding="utf-8")
            log.info("  Checkpoint guardado (%d/%d procesados)", processed, total_pending)

    # Guardar CSV final
    df.to_csv(METADATA_CSV, index=False, encoding="utf-8")

    # Resumen final
    total_ai  = df[df["is_ai_generated"].astype(str) == "True"].shape[0]
    total_hum = df[df["is_ai_generated"].astype(str) == "False"].shape[0]

    log.info("=" * 60)
    log.info("FASE 2 completada.")
    log.info("  Procesados esta ejecución : %d", processed)
    log.info("  Errores                   : %d", errors)
    log.info("  Total IA (filtered)       : %d", total_ai)
    log.info("  Total No-IA               : %d", total_hum)
    log.info("  Métodos usados            : %s", method_counts)
    log.info("  Threshold aplicado        : %.1f", PERPLEXITY_THRESHOLD)
    log.info("=" * 60)

    if total_ai == 0:
        log.warning("Criterio de aceptación NO cumplido: data/filtered/ está vacío.")
    else:
        log.info("✓ Criterio de aceptación CUMPLIDO.")

    return total_ai


# ── Entry point directo ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fase 2 — Perplexity Filter")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesa todos los archivos, incluso los ya evaluados.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "ollama", "offline"],
        default="auto",
        help="Estrategia: auto=detecta Ollama, ollama=solo LLM, offline=solo heurística (default: auto).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override del threshold de perplexity (default: valor en .env).",
    )
    args = parser.parse_args()

    if args.threshold is not None:
        globals()["PERPLEXITY_THRESHOLD"] = args.threshold
        log.info("Threshold sobreescrito a %.1f", args.threshold)

    n = run_phase2(force_reprocess=args.force, mode=args.mode)
    sys.exit(0 if n > 0 else 1)
