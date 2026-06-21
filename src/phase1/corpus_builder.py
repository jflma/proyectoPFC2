import logging
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_CORPUS_DIR,
    EXTENSION_MAP,
    GITHUB_TOKEN,
    METADATA_CSV,
    METADATA_DIR,
    PERPLEXITY_THRESHOLD,
    VALID_LANGUAGES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-1] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MIN_CODE_LENGTH = 100   
GITHUB_MAX_REPOS   = 100
GITHUB_MAX_FILES   = 50  
GITHUB_RATE_SLEEP  = 60  

VALID_EXT_MAP = {".py": "python", ".c": "c", ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp"}

GITHUB_LANGUAGES = ["Python", "C", "C++"]


def _calculate_perplexity_heuristic(code_text: str, language: str) -> float:
    lines = code_text.split("\n")
    non_empty = [l for l in lines if l.strip()]
    total_lines = max(len(non_empty), 1)
    text_lower = code_text.lower()

    score = 50.0  

    # 1. Ratio de comentarios 
    if language == "python":
        comment_lines = sum(1 for l in non_empty if l.strip().startswith("#"))
        docstring_count = code_text.count('"""') + code_text.count("'''")
    else:
        comment_lines = sum(1 for l in non_empty if "//" in l or "/*" in l)
        docstring_count = 0

    comment_ratio = comment_lines / total_lines
    if 0.15 <= comment_ratio <= 0.45:
        score -= 12.0
    elif comment_ratio > 0.45:
        score -= 8.0

    # 2. Docstrings 
    if docstring_count >= 2:
        score -= 8.0

    # 3. Vocabulario estilo IA en comentarios
    ai_style_words = [
        "efficiently", "optimized", "comprehensive", "robust", "ensure",
        "handles", "implementation", "returns", "raises", "note:", "example:",
        "todo:", "fixme:", "param", ":param", ":return", "args:", "returns:",
    ]
    ai_word_hits = sum(1 for w in ai_style_words if w in text_lower)
    score -= min(ai_word_hits * 2.0, 14.0)

    # 4. Longitud de identificadores descriptivos
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{4,})\b', code_text)
    if identifiers:
        avg_id_len = mean(len(i) for i in identifiers)
        if avg_id_len >= 8.0:
            score -= 8.0
        elif avg_id_len >= 6.0:
            score -= 4.0

    # 5. Type hints 
    # def saludar(nombre: str, edad: int) -> str:
    if language == "python":
        type_hint_count = len(re.findall(r'def \w+\([^)]*:\s*\w', code_text))
        if type_hint_count > 0:
            score -= min(type_hint_count * 3.0, 10.0)

    # 6. Manejo de errores explícito
    error_handling = (code_text.count("try:") + code_text.count("except ")
                      + code_text.count("catch(") + code_text.count("catch ("))
    if error_handling >= 2:
        score -= 6.0

    # 7. Código muy corto
    if total_lines < 10:
        score += 15.0

    # 8. Inconsistencia de indentación 
    indent_sizes = set()
    for l in non_empty:
        stripped = l.lstrip()
        if stripped and l != stripped:
            indent_size = len(l) - len(stripped)
            indent_sizes.add(indent_size % 4 if indent_size > 0 else 0)
    if len(indent_sizes) > 2:
        score += 8.0

    # 9. Variables de un carácter 
    single_char_vars = len(re.findall(r'\b([a-df-hj-np-tv-z])\b', code_text))
    if single_char_vars > total_lines * 0.3:
        score += 6.0

    return max(5.0, min(80.0, score))


def _make_record(file_id: str, language: str, source_url: str,
                 perplexity_score: float) -> dict:
    ext = EXTENSION_MAP[language]
    return {
        "file_id":           file_id,
        "filename":          f"{file_id}{ext}",
        "language":          language,
        "source_type":       "github",
        "source_url":        source_url,
        "perplexity_score":  round(perplexity_score, 4),
        "is_ai_generated":   "pending",   
        "vulnerability_ids": "",
        "cvss_max":          -1.0,
        "patch_status":      "pending",
        "iterations_used":   0,
        "final_cvss":        -1.0,
    }


def _save_code_file(content: str, language: str) -> str:
    """Guarda en data/corpus/{uuid}{ext}. Retorna el file_id."""
    file_id = str(uuid.uuid4())
    ext = EXTENSION_MAP[language]
    dest = os.path.join(DATA_CORPUS_DIR, f"{file_id}{ext}")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
    return file_id

def _mine_github(max_repos: int = GITHUB_MAX_REPOS,
                 max_files_per_repo: int = GITHUB_MAX_FILES) -> list[dict]:
    if not GITHUB_TOKEN:
        log.warning("GITHUB_TOKEN no configurado — omitiendo fuente GitHub.")
        return []

    try:
        from github import Github
        from github.GithubException import RateLimitExceededException, UnknownObjectException
    except ImportError:
        log.error("PyGithub no instalado. Ejecuta: pip install PyGithub")
        return []

    records: list[dict] = []
    g = Github(GITHUB_TOKEN)

    for lang in GITHUB_LANGUAGES:
        log.info("Minando repositorios recientes de lenguaje: %s", lang)
        try:
            # Búsqueda de repos recientes (ordenados por fecha de actualización)
            repos = g.search_repositories(
                query=f"language:{lang} is:public",
                sort="updated",
                order="desc",
            )

            repos_procesados = 0
            for repo in repos:
                if repos_procesados >= max_repos:
                    break
                try:
                    # Listar archivos del repo (solo raíz + subdirectorios de 1 nivel)
                    contents = repo.get_contents("")
                    archivos_candidatos = []
                    while contents:
                        item = contents.pop(0)
                        if item.type == "dir":
                            try:
                                contents.extend(repo.get_contents(item.path))
                            except Exception:
                                pass
                        elif item.type == "file":
                            ext = os.path.splitext(item.name.lower())[1]
                            if ext in VALID_EXT_MAP:
                                archivos_candidatos.append(item)

                    files_guardados = 0
                    for archivo in archivos_candidatos[:max_files_per_repo]:
                        try:
                            code = archivo.decoded_content.decode("utf-8", errors="replace")
                        except Exception:
                            continue

                        if len(code.strip()) < MIN_CODE_LENGTH:
                            continue

                        ext = os.path.splitext(archivo.name.lower())[1]
                        language = VALID_EXT_MAP[ext]

                        # FILTRO HEURÍSTICO EN MEMORIA
                        score = _calculate_perplexity_heuristic(code, language)
                        if score >= PERPLEXITY_THRESHOLD:
                            continue  # Descartar código humano sin guardar

                        # Solo si pasó el filtro → guardar a disco
                        file_id = _save_code_file(code.strip(), language)
                        records.append(
                            _make_record(
                                file_id=file_id,
                                language=language,
                                source_url=archivo.html_url or repo.html_url,
                                perplexity_score=score,
                            )
                        )
                        files_guardados += 1

                    log.debug("  %s → %d archivos candidatos IA guardados", repo.full_name, files_guardados)
                    repos_procesados += 1

                except RateLimitExceededException:
                    log.warning("Rate limit alcanzado. Esperando %ds...", GITHUB_RATE_SLEEP)
                    time.sleep(GITHUB_RATE_SLEEP)
                except UnknownObjectException:
                    log.debug("Repositorio inaccesible: %s", repo.full_name)
                except Exception as e:
                    log.debug("Error en repo %s: %s", repo.full_name, e)
                    continue

        except RateLimitExceededException:
            log.warning("Rate limit global. Saltando lenguaje '%s'.", lang)
        except Exception as e:
            log.error("Error buscando repos de %s: %s", lang, e)

    log.info("GitHub → %d archivos candidatos IA extraídos.", len(records))
    return records

def run_phase1(skip_github: bool = False) -> int:
    os.makedirs(DATA_CORPUS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    all_records: list[dict] = []

    if not skip_github:
        log.info("Minando repositorios de GitHub (lenguajes: %s)...", GITHUB_LANGUAGES)
        log.info("Threshold heurístico: %.1f (score < threshold → candidato IA)", PERPLEXITY_THRESHOLD)
        github_records = _mine_github()
        all_records.extend(github_records)
    else:
        log.info("Fuente GitHub omitida (skip_github=True).")

    if not all_records:
        log.error("No se encontró código candidato IA en ninguna fuente.")
        return 0

    # Si el CSV ya existe, añadir solo los nuevos 
    if os.path.exists(METADATA_CSV):
        df_existing = pd.read_csv(METADATA_CSV, dtype=str)
        existing_ids = set(df_existing["file_id"].tolist())
        new_records = [r for r in all_records if r["file_id"] not in existing_ids]
        if new_records:
            df_new = pd.DataFrame(new_records)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_existing
            log.info("No hay nuevos registros para añadir.")
    else:
        df_combined = pd.DataFrame(all_records)

    df_combined.to_csv(METADATA_CSV, index=False, encoding="utf-8")

    total = len(df_combined)
    new_added = len(all_records)

    log.info("=" * 60)
    log.info("FASE 1 completada.")
    log.info("  Total en registro_metadatos.csv : %d", total)
    log.info("  Archivos nuevos esta ejecución  : %d", new_added)
    log.info("  Corpus guardado en              : %s", DATA_CORPUS_DIR)
    log.info("  Threshold heurístico aplicado   : %.1f", PERPLEXITY_THRESHOLD)
    log.info("=" * 60)

    if total < 50:
        log.warning("Criterio de aceptación NO cumplido: se necesitan >=50 filas, hay %d.", total)
    else:
        log.info("- Criterio de aceptación CUMPLIDO (>=50 filas).")

    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fase 1 — Corpus Builder (Minería GitHub)")
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Omite la extracción desde GitHub API (para pruebas).",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=GITHUB_MAX_REPOS,
        help=f"Máximo de repositorios a procesar por lenguaje (default: {GITHUB_MAX_REPOS}).",
    )
    args = parser.parse_args()

    n = run_phase1(skip_github=args.skip_github)
    sys.exit(0 if n >= 50 else 1)
