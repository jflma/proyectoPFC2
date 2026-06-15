import os
import sys
import json
import shutil
import logging
import subprocess
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_FILTERED_DIR,
    DATA_PATCHED_DIR,
    DATA_REPORTS_DIR,
    DATA_VALIDATED_DIR,
    METADATA_CSV,
    EXTENSION_MAP,
    K_MAX,
)
from src.phase3.vulnerability_analyzer import analyze_file
from src.phase4.mitigation_agent import run_phase4_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-5] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def check_syntax(filepath: str, language: str) -> str | None:
    try:
        if language == "python":
            r = subprocess.run(
                [sys.executable, "-m", "py_compile", filepath],
                capture_output=True, text=True
            )
        elif language == "c":
            r = subprocess.run(
                ["gcc", "-fsyntax-only", "-Wall", filepath],
                capture_output=True, text=True
            )
        elif language == "cpp":
            r = subprocess.run(
                ["g++", "-fsyntax-only", "-Wall", filepath],
                capture_output=True, text=True
            )
        else:
            return None
        return r.stderr.strip() if r.returncode != 0 else None
    except FileNotFoundError as e:
        log.warning("Compilador no encontrado: %s", e)
        return None


def _mark_failed(df: pd.DataFrame, file_id: str):
    df.loc[df["file_id"] == file_id, "patch_status"] = "failed"
    df.to_csv(METADATA_CSV, index=False, encoding="utf-8")
    log.warning("Archivo %s marcado como FAILED.", file_id)


def validate_and_patch_loop(file_id: str) -> str:
    if not os.path.exists(METADATA_CSV):
        log.error("registro_metadatos.csv no encontrado.")
        return "failed"

    df = pd.read_csv(METADATA_CSV, dtype=str)
    row = df[df["file_id"] == file_id]
    if row.empty:
        log.warning("file_id %s no encontrado en CSV.", file_id)
        return "failed"

    language = str(row.iloc[0]["language"])
    ext = EXTENSION_MAP.get(language, ".py")
    vuln_ids_raw = str(row.iloc[0].get("vulnerability_ids", ""))
    cwes_originales = set(v for v in vuln_ids_raw.split("|") if v.strip())

    os.makedirs(DATA_VALIDATED_DIR, exist_ok=True)
    os.makedirs(DATA_PATCHED_DIR, exist_ok=True)

    for k in range(1, K_MAX + 1):
        log.info("--- %s: iteracion %d/%d ---", file_id, k, K_MAX)

        ok = run_phase4_single(file_id, k)
        if not ok:
            log.warning("Fase 4 no pudo generar parche en iter %d.", k)
            _mark_failed(df, file_id)
            return "failed"

        candidato = os.path.join(DATA_PATCHED_DIR, f"{file_id}_v{k}{ext}")

        # Verificar sintaxis
        err = check_syntax(candidato, language)
        if err:
            log.warning("Error de sintaxis iter %d:\n%s", k, err[:300])
            if k == K_MAX:
                _mark_failed(df, file_id)
                return "failed"
            continue

        # Re-escanear vulnerabilidades
        report_nuevo = analyze_file(file_id, candidato, language)
        cwes_restantes = {v["cwe_id"] for v in report_nuevo.get("vulnerabilities", [])}

        # Comprueba si las CWEs originales fueron eliminadas
        persistentes = cwes_originales & cwes_restantes
        if not persistentes:
            # APROBADO
            dest = os.path.join(DATA_VALIDATED_DIR, f"{file_id}{ext}")
            shutil.copy2(candidato, dest)

            df.loc[df["file_id"] == file_id, "patch_status"] = "validated"
            df.loc[df["file_id"] == file_id, "iterations_used"] = str(k)
            cvss_final = report_nuevo.get("max_cvss", -1.0)
            df.loc[df["file_id"] == file_id, "final_cvss"] = str(cvss_final)
            df.to_csv(METADATA_CSV, index=False, encoding="utf-8")

            log.info("VALIDADO: %s tras %d iteracion(es). CVSS final: %.2f",
                     file_id, k, cvss_final)
            return "validated"

        log.info("CWEs persistentes tras iter %d: %s. Reintentando...", k, persistentes)
        # Actualizar reporte para la siguiente iteracion con las vulns restantes
        report_path = os.path.join(DATA_REPORTS_DIR, f"{file_id}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_nuevo, f, indent=2)

    _mark_failed(df, file_id)
    return "failed"


def run_phase5() -> dict:
    if not os.path.exists(METADATA_CSV):
        log.error("registro_metadatos.csv no encontrado.")
        return {}

    df = pd.read_csv(METADATA_CSV, dtype=str)
    # Solo procesar archivos con vulnerabilidades y aun en pending
    mask = (
        (df["is_ai_generated"].astype(str) == "True") &
        (df["vulnerability_ids"].astype(str).str.strip().ne("")) &
        (df["vulnerability_ids"].astype(str) != "nan") &
        (df["patch_status"] == "pending")
    )
    targets = df[mask]

    if targets.empty:
        log.info("No hay archivos pendientes con vulnerabilidades para Fase 5.")
        return {}

    log.info("Iniciando bucle Fase 4/5 para %d archivos...", len(targets))

    resultados = {}
    for _, row in targets.iterrows():
        fid = row["file_id"]
        result = validate_and_patch_loop(fid)
        resultados[fid] = result
        log.info("[%s] → %s", fid, result)

    validated = sum(1 for v in resultados.values() if v == "validated")
    failed = sum(1 for v in resultados.values() if v == "failed")
    log.info("=" * 60)
    log.info("FASE 5 completada. Validados: %d | Fallidos: %d", validated, failed)
    log.info("=" * 60)
    return resultados


if __name__ == "__main__":
    run_phase5()
