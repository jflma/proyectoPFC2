"""
Tests para Fase 1 — Corpus Builder
=====================================
Archivo: tests/test_phase1.py
"""
import os
import sys
from pathlib import Path

import pytest

# Asegurar que el root del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.constants import DATA_CORPUS_DIR, METADATA_CSV


def test_metadata_csv_exists():
    """El CSV de metadatos debe existir tras correr Fase 1."""
    assert os.path.exists(METADATA_CSV), (
        f"registro_metadatos.csv no encontrado en {METADATA_CSV}. "
        "Ejecuta primero: python orchestrator.py --from-phase 1 --to-phase 1 --skip-github"
    )


def test_metadata_csv_min_rows():
    """El CSV debe tener al menos 50 filas (criterio de aceptación Fase 1)."""
    import pandas as pd
    if not os.path.exists(METADATA_CSV):
        pytest.skip("registro_metadatos.csv no existe aún.")
    df = pd.read_csv(METADATA_CSV)
    assert len(df) >= 50, f"Solo {len(df)} filas; se necesitan >=50."


def test_corpus_files_have_uuid_names():
    """Todos los archivos en data/corpus/ deben tener nombres UUID válidos."""
    import uuid
    if not os.path.exists(DATA_CORPUS_DIR):
        pytest.skip("data/corpus/ no existe aún.")
    files = [f for f in os.listdir(DATA_CORPUS_DIR) if not f.startswith(".")]
    if not files:
        pytest.skip("data/corpus/ está vacío.")
    for fname in files:
        stem = Path(fname).stem
        try:
            uuid.UUID(stem, version=4)
        except ValueError:
            pytest.fail(f"Archivo con nombre no-UUID encontrado: {fname}")


def test_metadata_fields():
    """El CSV debe contener todos los campos del contrato §2.1."""
    import pandas as pd
    if not os.path.exists(METADATA_CSV):
        pytest.skip("registro_metadatos.csv no existe aún.")
    df = pd.read_csv(METADATA_CSV)
    required_cols = [
        "file_id", "filename", "language", "source_type", "source_url",
        "original_prompt", "perplexity_score", "is_ai_generated",
        "vulnerability_ids", "cvss_max", "patch_status", "iterations_used", "final_cvss",
    ]
    for col in required_cols:
        assert col in df.columns, f"Columna faltante en CSV: '{col}'"


def test_language_values():
    """El campo 'language' solo debe contener python, c o cpp."""
    import pandas as pd
    if not os.path.exists(METADATA_CSV):
        pytest.skip("registro_metadatos.csv no existe aún.")
    df = pd.read_csv(METADATA_CSV)
    invalid = df[~df["language"].isin(["python", "c", "cpp"])]["language"].unique()
    assert len(invalid) == 0, f"Lenguajes inválidos en CSV: {invalid}"
