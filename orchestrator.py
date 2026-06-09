"""
Uso:
  python orchestrator.py                        # Ejecuta fases 1-6
  python orchestrator.py --from-phase 1 --to-phase 1 --skip-github  # Solo fase 1 sin GitHub

"""

import argparse
import logging
import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ORCHESTRATOR] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_pipeline(
    from_phase: int = 1,
    to_phase: int = 6,
    target_file_id: str | None = None,
    skip_github: bool = False,
    phase2_mode: str = "auto",
) -> None:
    """Ejecuta las fases del pipeline en orden."""

    # FASE 1
    if from_phase <= 1 <= to_phase:
        log.info("▶ FASE 1 — Corpus Builder")
        from src.phase1.corpus_builder import run_phase1
        n = run_phase1(skip_github=skip_github)
        log.info("◀ FASE 1 completada. %d archivos en corpus.", n)

    # FASE 2
    if from_phase <= 2 <= to_phase:
        log.info("▶ FASE 2 — Perplexity Filter (mode=%s)", phase2_mode)
        from src.phase2.perplexity_filter import run_phase2
        n = run_phase2(mode=phase2_mode)
        log.info("◀ FASE 2 completada. %d archivos confirmados IA.", n)

    # FASE 3
    if from_phase <= 3 <= to_phase:
        log.info("▶ FASE 3 — Vulnerability Analyzer")
        from src.phase3.vulnerability_analyzer import run_phase3
        run_phase3()
        log.info("◀ FASE 3 completada.")

    # FASES 4 + 5
    if from_phase <= 4 <= to_phase:
        import pandas as pd
        from src.shared.constants import METADATA_CSV
        from src.phase5.integrity_validator import validate_and_patch_loop

        log.info("▶ FASES 4+5 — Mitigación + Validación (bucle cerrado K≤2)")
        df = pd.read_csv(METADATA_CSV)
        mask = (
            (df["is_ai_generated"].astype(str) == "True") &
            (df["vulnerability_ids"].astype(str) != "") &
            (df["patch_status"] == "pending")
        )
        archivos = df[mask]
        if target_file_id:
            archivos = archivos[archivos["file_id"] == target_file_id]

        for _, row in archivos.iterrows():
            fid = row["file_id"]
            resultado = validate_and_patch_loop(fid)
            log.info("  [%s] → %s", fid, resultado)

        log.info("◀ FASES 4+5 completadas.")

    # FASE 6
    if from_phase <= 6 <= to_phase:
        log.info("▶ FASE 6 — Metrics Evaluator")
        from src.phase6.metrics_evaluator import run_phase6
        run_phase6()
        log.info("◀ FASE 6 completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RefactVulIA — Orquestador del pipeline de detección y refactorización."
    )
    parser.add_argument("--from-phase", type=int, default=1, metavar="N",
                        help="Fase inicial (1-6). Por defecto: 1.")
    parser.add_argument("--to-phase",   type=int, default=6, metavar="M",
                        help="Fase final (1-6). Por defecto: 6.")
    parser.add_argument("--file-id",    type=str, default=None, metavar="UUID",
                        help="Procesar solo un file_id específico (Fases 4-5).")
    parser.add_argument("--skip-github", action="store_true",
                        help="Omite la fuente GitHub en Fase 1 (más rápido para pruebas).")
    parser.add_argument("--phase2-mode", choices=["auto", "ollama", "offline"],
                        default="auto",
                        help="Estrategia Fase 2: auto|ollama|offline (default: auto).")
    args = parser.parse_args()

    run_pipeline(
        from_phase=args.from_phase,
        to_phase=args.to_phase,
        target_file_id=args.file_id,
        skip_github=args.skip_github,
        phase2_mode=args.phase2_mode,
    )
