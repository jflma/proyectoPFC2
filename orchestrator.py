import argparse
import logging
import sys
from pathlib import Path

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
) -> None:
    """Ejecuta las fases del pipeline en orden."""

    # FASE 1: Corpus Builder (Minería GitHub + Filtro Heurístico) 
    if from_phase <= 1 <= to_phase:
        log.info("▶ FASE 1 — Corpus Builder (Minería GitHub + Filtro Heurístico en Memoria)")
        from src.phase1.corpus_builder import run_phase1
        n = run_phase1(skip_github=skip_github)
        log.info("◀ FASE 1 completada. %d archivos candidatos IA en corpus.", n)

    # FASE 2: Perplexity Filter (PPL matemático con LLM logprobs) 
    if from_phase <= 2 <= to_phase:
        log.info("▶ FASE 2 — Filtro de Perplejidad Matemática PPL(Lᵢ) con LLM logprobs")
        from src.phase2.perplexity_filter import run_phase2
        n = run_phase2()
        log.info("◀ FASE 2 completada. %d archivos confirmados IA (PPL < threshold).", n)

    # FASE 3: Analizador de Vulnerabilidades (CodeQL + Semgrep)
    if from_phase <= 3 <= to_phase:
        log.info("▶ FASE 3 — Analizador de Flujo Arquitectónico (CodeQL + Semgrep)")
        from src.phase3.vulnerability_analyzer import run_phase3
        n = run_phase3()
        log.info("◀ FASE 3 completada. %d archivos con vulnerabilidades.", n)

    # FASES 4+5: Mitigación LMDF + Validación VIS (bucle cerrado K dinámico)
    if from_phase <= 4 <= to_phase:
        import pandas as pd
        from src.shared.constants import METADATA_CSV, K_BUFFER
        from src.phase5.integrity_validator import validate_and_patch_loop

        log.info("▶ FASES 4+5 — LMDF + VIS (bucle cerrado, K dinámico, buffer=%d)", K_BUFFER)

        df = pd.read_csv(METADATA_CSV)
        mask = (
            (df["is_ai_generated"].astype(str) == "True")
            & (df["vulnerability_ids"].astype(str).str.strip().ne(""))
            & (df["vulnerability_ids"].astype(str) != "nan")
            & (df["patch_status"] == "pending")
        )
        archivos = df[mask]

        if target_file_id:
            archivos = archivos[archivos["file_id"] == target_file_id]

        if archivos.empty:
            log.info("No hay archivos pendientes para Fases 4+5.")
        else:
            for _, row in archivos.iterrows():
                fid = row["file_id"]
                resultado = validate_and_patch_loop(fid)
                log.info("  [%s] → %s", fid, resultado)

        log.info("◀ FASES 4+5 completadas.")

    # FASE 6: Métricas y Gráficos
    if from_phase <= 6 <= to_phase:
        log.info("▶ FASE 6 — Evaluación de Impacto y Métricas")
        from src.phase6.metrics_evaluator import run_phase6
        run_phase6()
        log.info("◀ FASE 6 completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RefactVulIA — Orquestador del pipeline de detección y refactorización."
    )
    parser.add_argument(
        "--from-phase", type=int, default=1, metavar="N",
        help="Fase inicial (1-6). Por defecto: 1."
    )
    parser.add_argument(
        "--to-phase", type=int, default=6, metavar="M",
        help="Fase final (1-6). Por defecto: 6."
    )
    parser.add_argument(
        "--file-id", type=str, default=None, metavar="UUID",
        help="Procesar solo un file_id específico (Fases 4-5)."
    )
    parser.add_argument(
        "--skip-github", action="store_true",
        help="Omite la fuente GitHub en Fase 1 (para pruebas)."
    )
    args = parser.parse_args()

    run_pipeline(
        from_phase=args.from_phase,
        to_phase=args.to_phase,
        target_file_id=args.file_id,
        skip_github=args.skip_github,
    )
