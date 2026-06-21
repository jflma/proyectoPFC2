import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.constants import (
    DATA_GRAFICOS_DIR,
    METADATA_CSV,
    METADATA_DIR,
    METRICS_JSON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FASE-6] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _safe_float(val, default: float = -1.0) -> float:
    try:
        f = float(val)
        return f if f == f else default
    except (ValueError, TypeError):
        return default


def calcular_metricas(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"error": "No hay registros en el CSV."}

    by_language = df["language"].value_counts().to_dict()

    total_ai    = df[df["is_ai_generated"].astype(str) == "True"].shape[0]
    total_human = df[df["is_ai_generated"].astype(str) == "False"].shape[0]
    pending_cls = df[df["is_ai_generated"].astype(str) == "pending"].shape[0]
    tasa_ia     = round(total_ai / total * 100, 2) if total > 0 else 0.0

    df_ia          = df[df["is_ai_generated"].astype(str) == "True"]
    total_ia_count = len(df_ia)

    con_vulns = df_ia[
        df_ia["vulnerability_ids"].astype(str).str.strip().ne("") &
        df_ia["vulnerability_ids"].astype(str).ne("nan")
    ].shape[0]
    limpios  = df_ia[df_ia["patch_status"] == "clean"].shape[0]
    tasa_vul = round(con_vulns / total_ia_count * 100, 2) if total_ia_count > 0 else 0.0

    all_cwes: list[str] = []
    for ids in df_ia["vulnerability_ids"].astype(str):
        for cwe in ids.split("|"):
            cwe = cwe.strip()
            if cwe and cwe != "nan":
                all_cwes.append(cwe)
    cwe_distribution = dict(Counter(all_cwes).most_common())

    cvss_iniciales = [
        _safe_float(v) for v in df_ia["cvss_max"].tolist()
        if _safe_float(v) > 0
    ]
    cvss_inicial_promedio = round(sum(cvss_iniciales) / len(cvss_iniciales), 3) if cvss_iniciales else 0.0

    validated        = df[df["patch_status"] == "validated"].shape[0]
    failed           = df[df["patch_status"] == "failed"].shape[0]
    total_intentados = validated + failed
    tasa_exito       = round(validated / total_intentados * 100, 2) if total_intentados > 0 else 0.0

    df_validated = df[df["patch_status"] == "validated"]
    cvss_finales = [
        _safe_float(v) for v in df_validated["final_cvss"].tolist()
        if _safe_float(v) >= 0
    ]
    cvss_final_promedio = round(sum(cvss_finales) / len(cvss_finales), 3) if cvss_finales else 0.0
    reduccion_cvss      = round(cvss_inicial_promedio - cvss_final_promedio, 3)

    iter_counts = df_validated["iterations_used"].value_counts().to_dict()
    iter_counts = {str(k): int(v) for k, v in iter_counts.items()}
    avg_iters_raw = [
        _safe_float(v) for v in df_validated["iterations_used"].tolist()
        if _safe_float(v) > 0
    ]
    avg_iters = round(sum(avg_iters_raw) / len(avg_iters_raw), 2) if avg_iters_raw else 0.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "3.0",
        "corpus": {
            "total_archivos": total,
            "por_lenguaje": by_language,
        },
        "fase2_clasificacion": {
            "total_ia_detectados": total_ai,
            "total_humano": total_human,
            "pendientes": pending_cls,
            "tasa_deteccion_ia_pct": tasa_ia,
        },
        "fase3_vulnerabilidades": {
            "total_ia_analizados": total_ia_count,
            "con_vulnerabilidades": con_vulns,
            "sin_vulnerabilidades_clean": limpios,
            "tasa_vulnerabilidad_pct": tasa_vul,
            "cvss_inicial_promedio": cvss_inicial_promedio,
            "distribucion_cwes": cwe_distribution,
        },
        "fases45_mitigacion": {
            "total_intentados": total_intentados,
            "validados": validated,
            "fallidos": failed,
            "tasa_exito_pct": tasa_exito,
            "cvss_inicial_promedio": cvss_inicial_promedio,
            "cvss_final_promedio": cvss_final_promedio,
            "reduccion_cvss_promedio": reduccion_cvss,
            "iteraciones_promedio": avg_iters,
            "distribucion_iteraciones": iter_counts,
        },
    }


def generar_graficos(metricas: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib no disponible. Saltando generación de gráficos.")
        return

    os.makedirs(DATA_GRAFICOS_DIR, exist_ok=True)

    corpus_total  = metricas["corpus"]["total_archivos"]
    ia_detectados = metricas["fase2_clasificacion"]["total_ia_detectados"]
    con_vulns     = metricas["fase3_vulnerabilidades"]["con_vulnerabilidades"]
    validados     = metricas["fases45_mitigacion"]["validados"]

    fig, ax = plt.subplots(figsize=(9, 5))
    etapas  = ["Corpus\n(Fase 1)", "Confirmado IA\n(Fase 2)",
               "Con Vulns\n(Fase 3)", "Validado\n(Fase 5)"]
    valores = [corpus_total, ia_detectados, con_vulns, validados]
    colores = ["#4C6EF5", "#7950F2", "#F03E3E", "#2F9E44"]

    bars = ax.barh(etapas, valores, color=colores, edgecolor="white", height=0.5)
    ax.set_xlabel("Número de archivos", fontsize=11)
    ax.set_title("Embudo del Pipeline RefactVulIA", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, valores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(valores) * 1.15 if valores else 1)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_GRAFICOS_DIR, "embudo_pipeline.png"), dpi=150)
    plt.close()
    log.info("  embudo_pipeline.png")

    cwes = metricas["fase3_vulnerabilidades"]["distribucion_cwes"]
    if cwes:
        top_cwes = dict(sorted(cwes.items(), key=lambda x: x[1], reverse=True)[:10])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(list(top_cwes.keys()), list(top_cwes.values()),
                color="#F03E3E", edgecolor="white")
        ax.set_xlabel("Ocurrencias", fontsize=11)
        ax.set_title("Top 10 CWEs detectados (Fase 3)", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_GRAFICOS_DIR, "distribucion_cwes.png"), dpi=150)
        plt.close()
        log.info("  distribucion_cwes.png")

    cvss_ini = metricas["fases45_mitigacion"]["cvss_inicial_promedio"]
    cvss_fin = metricas["fases45_mitigacion"]["cvss_final_promedio"]
    if cvss_ini > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["CVSS Inicial\n(promedio)", "CVSS Final\n(promedio)"],
               [cvss_ini, cvss_fin],
               color=["#F03E3E", "#2F9E44"], edgecolor="white", width=0.4)
        ax.set_ylim(0, 10)
        ax.set_ylabel("Score CVSS", fontsize=11)
        ax.set_title("Reducción de CVSS tras mitigación", fontsize=13, fontweight="bold")
        for i, v in enumerate([cvss_ini, cvss_fin]):
            ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_GRAFICOS_DIR, "reduccion_cvss.png"), dpi=150)
        plt.close()
        log.info("  reduccion_cvss.png")

    validados = metricas["fases45_mitigacion"]["validados"]
    fallidos  = metricas["fases45_mitigacion"]["fallidos"]
    if validados + fallidos > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            [validados, fallidos],
            labels=["Validados", "Fallidos"],
            colors=["#2F9E44", "#F03E3E"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title("Resultado de Mitigación (Fases 4-5)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_GRAFICOS_DIR, "resultado_mitigacion.png"), dpi=150)
        plt.close()
        log.info("  resultado_mitigacion.png")


def run_phase6() -> dict:
    if not os.path.exists(METADATA_CSV):
        log.error("registro_metadatos.csv no encontrado.")
        return {}

    df = pd.read_csv(METADATA_CSV, dtype=str)
    if df.empty:
        log.warning("El CSV está vacio.")
        return {}

    log.info("Calculando métricas sobre %d registros", len(df))
    metricas = calcular_metricas(df)

    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    log.info("Métricas guardadas en: %s", METRICS_JSON)

    log.info("Generando gráficos en %s:", DATA_GRAFICOS_DIR)
    generar_graficos(metricas)

    f45 = metricas.get("fases45_mitigacion", {})
    log.info("=" * 60)
    log.info("FASE 6 completada — Resumen:")
    log.info("  Archivos en corpus          : %d", metricas["corpus"]["total_archivos"])
    log.info("  Confirmados IA              : %d (%.1f%%)",
             metricas["fase2_clasificacion"]["total_ia_detectados"],
             metricas["fase2_clasificacion"]["tasa_deteccion_ia_pct"])
    log.info("  Con vulnerabilidades        : %d (%.1f%%)",
             metricas["fase3_vulnerabilidades"]["con_vulnerabilidades"],
             metricas["fase3_vulnerabilidades"]["tasa_vulnerabilidad_pct"])
    log.info("  Validados / Fallidos        : %d / %d (%.1f%% éxito)",
             f45.get("validados", 0), f45.get("fallidos", 0),
             f45.get("tasa_exito_pct", 0.0))
    log.info("  CVSS: %.3f → %.3f  (Δ %.3f)",
             f45.get("cvss_inicial_promedio", 0),
             f45.get("cvss_final_promedio", 0),
             f45.get("reduccion_cvss_promedio", 0))
    log.info("=" * 60)

    return metricas


if __name__ == "__main__":
    metricas = run_phase6()
    sys.exit(0 if metricas else 1)
