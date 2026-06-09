"""
visualize_phase2.py — Dashboard de resultados de la Fase 2
===========================================================
Uso:
  python visualize_phase2.py             # Muestra resumen en consola + genera gráficos
  python visualize_phase2.py --no-plots  # Solo consola, sin matplotlib
  python visualize_phase2.py --top 20   # Muestra top N archivos por perplexity score

Genera:
  data/phase2_distribution.png   — Histograma de distribución de scores
  data/phase2_by_language.png    — Score promedio por lenguaje
  data/phase2_summary.png        — Dashboard completo (4 paneles)
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.shared.constants import (
    DATA_FILTERED_DIR,
    METADATA_CSV,
    PERPLEXITY_THRESHOLD,
    ROOT_DIR,
)


CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BOLD  = "\033[1m"
RESET = "\033[0m"
DIM   = "\033[2m"

def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def bar(value: float, max_val: float, width: int = 30, char: str = "█") -> str:
    filled = int(round((value / max_val) * width)) if max_val > 0 else 0
    return char * filled + "░" * (width - filled)


def load_data() -> pd.DataFrame:
    if not os.path.exists(METADATA_CSV):
        print(colored("✗ registro_metadatos.csv no encontrado.", RED))
        print("  Ejecuta primero: python orchestrator.py --from-phase 1 --to-phase 2")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV, dtype=str)
    df["perplexity_score"] = pd.to_numeric(df["perplexity_score"], errors="coerce")
    df["is_ai_generated"]  = df["is_ai_generated"].map(
        lambda x: True if str(x).strip().lower() == "true"
        else (False if str(x).strip().lower() == "false" else None)
    )
    return df

def print_summary(df: pd.DataFrame, top_n: int = 15) -> None:
    processed = df[df["perplexity_score"].notna() & (df["perplexity_score"] != -1.0)]
    pending   = df[df["perplexity_score"].isna() | (df["perplexity_score"] == -1.0)]
    ai_files  = processed[processed["is_ai_generated"] == True]
    hum_files = processed[processed["is_ai_generated"] == False]

    total = len(df)
    n_proc = len(processed)
    n_ai   = len(ai_files)
    n_hum  = len(hum_files)
    n_pend = len(pending)

    filtered_count = len([
        f for f in os.listdir(DATA_FILTERED_DIR)
        if not f.startswith(".")
    ]) if os.path.exists(DATA_FILTERED_DIR) else 0

    print()
    print(colored("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(colored("║        FASE 2 — ATRIBUCIÓN POR PERPLEJIDAD CRUZADA      ║", CYAN))
    print(colored("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()

    print(colored("  RESUMEN GENERAL", BOLD))
    print(f"  {'Total corpus':<30} {colored(str(total), BOLD)}")
    print(f"  {'Procesados':<30} {colored(str(n_proc), CYAN)} ({n_proc/total*100:.1f}%)")
    print(f"  {'Pendientes (score=-1)':<30} {colored(str(n_pend), YELLOW)}")
    print(f"  {'Threshold aplicado':<30} {colored(str(PERPLEXITY_THRESHOLD), BOLD)}")
    print()

    print(colored("  CLASIFICACIÓN", BOLD))
    ai_pct  = n_ai  / n_proc * 100 if n_proc > 0 else 0
    hum_pct = n_hum / n_proc * 100 if n_proc > 0 else 0

    print(f"  {colored('● IA confirmados', GREEN):<40} {colored(str(n_ai), GREEN)} ({ai_pct:.1f}%)")
    print(f"    {bar(n_ai, n_proc, 40, '█')}")
    print()
    print(f"  {colored('● No-IA (humanos)', RED):<40} {colored(str(n_hum), RED)} ({hum_pct:.1f}%)")
    print(f"    {bar(n_hum, n_proc, 40, '░')}")
    print()
    print(f"  {'Archivos en data/filtered/':<30} {colored(str(filtered_count), GREEN)}")
    print()

    if n_proc > 0:
        print(colored("  DISTRIBUCIÓN DE SCORES", BOLD))
        scores = processed["perplexity_score"]
        print(f"  {'Minimo':<20} {scores.min():.4f}")
        print(f"  {'Maximo':<20} {scores.max():.4f}")
        print(f"  {'Media':<20} {scores.mean():.4f}")
        print(f"  {'Mediana':<20} {scores.median():.4f}")
        print(f"  {'Desv. estándar':<20} {scores.std():.4f}")
        print()

        print(f"  Score < {PERPLEXITY_THRESHOLD} (IA):   {colored(str(n_ai), GREEN)}")
        print(f"  Score >= {PERPLEXITY_THRESHOLD} (No-IA): {colored(str(n_hum), RED)}")
        print()

        print(colored("  DESGLOSE POR LENGUAJE", BOLD))
        for lang in ["python", "c", "cpp"]:
            lang_df = processed[processed["language"] == lang]
            if len(lang_df) == 0:
                continue
            lang_ai  = lang_df[lang_df["is_ai_generated"] == True]
            lang_hum = lang_df[lang_df["is_ai_generated"] == False]
            avg_score = lang_df["perplexity_score"].mean()
            print(f"  {lang.upper():<8}  total={len(lang_df):>4}  "
                  f"IA={colored(str(len(lang_ai)), GREEN):>6}  "
                  f"No-IA={colored(str(len(lang_hum)), RED):>6}  "
                  f"avg_score={avg_score:.2f}")
        print()

        if "source_type" in processed.columns:
            print(colored("  DESGLOSE POR FUENTE", BOLD))
            for src in processed["source_type"].unique():
                src_df = processed[processed["source_type"] == src]
                src_ai = src_df[src_df["is_ai_generated"] == True]
                avg    = src_df["perplexity_score"].mean()
                print(f"  {src:<15}  total={len(src_df):>4}  IA={len(src_ai):>4}  avg_score={avg:.2f}")
            print()

    # Top archivos con score más bajo (más IA)
    if n_ai > 0:
        print(colored(f"  TOP {top_n} ARCHIVOS MÁS PROBABLEMENTE IA (score más bajo)", BOLD))
        top = ai_files.nsmallest(top_n, "perplexity_score")[
            ["filename", "language", "perplexity_score", "source_type"]
        ]
        print(f"  {'filename':<42} {'lang':<8} {'score':>8}  {'fuente'}")
        print("  " + "─" * 72)
        for _, row in top.iterrows():
            fname = str(row["filename"])[:40]
            score = float(row["perplexity_score"])
            color = GREEN if score < PERPLEXITY_THRESHOLD * 0.5 else CYAN
            print(f"  {fname:<42} {row['language']:<8} {colored(f'{score:>8.4f}', color)}  {row['source_type']}")
        print()

    # Archivos con score más alto (más humanos)
    if n_hum > 0:
        print(colored(f"  TOP {min(top_n, n_hum)} ARCHIVOS MÁS PROBABLEMENTE HUMANOS (score más alto)", BOLD))
        top_hum = hum_files.nlargest(min(top_n, n_hum), "perplexity_score")[
            ["filename", "language", "perplexity_score", "source_type"]
        ]
        print(f"  {'filename':<42} {'lang':<8} {'score':>8}  {'fuente'}")
        print("  " + "─" * 72)
        for _, row in top_hum.iterrows():
            fname = str(row["filename"])[:40]
            score = float(row["perplexity_score"])
            print(f"  {fname:<42} {row['language']:<8} {colored(f'{score:>8.4f}', RED)}  {row['source_type']}")
        print()

    # Estado del criterio de aceptación
    print(colored("  CRITERIO DE ACEPTACIÓN (Fase 2)", BOLD))
    crit1 = n_pend == 0
    crit2 = filtered_count > 0
    print(f"  {'✓' if crit1 else '✗'} Todos los archivos tienen score != -1.0  "
          f"({'CUMPLIDO' if crit1 else f'PENDIENTE: {n_pend} sin procesar'})")
    print(f"  {'✓' if crit2 else '✗'} data/filtered/ no está vacío  "
          f"({'CUMPLIDO' if crit2 else 'FALLIDO: sin archivos'})")
    print()


#  Gráficos

def generate_plots(df: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print(colored("  matplotlib no instalado. Instala con: pip install matplotlib", YELLOW))
        return

    processed = df[df["perplexity_score"].notna() & (df["perplexity_score"] != -1.0)].copy()
    if len(processed) == 0:
        print(colored("  Sin datos procesados para graficar.", YELLOW))
        return

    processed["is_ai_generated"] = processed["is_ai_generated"].astype(str).map(
        lambda x: True if x.lower() == "true" else False
    )

    # Estilos
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#0f3460",
        "axes.labelcolor":  "#e0e0e0",
        "text.color":       "#e0e0e0",
        "xtick.color":      "#a0a0b0",
        "ytick.color":      "#a0a0b0",
        "grid.color":       "#0f3460",
        "grid.alpha":       0.5,
        "font.family":      "monospace",
    })

    COLOR_AI  = "#00d4aa"   # teal - IA
    COLOR_HUM = "#ff6b6b"   # rojo - humano
    COLOR_THR = "#ffd700"   # dorado - threshold

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("FASE 2 — Atribución por Perplejidad Cruzada",
                 fontsize=16, fontweight="bold", color="#e0e0e0", y=0.98)

    scores_ai  = processed[processed["is_ai_generated"] == True]["perplexity_score"]
    scores_hum = processed[processed["is_ai_generated"] == False]["perplexity_score"]
    all_scores = processed["perplexity_score"]

    # Panel 1: Histograma de distribución
    ax1 = axes[0, 0]
    bins = min(60, max(20, len(processed) // 10))
    clip_max = all_scores.quantile(0.97)
    scores_clip = all_scores.clip(upper=clip_max)

    ax1.hist(scores_ai.clip(upper=clip_max),  bins=bins, alpha=0.7,
             color=COLOR_AI,  label=f"IA ({len(scores_ai)})", edgecolor="none")
    ax1.hist(scores_hum.clip(upper=clip_max), bins=bins, alpha=0.6,
             color=COLOR_HUM, label=f"No-IA ({len(scores_hum)})", edgecolor="none")
    ax1.axvline(PERPLEXITY_THRESHOLD, color=COLOR_THR, linewidth=2.0,
                linestyle="--", label=f"Threshold ({PERPLEXITY_THRESHOLD})")
    ax1.set_title("Distribución de Perplexity Scores", pad=10)
    ax1.set_xlabel("Perplexity Score")
    ax1.set_ylabel("Cantidad de archivos")
    ax1.legend(loc="upper right", framealpha=0.3)
    ax1.grid(True, axis="y")
    ax1.set_xlim(left=0)

    # Panel 2: Pie chart de clasificación
    ax2 = axes[0, 1]
    sizes  = [len(scores_ai), len(scores_hum)]
    colors = [COLOR_AI, COLOR_HUM]
    labels = [f"IA\n{len(scores_ai)} ({len(scores_ai)/len(processed)*100:.1f}%)",
              f"No-IA\n{len(scores_hum)} ({len(scores_hum)/len(processed)*100:.1f}%)"]
    wedges, texts = ax2.pie(sizes, labels=labels, colors=colors, startangle=90,
                             wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2})
    for text in texts:
        text.set_color("#e0e0e0")
        text.set_fontsize(11)
    ax2.set_title("Clasificación del Corpus", pad=10)

    # Panel 3: Score promedio por lenguaje
    ax3 = axes[1, 0]
    lang_data = []
    for lang in ["python", "c", "cpp"]:
        sub = processed[processed["language"] == lang]
        if len(sub) > 0:
            lang_data.append({
                "lang":    lang.upper(),
                "avg_ai":  sub[sub["is_ai_generated"] == True]["perplexity_score"].mean() if len(sub[sub["is_ai_generated"] == True]) > 0 else 0,
                "avg_hum": sub[sub["is_ai_generated"] == False]["perplexity_score"].mean() if len(sub[sub["is_ai_generated"] == False]) > 0 else 0,
                "count":   len(sub),
            })

    if lang_data:
        import numpy as np
        x      = range(len(lang_data))
        width  = 0.35
        ai_avgs  = [d["avg_ai"]  for d in lang_data]
        hum_avgs = [d["avg_hum"] for d in lang_data]
        langs    = [d["lang"] for d in lang_data]

        bars1 = ax3.bar([i - width/2 for i in x], ai_avgs,  width,
                        label="IA",    color=COLOR_AI,  alpha=0.8, edgecolor="none")
        bars2 = ax3.bar([i + width/2 for i in x], hum_avgs, width,
                        label="No-IA", color=COLOR_HUM, alpha=0.8, edgecolor="none")
        ax3.axhline(PERPLEXITY_THRESHOLD, color=COLOR_THR, linewidth=1.5,
                    linestyle="--", label=f"Threshold ({PERPLEXITY_THRESHOLD})")
        ax3.set_xticks(list(x))
        ax3.set_xticklabels(langs)
        ax3.set_title("Score Promedio por Lenguaje", pad=10)
        ax3.set_ylabel("Perplexity Score")
        ax3.legend(framealpha=0.3)
        ax3.grid(True, axis="y")

    # Panel 4: Boxplot por lenguaje
    ax4 = axes[1, 1]
    box_data  = []
    box_labels = []
    box_colors = []
    for lang in ["python", "c", "cpp"]:
        sub_ai  = processed[(processed["language"] == lang) & (processed["is_ai_generated"] == True)]["perplexity_score"]
        sub_hum = processed[(processed["language"] == lang) & (processed["is_ai_generated"] == False)]["perplexity_score"]
        if len(sub_ai) > 0:
            box_data.append(sub_ai.clip(upper=clip_max).values)
            box_labels.append(f"{lang.upper()}\nIA")
            box_colors.append(COLOR_AI)
        if len(sub_hum) > 0:
            box_data.append(sub_hum.clip(upper=clip_max).values)
            box_labels.append(f"{lang.upper()}\nNo-IA")
            box_colors.append(COLOR_HUM)

    if box_data:
        bp = ax4.boxplot(box_data, patch_artist=True, notch=False,
                         medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color("#a0a0b0")
        ax4.set_xticklabels(box_labels, fontsize=9)
        ax4.axhline(PERPLEXITY_THRESHOLD, color=COLOR_THR, linewidth=1.5,
                    linestyle="--", label=f"Threshold ({PERPLEXITY_THRESHOLD})")
        ax4.set_title("Distribución por Lenguaje y Clasificación", pad=10)
        ax4.set_ylabel("Perplexity Score")
        ax4.legend(framealpha=0.3)
        ax4.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(ROOT_DIR, "data", "phase2_summary.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(colored(f"  ✓ Dashboard guardado en: {out_path}", GREEN))

    # ── Gráfico de distribución simple ────────────────────────────────────────
    fig2, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bins2 = min(80, max(20, len(processed) // 8))
    ax.hist(scores_ai.clip(upper=clip_max),  bins=bins2, alpha=0.8,
            color=COLOR_AI,  label=f"IA ({len(scores_ai)})", edgecolor="none")
    ax.hist(scores_hum.clip(upper=clip_max), bins=bins2, alpha=0.7,
            color=COLOR_HUM, label=f"Humano ({len(scores_hum)})", edgecolor="none")
    ax.axvline(PERPLEXITY_THRESHOLD, color=COLOR_THR, linewidth=2.5,
               linestyle="--", label=f"Threshold = {PERPLEXITY_THRESHOLD}")
    ax.set_title("Distribución de Perplexity Scores — Corpus IA vs Humano",
                 color="#e0e0e0", fontsize=13, pad=12)
    ax.set_xlabel("Perplexity Score", color="#a0a0b0")
    ax.set_ylabel("Archivos", color="#a0a0b0")
    ax.legend(framealpha=0.3, facecolor="#1a1a2e", edgecolor="#0f3460")
    ax.tick_params(colors="#a0a0b0")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    out2 = os.path.join(ROOT_DIR, "data", "phase2_distribution.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(colored(f"  ✓ Histograma guardado en: {out2}", GREEN))


# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualización de resultados Fase 2")
    parser.add_argument("--no-plots", action="store_true",
                        help="Solo muestra resumen en consola sin generar imágenes.")
    parser.add_argument("--top", type=int, default=15, metavar="N",
                        help="Top N archivos a mostrar (default: 15).")
    args = parser.parse_args()

    df = load_data()
    print_summary(df, top_n=args.top)

    if not args.no_plots:
        print(colored("  Generando gráficos...", CYAN))
        generate_plots(df)
    print()
