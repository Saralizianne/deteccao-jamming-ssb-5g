"""
04_metricas_completas.py — Metricas completas + benchmark consolidado
Le: predicoes_modelos.pkl
Gera:
  - benchmark_consolidado.json      (para Script 06)
  - metricas_completas.json         (detalhado por modelo/representacao)
  - metricas_completas.csv
  - figuras/matriz_confusao_8modelos_psd1024.pdf
  - figuras/accuracy_8modelos.pdf
  - figuras/f1_8modelos.pdf
  - figuras/pfa_8modelos.pdf
  - figuras/pmd_8modelos.pdf
  - figuras/auc_roc_8modelos.pdf
  - figuras/latencia_8modelos.pdf
"""
import numpy as np
import pickle
import json
import csv
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

os.makedirs("figuras", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ============================================================
# CARREGAR PREDICOES
# ============================================================
print("Carregando predicoes_modelos.pkl...")

with open("predicoes_modelos.pkl", "rb") as f:
    dados = pickle.load(f)

y_test = dados["y_test"]
predicoes = dados["predicoes"]
scores = dados.get("scores", {})
latencias = dados.get("latencias", {})

MODELOS = ["CNN", "kNN", "SVM", "RF", "AE", "LOF", "KMeans", "IForest"]
MODELOS_LABEL = ["CNN", "k-NN", "SVM", "RF", "AE", "LOF", "K-Means", "IF"]
REPS = ["psd1024", "pc8", "pc16", "pc32", "pc64"]
REP_LABELS = ["PSD1024", "PCA8", "PCA16", "PCA32", "PCA64"]

print(f"  y_test: {len(y_test)} amostras ({np.sum(y_test==0)} normal, {np.sum(y_test==1)} jamming)")
print(f"  Configuracoes: {len(predicoes)}")

# ============================================================
# COMPUTAR TODAS AS METRICAS
# ============================================================
print("\nComputando metricas...")

benchmark = {}
metricas = {}

for modelo in MODELOS:
    metricas[modelo] = {}

    for rep in REPS:
        chave = f"{modelo}_{rep}"
        pred = predicoes[chave]
        y_score = scores.get(chave, pred.astype(float))

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        pmd = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        pdetect = 1.0 - pmd
        lat = latencias.get(chave, 0.0)

        try:
            auc_roc = roc_auc_score(y_test, y_score)
        except Exception:
            auc_roc = 0.0

        try:
            auc_pr = average_precision_score(y_test, y_score)
        except Exception:
            auc_pr = 0.0

        # benchmark_consolidado.json (formato plano para Script 06)
        benchmark[chave] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "far": float(pfa),
            "mdr": float(pmd),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "latency_ms": float(lat),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            }
        }

        # metricas_completas.json (formato detalhado)
        metricas[modelo][rep] = {
            "accuracy": round(acc * 100, 2),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc_roc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "pfa": round(pfa, 4),
            "pmd": round(pmd, 4),
            "p_detect": round(pdetect, 4),
            "latency_ms": round(lat, 4),
            "confusion_matrix": {
                "TN": int(tn), "FP": int(fp),
                "FN": int(fn), "TP": int(tp),
            },
        }

print("  Concluido!")

# ============================================================
# SALVAR benchmark_consolidado.json
# ============================================================
print("\nSalvando benchmark_consolidado.json...")

with open("benchmark_consolidado.json", "w", encoding="utf-8") as f:
    json.dump(benchmark, f, indent=2, ensure_ascii=False)
print("  OK")

# ============================================================
# SALVAR metricas_completas.json
# ============================================================
print("Salvando metricas_completas.json...")

with open("metricas_completas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)
print("  OK")

# ============================================================
# SALVAR metricas_completas.csv
# ============================================================
print("Salvando metricas_completas.csv...")

csv_rows = []
for modelo in MODELOS:
    for rep in REPS:
        r = metricas[modelo][rep]
        csv_rows.append({
            "Modelo": MODELOS_LABEL[MODELOS.index(modelo)],
            "Representacao": REP_LABELS[REPS.index(rep)],
            "Accuracy (%)": r["accuracy"],
            "F1-Score": r["f1_score"],
            "AUC-ROC": r["auc_roc"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "Pfa": r["pfa"],
            "Pmd": r["pmd"],
            "P_Detect": r["p_detect"],
            "Latencia (ms)": r["latency_ms"],
            "TN": r["confusion_matrix"]["TN"],
            "FP": r["confusion_matrix"]["FP"],
            "FN": r["confusion_matrix"]["FN"],
            "TP": r["confusion_matrix"]["TP"],
        })

with open("metricas_completas.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print("  OK")

# ============================================================
# FIGURA 1: MATRIZES DE CONFUSAO (8 modelos, PSD1024)
# ============================================================
print("\nGerando matriz_confusao_8modelos_psd1024.pdf...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes.flatten()

for idx, modelo in enumerate(MODELOS):
    ax = axes_flat[idx]
    r = metricas[modelo]["psd1024"]
    cm = np.array([[r["confusion_matrix"]["TN"], r["confusion_matrix"]["FP"]],
                   [r["confusion_matrix"]["FN"], r["confusion_matrix"]["TP"]]])

    im = ax.imshow(cm, cmap='Blues', aspect='equal', vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Jamming'], fontsize=9)
    ax.set_yticklabels(['Normal', 'Jamming'], fontsize=9)
    ax.set_xlabel('Predito', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f'{MODELOS_LABEL[idx]}\nAcc={r["accuracy"]:.1f}%  Pfa={r["pfa"]:.4f}  Pmd={r["pmd"]:.4f}',
                 fontsize=10, fontweight='bold')

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)

plt.suptitle('Matrizes de Confusao - PSD1024 (8 Modelos)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig("figuras/matriz_confusao_8modelos_psd1024.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# FIGURAS 2-7: UMA FIGURA POR METRICA
# ============================================================
print("\nGerando 6 figuras individuais por metrica...")

metricas_config = [
    ("accuracy", "Accuracy (%)", "accuracy_8modelos.pdf", False),
    ("f1_score", "F1-Score", "f1_8modelos.pdf", False),
    ("pfa", "Probabilidade de Falso Alarme (Pfa)", "pfa_8modelos.pdf", True),
    ("pmd", "Probabilidade de Deteccao Faltida (Pmd)", "pmd_8modelos.pdf", True),
    ("auc_roc", "AUC-ROC", "auc_roc_8modelos.pdf", False),
    ("latency_ms", "Latencia (ms)", "latencia_8modelos.pdf", True),
]

colors_rep = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for metric_key, ylabel, filename, use_log in metricas_config:
    print(f"  Gerando {filename}...")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(MODELOS))
    width = 0.15

    for i, rep_label in enumerate(REP_LABELS):
        vals = [metricas[modelo][REPS[i]][metric_key] for modelo in MODELOS]
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=rep_label,
               color=colors_rep[i], alpha=0.85)

    ax.set_xlabel("Modelo", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel} - 8 Modelos x 5 Representacoes", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELOS_LABEL, fontsize=9, rotation=30, ha='right')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1),
              framealpha=0.9, borderpad=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    if use_log:
        ax.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(f"figuras/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    OK")

# ============================================================
# RESUMO
# ============================================================
print(f"\n{'='*80}")
print(f"METRICAS COMPLETAS - RESUMO")
print(f"{'='*80}")

print(f"\n  --- PSD1024 ---")
print(f"  {'Modelo':<12s} {'Acc(%)':>7s} {'F1':>7s} {'AUC':>7s} {'Pfa':>8s} {'Pmd':>8s} {'Lat(ms)':>9s}")
print(f"  {'-'*65}")
for modelo in MODELOS:
    r = metricas[modelo]["psd1024"]
    label = MODELOS_LABEL[MODELOS.index(modelo)]
    print(f"  {label:<12s} {r['accuracy']:>7.1f} {r['f1_score']:>7.4f} {r['auc_roc']:>7.4f} {r['pfa']:>8.4f} {r['pmd']:>8.4f} {r['latency_ms']:>9.4f}")

print(f"\n  Arquivos gerados:")
print(f"    benchmark_consolidado.json")
print(f"    metricas_completas.json")
print(f"    metricas_completas.csv")
print(f"    figuras/matriz_confusao_8modelos_psd1024.pdf")
print(f"    figuras/accuracy_8modelos.pdf")
print(f"    figuras/f1_8modelos.pdf")
print(f"    figuras/pfa_8modelos.pdf")
print(f"    figuras/pmd_8modelos.pdf")
print(f"    figuras/auc_roc_8modelos.pdf")
print(f"    figuras/latencia_8modelos.pdf")