"""
05_rf_feature_importance.py — Random Forest Spectral Feature Importance
Treina RF no PSD1024 e gera:
  - figuras/rf_feature_importance.pdf        (visão completa + cumulative)
  - figuras/rf_feature_importance_zoom.pdf   (zoom na região do jammer, top-5 anotados)
  - rf_feature_importance.json               (metadados + top-20 bins)
"""
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

os.makedirs("figuras", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ============================================================
# PARÂMETROS DE FREQUÊNCIA (5G NR n78)
# ============================================================
FREQ_CENTRAL_MHZ = 3750.0       # Centro do canal 5G n78
CANAL_BW_MHZ = 120.0            # Largura de canal do watchdog
N_BINS = 1024

BIN_RESOLUTION_MHZ = CANAL_BW_MHZ / N_BINS   # 0.11719 MHz/bin
FREQ_START_MHZ = FREQ_CENTRAL_MHZ - CANAL_BW_MHZ / 2   # 3690 MHz
FREQ_END_MHZ = FREQ_CENTRAL_MHZ + CANAL_BW_MHZ / 2     # 3810 MHz

# Jammer: SSB da operadora a 3709.92 MHz, BW = 7.2 MHz
JAMMER_CENTER_MHZ = 3709.92
JAMMER_BW_MHZ = 7.2
JAMMER_FREQ_START = JAMMER_CENTER_MHZ - JAMMER_BW_MHZ / 2   # 3706.32
JAMMER_FREQ_END = JAMMER_CENTER_MHZ + JAMMER_BW_MHZ / 2     # 3713.52

def bin_to_freq(bin_idx):
    return FREQ_START_MHZ + bin_idx * BIN_RESOLUTION_MHZ

def freq_to_bin(freq_mhz):
    return int(round((freq_mhz - FREQ_START_MHZ) / BIN_RESOLUTION_MHZ))

# ============================================================
# CARREGAR DADOS
# ============================================================
print("Carregando dataset_completo.npz...")
data = np.load("dataset_completo.npz", allow_pickle=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

n_features = X_train.shape[1]
print(f"  Features: {n_features} bins, Faixa: {FREQ_START_MHZ:.2f}-{FREQ_END_MHZ:.2f} MHz")

# ============================================================
# TREINAR RANDOM FOREST
# ============================================================
print("\nTreinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test) * 100
print(f"  Accuracy: {acc:.1f}%")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

top20_bins = indices[:20]
top20_freqs = np.array([bin_to_freq(b) for b in top20_bins])
top20_imps = importances[top20_bins]

# ============================================================
# FIGURA 1: VISÃO COMPLETA
# ============================================================
print("\nGerando rf_feature_importance.pdf...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

freqs = np.array([bin_to_freq(i) for i in range(n_features)])

ax1.plot(freqs, importances, color='#808080', linewidth=0.5, zorder=2)
ax1.scatter(top20_freqs, top20_imps, c='#2ca02c', s=25, zorder=3,
            label='Top 20 spectral features')
ax1.axvspan(JAMMER_FREQ_START, JAMMER_FREQ_END, alpha=0.15, color='#90EE90',
            label='Nominal jammer band')
ax1.axvline(x=JAMMER_CENTER_MHZ, color='#800020', linestyle='--', linewidth=1.5, alpha=0.8,
            label='Nominal jammer center')
ax1.axvline(x=FREQ_CENTRAL_MHZ, color='gray', linestyle=':', linewidth=1, alpha=0.4,
            label='5G center')

ax1.set_xlabel("Frequency (MHz)", fontsize=11)
ax1.set_ylabel("Feature Importance", fontsize=11)
ax1.set_title(f"Random Forest Spectral Feature Importance (n78, {CANAL_BW_MHZ:.0f} MHz)",
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.9,
           edgecolor='gray', fancybox=False, borderpad=0.5,
           handletextpad=0.3, labelspacing=0.3)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(FREQ_START_MHZ, FREQ_END_MHZ)
ax1.set_xticks(np.arange(FREQ_START_MHZ, FREQ_END_MHZ + 1, 10))
ax1.tick_params(axis='x', labelsize=8)
ax1.set_ylim(bottom=0)

importance_cumul = np.cumsum(importances[indices])
ax2.plot(np.arange(1, len(importance_cumul) + 1), importance_cumul,
         color='#d62728', linewidth=2)
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% importance')
n_80 = np.searchsorted(importance_cumul, 0.8) + 1
ax2.axvline(x=n_80, color='green', linestyle=':', alpha=0.7,
            label=f'{n_80} bins = 80% importance')
ax2.set_xlabel("Number of Bins (sorted by importance)", fontsize=11)
ax2.set_ylabel("Cumulative Importance", fontsize=11)
ax2.legend(fontsize=9, loc='lower right', frameon=True, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("figuras/rf_feature_importance.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# FIGURA 2: ZOOM — APENAS TOP-5 ANOTADOS
# ============================================================
print("Gerando rf_feature_importance_zoom.pdf...")

fig, ax = plt.subplots(figsize=(10, 6))

zoom_margin = 15.0
zoom_freq_start = JAMMER_FREQ_START - zoom_margin
zoom_freq_end = JAMMER_FREQ_END + zoom_margin
zoom_bin_start = max(0, freq_to_bin(zoom_freq_start))
zoom_bin_end = min(n_features, freq_to_bin(zoom_freq_end) + 1)

zoom_freqs = np.array([bin_to_freq(i) for i in range(zoom_bin_start, zoom_bin_end)])
zoom_imps = importances[zoom_bin_start:zoom_bin_end]

ax.plot(zoom_freqs, zoom_imps, color='#808080', linewidth=0.8, zorder=2)

mask_in_zoom = (top20_freqs >= zoom_freq_start) & (top20_freqs <= zoom_freq_end)
zoom_top_freqs = top20_freqs[mask_in_zoom]
zoom_top_imps = top20_imps[mask_in_zoom]
zoom_top_bins = top20_bins[mask_in_zoom]

ax.scatter(zoom_top_freqs, zoom_top_imps, c='#2ca02c', s=50, zorder=3,
           label='Top 20 spectral features')

ax.axvspan(JAMMER_FREQ_START, JAMMER_FREQ_END, alpha=0.15, color='#90EE90',
           label='Nominal jammer band')

ax.axvline(x=JAMMER_CENTER_MHZ, color='#800020', linestyle='--', linewidth=1.5, alpha=0.8,
           label='Nominal jammer center')

# =============== ANOTAR APENAS TOP-5 ===============
top5_local_idx = np.argsort(zoom_top_imps)[::-1][:5]
top5_local_idx = sorted(top5_local_idx, key=lambda i: zoom_top_freqs[i])

max_imp = max(zoom_imps)
freq_range = zoom_freq_end - zoom_freq_start

positions = [
    (-0.12, 1.20),
    (0.12, 1.20),
    (-0.08, 1.45),
    (0.08, 1.45),
    (0.0, 1.65),
]

for i, idx in enumerate(top5_local_idx):
    freq = zoom_top_freqs[idx]
    imp = zoom_top_imps[idx]
    binn = zoom_top_bins[idx]

    x_text = freq + freq_range * positions[i][0]
    y_text = max_imp * positions[i][1]

    ax.annotate(f'Bin {binn}',
                xy=(freq, imp),
                xytext=(x_text, y_text),
                fontsize=9, ha='center', va='bottom',
                color='#1a6b1a', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#2ca02c', alpha=0.95, linewidth=0.5),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=0.8,
                               connectionstyle='arc3,rad=0.15'))

ax.set_xlabel("Frequency (MHz)", fontsize=11)
ax.set_ylabel("Feature Importance", fontsize=11)
ax.set_title(f"Zoom: Spectral Feature Importance Near Jammer Band "
             f"({zoom_freq_start:.1f}-{zoom_freq_end:.1f} MHz)",
             fontsize=11, fontweight='bold')

ax.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.9,
          edgecolor='gray', fancybox=False, borderpad=0.5,
          handletextpad=0.3, labelspacing=0.3)

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(zoom_freq_start, zoom_freq_end)

zoom_tick_start = int(np.floor(zoom_freq_start / 2.5)) * 2.5
zoom_tick_end = int(np.ceil(zoom_freq_end / 2.5)) * 2.5
ax.set_xticks(np.arange(zoom_tick_start, zoom_tick_end + 0.1, 2.5))
ax.tick_params(axis='x', labelsize=8)
ax.set_ylim(bottom=0, top=max_imp * 1.8)

plt.tight_layout()
plt.savefig("figuras/rf_feature_importance_zoom.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# JSON
# ============================================================
print("Gerando rf_feature_importance.json...")

JAMMER_BIN_START = freq_to_bin(JAMMER_FREQ_START)
JAMMER_BIN_END = freq_to_bin(JAMMER_FREQ_END)

resumo = {
    "frequencia_central_5g_mhz": FREQ_CENTRAL_MHZ,
    "largura_canal_mhz": CANAL_BW_MHZ,
    "faixa_frequencias_mhz": {
        "inicio": round(FREQ_START_MHZ, 2),
        "fim": round(FREQ_END_MHZ, 2),
    },
    "resolucao_por_bin_khz": round(BIN_RESOLUTION_MHZ * 1000, 2),
    "n_features": int(n_features),
    "accuracy_test": round(acc, 2),
    "banda_jammer": {
        "freq_central_mhz": JAMMER_CENTER_MHZ,
        "largura_mhz": JAMMER_BW_MHZ,
        "freq_start_mhz": round(JAMMER_FREQ_START, 2),
        "freq_end_mhz": round(JAMMER_FREQ_END, 2),
        "bin_start": int(JAMMER_BIN_START),
        "bin_end": int(JAMMER_BIN_END),
        "n_bins": int(JAMMER_BIN_END - JAMMER_BIN_START),
        "soma_importancia": round(float(np.sum(importances[JAMMER_BIN_START:JAMMER_BIN_END])), 4),
    },
    "importancia_acumulada_80pct": int(n_80),
    "top_20_bins": [
        {
            "bin": int(indices[i]),
            "freq_mhz": round(bin_to_freq(indices[i]), 2),
            "importance": round(float(importances[indices[i]]), 6),
        }
        for i in range(20)
    ],
}

with open("rf_feature_importance.json", "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)
print("  OK")

print(f"\n  Banda jammer: {JAMMER_FREQ_START:.2f}-{JAMMER_FREQ_END:.2f} MHz (bins {JAMMER_BIN_START}-{JAMMER_BIN_END})")
print(f"  {n_80} bins = 80% da importancia")
print(f"  Top-5 bins anotados no zoom")