"""
07_mcnemar.py — Teste de McNemar para comparação pareada de modelos
Usa predições do PSD1024 (representação principal).
Gera:
  - figuras/mcnemar_heatmap.pdf — matriz 8x8 de p-values
  - mcnemar_results.json — tabela completa
"""
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

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

MODELOS = ["CNN", "kNN", "SVM", "RF", "AE", "LOF", "KMeans", "IForest"]
MODELOS_LABEL = ["CNN", "k-NN", "SVM", "RF", "AE", "LOF", "K-Means", "IF"]
n = len(MODELOS)

# Usar apenas PSD1024
preds_psd = {}
for modelo in MODELOS:
    preds_psd[modelo] = predicoes[f"{modelo}_psd1024"]

print(f"  Modelos: {n}")
print(f"  Amostras: {len(y_test)}")

# ============================================================
# TESTE DE McNEMAR
# ============================================================
print("\nComputando teste de McNemar...")

p_values = np.zeros((n, n))
statistics = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            p_values[i, j] = 1.0
            statistics[i, j] = 0.0
            continue

        pred_i = preds_psd[MODELOS[i]]
        pred_j = preds_psd[MODELOS[j]]

        correct_i = (pred_i == y_test)
        correct_j = (pred_j == y_test)

        n01 = int(np.sum(~correct_i & correct_j))
        n10 = int(np.sum(correct_i & ~correct_j))

        tabela = [[0, n01], [n10, 0]]

        exact = (n01 + n10) < 25
        resultado = mcnemar(tabela, exact=exact, correction=True)

        p_values[i, j] = resultado.pvalue
        statistics[i, j] = resultado.statistic

print("  Concluido!")

# ============================================================
# FIGURA: HEATMAP DE P-VALUES
# ============================================================
print("\nGerando mcnemar_heatmap.pdf...")

fig, ax = plt.subplots(figsize=(9, 7.5))

mask = np.triu(np.ones_like(p_values, dtype=bool), k=1)
p_display = np.where(mask, np.nan, p_values)

im = ax.imshow(p_display, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='equal')

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(MODELOS_LABEL, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(MODELOS_LABEL, fontsize=9)

for i in range(n):
    for j in range(n):
        if i > j:
            val = p_values[i, j]
            color = 'white' if val < 0.5 else 'black'
            if val < 0.001:
                texto = f'{val:.1e}'
            elif val < 0.01:
                texto = f'{val:.3f}'
            else:
                texto = f'{val:.3f}'
            ax.text(j, i, texto, ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

for i in range(n):
    ax.plot([i, i], [i, n-1], color='gray', linewidth=0.3, alpha=0.5)
    ax.plot([0, i], [i, i], color='gray', linewidth=0.3, alpha=0.5)

ax.set_title("Teste de McNemar (p-values) — PSD1024\n"
             "Verde = diferenca significativa (p<0,05)\n"
             "Vermelho = sem diferenca significativa",
             fontsize=11, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('p-value', fontsize=10)

plt.tight_layout()
plt.savefig("figuras/mcnemar_heatmap.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# JSON
# ============================================================
print("Gerando mcnemar_results.json...")

resultados_lista = []
for i in range(n):
    for j in range(n):
        if i != j:
            resultados_lista.append({
                "modelo_1": MODELOS_LABEL[i],
                "modelo_2": MODELOS_LABEL[j],
                "statistic": round(float(statistics[i, j]), 4),
                "p_value": round(float(p_values[i, j]), 6),
                "significativo": bool(p_values[i, j] < 0.05),
            })

resumo = {
    "representacao": "PSD1024",
    "n_amostras": int(len(y_test)),
    "teste": "McNemar (statsmodels)",
    "nivel_significancia": 0.05,
    "resultados": resultados_lista,
    "matriz_pvalues": [
        [round(float(p_values[i, j]), 6) for j in range(n)]
        for i in range(n)
    ],
    "modelos": MODELOS_LABEL,
}

with open("mcnemar_results.json", "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)
print("  OK")

# ============================================================
# RESUMO
# ============================================================
print(f"\n{'='*60}")
print(f"TESTE DE McNEMAR — RESUMO")
print(f"{'='*60}")
print(f"  Representacao: PSD1024")
print(f"  Amostras: {len(y_test)}")
print(f"  Nivel de significancia: 0,05")

print(f"\n  Comparacoes significativas (p<0,05):")
sig_count = 0
for r in resultados_lista:
    if r["significativo"]:
        sig_count += 1
        print(f"    {r['modelo_1']:<8s} vs {r['modelo_2']:<8s}  p={r['p_value']:.4f}")

print(f"\n  Total: {sig_count} comparacoes significativas de {len(resultados_lista)}")

print(f"\n  Matriz de p-values (triangulo inferior):")
print(f"    {'':8s}", end="")
for label in MODELOS_LABEL:
    print(f"  {label:>8s}", end="")
print()
for i in range(n):
    print(f"    {MODELOS_LABEL[i]:8s}", end="")
    for j in range(n):
        if i > j:
            print(f"  {p_values[i,j]:8.4f}", end="")
        elif i == j:
            print(f"  {'1.0000':>8s}", end="")
        else:
            print(f"  {'---':>8s}", end="")
    print()

print(f"\n  Figuras geradas:")
print(f"    figuras/mcnemar_heatmap.pdf")
print(f"    mcnemar_results.json")