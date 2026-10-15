"""
06_ablation.py — Analise de Ablacao Consolidada
1. PCA: accuracy e speedup para 5 representacoes
2. AE Espaco Latente: PCA 2D + t-SNE 2D (variancias separadas)
3. CNN c/ vs s/ Dropout+BatchNorm
4. K-Means Cotovelo: elbow + silhouette + ARI + NMI
5. K-Means k=2 vs k=3

Gera:
  - figuras/ablation_pca.pdf
  - figuras/ae_latent_space.pdf
  - figuras/cnn_ablation_dropout_bn.pdf
  - figuras/kmeans_elbow.pdf
  - figuras/kmeans_silhouette.pdf
  - figuras/kmeans_k2_vs_k3.pdf
  - ablation_consolidada.json
"""
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score, accuracy_score,
                             confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("figuras", exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ============================================================
# CARREGAR DADOS
# ============================================================
print("Carregando dados...")
data = np.load("dataset_completo.npz", allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

mask_n = y_test == 0
mask_j = y_test == 1
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ============================================================
# PARTE 1: ABLACAO PCA
# ============================================================
print(f"\n{'='*60}")
print("PARTE 1: Ablacao PCA")
print(f"{'='*60}")

with open("benchmark_consolidado.json", "r") as f:
    bench = json.load(f)

MODELOS = ["CNN", "kNN", "SVM", "RF", "AE", "LOF", "KMeans", "IForest"]
MODELOS_LABEL = ["CNN", "k-NN", "SVM", "RF", "AE", "LOF", "K-Means", "IF"]
REPS = ["psd1024", "pc8", "pc16", "pc32", "pc64"]
REP_LABELS = ["PSD1024", "PCA8", "PCA16", "PCA32", "PCA64"]

results_pca = {}
for rep_label, rep_key in zip(REP_LABELS, REPS):
    accs = []
    lats = []
    for modelo in MODELOS:
        chave = f"{modelo}_{rep_key}"
        accs.append(bench[chave]["accuracy"] * 100)
        lats.append(bench[chave]["latency_ms"])
    results_pca[rep_label] = {
        "acc_mean": round(float(np.mean(accs)), 2),
        "acc_std": round(float(np.std(accs)), 2),
        "lat_mean": round(float(np.mean(lats)), 4),
        "accs": accs,
        "lats": lats,
    }

psd_lat = results_pca["PSD1024"]["lat_mean"]
for rep_label in REP_LABELS:
    results_pca[rep_label]["speedup"] = round(psd_lat / results_pca[rep_label]["lat_mean"], 2)

print("  Gerando ablation_pca.pdf...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(MODELOS))
width = 0.15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, rep_label in enumerate(REP_LABELS):
    offset = (i - 2) * width
    ax1.bar(x + offset, results_pca[rep_label]["accs"], width,
            label=rep_label, color=colors[i], alpha=0.85)

ax1.set_xlabel("Modelo", fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("Accuracy por Representacao", fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(MODELOS_LABEL, fontsize=8, rotation=30, ha='right')
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(30, 105)

speedups = [results_pca[r]["speedup"] for r in REP_LABELS]
bars = ax2.bar(REP_LABELS, speedups, color=colors, alpha=0.85)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (PSD1024)')
ax2.set_xlabel("Representacao", fontsize=11)
ax2.set_ylabel("Speedup (x)", fontsize=11)
ax2.set_title("Speedup Medio vs PSD1024", fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}x', ha='center', fontsize=9, fontweight='bold')

plt.suptitle("Ablacao: Reducao de Dimensionalidade (PCA)", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figuras/ablation_pca.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# PARTE 2: AE ESPACO LATENTE
# ============================================================
print(f"\n{'='*60}")
print("PARTE 2: AE Espaco Latente")
print(f"{'='*60}")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LATENT_DIM = 32
EPOCHS = 15
input_dim = X_train.shape[1]

print(f"  Treinando AE ({EPOCHS} epochs, latent_dim={LATENT_DIM})...")

encoder_input = keras.Input(shape=(input_dim,))
x = layers.Dense(512, activation='relu')(encoder_input)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
latent = layers.Dense(LATENT_DIM, activation='relu')(x)

decoder_input = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(128, activation='relu')(decoder_input)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
reconstructed = layers.Dense(input_dim, activation='linear')(x)

encoder = keras.Model(encoder_input, latent, name='encoder')
decoder = keras.Model(decoder_input, reconstructed, name='decoder')
ae = keras.Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')
ae.compile(optimizer='adam', loss='mse')

hist_ae = ae.fit(X_train, X_train, epochs=EPOCHS, batch_size=32,
                 shuffle=True, validation_data=(X_test, X_test), verbose=0)

X_test_latent = encoder.predict(X_test, verbose=0)
print(f"  Loss: {hist_ae.history['loss'][-1]:.6f}  Val: {hist_ae.history['val_loss'][-1]:.6f}")

pca_orig = PCA(n_components=2, random_state=42)
X_orig_2d = pca_orig.fit_transform(X_test)
var_orig = pca_orig.explained_variance_ratio_

pca_lat = PCA(n_components=2, random_state=42)
X_lat_2d = pca_lat.fit_transform(X_test_latent)
var_lat = pca_lat.explained_variance_ratio_

print(f"  Var Original: {var_orig[0]*100:.1f}%, {var_orig[1]*100:.1f}%")
print(f"  Var Latente:  {var_lat[0]*100:.1f}%, {var_lat[1]*100:.1f}%")

print("  Calculando t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne_2d = tsne.fit_transform(X_test_latent)

print("  Gerando ae_latent_space.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.scatter(X_orig_2d[mask_n, 0], X_orig_2d[mask_n, 1], c='#2ca02c', marker='o', s=15, alpha=0.5, label='Normal')
ax.scatter(X_orig_2d[mask_j, 0], X_orig_2d[mask_j, 1], c='#d62728', marker='^', s=15, alpha=0.5, label='Jamming')
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.set_title(f"PSD1024 (Original)\nVar: {var_orig[0]*100:.1f}%, {var_orig[1]*100:.1f}%", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(X_lat_2d[mask_n, 0], X_lat_2d[mask_n, 1], c='#2ca02c', marker='o', s=15, alpha=0.5, label='Normal')
ax.scatter(X_lat_2d[mask_j, 0], X_lat_2d[mask_j, 1], c='#d62728', marker='^', s=15, alpha=0.5, label='Jamming')
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.set_title(f"AE Latent Space (PCA 2D)\nVar: {var_lat[0]*100:.1f}%, {var_lat[1]*100:.1f}%", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(X_tsne_2d[mask_n, 0], X_tsne_2d[mask_n, 1], c='#2ca02c', marker='o', s=15, alpha=0.5, label='Normal')
ax.scatter(X_tsne_2d[mask_j, 0], X_tsne_2d[mask_j, 1], c='#d62728', marker='^', s=15, alpha=0.5, label='Jamming')
ax.set_xlabel("t-SNE Component 1", fontsize=10)
ax.set_ylabel("t-SNE Component 2", fontsize=10)
ax.set_title("AE Latent Space (t-SNE 2D)", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Comparacao de Espacos de Representacao: Original vs Autoencoder", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figuras/ae_latent_space.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# PARTE 3: CNN c/ vs s/ DROPOUT + BATCHNORM
# ============================================================
print(f"\n{'='*60}")
print("PARTE 3: CNN - Ablacao Dropout + BatchNorm")
print(f"{'='*60}")

# Normalizar antes da CNN (z-score)
mean_train = X_train.mean()
std_train = X_train.std()
X_train_norm = (X_train - mean_train) / (std_train + 1e-8)
X_test_norm = (X_test - mean_train) / (std_train + 1e-8)

X_train_cnn = X_train_norm.reshape(-1, input_dim, 1)
X_test_cnn = X_test_norm.reshape(-1, input_dim, 1)

def build_cnn(with_dropout_bn):
    inp = keras.Input(shape=(input_dim, 1))
    x = layers.Conv1D(32, kernel_size=7, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if with_dropout_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    if with_dropout_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    if with_dropout_bn:
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out, name=f"CNN_{'with' if with_dropout_bn else 'without'}_DBN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

results_cnn = {}

for config_name, with_dbn in [("CNN (com Dropout+BN)", True),
                               ("CNN (sem Dropout+BN)", False)]:
    print(f"\n  Treinando: {config_name}...")
    tf.keras.utils.set_random_seed(42)
    model = build_cnn(with_dbn)
    n_params = model.count_params()

    t0 = time.time()
    hist = model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test),
                     epochs=50, batch_size=32, shuffle=True, verbose=0)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred_prob = model.predict(X_test_cnn, verbose=0)
    latency_ms = (time.time() - t0) / len(X_test_cnn) * 1000
    y_pred = (y_pred_prob.flatten() > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results_cnn[config_name] = {
        "accuracy": round(acc, 2),
        "latency_ms": round(latency_ms, 4),
        "train_time_s": round(train_time, 2),
        "n_params": int(n_params),
        "loss_final": round(hist.history['loss'][-1], 6),
        "val_loss_final": round(hist.history['val_loss'][-1], 6),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "history_loss": [round(float(x), 4) for x in hist.history['loss']],
        "history_val_loss": [round(float(x), 4) for x in hist.history['val_loss']],
        "history_acc": [round(float(x), 4) for x in hist.history['accuracy']],
        "history_val_acc": [round(float(x), 4) for x in hist.history['val_accuracy']],
    }
    print(f"    Acc={acc:.1f}%  Lat={latency_ms:.3f}ms  Params={n_params}")

print("  Gerando cnn_ablation_dropout_bn.pdf...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
for config_name, color, ls in [("CNN (com Dropout+BN)", '#1f77b4', '-'),
                                  ("CNN (sem Dropout+BN)", '#d62728', '--')]:
    r = results_cnn[config_name]
    ax.plot(r["history_loss"], color=color, linestyle=ls, linewidth=1.5, label=f'{config_name} (train)')
    ax.plot(r["history_val_loss"], color=color, linestyle=ls, linewidth=1.5, alpha=0.5, label=f'{config_name} (val)')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Training/Validation Loss", fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[1]
for config_name, color, ls in [("CNN (com Dropout+BN)", '#1f77b4', '-'),
                                  ("CNN (sem Dropout+BN)", '#d62728', '--')]:
    r = results_cnn[config_name]
    ax.plot(r["history_acc"], color=color, linestyle=ls, linewidth=1.5, label=f'{config_name} (train)')
    ax.plot(r["history_val_acc"], color=color, linestyle=ls, linewidth=1.5, alpha=0.5, label=f'{config_name} (val)')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Training/Validation Accuracy", fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)

ax = axes[2]
configs_short = ["Com\nDropout+BN", "Sem\nDropout+BN"]
accs = [results_cnn["CNN (com Dropout+BN)"]["accuracy"],
        results_cnn["CNN (sem Dropout+BN)"]["accuracy"]]
lats = [results_cnn["CNN (com Dropout+BN)"]["latency_ms"],
        results_cnn["CNN (sem Dropout+BN)"]["latency_ms"]]

x_bar = np.arange(len(configs_short))
width = 0.35
bars1 = ax.bar(x_bar - width/2, accs, width, color=['#1f77b4', '#d62728'], label='Accuracy (%)')
ax.set_ylabel("Accuracy (%)", fontsize=11, color='#1f77b4')
ax.tick_params(axis='y', labelcolor='#1f77b4')
ax.set_ylim(95, 101)
ax.set_xticks(x_bar)
ax.set_xticklabels(configs_short, fontsize=10)

ax2 = ax.twinx()
bars2 = ax2.bar(x_bar + width/2, lats, width, color=['#ff7f0e', '#2ca02c'], alpha=0.7, label='Latency (ms)')
ax2.set_ylabel("Latency (ms)", fontsize=11, color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f77b4')
for bar, val in zip(bars2, lats):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#ff7f0e')

ax.set_title("Accuracy & Latency", fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.suptitle("CNN Ablation: Dropout + BatchNorm", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figuras/cnn_ablation_dropout_bn.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# PARTE 4: K-MEANS COTOVELO
# ============================================================
print(f"\n{'='*60}")
print("PARTE 4: K-Means - Analise de Cotovelo")
print(f"{'='*60}")

def kneedle(ks, inertias):
    x = np.array(ks, dtype=float)
    y = np.array(inertias, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    m = (y_norm[-1] - y_norm[0]) / (x_norm[-1] - x_norm[0] + 1e-12)
    b = y_norm[0] - m * x_norm[0]
    distancias = np.abs(y_norm - (m * x_norm + b)) / np.sqrt(m**2 + 1)
    idx = int(np.argmax(distancias))
    return idx + 1

k_range = range(1, 16)
inertias = []
silhouettes = []
aris = []
nmis = []

print("  Calculando K-Means k=1..15...")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_km = km.fit_predict(X_test)
    inertias.append(km.inertia_)
    if k >= 2:
        sil = silhouette_score(X_test, labels_km)
        ari = adjusted_rand_score(y_test, labels_km)
        nmi = normalized_mutual_info_score(y_test, labels_km)
        silhouettes.append(sil)
        aris.append(ari)
        nmis.append(nmi)
        print(f"    k={k:2d}  inertia={km.inertia_:.2f}  sil={sil:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")
    else:
        silhouettes.append(0)
        aris.append(0)
        nmis.append(0)
        print(f"    k={k:2d}  inertia={km.inertia_:.2f}")

cotovelo_k = kneedle(list(k_range), inertias)
melhor_k_sil = int(np.argmax(silhouettes[1:]) + 2)
melhor_k_ari = int(np.argmax(aris[1:]) + 2)
print(f"  Cotovelo (kneedle): k={cotovelo_k}")

print("  Gerando kmeans_elbow.pdf...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), inertias, 'o-', color='#1f77b4', linewidth=2, markersize=8,
        markeredgecolor='black', markeredgewidth=0.5)
ax.axvline(x=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='k=2 (benchmark)')
ax.axvline(x=cotovelo_k, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Cotovelo (kneedle) = k={cotovelo_k}')
ax.set_xlabel("Numero de Clusters (k)", fontsize=11)
ax.set_ylabel("Inertia (WCSS)", fontsize=11)
ax.set_title("Metodo do Cotovelo - K-Means", fontsize=12, fontweight='bold')
ax.set_xticks(list(k_range))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figuras/kmeans_elbow.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

print("  Gerando kmeans_silhouette.pdf...")
fig, ax1 = plt.subplots(figsize=(8, 5))
k_range_sil = list(range(2, 16))
color1 = '#2ca02c'
ax1.set_xlabel("Numero de Clusters (k)", fontsize=11)
ax1.set_ylabel("Silhouette Score", fontsize=11, color=color1)
ax1.plot(k_range_sil, silhouettes[1:], 'o-', color=color1, linewidth=2, markersize=8,
         markeredgecolor='black', markeredgewidth=0.5, label='Silhouette')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axvline(x=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='k=2 (benchmark)')
ax1.axvline(x=cotovelo_k, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'Cotovelo = k={cotovelo_k}')

ax2 = ax1.twinx()
color2 = '#d62728'
ax2.set_ylabel("ARI / NMI", fontsize=11, color=color2)
ax2.plot(k_range_sil, aris[1:], 's--', color=color2, linewidth=1.5, markersize=7,
         markeredgecolor='black', markeredgewidth=0.4, label='ARI')
ax2.plot(k_range_sil, nmis[1:], '^--', color='#ff7f0e', linewidth=1.5, markersize=7,
         markeredgecolor='black', markeredgewidth=0.4, label='NMI')
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
plt.title("Silhouette, ARI e NMI vs Numero de Clusters", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(list(k_range))
plt.tight_layout()
plt.savefig("figuras/kmeans_silhouette.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# PARTE 5: K-MEANS k=2 vs k=3
# ============================================================
print(f"\n{'='*60}")
print("PARTE 5: K-Means k=2 vs k=3")
print(f"{'='*60}")

results_kmeans = {}

for k in [2, 3]:
    print(f"\n  K-Means k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_km = km.fit_predict(X_test)

    if k == 2:
        cm_temp = confusion_matrix(y_test, labels_km, labels=[0, 1])
        if cm_temp[0, 0] < cm_temp[0, 1]:
            labels_mapped = 1 - labels_km
        else:
            labels_mapped = labels_km.copy()
        acc = accuracy_score(y_test, labels_mapped) * 100
        cm = confusion_matrix(y_test, labels_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results_kmeans[f"k={k}"] = {
            "accuracy": round(acc, 2),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }
    else:
        labels_mapped = np.zeros_like(labels_km)
        for c in range(k):
            mask_c = labels_km == c
            if np.sum(mask_c) > 0:
                majoritaria = int(np.argmax(np.bincount(y_test[mask_c])))
                labels_mapped[mask_c] = majoritaria
        acc = accuracy_score(y_test, labels_mapped) * 100
        cm = confusion_matrix(y_test, labels_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cluster_composition = []
        for c in range(k):
            mask_c = labels_km == c
            n_normal = int(np.sum(y_test[mask_c] == 0))
            n_jamming = int(np.sum(y_test[mask_c] == 1))
            cluster_composition.append({
                "cluster": c, "n_total": int(np.sum(mask_c)),
                "n_normal": n_normal, "n_jamming": n_jamming,
                "pct_normal": round(n_normal / max(1, np.sum(mask_c)) * 100, 1),
                "pct_jamming": round(n_jamming / max(1, np.sum(mask_c)) * 100, 1),
            })
        results_kmeans[f"k={k}"] = {
            "accuracy": round(acc, 2),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "cluster_composition": cluster_composition,
        }
    print(f"    Acc={acc:.1f}%  TN={tn} FP={fp} FN={fn} TP={tp}")

print("  Gerando kmeans_k2_vs_k3.pdf...")
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_test)

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = km2.fit_predict(X_test)
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_k3 = km3.fit_predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.scatter(X_2d[mask_n, 0], X_2d[mask_n, 1], c='#2ca02c', marker='o', s=15, alpha=0.5, label='Normal')
ax.scatter(X_2d[mask_j, 0], X_2d[mask_j, 1], c='#d62728', marker='^', s=15, alpha=0.5, label='Jamming')
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.set_title("Ground Truth", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
cores_k2 = ['#1f77b4', '#ff7f0e']
for c in range(2):
    mask_c = labels_k2 == c
    ax.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1], c=cores_k2[c], marker='o', s=15, alpha=0.5,
               label=f'Cluster {c} (n={np.sum(mask_c)})')
centroids_2d = pca_2d.transform(km2.cluster_centers_)
ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='X', s=150,
           edgecolors='white', linewidth=1.5, label='Centroides', zorder=5)
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.set_title(f"K-Means (k=2)\nAcc={results_kmeans['k=2']['accuracy']:.1f}%", fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2]
cores_k3 = ['#1f77b4', '#ff7f0e', '#2ca02c']
for c in range(3):
    mask_c = labels_k3 == c
    ax.scatter(X_2d[mask_c, 0], X_2d[mask_c, 1], c=cores_k3[c], marker='o', s=15, alpha=0.5,
               label=f'Cluster {c} (n={np.sum(mask_c)})')
centroids_2d_k3 = pca_2d.transform(km3.cluster_centers_)
ax.scatter(centroids_2d_k3[:, 0], centroids_2d_k3[:, 1], c='black', marker='X', s=150,
           edgecolors='white', linewidth=1.5, label='Centroides', zorder=5)
ax.set_xlabel("PCA Component 1", fontsize=10)
ax.set_ylabel("PCA Component 2", fontsize=10)
ax.set_title(f"K-Means (k={cotovelo_k}, cotovelo)\nAcc={results_kmeans['k=3']['accuracy']:.1f}%", fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("K-Means: k=2 vs k=3 - Analise de Clusters", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("figuras/kmeans_k2_vs_k3.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("  OK")

# ============================================================
# JSON CONSOLIDADO
# ============================================================
print(f"\nGerando ablation_consolidada.json...")

resumo = {
    "pca_ablation": {
        rep: {
            "acc_mean": results_pca[rep]["acc_mean"],
            "lat_mean": results_pca[rep]["lat_mean"],
            "speedup": results_pca[rep]["speedup"],
        }
        for rep in REP_LABELS
    },
    "ae_latent_space": {
        "latent_dim": LATENT_DIM,
        "epochs": EPOCHS,
        "loss_final": round(float(hist_ae.history['loss'][-1]), 6),
        "val_loss_final": round(float(hist_ae.history['val_loss'][-1]), 6),
        "pca_variance_original": [round(float(v), 4) for v in var_orig],
        "pca_variance_latent": [round(float(v), 4) for v in var_lat],
    },
    "cnn_ablation": results_cnn,
    "kmeans_cotovelo": {
        "cotovelo_k": cotovelo_k,
        "melhor_k_silhouette": melhor_k_sil,
        "melhor_k_ari": melhor_k_ari,
        "silhouette_k2": round(silhouettes[1], 4),
        "ari_k2": round(aris[1], 4),
        "nmi_k2": round(nmis[1], 4),
        "silhouette_k3": round(silhouettes[2], 4),
        "ari_k3": round(aris[2], 4),
        "nmi_k3": round(nmis[2], 4),
    },
    "kmeans_k2_vs_k3": results_kmeans,
}

with open("ablation_consolidada.json", "w", encoding="utf-8") as f:
    json.dump(resumo, f, indent=2, ensure_ascii=False)
print("  OK")

# ============================================================
# RESUMO
# ============================================================
print(f"\n{'='*60}")
print(f"ABLAÇÃO CONSOLIDADA - RESUMO")
print(f"{'='*60}")
print(f"\n  1. PCA Ablation:")
for rep in REP_LABELS:
    r = results_pca[rep]
    print(f"     {rep:<10s}  Acc={r['acc_mean']:.1f}%  Lat={r['lat_mean']:.3f}ms  Speedup={r['speedup']}x")
print(f"\n  2. AE Latent Space:")
print(f"     Var Original: {var_orig[0]*100:.1f}%, {var_orig[1]*100:.1f}%")
print(f"     Var Latente:  {var_lat[0]*100:.1f}%, {var_lat[1]*100:.1f}%")
print(f"\n  3. CNN Ablation:")
for k in results_cnn:
    r = results_cnn[k]
    print(f"     {k}: Acc={r['accuracy']:.1f}%  Lat={r['latency_ms']:.3f}ms  Params={r['n_params']}")
print(f"\n  4. K-Means Cotovelo: k={cotovelo_k}")
print(f"     Sil k=2: {silhouettes[1]:.4f}  ARI k=2: {aris[1]:.4f}")
print(f"     Sil k=3: {silhouettes[2]:.4f}  ARI k=3: {aris[2]:.4f}")
print(f"\n  5. K-Means k=2 vs k=3:")
for k in ["k=2", "k=3"]:
    r = results_kmeans[k]
    cm = r['confusion_matrix']
    print(f"     {k}: Acc={r['accuracy']:.1f}%  TN={cm['TN']} FP={cm['FP']} FN={cm['FN']} TP={cm['TP']}")
print(f"\n  Figuras geradas:")
print(f"    figuras/ablation_pca.pdf")
print(f"    figuras/ae_latent_space.pdf")
print(f"    figuras/cnn_ablation_dropout_bn.pdf")
print(f"    figuras/kmeans_elbow.pdf")
print(f"    figuras/kmeans_silhouette.pdf")
print(f"    figuras/kmeans_k2_vs_k3.pdf")
print(f"    ablation_consolidada.json")