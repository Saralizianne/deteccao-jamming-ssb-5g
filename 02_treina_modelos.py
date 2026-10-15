"""
treinar_modelos.py — VERSÃO NOVA METODOLOGIA
Treina 8 modelos × 5 representações = 40 configurações.
Representações: PSD1024, PCA8, PCA16, PCA32, PCA64.
Modelos: CNN, k-NN, SVM, RF, Autoencoder, LOF, K-Means, Isolation Forest.
"""
import numpy as np
import pickle
import time
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization, AveragePooling1D, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.makedirs("modelos", exist_ok=True)

# ============================================================
# CARREGAR DATASET
# ============================================================
print("Carregando dataset...")
data = np.load("dataset_completo.npz", allow_pickle=True)

X_train = data["X_train"]
X_train_unsup = data["X_train_unsup"]
X_val = data["X_val"]
X_test = data["X_test"]
y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

print(f"  Treino (superv): {X_train.shape}")
print(f"  Treino (não sup): {X_train_unsup.shape}")
print(f"  Validação: {X_val.shape}")

# ============================================================
# FIT PCA E TRANSFORMAR X_train_unsup
# ============================================================
print("Ajustando PCA e transformando dados não supervisionados...")
pca_fitters = {}
X_train_unsup_pc = {}

for n in [8, 16, 32, 64]:
    pca = PCA(n_components=n)
    pca.fit(X_train)
    pca_fitters[n] = pca
    X_train_unsup_pc[n] = pca.transform(X_train_unsup).astype(np.float32)
    var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    print(f"  PCA{n}: {var*100:.1f}% variância explicada")

# ============================================================
# DICIONÁRIO DE REPRESENTAÇÕES
# ============================================================
REPS = ["psd1024", "pc8", "pc16", "pc32", "pc64"]

def get_train_data(rep):
    if rep == "psd1024":
        return X_train, X_train_unsup
    elif rep == "pc8":
        return data["X_train_PC8"], X_train_unsup_pc[8]
    elif rep == "pc16":
        return data["X_train_PC16"], X_train_unsup_pc[16]
    elif rep == "pc32":
        return data["X_train_PC32"], X_train_unsup_pc[32]
    elif rep == "pc64":
        return data["X_train_PC64"], X_train_unsup_pc[64]

def get_val_data(rep):
    if rep == "psd1024":
        return X_val
    elif rep == "pc8":
        return data["X_val_PC8"]
    elif rep == "pc16":
        return data["X_val_PC16"]
    elif rep == "pc32":
        return data["X_val_PC32"]
    elif rep == "pc64":
        return data["X_val_PC64"]

def get_input_dim(rep):
    return {"psd1024": 1024, "pc8": 8, "pc16": 16, "pc32": 32, "pc64": 64}[rep]

# ============================================================
# FUNÇÕES CONSTRUTORAS
# ============================================================
def build_cnn(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim, 1)))
    if input_dim >= 256:
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling1D(2))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling1D(2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(AveragePooling1D(2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
    elif input_dim >= 32:
        model.add(Conv1D(32, 3, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(AveragePooling1D(2))
        model.add(Conv1D(64, 3, padding='same', activation='relu'))
        model.add(AveragePooling1D(2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
    else:
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_autoencoder(input_dim):
    hidden = min(128, max(4, input_dim // 2))
    latent = min(32, max(2, input_dim // 4))
    inp = Input(shape=(input_dim,))
    enc = Dense(hidden, activation='relu')(inp)
    enc = Dense(latent, activation='relu')(enc)
    dec = Dense(hidden, activation='relu')(enc)
    out = Dense(input_dim, activation='linear')(dec)
    ae = Model(inp, out)
    ae.compile(optimizer='adam', loss='mse')
    return ae, latent

# ============================================================
# TREINAMENTO — 8 MODELOS × 5 REPRESENTAÇÕES
# ============================================================
total = len(REPS) * 8
contador = 0

for rep in REPS:
    print(f"\n{'='*60}")
    print(f"REPRESENTAÇÃO: {rep.upper()} (dim={get_input_dim(rep)})")
    print(f"{'='*60}")

    X_tr, X_tr_unsup = get_train_data(rep)
    X_vl = get_val_data(rep)
    dim = get_input_dim(rep)

    # --- 1. CNN ---
    contador += 1
    print(f"\n[{contador}/{total}] CNN — {rep}...")
    cnn = build_cnn(dim)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    t0 = time.time()
    hist = cnn.fit(
        X_tr.reshape(-1, dim, 1), y_train,
        validation_data=(X_vl.reshape(-1, dim, 1), y_val),
        epochs=100, batch_size=32, callbacks=[es, rlr], verbose=1
    )
    t1 = time.time()
    cnn.save(f"modelos/cnn_{rep}.h5")
    with open(f"modelos/cnn_history_{rep}.pkl", "wb") as f:
        pickle.dump(hist.history, f)
    print(f"  CNN {rep} treinada em {t1-t0:.1f}s, {len(hist.epoch)} épocas")

    # --- 2. k-NN ---
    contador += 1
    print(f"[{contador}/{total}] k-NN — {rep}...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    t1 = time.time()
    with open(f"modelos/knn_{rep}.pkl", "wb") as f:
        pickle.dump(knn, f)
    print(f"  k-NN {rep} treinado em {t1-t0:.1f}s, acc treino={knn.score(X_tr, y_train)*100:.2f}%")

    # --- 3. SVM ---
    contador += 1
    print(f"[{contador}/{total}] SVM — {rep}...")
    t0 = time.time()
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_tr, y_train)
    t1 = time.time()
    with open(f"modelos/svm_{rep}.pkl", "wb") as f:
        pickle.dump(svm, f)
    print(f"  SVM {rep} treinado em {t1-t0:.1f}s, acc treino={svm.score(X_tr, y_train)*100:.2f}%")

    # --- 4. Random Forest ---
    contador += 1
    print(f"[{contador}/{total}] RF — {rep}...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    t1 = time.time()
    with open(f"modelos/rf_{rep}.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"  RF {rep} treinado em {t1-t0:.1f}s, acc treino={rf.score(X_tr, y_train)*100:.2f}%")

    # --- 5. Autoencoder ---
    contador += 1
    print(f"[{contador}/{total}] Autoencoder — {rep}...")
    ae, latent_dim = build_autoencoder(dim)
    es_ae = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlr_ae = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    t0 = time.time()
    ae.fit(
        X_tr_unsup, X_tr_unsup,
        validation_data=(X_vl, X_vl),
        epochs=100, batch_size=32, callbacks=[es_ae, rlr_ae], verbose=1
    )
    t1 = time.time()
    ae.save(f"modelos/ae_{rep}.h5")
    # Calcular threshold (percentil 95 dos erros de reconstrução no treino não-sup)
    recon_train = ae.predict(X_tr_unsup, verbose=0)
    errors_train = np.mean((X_tr_unsup - recon_train) ** 2, axis=1)
    tau_ae = float(np.percentile(errors_train, 95))
    with open(f"modelos/ae_tau_{rep}.pkl", "wb") as f:
        pickle.dump(tau_ae, f)
    print(f"  AE {rep} treinado em {t1-t0:.1f}s, latent={latent_dim}, τ={tau_ae:.4f}")

    # --- 6. LOF ---
    contador += 1
    print(f"[{contador}/{total}] LOF — {rep}...")
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_tr_unsup)
    t1 = time.time()
    with open(f"modelos/lof_{rep}.pkl", "wb") as f:
        pickle.dump(lof, f)
    scores_train = -lof.score_samples(X_tr_unsup)
    tau_lof = float(np.percentile(scores_train, 95))
    with open(f"modelos/lof_tau_{rep}.pkl", "wb") as f:
        pickle.dump(tau_lof, f)
    print(f"  LOF {rep} treinado em {t1-t0:.1f}s, τ={tau_lof:.4f}")

    # --- 7. K-Means K=2 ---
    contador += 1
    print(f"[{contador}/{total}] K-Means K=2 — {rep}...")
    t0 = time.time()
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(X_tr_unsup)
    t1 = time.time()
    with open(f"modelos/kmeans_{rep}.pkl", "wb") as f:
        pickle.dump(km, f)
    print(f"  K-Means {rep} treinado em {t1-t0:.1f}s")

    # --- 8. Isolation Forest ---
    contador += 1
    print(f"[{contador}/{total}] Isolation Forest — {rep}...")
    t0 = time.time()
    iforest = IsolationForest(random_state=42, contamination='auto')
    iforest.fit(X_tr_unsup)
    t1 = time.time()
    with open(f"modelos/iforest_{rep}.pkl", "wb") as f:
        pickle.dump(iforest, f)
    scores_if = -iforest.score_samples(X_tr_unsup)
    tau_if = float(np.percentile(scores_if, 95))
    with open(f"modelos/iforest_tau_{rep}.pkl", "wb") as f:
        pickle.dump(tau_if, f)
    print(f"  IForest {rep} treinado em {t1-t0:.1f}s, τ={tau_if:.4f}")

# ============================================================
# RESUMO
# ============================================================
print(f"\n{'='*60}")
print(f"TREINAMENTO COMPLETO — 40 configurações (8 modelos × 5 reps)")
print(f"{'='*60}")

arquivos = sorted(os.listdir("modelos"))
total_size = sum(os.path.getsize(os.path.join("modelos", f)) for f in arquivos)
for f in arquivos:
    size = os.path.getsize(os.path.join("modelos", f))
    print(f"  {f:40s} {size/1024:.1f} KB")
print(f"  {'TOTAL':40s} {total_size/1024:.1f} KB ({total_size/(1024*1024):.1f} MB)")