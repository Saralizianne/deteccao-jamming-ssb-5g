"""
testar_modelos.py — VERSÃO NOVA METODOLOGIA
Testa 8 modelos × 5 representações = 40 configurações.
"""
import numpy as np
import pickle
import time
import os
from tensorflow.keras.models import load_model

# ============================================================
# CARREGAR DATASET
# ============================================================
print("Carregando dataset...")
data = np.load("dataset_completo.npz", allow_pickle=True)

y_test = data["y_test"]

REPS = ["psd1024", "pc8", "pc16", "pc32", "pc64"]

def get_test_data(rep):
    if rep == "psd1024":
        return data["X_test"]
    elif rep == "pc8":
        return data["X_test_PC8"]
    elif rep == "pc16":
        return data["X_test_PC16"]
    elif rep == "pc32":
        return data["X_test_PC32"]
    elif rep == "pc64":
        return data["X_test_PC64"]

def get_input_dim(rep):
    return {"psd1024": 1024, "pc8": 8, "pc16": 16, "pc32": 32, "pc64": 64}[rep]

# ============================================================
# FUNÇÃO: MAPEAR CLUSTERS K-MEANS
# ============================================================
def kmeans_predict(km_model, X):
    centroids = km_model.cluster_centers_
    energia = np.mean(centroids, axis=1)
    cluster_ataque = np.argmax(energia)
    clusters = km_model.predict(X)
    return (clusters == cluster_ataque).astype(int)

# ============================================================
# TESTAR 40 CONFIGURAÇÕES
# ============================================================
predicoes = {}
scores = {}
latencias = {}

for rep in REPS:
    print(f"\n{'='*50}")
    print(f"TESTANDO — {rep.upper()}")
    print(f"{'='*50}")

    X_te = get_test_data(rep)
    dim = get_input_dim(rep)

    # 1. CNN
    print(f"  CNN...")
    cnn = load_model(f"modelos/cnn_{rep}.h5", compile=False)
    prob = cnn.predict(X_te.reshape(-1, dim, 1), verbose=0).flatten()
    predicoes[f"CNN_{rep}"] = (prob > 0.5).astype(int)
    scores[f"CNN_{rep}"] = prob

    # 2. k-NN
    print(f"  k-NN...")
    with open(f"modelos/knn_{rep}.pkl", "rb") as f:
        knn = pickle.load(f)
    predicoes[f"kNN_{rep}"] = knn.predict(X_te)
    scores[f"kNN_{rep}"] = knn.predict_proba(X_te)[:, 1]

    # 3. SVM
    print(f"  SVM...")
    with open(f"modelos/svm_{rep}.pkl", "rb") as f:
        svm = pickle.load(f)
    predicoes[f"SVM_{rep}"] = svm.predict(X_te)
    scores[f"SVM_{rep}"] = svm.predict_proba(X_te)[:, 1]

    # 4. RF
    print(f"  RF...")
    with open(f"modelos/rf_{rep}.pkl", "rb") as f:
        rf = pickle.load(f)
    predicoes[f"RF_{rep}"] = rf.predict(X_te)
    scores[f"RF_{rep}"] = rf.predict_proba(X_te)[:, 1]

    # 5. Autoencoder
    print(f"  Autoencoder...")
    ae = load_model(f"modelos/ae_{rep}.h5", compile=False)
    with open(f"modelos/ae_tau_{rep}.pkl", "rb") as f:
        tau_ae = pickle.load(f)
    recon = ae.predict(X_te, verbose=0)
    errors = np.mean((X_te - recon) ** 2, axis=1)
    predicoes[f"AE_{rep}"] = (errors > tau_ae).astype(int)
    scores[f"AE_{rep}"] = errors

    # 6. LOF
    print(f"  LOF...")
    with open(f"modelos/lof_{rep}.pkl", "rb") as f:
        lof = pickle.load(f)
    with open(f"modelos/lof_tau_{rep}.pkl", "rb") as f:
        tau_lof = pickle.load(f)
    raw = lof.predict(X_te)
    predicoes[f"LOF_{rep}"] = (raw == -1).astype(int)
    scores[f"LOF_{rep}"] = -lof.score_samples(X_te)

    # 7. K-Means
    print(f"  K-Means...")
    with open(f"modelos/kmeans_{rep}.pkl", "rb") as f:
        km = pickle.load(f)
    predicoes[f"KMeans_{rep}"] = kmeans_predict(km, X_te)
    dist = km.transform(X_te)
    cluster_ataque = np.argmax(np.mean(km.cluster_centers_, axis=1))
    scores[f"KMeans_{rep}"] = dist[:, cluster_ataque]

    # 8. IForest
    print(f"  Isolation Forest...")
    with open(f"modelos/iforest_{rep}.pkl", "rb") as f:
        iforest = pickle.load(f)
    with open(f"modelos/iforest_tau_{rep}.pkl", "rb") as f:
        tau_if = pickle.load(f)
    raw_if = iforest.predict(X_te)
    predicoes[f"IForest_{rep}"] = (raw_if == -1).astype(int)
    scores[f"IForest_{rep}"] = -iforest.score_samples(X_te)

# ============================================================
# LATÊNCIAS (500 inferências por configuração)
# ============================================================
print(f"\n{'='*50}")
print("Medindo latências (500 inferências por configuração)...")
print(f"{'='*50}")

modelos_cache = {}

def medir_latencia(modelo, X, nome, rep, n=500):
    dim = get_input_dim(rep)
    latencias = []
    for i in range(n):
        amostra = X[i % len(X)]
        if nome in ("CNN", "AE"):
            inp = amostra.reshape(1, dim, 1) if nome == "CNN" else amostra.reshape(1, -1)
        else:
            inp = amostra.reshape(1, -1)
        t0 = time.perf_counter()
        if nome in ("CNN", "AE"):
            modelo.predict(inp, verbose=0)
        else:
            modelo.predict(inp)
        t1 = time.perf_counter()
        latencias.append((t1 - t0) * 1000)
    return float(np.mean(latencias))

for rep in REPS:
    X_te = get_test_data(rep)
    dim = get_input_dim(rep)

    for nome, arquivo, is_keras in [
        ("CNN", f"modelos/cnn_{rep}.h5", True),
        ("kNN", f"modelos/knn_{rep}.pkl", False),
        ("SVM", f"modelos/svm_{rep}.pkl", False),
        ("RF", f"modelos/rf_{rep}.pkl", False),
        ("AE", f"modelos/ae_{rep}.h5", True),
        ("LOF", f"modelos/lof_{rep}.pkl", False),
        ("KMeans", f"modelos/kmeans_{rep}.pkl", False),
        ("IForest", f"modelos/iforest_{rep}.pkl", False),
    ]:
        chave = f"{nome}_{rep}"
        print(f"  {chave}...", end=" ", flush=True)
        if is_keras:
           modelo = load_model(arquivo, compile=False)
        else:
            with open(arquivo, "rb") as f:
                modelo = pickle.load(f)
        lat = medir_latencia(modelo, X_te, nome, rep, n=500)
        latencias[chave] = lat
        print(f"{lat:.4f} ms")

# ============================================================
# SALVAR
# ============================================================
print(f"\nSalvando predicoes_modelos.pkl...")

with open("predicoes_modelos.pkl", "wb") as f:
    pickle.dump({
        "y_test": y_test,
        "predicoes": predicoes,
        "scores": scores,
        "latencias": latencias,
        "representacoes": REPS,
        "modelos": ["CNN", "kNN", "SVM", "RF", "AE", "LOF", "KMeans", "IForest"],
        "distribuicao_teste": {
            "legitimo": int(np.sum(y_test == 0)),
            "ataque": int(np.sum(y_test == 1)),
            "total": len(y_test)
        }
    }, f)

print(f"✅ predicoes_modelos.pkl salvo")
print(f"  Configurações testadas: {len(predicoes)}")
print(f"  Teste: {np.sum(y_test==0)} legítimas + {np.sum(y_test==1)} ataque = {len(y_test)} total")