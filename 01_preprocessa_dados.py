"""
01_preprocessa_dados.py — Pré-processamento consolidado
Etapa 1: Calibração espectral (carrega .c64, calcula PSD médio, identifica bins do SSB)
Etapa 2: Geração do dataset (extrai PSDs, particiona treino/val/teste, calcula PCA)

Baseado em: Varotto et al. — 120 MHz, FFT 1024, Δf = 117,19 kHz/bin
Entrada: teste_lab/casoA/, teste_lab/casoB/, teste_lab/casoC/ (.c64 + .json)
Saída:
  - calibracao_psd.pdf (figura de calibração espectral)
  - dataset_completo.npz (PSD1024 + PCA 8/16/32/64 + labels)
"""
import numpy as np
import json
import os
import glob
import time
from scipy.signal import welch
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PARÂMETROS GLOBAIS (Varotto et al.)
# ============================================================
FS = 120e6                    # 120 MHz — taxa de amostragem do watchdog
N_FFT = 1024                  # Tamanho da FFT no Welch
JANELA_MS = 5                 # Janela de captura 5 ms
EPSILON = 1e-12               # Regularização log
DELTA_F_BIN = FS / N_FFT      # 117.187,5 Hz/bin ≈ 117,19 kHz

# SSB = 240 subportadoras × 30 kHz = 7,2 MHz → ~61 bins centrados em DC
# Após fftshift: DC = bin 512 → bins 482 a 542
SSB_BIN_START = 482
SSB_BIN_END = 542

# Diretório dos dados brutos
DIR_DADOS = "teste_lab/"

# Particionamento (Varotto et al.)
N_TREINO_POR_CENARIO = 2000
N_VAL_POR_CENARIO = 1000
N_TESTE_POR_CENARIO = 400
N_AMOSTRAS_POR_CENARIO = N_TREINO_POR_CENARIO + N_VAL_POR_CENARIO + N_TESTE_POR_CENARIO  # 3400

# Cores para figuras
COR_A = '#74c476'
COR_B = '#2c7fb8'
COR_C = '#d7301f'

plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def carregar_c64(arquivo_c64):
    """Carrega um arquivo .c64 (complex64) e seu .json de metadados."""
    arq_json = arquivo_c64 + ".json"
    if os.path.exists(arq_json):
        with open(arq_json, 'r') as f:
            meta = json.load(f)
        fs = float(meta.get('fs', FS))
    else:
        print(f"  [AVISO] JSON nao encontrado: {arq_json}, usando fs=120 MHz")
        meta = {}
        fs = FS
    iq = np.fromfile(arquivo_c64, dtype=np.complex64)
    return iq, fs, meta

def extrair_psd(iq_window, fs):
    """Calcula PSD via Welch a partir de amostras IQ complexas.
    Retorna vetor de 1024 bins em dB com fftshift (DC no centro)."""
    freqs, psd = welch(
        iq_window, fs=fs, window='hann',
        nperseg=N_FFT, noverlap=N_FFT // 2, nfft=N_FFT,
        return_onesided=False, scaling='density'
    )
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(psd + EPSILON)
    return psd_db

def calcular_psd_medio(iq, fs, n_janelas=50):
    """Calcula PSD médio de várias janelas de 5 ms."""
    n_janela = int(fs * JANELA_MS / 1000)
    psd_acumulado = np.zeros(N_FFT)
    n_disponivel = len(iq)
    step = max(1, (n_disponivel - n_janela) // max(1, n_janelas - 1))
    count = 0
    for i in range(n_janelas):
        inicio = i * step
        fim = inicio + n_janela
        if fim > n_disponivel:
            break
        janela = iq[inicio:fim]
        psd = extrair_psd(janela, fs)
        psd_acumulado += psd
        count += 1
    return psd_acumulado / max(count, 1)

def calcular_eixo_frequencia(fs, n_bins=1024):
    freqs = np.fft.fftshift(np.fft.fftfreq(n_bins, 1/fs))
    return freqs / 1e6

def carregar_cenario(dir_cenario, n_amostras=3400):
    """Carrega arquivos .c64, concatena, e extrai PSDs de janelas de 5 ms."""
    arquivos_c64 = sorted(glob.glob(os.path.join(dir_cenario, "*.c64")))
    print(f"  Arquivos encontrados: {len(arquivos_c64)}")
    if len(arquivos_c64) == 0:
        raise FileNotFoundError(f"Nenhum arquivo .c64 em {dir_cenario}")
    iq_total = []
    fs_eff = None
    for arq in arquivos_c64:
        print(f"  Carregando {os.path.basename(arq)}...")
        iq, fs, meta = carregar_c64(arq)
        if fs_eff is None:
            fs_eff = fs
        iq_total.append(iq)
        print(f"    {len(iq):,} amostras ({len(iq)/fs:.2f}s)")
    iq_concat = np.concatenate(iq_total)
    print(f"  Total: {len(iq_concat):,} amostras ({len(iq_concat)/fs_eff:.2f}s)")
    n_janela = int(fs_eff * JANELA_MS / 1000)
    print(f"  Janela de {JANELA_MS}ms = {n_janela:,} amostras")
    n_disponivel = len(iq_concat)
    if n_disponivel < n_janela:
        raise ValueError(f"Dados insuficientes: {n_disponivel} < {n_janela}")
    if n_amostras > 1:
        step = max(1, (n_disponivel - n_janela) // (n_amostras - 1))
    else:
        step = n_janela
    overlap_pct = (1 - step / n_janela) * 100
    print(f"  Step: {step:,} amostras ({step/fs_eff*1000:.2f}ms), overlap: {overlap_pct:.1f}%")
    psd_list = []
    for i in range(n_amostras):
        inicio = i * step
        fim = inicio + n_janela
        if fim > n_disponivel:
            break
        janela = iq_concat[inicio:fim]
        psd = extrair_psd(janela, fs_eff)
        psd_list.append(psd)
    psd_array = np.array(psd_list[:n_amostras])
    if len(psd_array) < n_amostras:
        print(f"  [AVISO] Apenas {len(psd_array)} amostras extraidas (solicitadas: {n_amostras})")
        while len(psd_array) < n_amostras:
            idx = np.random.randint(0, max(1, len(psd_array)))
            psd_array = np.vstack([psd_array, psd_array[idx:idx+1]])
    return psd_array

def particionar_cenario(psd_cenario, n_treino, n_val, n_teste):
    idx = np.random.permutation(len(psd_cenario))
    idx_treino = idx[:n_treino]
    idx_val = idx[n_treino:n_treino + n_val]
    idx_teste = idx[n_treino + n_val:n_treino + n_val + n_teste]
    return psd_cenario[idx_treino], psd_cenario[idx_val], psd_cenario[idx_teste]

def embaralhar(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ============================================================
# ETAPA 1: CALIBRAÇÃO ESPECTRAL
# ============================================================
print("=" * 60)
print("ETAPA 1: CALIBRACAO ESPECTRAL — 120 MHz (Varotto et al.)")
print("=" * 60)
print(f"  fs = {FS/1e6:.0f} MHz | FFT = {N_FFT} | df = {DELTA_F_BIN/1e3:.2f} kHz/bin")
print(f"  SSB = {SSB_BIN_END - SSB_BIN_START + 1} bins ({(SSB_BIN_END-SSB_BIN_START+1)*DELTA_F_BIN/1e6:.1f} MHz)")

cenarios_cal = {
    "A - Canal Ocioso": ("casoA/", COR_A),
    "B - Trafego Legitimo": ("casoB/", COR_B),
    "C - Ataque de Jamming": ("casoC/", COR_C),
}

psd_medios = {}
freqs_mhz = None

for nome_cenario, (subdir, cor) in cenarios_cal.items():
    dir_cenario = os.path.join(DIR_DADOS, subdir)
    arquivos = sorted(glob.glob(os.path.join(dir_cenario, "*.c64")))
    if len(arquivos) == 0:
        print(f"  Nenhum .c64 encontrado em {dir_cenario}")
        continue
    arq = arquivos[0]
    print(f"\n  {nome_cenario}: {os.path.basename(arq)}")
    iq, fs_orig, meta = carregar_c64(arq)
    print(f"    fs: {fs_orig/1e6:.2f} MHz, amostras: {len(iq):,}")
    psd_medio = calcular_psd_medio(iq, fs_orig, n_janelas=50)
    psd_medios[nome_cenario] = (psd_medio, cor)
    if freqs_mhz is None:
        freqs_mhz = calcular_eixo_frequencia(fs_orig, N_FFT)
    print(f"    PSD medio calculado — {len(psd_medio)} bins")

# Figura de calibração
print(f"\n  Gerando calibracao_psd.pdf...")
fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

ax1 = axes[0]
for nome_cenario, (psd, cor) in psd_medios.items():
    ax1.plot(freqs_mhz, psd, color=cor, linewidth=0.8, alpha=0.85, label=nome_cenario)
ssb_limite = 3.6
ax1.axvspan(-ssb_limite, ssb_limite, alpha=0.1, color='red',
            label=f'Regiao SSB (+/-{ssb_limite} MHz)')
ax1.set_xlabel('Frequencia relativa ao centro (MHz)')
ax1.set_ylabel('PSD (dB)')
ax1.set_title('Calibracao Espectral — 120 MHz, sem decimacao')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_xlim(freqs_mhz.min(), freqs_mhz.max())
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for nome_cenario, (psd, cor) in psd_medios.items():
    mask = (freqs_mhz >= -5) & (freqs_mhz <= 5)
    ax2.plot(freqs_mhz[mask], psd[mask], color=cor, linewidth=1.2, alpha=0.85, label=nome_cenario)
ax2.axvspan(-ssb_limite, ssb_limite, alpha=0.1, color='red')
ax2.set_xlabel('Frequencia relativa ao centro (MHz)')
ax2.set_ylabel('PSD (dB)')
ax2.set_title('Zoom — Regiao do SSB')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibracao_psd.pdf', format='pdf')
plt.close()
print("  calibracao_psd.pdf gerado")

# Análise de bins com maior diferença espectral
if "A - Canal Ocioso" in psd_medios and "C - Ataque de Jamming" in psd_medios:
    psd_a = psd_medios["A - Canal Ocioso"][0]
    psd_c = psd_medios["C - Ataque de Jamming"][0]
    delta = psd_c - psd_a
    top_bins = np.argsort(np.abs(delta))[::-1][:20]
    print(f"\n  Top-20 bins com MAIOR diferenca |C - A|:")
    print(f"  {'Bin':>6s} {'Freq (MHz)':>12s} {'Delta (dB)':>12s}")
    print(f"  {'-'*35}")
    for idx in np.sort(top_bins):
        print(f"  {idx:6d} {freqs_mhz[idx]:12.3f} {delta[idx]:12.2f}")

# ============================================================
# ETAPA 2: GERAÇÃO DO DATASET
# ============================================================
print(f"\n{'='*60}")
print("ETAPA 2: GERACAO DO DATASET — .c64 @ 120 MHz + Welch PSD")
print(f"{'='*60}")

np.random.seed(42)

print("\n  Cenario A (Canal Ocioso)...")
psd_A = carregar_cenario(os.path.join(DIR_DADOS, "casoA/"), N_AMOSTRAS_POR_CENARIO)
print(f"    {psd_A.shape[0]} espectrogramas extraidos")

print("\n  Cenario B (Trafego Legitimo)...")
psd_B = carregar_cenario(os.path.join(DIR_DADOS, "casoB/"), N_AMOSTRAS_POR_CENARIO)
print(f"    {psd_B.shape[0]} espectrogramas extraidos")

print("\n  Cenario C (Ataque de Jamming)...")
psd_C = carregar_cenario(os.path.join(DIR_DADOS, "casoC/"), N_AMOSTRAS_POR_CENARIO)
print(f"    {psd_C.shape[0]} espectrogramas extraidos")

assert psd_A.shape[0] == N_AMOSTRAS_POR_CENARIO
assert psd_B.shape[0] == N_AMOSTRAS_POR_CENARIO
assert psd_C.shape[0] == N_AMOSTRAS_POR_CENARIO

# Particionamento
A_treino, A_val, A_teste = particionar_cenario(psd_A, N_TREINO_POR_CENARIO, N_VAL_POR_CENARIO, N_TESTE_POR_CENARIO)
B_treino, B_val, B_teste = particionar_cenario(psd_B, N_TREINO_POR_CENARIO, N_VAL_POR_CENARIO, N_TESTE_POR_CENARIO)
C_treino, C_val, C_teste = particionar_cenario(psd_C, N_TREINO_POR_CENARIO, N_VAL_POR_CENARIO, N_TESTE_POR_CENARIO)

# Montar conjuntos finais
# Rótulos: 0 = legítimo (A+B), 1 = ataque (C)
X_train = np.vstack([A_treino, B_treino, C_treino])
y_train = np.concatenate([
    np.zeros(N_TREINO_POR_CENARIO),
    np.zeros(N_TREINO_POR_CENARIO),
    np.ones(N_TREINO_POR_CENARIO)
]).astype(int)

X_train_unsup = np.vstack([A_treino, B_treino])

X_val = np.vstack([A_val, B_val, C_val])
y_val = np.concatenate([
    np.zeros(N_VAL_POR_CENARIO),
    np.zeros(N_VAL_POR_CENARIO),
    np.ones(N_VAL_POR_CENARIO)
]).astype(int)

X_test = np.vstack([A_teste, B_teste, C_teste])
y_test = np.concatenate([
    np.zeros(N_TESTE_POR_CENARIO),
    np.zeros(N_TESTE_POR_CENARIO),
    np.ones(N_TESTE_POR_CENARIO)
]).astype(int)

# Embaralhar
X_train, y_train = embaralhar(X_train, y_train)
X_val, y_val = embaralhar(X_val, y_val)
X_test, y_test = embaralhar(X_test, y_test)

print(f"\n  DATASET FINAL — Varotto et al. (6000/3000/1200)")
print(f"  Treino:          {X_train.shape[0]} amostras")
print(f"    Legitimas (A+B): {np.sum(y_train==0)}")
print(f"    Ataque (C):      {np.sum(y_train==1)}")
print(f"  Treino (nao sup): {X_train_unsup.shape[0]} amostras (A+B)")
print(f"  Validacao:       {X_val.shape[0]} amostras")
print(f"  Teste:           {X_test.shape[0]} amostras")
print(f"    Legitimas (A+B): {np.sum(y_test==0)}")
print(f"    Ataque (C):      {np.sum(y_test==1)}")
print(f"  Total:           {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} amostras")
print(f"  Features:        {X_train.shape[1]} bins (PSD1024)")
print(f"  fs:              {FS/1e6:.0f} MHz | df_bin: {DELTA_F_BIN/1e3:.2f} kHz")

# Gerar variâncias PCA
print(f"\n  Gerando variacoes PCA...")
pca_configs = {"PC8": 8, "PC16": 16, "PC32": 32, "PC64": 64}
pca_data = {}
for nome, n_comp in pca_configs.items():
    pca = PCA(n_components=n_comp)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    var_exp = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    pca_data[nome] = {
        "X_train": X_train_pca, "X_val": X_val_pca, "X_test": X_test_pca,
        "variancia": var_exp
    }
    print(f"    {nome}: {n_comp} componentes, {var_exp*100:.1f}% variancia")

# Salvar
print(f"\n  Salvando dataset_completo.npz...")
np.savez_compressed(
    "dataset_completo.npz",
    X_train=X_train.astype(np.float32), y_train=y_train,
    X_train_unsup=X_train_unsup.astype(np.float32),
    X_val=X_val.astype(np.float32), y_val=y_val,
    X_test=X_test.astype(np.float32), y_test=y_test,
    X_train_PC8=pca_data["PC8"]["X_train"].astype(np.float32),
    X_train_PC16=pca_data["PC16"]["X_train"].astype(np.float32),
    X_train_PC32=pca_data["PC32"]["X_train"].astype(np.float32),
    X_train_PC64=pca_data["PC64"]["X_train"].astype(np.float32),
    X_val_PC8=pca_data["PC8"]["X_val"].astype(np.float32),
    X_val_PC16=pca_data["PC16"]["X_val"].astype(np.float32),
    X_val_PC32=pca_data["PC32"]["X_val"].astype(np.float32),
    X_val_PC64=pca_data["PC64"]["X_val"].astype(np.float32),
    X_test_PC8=pca_data["PC8"]["X_test"].astype(np.float32),
    X_test_PC16=pca_data["PC16"]["X_test"].astype(np.float32),
    X_test_PC32=pca_data["PC32"]["X_test"].astype(np.float32),
    X_test_PC64=pca_data["PC64"]["X_test"].astype(np.float32),
    var_PC8=pca_data["PC8"]["variancia"],
    var_PC16=pca_data["PC16"]["variancia"],
    var_PC32=pca_data["PC32"]["variancia"],
    var_PC64=pca_data["PC64"]["variancia"],
    N_FFT=N_FFT, FS=FS,
    SSB_BIN_START=SSB_BIN_START, SSB_BIN_END=SSB_BIN_END,
    DELTA_F_BIN=DELTA_F_BIN,
    N_TREINO=len(X_train), N_VAL=len(X_val), N_TESTE=len(X_test),
    particionamento="Varotto et al. — 6000 treino / 3000 val / 1200 teste",
    distribuicao="2000+2000+2000 treino | 1000+1000+1000 val | 400+400+400 teste",
)
print("  dataset_completo.npz salvo")
print(f"\n  Saida:")
print(f"    calibracao_psd.pdf")
print(f"    dataset_completo.npz")