# Experimental Analysis of Machine Learning Paradigms for SSB Jamming Detection in 5G Networks

This repository contains the software routines, experimental acquisition scripts, and dataset processing tools used in the paper **"Experimental Analysis of Machine Learning Paradigms for SSB Jamming Detection in 5G Networks"**.

The framework provides an end-to-end evaluation pipeline comparing supervised (1D CNN, k-NN, Linear SVM, Random Forest) and unsupervised / anomaly detection models (Autoencoder, Local Outlier Factor, Isolation Forest, K-means) against selective Synchronization Signal Block (SSB) jamming in 5G NR networks.

## Experimental Setup & Laboratory Testbed

The experimental dataset was acquired using an over-the-air (OTA) 5G Standalone (SA) testbed operating in **band n78 (3.75 GHz)** with a 100 MHz channel bandwidth:
- **gNodeB (Transmitter):** NI USRP X310 (UBX-160) running [srsRAN](https://github.com/srsran/srsRAN) (v23.10) integrated with an [Open5GS](https://open5gs.org/) (v2.6.6) core network.
- **User Equipment (Legitimate Receiver):** Commercial Telit Cinterion FN990A40 module (Qualcomm Snapdragon X65 modem-RF system).
- **Watchdog (Passive Monitoring Node):** NI USRP X310 dedicated to collecting raw complex IQ samples at $f_s = 120\text{ MS/s}$.
- **Jammer (Adversary):** NI USRP X410 generating band-limited selective interference over the SSB (7.2 MHz bandwidth, 240 subcarriers with $\Delta f = 30\text{ kHz}$) centered at 3.70992 GHz with 100% duty cycle.

## Dataset Structure

Data was captured under three operational conditions ($T_w = 5\text{ ms}$ observation windows, $6 \times 10^5$ complex samples per window):
- **Scenario A (Idle Channel):** Background environmental noise and spectral baseline (threat-free).
- **Scenario B (Legitimate Traffic):** Active gNB–UE data session without interference.
- **Scenario C (Selective Jamming):** Continuous interference injected across the SSB bandwidth.

The consolidated dataset contains **10,200 samples** (3,400 per scenario) transformed into 1024-point Welch Power Spectral Density (PSD1024) vectors and Principal Component Analysis (PCA) projections (PCA64, PCA32, PCA16, PCA8).


## Repository Structure & Script Descriptions

### 1. SDR Data Acquisition & Jamming Transmission
- `jammer_tx_ssb_noise.py`: Controls the NI USRP X410 to generate and transmit band-limited uniform noise aligned with the 5G SSB time-frequency resources.
- `watchdog_capture_classA.py`: Captures raw IQ samples for Scenario A (Idle Channel baseline).
- `watchdog_capture_classB.py`: Captures raw IQ samples for Scenario B (Legitimate 5G SA traffic).
- `watchdog_capture_classC.py`: Captures raw IQ samples for Scenario C (Active selective jamming attack).

### 2. Preprocessing, Training & Evaluation Pipeline
- `01_preprocessa_dados.py`: Processes raw complex binary `.c64` files, estimates logarithmic Welch PSD ($N=1024$), computes PCA projections (64, 32, 16, 8 components), and prepares stratified train/validation/test splits.
- `02_treina_modelos.py`: Trains the eight evaluated ML models (Supervised: CNN 1D, k-NN, Linear SVM, Random Forest; Unsupervised: Autoencoder, LOF, Isolation Forest, K-means).
- `03_testa_modelos.py`: Evaluates fitted models on the unseen test set (1,200 samples: 800 legitimate, 400 jamming).
- `04_metricas_completas.py`: Computes complete statistical and operational metrics ($P_{FA}$, $P_{MD}$, Accuracy, F1-Score, AUC-ROC, and per-sample inference latency).
- `05_rf_feature_importance.py`: Performs Random Forest Gini impurity interpretability analysis to identify dominant spectral bins.
- `06_ablation.py`: Executes ablation studies (e.g., 1D CNN architecture with/without Dropout and Batch Normalization, and Autoencoder latent space dimensions).
- `07_mcnemar.py`: Conducts paired McNemar statistical significance tests ($\alpha = 0.05$) across model predictions.
