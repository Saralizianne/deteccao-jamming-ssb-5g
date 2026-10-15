#!/usr/bin/env python3
"""
watchdog_capture_classC.py

Classe C: Rede ativa com JAMMER (captura quando o operador ativou o jammer manualmente).
**AVISO DE SEGURANÇA**: Este script NÃO liga, ajusta, nem controla o jammer. Ele assume que
o jammer foi ativado manualmente por pessoal autorizado e apenas registra nos metadados os
parâmetros informados pelo usuário (gain, potência medida, etc.).

Funcionalidades:
- Captura IQ do USRP (watchdog) e salva em "teste_lab/"
- Inclui campos de metadados específicos do jammer (gain_setting, measured_dbm, attenuator_dB)
- Não executa nenhum comando para iniciar transmissões de jammer
- Permite adicionar notas e marcar que o jammer estava ON durante a captura

Exemplo de uso (captura + metadados do jammer preenchidos manualmente):
  python watchdog_capture_classC.py --seconds 10 --addr 192.168.10.2 --rate 5e6 --center 3.75e9 --gain 20 \
      --outdir teste_lab --notes "Classe C - jammer manual" --jammer_on True --jammer_gain_setting 80 \
      --jammer_measured_dbm 10.0 --jammer_attenuator_db 20

OBS: Se você precisa automatizar o controle do jammer, utilize apenas ferramentas do fabricante e
procedimentos autorizados pelo seu laboratório/órgão regulador. Não inclua automação de jamming sem
autorização expressa. Este script evita isso por motivos de segurança/legais.
"""

import os
import time
import json
import argparse

# Try importing GNU Radio UHD blocks.
try:
    from gnuradio import gr, blocks, uhd
except Exception as e:
    raise RuntimeError("Este script requer GNU Radio (gr-uhd) com UHD. Erro ao importar: " + str(e))

class CaptureTB(gr.top_block):
    def __init__(self, rate, center, gain, addr, num_samps, out_path, label, metadata):
        gr.top_block.__init__(self, "watchdog-capture-classC")
        # UHD source (USRP)
        self.src = uhd.usrp_source(
            ",".join([f"type=x300", f"addr={addr}"]),
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.src.set_samp_rate(float(rate))
        self.src.set_center_freq(float(center), 0)
        self.src.set_gain(float(gain), 0)

        # Limit samples + sink to file (complex64)
        self.head = blocks.head(gr.sizeof_gr_complex, int(num_samps))
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, out_path, False)
        self.sink.set_unbuffered(False)

        self.connect(self.src, self.head, self.sink)

        # metadata base
        self._meta = metadata
        self._meta.update({
            "fs": float(rate),
            "center_hz": float(center),
            "label": int(label),
            "format": "c64",
            "samples_requested": int(num_samps)
        })
        self._meta_path = out_path + ".json"

    def write_meta(self):
        try:
            with open(self._meta_path, "w") as f:
                json.dump(self._meta, f, indent=2)
        except Exception as e:
            print("[meta] erro ao salvar metadata:", e)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def main():
    ap = argparse.ArgumentParser(description="Watchdog capture (Classe C) -> grava em teste_lab/ (jammer MANUAL)")
    # RF / device
    ap.add_argument("--seconds", type=float, default=10.0, help="Duração da captura (s)")
    ap.add_argument("--addr", type=str, default="192.168.10.2", help="Endereço do USRP (device addr)")
    ap.add_argument("--rate", type=float, default=5e6, help="Taxa de amostragem (Hz)")
    ap.add_argument("--center", type=float, default=3.75e9, help="Frequência central (Hz)")
    ap.add_argument("--gain", type=float, default=20.0, help="Ganho RX do USRP (dB)")
    ap.add_argument("--outdir", type=str, default="teste_lab", help="Pasta de saída")
    ap.add_argument("--label", type=int, default=1, help="Rótulo (1 = Classe C com jammer)")
    ap.add_argument("--notes", type=str, default="", help="Observações adicionais")

    # Jammer metadata (user-provided; script will NOT activate jammer)
    ap.add_argument("--jammer_on", type=lambda s: s.lower() in ["1","true","yes"], default=False,
                    help="Indica se o jammer estava ligado (True/False).")
    ap.add_argument("--jammer_gain_setting", type=float, default=None, help="Valor do 'gain' configurado no transmissor (unidade do dispositivo)")
    ap.add_argument("--jammer_measured_dbm", type=float, default=None, help="Potência medida na saída (dBm) — registre com medidor de potência")
    ap.add_argument("--jammer_attenuator_db", type=float, default=None, help="Valor do atenuador em linha (dB), se usado")
    ap.add_argument("--jammer_center_hz", type=float, default=None, help="Frequência central do jammer (Hz)")
    ap.add_argument("--jammer_bw_hz", type=float, default=None, help="Largura de banda do jammer (Hz)")

    args = ap.parse_args()

    # Safety check: warn if jammer_on is True but no measured_dbm provided
    if args.jammer_on and args.jammer_measured_dbm is None:
        print("[AVISO] Você marcou jammer_on=True mas não forneceu --jammer_measured_dbm. Recomendamos registrar a potência medida (dBm).")

    # Ensure outdir exists
    outdir = ensure_dir(args.outdir)

    # Build filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"casoC_{ts}.c64"
    out_path = os.path.join(outdir, fname)

    # Compute num samples
    num_samps = int(args.rate * args.seconds)

    # Prepare metadata
    metadata = {
        "script": "watchdog_capture_classC.py",
        "start_utc": iso_now(),
        "user_notes": args.notes,
        "class": "C",
        "jammer_on": bool(args.jammer_on),
        "jammer_info": {
            "gain_setting": args.jammer_gain_setting,
            "measured_dbm": args.jammer_measured_dbm,
            "attenuator_db": args.jammer_attenuator_db,
            "center_hz": args.jammer_center_hz,
            "bw_hz": args.jammer_bw_hz
        },
        "jammer_controlled_by_script": False,
        "safety_note": "This script DOES NOT control or activate any jammer. Jammer must be activated manually by authorized personnel."
    }

    print(f"[watchdog] Classe C: captura -> {out_path}")
    print(f"  USRP addr={args.addr} rate={args.rate} center={args.center} gain={args.gain} seconds={args.seconds}")
    if metadata["jammer_on"]:
        print("  Jammer metadata provided: ", metadata["jammer_info"])

    try:
        tb = CaptureTB(args.rate, args.center, args.gain, args.addr, num_samps, out_path, args.label, metadata)
        t0 = time.time()
        tb.start()
        tb.wait()
        dt = time.time() - t0

        # finalize metadata
        metadata["end_utc"] = iso_now()
        metadata["duration_s"] = dt

        # write metadata
        with open(out_path + ".json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[watchdog] OK: {num_samps} amostras gravadas em {dt:.2f}s -> {out_path}")
        print("[watchdog] metadata salvo em:", out_path + ".json")
    except Exception as e:
        print("[watchdog] erro durante captura:", e)


if __name__ == "__main__":
    main()