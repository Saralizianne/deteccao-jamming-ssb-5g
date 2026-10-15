#!/usr/bin/env python3
"""
watchdog_capture_classA.py

Capture script for "Classe A" (sem transmissão) — gravação automática para pasta "teste_lab".
Uses GNU Radio UHD USRP source (requires gr-uhd / uhd + GNU Radio 3.10+).

Exemplo de uso (no host conectado ao USRP):
    python watchdog_capture_classA.py --seconds 20 --addr 192.168.10.2 --rate 120e6 --center 3.75e9 --gain 20

O script grava:
 - arquivo .c64 com amostras complex64 (I/Q) em teste_lab/
 - arquivo .json com metadados junto ao .c64
 - escreve logs simples no stdout

Notas:
 - Classe A refere-se a "sem transmissão" (gNB sem tráfego). O script não desliga nem verifica o gNB;
   cabe ao operador garantir o estado da rede antes de rodar.
"""

import os
import time
import json
import argparse

# Try importing GNU Radio UHD blocks. If not available, instruct user.
try:
    from gnuradio import gr, blocks, uhd
except Exception as e:
    raise RuntimeError("Este script requer GNU Radio (gr-uhd) instalado com suporte UHD. Erro: " + str(e))

class CaptureTB(gr.top_block):
    def __init__(self, rate, center, gain, addr, num_samps, out_path, label, metadata):
        gr.top_block.__init__(self, "watchdog-capture-classA")
        # UHD source (USRP)
        self.src = uhd.usrp_source(
            ",".join([f"type=x300", f"addr={addr}"]),
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.src.set_samp_rate(float(rate))
        self.src.set_center_freq(float(center), 0)
        self.src.set_gain(float(gain), 0)

        # head to limit samples + file sink
        self.head = blocks.head(gr.sizeof_gr_complex, int(num_samps))
        # file sink writing complex64 (cfloat) raw binary
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, out_path, False)
        self.sink.set_unbuffered(False)

        # connect blocks
        self.connect(self.src, self.head, self.sink)

        # meta
        self._meta = metadata
        self._meta.update({
            "fs": float(rate),
            "center_hz": float(center),
            "label": int(label),
            "format": "c64",
            "samples_requested": int(num_samps)
        })
        self._meta_path = out_path + ".json"

    def done(self):
        # save metadata JSON
        try:
            with open(self._meta_path, "w") as f:
                json.dump(self._meta, f, indent=2)
            print(f"[meta] salvo {self._meta_path}")
        except Exception as e:
            print("[meta] erro ao salvar metadata:", e)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def iso_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser(description="Watchdog capture (Classe A) -> grava em teste_lab/")
    ap.add_argument("--seconds", type=float, default=20.0, help="Duração da captura em segundos")
    ap.add_argument("--addr", type=str, default="192.168.10.2", help="Endereço do USRP (device addr)")
    ap.add_argument("--rate", type=float, default=120e6, help="Taxa de amostragem (Hz)")
    ap.add_argument("--center", type=float, default=3.75e9, help="Frequência central (Hz)")
    ap.add_argument("--gain", type=float, default=20.0, help="Ganho RX do USRP (dB)")
    ap.add_argument("--outdir", type=str, default="teste_lab", help="Pasta de saída (será criada se necessário)")
    ap.add_argument("--label", type=int, default=0, help="Rótulo da captura (0 = Classe A)")
    ap.add_argument("--notes", type=str, default="", help="Observações para o metadado")
    args = ap.parse_args()

    # ensure outdir exists
    outdir = ensure_dir(args.outdir)

    # construct filename with timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"casoA_{ts}.c64"
    out_path = os.path.join(outdir, fname)

    # number of samples to capture
    num_samps = int(args.rate * args.seconds)

    # basic metadata
    metadata = {
        "script": "watchdog_capture_classA.py",
        "start_utc": iso_now(),
        "user_notes": args.notes
    }

    print(f"[watchdog] inicio captura Classe A -> {out_path}")
    print(f"  addr={args.addr} rate={args.rate} center={args.center} gain={args.gain} seconds={args.seconds}")
    try:
        tb = CaptureTB(args.rate, args.center, args.gain, args.addr, num_samps, out_path, args.label, metadata)
        t0 = time.time()
        tb.start()
        tb.wait()
        dt = time.time() - t0
        tb.done()
        # update metadata with end time and actual duration
        metadata["end_utc"] = iso_now()
        metadata["duration_s"] = dt
        # rewrite meta with timing info
        with open(out_path + ".json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[watchdog] gravados {num_samps} amostras em {dt:.2f}s -> {out_path}")
    except Exception as e:
        print("[watchdog] erro durante captura:", e)


if __name__ == "__main__":
    main()