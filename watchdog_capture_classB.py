#!/usr/bin/env python3
"""
watchdog_capture_classB.py

Classe B: Rede 5G ativa (com transmissão), sem jammer.
- Captura IQ do USRP (watchdog) e grava em "teste_lab/"
- Opcional: roda iperf3 em paralelo para registrar throughput do UE
- Gera metadata JSON com parâmetros e (se houver) saída do iperf

Exemplo (somente captura):
  python watchdog_capture_classB.py --seconds 10 --addr 192.168.10.2 --rate 5e6 --center 3.75e9 --gain 20

Exemplo (captura + iperf):
  python watchdog_capture_classB.py --seconds 10 --addr 192.168.10.2 --rate 5e6 --center 3.75e9 --gain 20 \
      --iperf-server 192.168.70.1 --iperf-seconds 10 --iperf-port 5201 --iperf-reverse

Notas:
- Garanta que o UE está conectado e gerando tráfego (o iperf client pode rodar no próprio host ou no UE, conforme sua topologia).
- Se você rodar iperf3 neste host, ele enviará tráfego para o servidor especificado (por ex., no core/UPF).
"""

import os
import time
import json
import argparse
import subprocess
import threading
from typing import Optional

# Try importing GNU Radio UHD blocks.
try:
    from gnuradio import gr, blocks, uhd
except Exception as e:
    raise RuntimeError("Este script requer GNU Radio (gr-uhd) com UHD. Erro ao importar: " + str(e))

class CaptureTB(gr.top_block):
    def __init__(self, rate, center, gain, addr, num_samps, out_path, label, metadata):
        gr.top_block.__init__(self, "watchdog-capture-classB")
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

def run_iperf(server: str, seconds: int, port: int, reverse: bool=False, extra_args: Optional[list]=None) -> dict:
    """Run iperf3 client and return dict with stdout/stderr/returncode."""
    if not server:
        return {"skipped": True, "reason": "no-server", "stdout": "", "stderr": "", "returncode": None}
    cmd = ["iperf3", "-c", server, "-t", str(seconds), "-p", str(port)]
    if reverse:
        cmd.append("-R")
    if extra_args:
        cmd.extend(extra_args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"skipped": False, "cmd": cmd, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
    except FileNotFoundError as e:
        return {"skipped": True, "reason": "iperf3-not-found", "stdout": "", "stderr": str(e), "returncode": None}
    except Exception as e:
        return {"skipped": True, "reason": f"exception:{type(e).__name__}", "stdout": "", "stderr": str(e), "returncode": None}

def main():
    ap = argparse.ArgumentParser(description="Watchdog capture (Classe B) -> rede ativa sem jammer")
    # RF / device
    ap.add_argument("--seconds", type=float, default=10.0, help="Duração da captura (s)")
    ap.add_argument("--addr", type=str, default="192.168.10.2", help="Endereço do USRP (device addr)")
    ap.add_argument("--rate", type=float, default=5e6, help="Taxa de amostragem (Hz)")
    ap.add_argument("--center", type=float, default=3.75e9, help="Frequência central (Hz)")
    ap.add_argument("--gain", type=float, default=20.0, help="Ganho RX do USRP (dB)")
    ap.add_argument("--outdir", type=str, default="teste_lab", help="Pasta de saída")
    ap.add_argument("--label", type=int, default=0, help="Rótulo (0 = Classe B sem jammer)")
    ap.add_argument("--notes", type=str, default="", help="Observações adicionais")

    # iperf options (optional)
    ap.add_argument("--iperf-server", type=str, default=None, help="IP/DNS do servidor iperf3 (opcional)")
    ap.add_argument("--iperf-seconds", type=int, default=None, help="Duração do iperf (s). Se vazio, usa --seconds")
    ap.add_argument("--iperf-port", type=int, default=5201, help="Porta do iperf3 (padrão 5201)")
    ap.add_argument("--iperf-reverse", action="store_true", help="Usa modo reverso (-R)")
    ap.add_argument("--iperf-extra", type=str, nargs="*", default=None, help="Args extras para o iperf3")

    args = ap.parse_args()

    # Ensure outdir
    outdir = ensure_dir(args.outdir)

    # Build filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"casoB_{ts}.c64"
    out_path = os.path.join(outdir, fname)

    # Compute num samples
    num_samps = int(args.rate * args.seconds)

    # Prepare metadata
    metadata = {
        "script": "watchdog_capture_classB.py",
        "start_utc": iso_now(),
        "user_notes": args.notes,
        "class": "B",
        "jammer_on": False
    }

    # If iperf server provided, we'll run it in parallel thread
    iperf_result = {}
    iperf_duration = args.iperf_seconds if args.iperf_seconds is not None else int(args.seconds)
    def iperf_thread_func():
        nonlocal iperf_result
        iperf_result = run_iperf(
            server=args.iperf_server,
            seconds=iperf_duration,
            port=args.iperf_port,
            reverse=args.iperf_reverse,
            extra_args=args.iperf_extra
        )

    iperf_thread = None
    if args.iperf_server:
        iperf_thread = threading.Thread(target=iperf_thread_func, daemon=True)
        iperf_thread.start()

    # Start capture
    print(f"[watchdog] Classe B: captura -> {out_path}")
    print(f"  USRP addr={args.addr} rate={args.rate} center={args.center} gain={args.gain} seconds={args.seconds}")
    if args.iperf_server:
        print(f"  iperf3 -> server={args.iperf_server} seconds={iperf_duration} port={args.iperf_port} reverse={args.iperf_reverse}")

    try:
        tb = CaptureTB(args.rate, args.center, args.gain, args.addr, num_samps, out_path, args.label, metadata)
        t0 = time.time()
        tb.start()
        tb.wait()
        dt = time.time() - t0

        # Join iperf thread if running
        if iperf_thread is not None:
            iperf_thread.join(timeout=5)

        # finalize metadata
        metadata["end_utc"] = iso_now()
        metadata["duration_s"] = dt
        if args.iperf_server:
            metadata["iperf"] = iperf_result

        # write metadata
        with open(out_path + ".json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[watchdog] OK: {num_samps} amostras gravadas em {dt:.2f}s -> {out_path}")
        if args.iperf_server:
            print(f"[watchdog] iperf returncode: {iperf_result.get('returncode')}")
    except Exception as e:
        print("[watchdog] erro durante captura:", e)

if __name__ == "__main__":
    main()
