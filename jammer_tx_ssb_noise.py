#!/usr/bin/env python3
# Jammer TX: ruído estreito (~7.2 MHz) centrado no SSB @ 3709.92 MHz
from gnuradio import gr, analog, filter, uhd
from gnuradio.filter import firdes
import argparse, time

class JammerTX(gr.top_block):
    def __init__(self, samp_rate=30.72e6, fc=3709.92e6, bw=7.2e6,
                 tx_gain_db=0.0, antenna="TX/RX", device_addr=""):
        gr.top_block.__init__(self, "Jammer TX SSB Noise")
        self.samp_rate = samp_rate
        self.fc = fc
        self.bw = bw

        # Fonte de ruído complexa (σ ajustado para headroom)
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, 0.20, 0)

        # Filtro passa-baixas p/ limitar ~7.2 MHz (3.6 MHz de cada lado)
        trans = bw/6.0
        self.lp = filter.fir_filter_ccf(1, firdes.low_pass(
            gain=1.0, sampling_freq=samp_rate,
            cutoff_freq=bw/2.0, transition_width=trans,
            window=firdes.WIN_HANN))

        # UHD Sink
        self.usrp = uhd.usrp_sink(",".join([device_addr]),
                                  uhd.stream_args(cpu_format="fc32", channels=[0]))
        self.usrp.set_samp_rate(samp_rate)
        self.usrp.set_center_freq(fc, 0)
        self.usrp.set_gain(tx_gain_db, 0)
        self.usrp.set_antenna(antenna, 0)  # X310: "TX/RX"; X410: "TX/RX", "TX0", etc.
        try:
            self.usrp.set_bandwidth(bw, 0)
        except Exception:
            pass

        self.connect(self.noise, self.lp, self.usrp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=30.72e6)
    ap.add_argument("--fc", type=float, default=3709.92e6)
    ap.add_argument("--bw", type=float, default=7.2e6)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--ant", type=str, default="TX/RX")
    ap.add_argument("--dev", type=str, default="")  # ex.: "addr=192.168.10.2"
    args = ap.parse_args()

    tb = JammerTX(samp_rate=args.fs, fc=args.fc, bw=args.bw,
                  tx_gain_db=args.gain, antenna=args.ant, device_addr=args.dev)
    tb.start()
    try:
        print(f"[TX] Jammer @ {args.fc/1e6:.3f} MHz | BW={args.bw/1e6:.1f} MHz | fs={args.fs/1e6:.2f} MS/s")
        print("Aumente --gain com cautela (atenuadores!)  — Ctrl+C para parar")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    tb.stop(); tb.wait()
