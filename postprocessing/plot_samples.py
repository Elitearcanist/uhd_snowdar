import sys
import argparse
import numpy as np
import scipy.signal as sp
import processing as pr
import matplotlib.pyplot as plt
from ruamel.yaml import YAML as ym

sys.path.append("../preprocessing")
from generate_chirp import generate_chirp


prefix = "../data/20250812_135114"
yaml_file = prefix + "_config.yaml"
bin_file = prefix + "_rx_samps.bin"

# Initialize Constants
yaml = ym()  # Always use safe load if not dumping
with open(yaml_file) as stream:
    config = yaml.load(stream)
    rx_params = config["PLOT"]
    sample_rate = rx_params["sample_rate"]  # Hertz
    direct_start = rx_params["direct_start"]
    echo_start = rx_params["echo_start"]
    sig_speed = rx_params["sig_speed"]

rx_samps = bin_file  # Received data to analyze


print("--- Loaded constants from config.yaml ---")

# Read and plot RX/TX
rx_sig = pr.extractSig(rx_samps)
print("--- Plotting real samples read from %s ---" % rx_samps)
pr.plotChirpVsTime(rx_sig, "Received Samples", sample_rate)

_, tx_sig = generate_chirp(config)  # generate chirp based on YAML config
print("--- Plotting transmited chirp, stored in %s ---" % yaml_file)
pr.plotChirpVsTime(tx_sig, "Transmitted Chirp", sample_rate)

# Correlate the two chirps to determine time difference
print("--- Match filtering received chirp with transmitted chirp ---")
xcorr_sig = sp.correlate(rx_sig, tx_sig, mode="valid", method="auto")
# as finddirectpath is written right now, it must be called before taking log of the signal
# because if not, negative log values could have a greater absolute value than positive log values.
dir_peak = pr.findDirectPath(xcorr_sig, direct_start, True)
xcorr_sig = 20 * np.log10(np.absolute(xcorr_sig))

print("--- Plotting result of match filter ---")
xcorr_samps = np.shape(xcorr_sig)[0]
xcorr_time = np.zeros(xcorr_samps)
for x in range(xcorr_samps):
    xcorr_time[x] = x * 1e6 / sample_rate

plt.figure()
plt.plot(xcorr_time, xcorr_sig)
plt.title("Output of Match Filter: Signal")
plt.xlabel("Time (us)")
plt.ylabel("Power [dB]")
plt.grid()

plt.figure()
plt.plot(range(-10, 60), xcorr_sig[dir_peak - 10 : dir_peak + 60])
plt.title("Output of Match Filter: Peaks")
plt.xlabel("Sample")
plt.ylabel("Power [dB]")
plt.grid()

[echo_samp, echo_dist] = pr.findEcho(
    xcorr_sig, sample_rate, dir_peak, echo_start, sig_speed, True
)

sys.stdout.flush()
plt.show()
