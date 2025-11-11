# Written by Levi Powell, August 2023, edited by Andrew McDonald July 2025
# This code plots the data saved by the SDR
# Turned into a callable function for use in other scripts,
# added comments,
# replaced numpy fft with scipy,
# by Trevor Wiseman August 2025

"""TODO:
1. see if empty data can be collected at end of buffer, remove trailing zeros
https://github.com/righthalfplane/rfspace/blob/master/universalSend.cpp
https://discourse.myriadrf.org/t/soapysdr-read-stream-samples-lost/8014

2. Improve data plotter
https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=10036&context=etd
https://github.com/kohlsne/LimeSDR_LFMCW_Radar/blob/main/main.cpp
https://github.com/kohlsne/LimeSDR_LFMCW_Radar/blob/main/MatlabScripts/ProcessDataMethod3.m
"""

import sys
import time
import array
import matplotlib.pyplot as plt
import numpy as np

# from numpy.fft import fft
# from numpy.fft import fftshift
from scipy.fft import fft, fftshift, fftfreq

import scipy.signal as sp
import processing as pr
from ruamel.yaml import YAML as ym

sys.path.append("../preprocessing")
from generate_chirp import generate_chirp


n_stack = 1  # Do additional stacking in this notebook - set > 1 to enable
zero_sample_idx = 0  # The index of the 0 distance sample -- will change with platforms and config changes -- if unsure, just set to 0

##### FILE NAMES #####
PREFIX = "../data/20250815_150241"
YAML_FILE = PREFIX + "_config.yaml"
BIN_FILE = PREFIX + "_rx_samps.bin"

##### CONSTANTS #####
FFT_LENGTH = 100  # length of the FFT, must be less than or equal to numSamples

LOCK_AXIS = False  # Sets axis to fixed values


# Funciton for importing the data, mixing with a chirp, windowing and calculating the FFT
def handle_data(
    file_name: str,
    yaml_config,
    numChirps: int,
    numSamples: int,
):
    # instantiate the data matrices
    rx_sig = np.empty([numChirps, numSamples], np.csingle)
    mixed_sig = np.empty([numChirps, numSamples], np.csingle)
    realMatrix = np.empty([numChirps, numSamples], np.float32)
    FFTMagMatrix = np.empty([numChirps, FFT_LENGTH], np.float32)

    # read data
    print(f"Reading {file_name}...")  # debug statement
    startTime = time.time()  # start timer

    rx_sig = pr.extractSig(file_name)

    rx_sig_reshaped = np.reshape(rx_sig, (numChirps, numSamples))

    _, tx_sig = generate_chirp(yaml_config)  # generate chirp based on YAML config

    for row in range(numChirps):
        for i in range(numSamples):
            mixed_sig[row][i] = (tx_sig[i].real + tx_sig[i].imag * 1j) * (
                rx_sig_reshaped[row][i].real - rx_sig_reshaped[row][i].imag * 1j
            )

    realMatrix = mixed_sig.real

    # stop timer and print time it takes to save data
    endTime = time.time()
    print(f"\tTotal time was {endTime - startTime} s")

    # start timer to calculate the FFT of the data set
    print(f"Calculating FFT for {file_name}...")
    startTime = time.time()

    # calculate FFT
    for i in range(len(mixed_sig)):

        # save FFT of data to a matrix
        X = fftshift(fft(mixed_sig[i]))  # take the fft of the data in the matrix
        X = X[
            numSamples // 2 - FFT_LENGTH // 2 : numSamples // 2 + FFT_LENGTH // 2
        ]  # only keep data in the FFT length

        # Save the magnitude
        FFTMagMatrix[i] = np.abs(X)
        FFTMagMatrix[i] = 10 * np.log10(
            FFTMagMatrix[i]
        )  # TODO sometimes numbers are 0 here

    # free memory
    del mixed_sig

    # stop timer and print time it takes to do the fft
    endTime = time.time()
    print(f"\tTotal time was {endTime - startTime} s")

    return realMatrix, FFTMagMatrix


def plotter():
    # Plot settings

    # Configure the plots with the config file
    print("Reading configuration file...")  # debug statement

    # Initialize Constants
    yaml = ym()
    with open(YAML_FILE) as stream:
        config = yaml.load(stream)
        sampleRate = config["PLOT"]["sample_rate"]  # Hertz
        sig_speed = config["PLOT"]["sig_speed"]

        bandwidth = config["GENERATE"]["chirp_bandwidth"]  # bandwidth from config file
        numChirps = int(
            config["CHIRP"]["num_pulses"] / config["CHIRP"]["num_presums"]
        )  # number of chirps from config file

        # expected_n_rxs = int(config['CHIRP']['num_pulses'] / config['CHIRP']['num_presums'])

        numSamples = int(
            config["CHIRP"]["rx_duration"] * config["GENERATE"]["sample_rate"]
        )

    rx_samps = BIN_FILE

    # print the configuration
    print(f"\tSample rate was set to {sampleRate / 1e6} MS/s")
    print(f"\tBandwidth was set to {bandwidth / 1e6} MHz")
    print(f"\tNumber of samples was set to {numSamples}")
    print(f"\tNumber of chirps was set to {numChirps}")

    # Get target data
    realMatrix2, FFTMagMatrix2 = handle_data(rx_samps, config, numChirps, numSamples)

    # start plotting
    print("Plotting...")

    # set bounds
    vmaxT = "0.01"
    vminT = "-0.01"
    vmaxF = "20"
    vminF = "-20"

    # set subplots
    subplot1 = 210

    plt.figure(1)

    plt.subplot(subplot1 + 1)
    plt.title("Target Data")
    if LOCK_AXIS:
        plot = plt.matshow(
            realMatrix2,
            cmap="hot",
            vmax=vmaxT,
            vmin=vminT,
            aspect="auto",
            fignum=False,
        )
    else:
        plot = plt.matshow(realMatrix2, cmap="hot", aspect="auto", fignum=False)
    plt.colorbar(plot)
    plt.xlabel("Sample Number")
    plt.ylabel("Chirp Number")

    plt.subplot(subplot1 + 2)
    if LOCK_AXIS:
        plot = plt.matshow(
            FFTMagMatrix2,
            cmap="hot",
            vmax=vmaxF,
            vmin=vminF,
            aspect="auto",
            fignum=False,
        )
    else:
        plot = plt.matshow(FFTMagMatrix2, cmap="hot", aspect="auto", fignum=False)
    plt.colorbar(plot)
    plt.xlabel("Frequency Bin")
    plt.ylabel("Chirp Number")

    # Free up memory
    del FFTMagMatrix2
    del realMatrix2

    plt.show()
    print("Done")


if __name__ == "__main__":
    plotter()
