import os
import re

import xarray as xr
import dask.array as da
import dask

import numpy as np
import scipy.signal

import processing as old_processing


def uninterleave_data(prefix, expected_base_name_regex=r"\d{8}_\d{6}"):
    """Function for seperating a single multi channel data file into several channel data files

    File comes in this format:
    {Full_File} = {Chirp1-Chirp2...ChirpX}
        [ChirpX] = [Chan0-Chan1...ChanX]
            (ChanX) = (Samp1Real, Samp1Imag-Samp2Real, Samp2Imag...SampXReal, SampXImag)

    Files goes out in this format:
    {Chan1_File} = {Chirp1-Chirp2...ChirpX}
        (ChanX) = (Samp1Real, Samp1Imag-Samp2Real, Samp2Imag...SampXReal, SampXImag)
    ...
    {ChanX_File} = {Chirp1-Chirp2...ChirpX}
        (ChanX) = (Samp1Real, Samp1Imag-Samp2Real, Samp2Imag...SampXReal, SampXImag)


    A lot of the beginning parsing code is pulled from "save_radar_data_to_zarr()" in "processing_dask"
    """

    #
    # Validation and file name generation
    #
    basename = os.path.basename(prefix)
    if (
        expected_base_name_regex
    ):  # Optional sanity check that basename makes sense -- set expected_base_name_regex to None to skip
        if not re.match(expected_base_name_regex, basename):
            raise ValueError(
                f"Prefix basename {basename} does not match expected regex {expected_base_name_regex}"
            )

    # Build filenames from prefix
    rx_samps_file = prefix + "_rx_samps.bin"
    log_file = prefix + "_uhd_stdout.log"

    #
    # Data loading
    #
    # Load configuration YAML
    config = old_processing.load_config(prefix)

    cpu_format = config["DEVICE"].get("cpu_format", "fc32")
    if cpu_format == "fc32":
        output_dtype = np.float32
        scale_factor = 1.0
    elif cpu_format == "sc16":
        output_dtype = np.int16
        scale_factor = np.iinfo(output_dtype).max
    elif cpu_format == "sc8":
        output_dtype = np.int8
        scale_factor = np.iinfo(output_dtype).max
    else:
        raise Exception(
            f"Unrecognized cpu_format '{cpu_format}'. Must be one of 'fc32', 'sc16', or 'sc8'."
        )

    # Load raw RX samples
    rx_len_samples = int(
        config["CHIRP"]["rx_duration"] * config["GENERATE"]["sample_rate"]
    )
    rx_sig = da.from_array(
        np.memmap(rx_samps_file, dtype=output_dtype, mode="r", order="C"),
        chunks=str(rx_len_samples * 2 * 100),
    )
    rx_sig = (rx_sig[::2] + (1j * rx_sig[1::2])).astype(np.complex64) / scale_factor
    n_rxs = rx_sig.size // rx_len_samples
    radar_data = da.transpose(
        da.reshape(rx_sig, (n_rxs, rx_len_samples), merge_chunks=True)
    )

    num_chans = len(str(config["DEVICE"]["rx_channels"]).split(sep=","))

    print(radar_data.shape)

    # Do for each channel
    out_files = []  # store output filenames
    for i in range(num_chans):
        chan_data = np.array(
            radar_data[::, i::num_chans]
        )  # change to a np.array because dask cant save directly to a binary file
        print(chan_data.shape)
        chan_file_name = prefix + "_rx_samps_chan_" + str(i) + ".bin"
        # Save newly seperated data
        chan_data.tofile(chan_file_name)
        out_files.append(chan_file_name)

    return out_files


def save_chan_data_to_zarr(
    prefix,
    skip_if_cached=True,
    zarr_base_location=None,
    expected_base_name_regex=r"\d{8}_\d{6}",
    log_required=True,
    dryrun=False,
):
    """
    Load raw radar data from a given prefix, and save it to a zarr file.

    `prefix` is the path to the raw data, without the _rx_samps.bin/_config.yaml/_uhd_stdout.log suffixes.
    (`log_required` can be set to False if no log file is available)

    As a safety precaution, this function will check that the prefix basename matches the `expected_base_name_regex` expression.
    If using the python code to run the radar system, the prefixes will be of the form YYYYMMDD_HHMMSS,
    which is what the default regex is looking for. `expected_base_name_regex` can be set to None to skip this check.

    The location for the zarr file is the same directory containing the prefix, unless you provide
    an alternate `zarr_base_location` argument.

    By default, this will first look for an existing zarr file that matches the expected filename.
    If you want to force reprocessing, set `skip_if_cached` to False.

    Setting `dryrun` to True will cause this function to return the path to the zarr file
    that it would have created without actually writing anything to disk.

    Returns the path to the zarr file only. You are responsible for re-loading the data from the zarr file.
    """

    #
    # Validation and file name generation
    #
    basename = os.path.basename(prefix)
    if (
        expected_base_name_regex
    ):  # Optional sanity check that basename makes sense -- set expected_base_name_regex to None to skip
        if not re.match(expected_base_name_regex, basename):
            raise ValueError(
                f"Prefix basename {basename} does not match expected regex {expected_base_name_regex}"
            )

    # Build filenames from prefix
    rx_chan_file_prefix = prefix + "_rx_samps_chan_"
    log_file = prefix + "_uhd_stdout.log"

    #
    # Data loading
    #
    # Load configuration YAML
    config = old_processing.load_config(prefix)

    cpu_format = config["DEVICE"].get("cpu_format", "fc32")
    if cpu_format == "fc32":
        output_dtype = np.float32
        scale_factor = 1.0
    elif cpu_format == "sc16":
        output_dtype = np.int16
        scale_factor = np.iinfo(output_dtype).max
    elif cpu_format == "sc8":
        output_dtype = np.int8
        scale_factor = np.iinfo(output_dtype).max
    else:
        raise Exception(
            f"Unrecognized cpu_format '{cpu_format}'. Must be one of 'fc32', 'sc16', or 'sc8'."
        )

    num_chans = len(str(config["DEVICE"]["rx_channels"]).split(sep=","))

    zarr_paths = []
    # loop for each channel
    for chan in range(num_chans):

        # complete chan file name
        rx_chan_file = rx_chan_file_prefix + str(chan) + ".bin"

        # Generate expected zarr output location
        if zarr_base_location is None:
            zarr_path = prefix + "_chan_" + str(chan) + ".zarr"
        else:
            zarr_path = os.path.join(
                zarr_base_location, basename + "_chan_" + str(chan) + ".zarr"
            )

        zarr_paths.append(zarr_path)

        # Check if zarr file already exists, if so continue to next channel
        if skip_if_cached and os.path.exists(zarr_path):
            continue

        # Load raw RX samples
        rx_len_samples = int(
            config["CHIRP"]["rx_duration"] * config["GENERATE"]["sample_rate"]
        )
        rx_sig = da.from_array(
            np.memmap(rx_chan_file, dtype=output_dtype, mode="r", order="C"),
            chunks=rx_len_samples * 2 * 100,
        )
        rx_sig = (rx_sig[::2] + (1j * rx_sig[1::2])).astype(np.complex64) / scale_factor
        n_rxs = rx_sig.size // rx_len_samples
        radar_data = da.transpose(
            da.reshape(rx_sig, (n_rxs, rx_len_samples), merge_chunks=True)
        )

        # Create time axes
        slow_time = np.linspace(
            0,
            config["CHIRP"]["pulse_rep_int"]
            * config["CHIRP"].get("num_presums", 1)
            * n_rxs,
            radar_data.shape[1],
        )
        fast_time = np.linspace(0, config["CHIRP"]["rx_duration"], radar_data.shape[0])

        # Load raw data from log
        log = None
        if os.path.exists(log_file):
            with open(log_file, "r") as log_f:
                log = log_f.read()
        else:
            if log_required:
                raise FileNotFoundError(
                    f"Log file not found: {log_file}. If a log file is not required, set log_required=False"
                )

        # Save radar_data, slow_time, and fs to an xarray datarray
        data = xr.Dataset(
            data_vars={
                "radar_data": (
                    ["sample_idx", "pulse_idx"],
                    radar_data,
                    {"description": "complex radar data"},
                ),
            },
            coords={
                "sample_idx": (
                    "sample_idx",
                    np.arange(radar_data.shape[0]),
                    {"description": "Index of this sample in the chirp"},
                ),
                "fast_time": (
                    "sample_idx",
                    fast_time,
                    {
                        "description": "time relative to start of this recording interval in seconds"
                    },
                ),
                "pulse_idx": (
                    "pulse_idx",
                    np.arange(radar_data.shape[1]),
                    {"description": "Index of this chirp in the sequence"},
                ),
                "slow_time": (
                    "pulse_idx",
                    slow_time,
                    {"description": "time in seconds"},
                ),
            },
            attrs={
                "config": config,
                "stdout_log": log,
                "prefix": prefix,
                "basename": basename,
            },
        )

        # TODO: Due to the currently hard-coded increase in pulse repetition interval after an error,
        # the slow time may not be correct.

        if not dryrun:
            with dask.config.set(scheduler="single-threaded"):
                data.to_zarr(zarr_path, mode="w")
        else:
            print("This is a dry run: not saving data to disk")
            print(data)

    return zarr_paths


uninterleave_data("/home/amcdona4/Documents/uhd_snowdar/data/20260108_154718")
save_chan_data_to_zarr("/home/amcdona4/Documents/uhd_snowdar/data/20260108_154718")
