#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBAT + Thymio with a 4-microphone circular array (UCA).

Sequential processing:
    wait for clear path
    -> acquire one passive audio window
    -> SPL threshold
    -> map full 360-degree DMAS response to 120 ring neurons
    -> run the original ring attractor
    -> turn
    -> move forward for 0.5 s
    -> stop
    -> wait 5 s
    -> next cycle

No acoustic output/chirp is emitted. Audio capture is continuous through a
sounddevice InputStream callback; there is no separate robot-movement thread.

Null_sensitivity.csv is loaded from the same folder as this script. If it is
not present, the script falls back to unity sensitivity (no compensation).
SPL values are therefore not guaranteed to be calibrated absolute dB SPL
measurements.
"""

import argparse
import csv
import datetime
import logging
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import scipy.signal as signal
import sounddevice as sd

from thymiodirect import Connection, Thymio


# =============================================================================
# CONFIGURATION — circular-array YAML values + confirmed ROBAT settings
# =============================================================================

FS = 16000
CHANNELS = 5                 # channel 0 + four physical microphones on 1..4
ARRAY_RADIUS = 0.0323         # metres
ARRAY_SHIFT_DEG = 45.0
REF_MIC = 2                   # index within selected four-microphone array
DOA_FREQ_RANGE = (1000.0, 2800.0)
ANALYZED_BUFFER_TIME = 0.02  # seconds
INPUT_BLOCK_MS = 1
DMAS_WLEN = 64
THETA_DAS = np.arange(0.0, 360.0, 3.0)  # 120 directions
N_PEAKS = 1

TRIGGER_LEVEL = 48.0
CRITICAL_LEVEL = 99.0

# Original ROBAT ring-attractor parameters — unchanged
RING_NEURONS = 120
RING_V = 0.5
RING_H0 = 0.051
RING_HB = 0.045
RING_BETA = 400.0
RING_UPDATES_PER_STEP = 1200

# Original ROBAT normalization limits — unchanged
DOA_GLOBAL_MIN = 0.0000000005
DOA_GLOBAL_MAX = 0.0000068100
DOA_NORMALIZING_FACTOR = 0.1900

# Original ROBAT robot parameters
FORWARD_SPEED = 80
TURN_SPEED = 100
LEFT_SENSOR_THRESHOLD = 300
RIGHT_SENSOR_THRESHOLD = 300
FORWARD_DURATION_S = 0.5
POST_MOVEMENT_WAIT_S = 5.0
OBSTACLE_THRESHOLD = 3000
GROUND_LIFT_THRESHOLD = 10

# Same buffer-sizing rule as the circular-array controller:
# (1 / 4 Hz + 0.05 s) * 16000 = 4800 samples.
CALL_RATE_FOR_WINDOW = 4.0
BLOCK_SIZE = int((1.0 / CALL_RATE_FOR_WINDOW + 0.05) * FS)


# =============================================================================
# LOGGING
# =============================================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"target_tracking_circular_{LOG_TIMESTAMP}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

logger = logging.getLogger("ROBAT_CIRCULAR")


# =============================================================================
# AUDIO / UCA UTILITY FUNCTIONS
# Inlined from your uploaded utilities.py
# =============================================================================

def pascal_to_dbspl(value):
    """Convert RMS pressure in Pa to dB SPL re 20 micro-Pa."""
    value = np.asarray(value)
    return 20.0 * np.log10(
        np.maximum(np.abs(value), 1e-20) / 20e-6
    )


def calc_native_freqwise_rms(x, fs):
    """Same native FFT-bin RMS calculation as utilities.calc_native_freqwise_rms."""
    x = np.asarray(x)
    rfft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)

    abs_rfft_squared = np.abs(rfft) ** 2
    mean_sq_freq = abs_rfft_squared / rfft.size
    freqwise_rms = np.sqrt(mean_sq_freq / (2 * rfft.size - 1))

    return freqs, freqwise_rms


def load_null_sensitivity(center_freqs):
    """
    Load Null_sensitivity.csv beside this script.

    The expected Null_sensitivity file is the flat/no-compensation curve.
    If it is absent, use unity sensitivity.
    """
    csv_path = Path(__file__).resolve().with_name("Null_sensitivity.csv")

    if not csv_path.exists():
        logger.warning(
            "Null_sensitivity.csv not found beside this script; "
            "using unity sensitivity (no compensation)."
        )
        return np.ones_like(center_freqs, dtype=float)

    rows = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.reader(csv_file):
            if len(row) < 2:
                continue

            try:
                rows.append((float(row[0]), float(row[1])))
            except (TypeError, ValueError):
                # Skip the header row, as in pandas.read_csv in the source.
                continue

    if len(rows) < 2:
        logger.warning(
            "Null_sensitivity.csv has no usable two-column data; "
            "using unity sensitivity."
        )
        return np.ones_like(center_freqs, dtype=float)

    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0])

    frequencies = values[order, 0]
    sensitivity = values[order, 1]

    logger.info("Loaded null sensitivity curve from %s", csv_path)

    return np.interp(
        center_freqs,
        frequencies,
        sensitivity,
        left=float(np.min(sensitivity)),
        right=float(np.min(sensitivity)),
    )


def find_peaks_circular(magnitude, wrap=3):
    """Circular-aware peak detection used by the uploaded UCA processor."""
    n = len(magnitude)

    if n == 0:
        return np.array([], dtype=int)

    extended = np.concatenate(
        (magnitude[-wrap:], magnitude, magnitude[:wrap])
    )

    peaks_ext, _ = signal.find_peaks(extended)
    peaks = []

    for p in peaks_ext:
        real_idx = p - wrap

        if 0 <= real_idx < n:
            peaks.append(real_idx)
        elif real_idx < 0:
            peaks.append(real_idx + n)
        elif real_idx >= n:
            peaks.append(real_idx - n)

    return np.unique(peaks)


def uca_steering_vectors(
    fs,
    nch,
    radius,
    shift_deg,
    frequency_band,
    theta_deg,
    c=343.0,
    wlen=64,
):
    """Precompute UCA steering vectors using the uploaded geometry convention."""
    radius = float(np.squeeze(radius))
    shift_deg = float(np.squeeze(shift_deg))

    f_spec_axis = np.fft.rfftfreq(wlen, 1.0 / fs)

    band_indices = np.where(
        (f_spec_axis >= frequency_band[0])
        & (f_spec_axis <= frequency_band[1])
    )[0]

    if band_indices.size == 0:
        raise ValueError(
            f"No FFT bin falls in DOA_FREQ_RANGE={frequency_band}; "
            f"wlen={wlen}, fs={fs}."
        )

    bands = f_spec_axis[band_indices]
    theta_rad = np.deg2rad(theta_deg)

    phi_n = (
        np.linspace(2 * np.pi, 0, nch, endpoint=False)
        + np.deg2rad(shift_deg)
    )

    mic_x = radius * np.cos(phi_n)
    mic_y = radius * np.sin(phi_n)

    projection = (
        np.outer(mic_x, np.cos(theta_rad))
        + np.outer(mic_y, np.sin(theta_rad))
    )

    wave_number = 2 * np.pi * bands / c

    steering = np.exp(
        1j * wave_number[None, :, None] * projection[:, None, :]
    )

    return band_indices, bands, steering


def higher_order_dmas_UCA(
    y,
    fs,
    nch,
    radius,
    shift_deg,
    frequency_band,
    theta_deg,
    c=343.0,
    wlen=64,
    order=2,
    apply_cf=True,
    steering=None,
):
    """
    Higher-order DMAS-CF algorithm from the uploaded utilities.py.

    The STFT implementation here is the direct scipy version. It produces
    the same spatial response as the optimized stft_bins implementation
    in your uploaded utilities.py.
    """

    if steering is None:
        f_spec_axis, _, spectrum = signal.stft(
            y,
            fs=fs,
            window=np.ones(wlen),
            nperseg=wlen,
            noverlap=wlen - 1,
            axis=0,
        )

        steering = uca_steering_vectors(
            fs,
            nch,
            radius,
            shift_deg,
            frequency_band,
            theta_deg,
            c=c,
            wlen=wlen,
        )

        band_indices, bands, a_mat = steering
        spec = spectrum[band_indices]

    else:
        band_indices, bands, a_mat = steering

        f_spec_axis = np.fft.rfftfreq(wlen, 1.0 / fs)

        _, _, spectrum = signal.stft(
            y,
            fs=fs,
            window=np.ones(wlen),
            nperseg=wlen,
            noverlap=wlen - 1,
            axis=0,
        )

        spec = spectrum[band_indices]

    # scipy.signal.stft with axis=0 returns:
    # (frequency, microphone, frame)
    spec = spec - spec.mean(axis=2, keepdims=True)

    covariance = (
        spec @ spec.conj().transpose(0, 2, 1)
        / spec.shape[2]
    )

    a_bt = a_mat.transpose(1, 0, 2)

    def beam_power(cov):
        return np.sum(
            a_bt.conj() * (cov @ a_bt),
            axis=1,
        )

    # Coherence factor
    if apply_cf:
        p_das = np.real(beam_power(covariance))

        incoherent_power = nch * np.real(
            np.trace(covariance, axis1=1, axis2=2)
        )

        cf = np.clip(
            p_das / (incoherent_power[:, None] + 1e-12),
            0.0,
            1.0,
        )
    else:
        cf = 1.0

    # Higher-order DMAS modifications
    abs_cov = np.abs(covariance)

    cov_hdmas = np.power(abs_cov, 1.0 / order) * (
        covariance / (abs_cov + 1e-12)
    )

    diagonal = np.arange(nch)
    cov_hdmas[:, diagonal, diagonal] = 0

    num_cross_terms = (nch ** 2) - nch

    p_pseudo = beam_power(cov_hdmas) / num_cross_terms
    p_final = np.abs(p_pseudo) ** order * cf

    spatial_response = p_final.sum(axis=0) / len(bands)

    return theta_deg, spatial_response, f_spec_axis, spectrum, bands


# =============================================================================
# RING ATTRACTOR — original ROBAT implementation and parameters
# =============================================================================

class RingAttractor:

    def __init__(self):
        self.Ns = RING_NEURONS
        self.v = RING_V
        self.h_0 = RING_H0
        self.h_b = RING_HB
        self.h_ext = None
        self.beta = RING_BETA

        self.sigma_ang = 2 * np.pi / self.Ns

        self.thetas = np.linspace(
            -np.pi,
            np.pi,
            self.Ns,
            endpoint=False,
        )

        self.spins = np.random.choice(
            [1, 0],
            size=self.Ns,
        )

        self.updates_per_step = RING_UPDATES_PER_STEP

        logger.info(
            "Ring attractor initialized: Ns=%d, beta=%.2f, updates_per_step=%d",
            self.Ns,
            self.beta,
            self.updates_per_step,
        )


def compute_J(ring, i):
    alpha_i = ring.thetas[i]

    angle_diffs = np.abs(
        (ring.thetas - alpha_i + np.pi) % (2 * np.pi) - np.pi
    )

    return np.cos(
        np.pi * ((angle_diffs / np.pi) ** ring.v)
    )


def compute_delta_H(ring, i):
    j_curr = compute_J(ring, i)
    j_curr[i] = 0

    interaction_sum = np.dot(j_curr, ring.spins)

    spin_flip = np.array([
        ring.spins[i],
        1 - ring.spins[i],
    ])

    interaction_energy = (
        interaction_sum * spin_flip
    ) / (ring.Ns - 1)

    if ring.h_ext is None:
        input_energy = 0
    else:
        input_energy = ring.h_ext[i] * spin_flip

    leak_energy = ring.h_b * spin_flip

    hamiltonian = -(
        interaction_energy
        + input_energy
        - leak_energy
    )

    return hamiltonian[1] - hamiltonian[0]


def doa_to_row(
    doa_estimate,
    global_min=DOA_GLOBAL_MIN,
    global_max=DOA_GLOBAL_MAX,
    normalizing_factor=DOA_NORMALIZING_FACTOR,
):
    """Preserve ROBAT's fixed min/max normalization."""
    denom = global_max - global_min
    result = []

    for value in doa_estimate:
        if value == 0.0:
            result.append(0.0)

        elif denom == 0:
            result.append(0.0)

        else:
            norm = (value - global_min) / denom
            result.append(norm * normalizing_factor)

    return result


def run_ring_attractor(ring, doa_estimate):
    """Run the original ring-attractor dynamics and return the bump angle."""
    ring_start = time.perf_counter()

    row_from_doa = doa_to_row(doa_estimate)
    ring.h_ext = row_from_doa

    for _ in range(ring.updates_per_step):
        i = np.random.randint(0, ring.Ns)

        delta_h = compute_delta_H(ring, i)

        if delta_h < 0:
            ring.spins[i] = 1 - ring.spins[i]

        else:
            probability = np.exp(-ring.beta * delta_h)

            if np.random.rand() < probability:
                ring.spins[i] = 1 - ring.spins[i]

    if np.any(ring.spins == 1):
        phi = np.angle(
            np.sum(
                np.exp(
                    1j * ring.thetas[ring.spins == 1]
                )
            )
        )
    else:
        phi = 0.0

    bump_angle_deg = math.degrees(phi)
    elapsed = time.perf_counter() - ring_start

    logger.info(
        "Ring complete: bump=%.2f deg, active neurons=%d, time=%.4f s",
        bump_angle_deg,
        int(np.sum(ring.spins == 1)),
        elapsed,
    )

    return bump_angle_deg, row_from_doa, elapsed


# =============================================================================
# CIRCULAR ARRAY AUDIO PROCESSOR — passive input only
# =============================================================================

class CircularAudioProcessor:

    def __init__(self, device_index):
        self.fs = FS
        self.channels = CHANNELS
        self.radius = ARRAY_RADIUS
        self.shift = ARRAY_SHIFT_DEG
        self.block_size = BLOCK_SIZE
        self.analyzed_buffer_time = ANALYZED_BUFFER_TIME
        self.trigger_level = TRIGGER_LEVEL
        self.critical_level = CRITICAL_LEVEL
        self.ref = REF_MIC
        self.theta_das = THETA_DAS
        self.N_peaks = N_PEAKS
        self.device_index = device_index
        self.dmas_wlen = DMAS_WLEN

        self.input_blocksize = max(
            1,
            int(round(FS * INPUT_BLOCK_MS / 1000)),
        )

        self.min_window_rows = int(
            FS * ANALYZED_BUFFER_TIME
        )

        self.filter_margin = 128

        # Buffer containing the most recent audio samples.
        self.ring_size = int(self.block_size * 2)

        self.audio_ring = np.zeros(
            (self.ring_size, self.channels),
            dtype=np.float32,
        )

        self.ring_pos = 0
        self.samples_written = 0
        self.lock = threading.Lock()
        self.stream = None
        self.last_callback_status = None

        self.sos = signal.butter(
            1,
            DOA_FREQ_RANGE[0],
            btype="highpass",
            fs=FS,
            output="sos",
        )

        self.steering = uca_steering_vectors(
            FS,
            CHANNELS - 1,
            ARRAY_RADIUS,
            ARRAY_SHIFT_DEG,
            DOA_FREQ_RANGE,
            THETA_DAS,
            wlen=DMAS_WLEN,
        )

        # Use the same Null_sensitivity.csv curve as the circular-array source.
        dummy_freqs = np.fft.rfftfreq(
            int(FS * ANALYZED_BUFFER_TIME),
            1.0 / FS,
        )

        self.interp_sensitivity = load_null_sensitivity(dummy_freqs)

    def _callback(self, indata, frames, time_info, status):
        """Continuously store microphone input in the circular audio buffer."""
        if status:
            self.last_callback_status = str(status)

        if indata.shape[1] != self.channels:
            return

        with self.lock:
            pos = self.ring_pos
            end = pos + frames

            if end <= self.ring_size:
                self.audio_ring[pos:end] = indata

            else:
                first = self.ring_size - pos

                self.audio_ring[pos:] = indata[:first]
                self.audio_ring[:frames - first] = indata[first:]

            self.ring_pos = end % self.ring_size
            self.samples_written += frames

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.fs,
            device=self.device_index,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            latency="low",
            blocksize=self.input_blocksize,
        )

        self.stream.start()

        logger.info(
            "Passive circular-array input started: device=%s, fs=%d, "
            "channels=%d, latency=%s",
            self.device_index,
            self.fs,
            self.channels,
            self.stream.latency,
        )

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None

    def get_latest_chunk(self):
        """Return the latest audio window, oldest sample first."""
        with self.lock:
            available = min(
                self.samples_written,
                self.ring_size,
            )

            num_rows = min(
                self.block_size,
                available,
            )

            num_rows -= num_rows % 32

            if num_rows < self.min_window_rows:
                return None

            stop = self.ring_pos
            start = (stop - num_rows) % self.ring_size

            if start < stop:
                chunk = self.audio_ring[start:stop].copy()

            else:
                chunk = np.concatenate(
                    (
                        self.audio_ring[start:],
                        self.audio_ring[:stop],
                    ),
                    axis=0,
                )

        return chunk

    def update_das_UCA(self):
        """
        Return:
            peak_angles
            dB_SPL_level
            complete 120-value circular spatial response
        """

        in_buffer = self.get_latest_chunk()

        if in_buffer is None:
            return (
                [None],
                float("-inf"),
                np.zeros(120, dtype=float),
            )

        # The uploaded circular-array processor selects raw channels 1..4.
        microphones = in_buffer[:, 1:5]

        # High-pass reference microphone to locate the loudest part.
        reference_signal = signal.sosfiltfilt(
            self.sos,
            microphones[:, self.ref],
        )

        envelope = np.abs(
            signal.hilbert(reference_signal)
        )

        max_envelope_idx = int(
            np.argmax(envelope)
        )

        # Trim around the maximum envelope.
        trim_samples = int(
            self.fs * self.analyzed_buffer_time
        )

        half_trim = trim_samples // 2

        if max_envelope_idx - half_trim < 0:
            start_idx = 0
            end_idx = trim_samples

        elif max_envelope_idx + half_trim > microphones.shape[0]:
            end_idx = microphones.shape[0]
            start_idx = end_idx - trim_samples

        else:
            start_idx = max_envelope_idx - half_trim
            end_idx = start_idx + trim_samples

        # Filter all microphones around the trimmed segment.
        seg_start = max(
            start_idx - self.filter_margin,
            0,
        )

        seg_end = min(
            end_idx + self.filter_margin,
            microphones.shape[0],
        )

        filtered_segment = signal.sosfiltfilt(
            self.sos,
            microphones[seg_start:seg_end],
            axis=0,
        )

        trimmed_signal = filtered_segment[
            start_idx - seg_start:end_idx - seg_start
        ]

        # SPL calculation using the Null_sensitivity curve.
        center_freqs, freq_rms = calc_native_freqwise_rms(
            trimmed_signal[:, self.ref],
            self.fs,
        )

        freqwise_parameters = (
            freq_rms / self.interp_sensitivity
        )

        relevant = np.logical_and(
            center_freqs >= DOA_FREQ_RANGE[0],
            center_freqs <= DOA_FREQ_RANGE[1],
        )

        total_rms = np.sqrt(
            np.sum(freqwise_parameters[relevant] ** 2)
        )

        db_spl_level = float(
            pascal_to_dbspl(total_rms)
        )

        # Full 360-degree higher-order DMAS spatial response.
        theta, spatial_response, _, _, _ = higher_order_dmas_UCA(
            trimmed_signal,
            self.fs,
            self.channels - 1,
            self.radius,
            self.shift,
            DOA_FREQ_RANGE,
            theta_deg=self.theta_das,
            wlen=self.dmas_wlen,
            steering=self.steering,
        )

        peaks = find_peaks_circular(
            spatial_response,
            wrap=3,
        )

        if peaks.size:
            peak_order = np.argsort(
                spatial_response[peaks]
            )[::-1]

            strongest_index = int(
                peaks[peak_order[0]]
            )

            peak_angles = [
                float(theta[strongest_index])
            ]

        else:
            peak_angles = [None]

        # Return the full spatial response regardless of peak detection.
        return (
            peak_angles,
            db_spl_level,
            np.asarray(spatial_response, dtype=float),
        )


# =============================================================================
# THYMIO ROBOT — sequential movement/safety methods from ROBAT
# =============================================================================

class RobotMove:

    def __init__(self, ring, ground_sensors_bool=False):
        self.forward_speed = FORWARD_SPEED
        self.turn_speed = TURN_SPEED

        self.left_sensor_threshold = LEFT_SENSOR_THRESHOLD
        self.right_sensor_threshold = RIGHT_SENSOR_THRESHOLD

        self.ground_sensors_bool = ground_sensors_bool

        self.critical_level = CRITICAL_LEVEL
        self.trigger_level = TRIGGER_LEVEL

        self.running = True
        self.turning_angle = 5
        self.ring = ring
        self.heading = 0
        self.max_count = 0
        self.time_step = 0
        self.log_ring_state = {}
        self.stop_bool = False

        logger.info("Initializing Thymio robot")

        port = Connection.serial_default_port()

        thymio = Thymio(
            serial_port=port,
            on_connect=lambda node_id: logger.info(
                "Thymio %s connected",
                node_id,
            ),
        )

        thymio.connect()

        self.robot = thymio[
            thymio.first_node()
        ]

        logger.info("Thymio connected")

    def wait_until_path_clear(self, poll_interval=0.02):
        """Do not process a new DOA while an obstacle is in front."""
        obstacle_logged = False

        while self.running:

            if self.obstacle_in_front():

                if not obstacle_logged:
                    logger.warning(
                        "OBSTACLE IN FRONT — stopping and waiting"
                    )
                    obstacle_logged = True

                self.stop()
                time.sleep(poll_interval)
                continue

            if obstacle_logged:
                logger.info("Obstacle cleared — resuming")

            return True

        return False

    def angle_to_time(self, angle, speed):
        A = 612.33
        B = -0.94

        t = A * speed ** B

        return t * abs(angle) / 360.0

    def check_stop_all_motion(self):
        if self.robot is None:
            logger.error("Robot object is None")
            self.stop()
            return True

        try:
            ground = self.robot["prox.ground.delta"]

            if (
                ground[0] < GROUND_LIFT_THRESHOLD
                or ground[1] < GROUND_LIFT_THRESHOLD
            ):
                logger.warning(
                    "Safety stop: ground sensor indicates robot lift"
                )
                return True

        except Exception:
            logger.exception("Could not read ground sensors")
            return True

        return False

    def obstacle_in_front(self, threshold=OBSTACLE_THRESHOLD):
        if self.robot is None:
            return False

        try:
            front = self.robot["prox.horizontal"]

            return any(
                front[i] > threshold
                for i in range(5)
            )

        except Exception:
            logger.exception("Could not read front proximity sensors")
            return True

    def move_forward(self):
        if self.robot is None or self.check_stop_all_motion():
            self.stop_bool = True
            self.stop()
            return False

        self.robot["leds.top"] = [0, 255, 0]
        self.robot["leds.bottom.right"] = [0, 255, 0]
        self.robot["leds.bottom.left"] = [0, 255, 0]

        self.robot["motor.left.target"] = self.forward_speed
        self.robot["motor.right.target"] = self.forward_speed

        logger.info(
            "Moving forward: speed=%d",
            self.forward_speed,
        )

        return True

    def smooth_rotate_right(self, angle):
        if self.robot is None:
            self.stop()
            return

        duration = self.angle_to_time(
            angle,
            self.turn_speed,
        )

        logger.info(
            "Smooth rotate RIGHT %.2f deg for %.3f s",
            angle,
            duration,
        )

        start = time.time()

        while time.time() - start < duration:
            if self.check_stop_all_motion():
                self.stop_bool = True
                break

            self.robot["motor.left.target"] = (
                self.turn_speed + self.turn_speed // 2
            )

            self.robot["motor.right.target"] = (
                -self.turn_speed + self.turn_speed // 2
            )

        self.stop()

    def smooth_rotate_left(self, angle):
        if self.robot is None:
            self.stop()
            return

        duration = self.angle_to_time(
            angle,
            self.turn_speed,
        )

        logger.info(
            "Smooth rotate LEFT %.2f deg for %.3f s",
            angle,
            duration,
        )

        start = time.time()

        while time.time() - start < duration:
            if self.check_stop_all_motion():
                self.stop_bool = True
                break

            self.robot["motor.left.target"] = (
                -self.turn_speed + self.turn_speed // 2
            )

            self.robot["motor.right.target"] = (
                self.turn_speed + self.turn_speed // 2
            )

        self.stop()

    def stop(self):
        if self.robot is None:
            return

        try:
            self.robot["motor.left.target"] = 0
            self.robot["motor.right.target"] = 0

            self.robot["leds.top"] = [0, 0, 0]
            self.robot["leds.bottom.right"] = [0, 0, 0]
            self.robot["leds.bottom.left"] = [0, 0, 0]

            self.robot["leds.circle"] = [
                0, 0, 0, 0, 0, 0, 0, 0
            ]

        except Exception:
            logger.exception("Error while stopping Thymio")


# =============================================================================
# ONE ROBAT CONTROL CYCLE
# =============================================================================

def process_one_doa_cycle(
    audio_processor,
    robot_move,
    ring,
    counter,
):
    logger.info("------------------------------------------------------------")

    # 1. Wait until the path is clear.
    if not robot_move.wait_until_path_clear():
        logger.info("Robot stopped; ending processing")
        return counter

    # 2. Acquire one passive DOA measurement.
    logger.info("Path clear. Acquiring one passive DOA measurement")

    doa_start = time.perf_counter()

    peak_angles, db_spl_level, circular_response = (
        audio_processor.update_das_UCA()
    )

    doa_time = time.perf_counter() - doa_start

    # 3. Obstacle check after acquisition.
    if robot_move.obstacle_in_front():
        logger.warning(
            "Obstacle appeared after audio acquisition; discarding DOA"
        )
        robot_move.stop()
        return counter

    counter += 1
    db_spl = float(db_spl_level)

    logger.info(
        "Measurement #%d: strongest UCA angle=%s deg, "
        "SPL=%.2f dB, audio time=%.4f s",
        counter,
        peak_angles,
        db_spl,
        doa_time,
    )

    # 4. Convert UCA order [0, 3, ..., 357] to ring order
    #    [-180, -177, ..., 177].
    #
    # Ring neuron 0 represents -180 degrees, corresponding to UCA 180 degrees
    # (UCA index 60). Therefore roll the UCA response by -60 positions.
    full_array_doa = np.roll(
        circular_response,
        -60,
    )

    if full_array_doa.shape != (RING_NEURONS,):
        raise RuntimeError(
            f"Expected {RING_NEURONS} UCA spatial values; "
            f"got {full_array_doa.shape}"
        )

    # 5. SPL threshold.
    if db_spl <= TRIGGER_LEVEL:
        logger.info(
            "DOA #%d rejected: %.2f dB <= trigger %.2f dB",
            counter,
            db_spl,
            TRIGGER_LEVEL,
        )

        robot_move.time_step += 1
        return counter

    logger.info(
        "DOA #%d accepted: %.2f dB > trigger %.2f dB",
        counter,
        db_spl,
        TRIGGER_LEVEL,
    )

    if robot_move.obstacle_in_front():
        logger.warning(
            "Obstacle before ring processing; discarding current DOA"
        )
        robot_move.stop()
        return counter

    # 6. Run ring attractor using all 120 circular spatial-response values.
    bump_angle, normalized_doa, ring_time = run_ring_attractor(
        ring=ring,
        doa_estimate=full_array_doa,
    )

    robot_move.heading = bump_angle
    robot_move.time_step += 1

    # Save 120 spins + 120 external inputs + bump angle + timestamp.
    timestamp = datetime.datetime.now().isoformat(
        timespec="milliseconds"
    )

    output_exists = os.path.exists("output.csv")

    with open("output.csv", "a", newline="") as output_file:
        writer = csv.writer(output_file)

        if not output_exists or os.path.getsize("output.csv") == 0:
            header = (
                [f"spin_{i}" for i in range(RING_NEURONS)]
                + [f"hext_{i}" for i in range(RING_NEURONS)]
                + ["bump_angle", "timestamp"]
            )
            writer.writerow(header)

        writer.writerow(
            list(ring.spins)
            + list(normalized_doa)
            + [bump_angle, timestamp]
        )

    logger.info(
        "Cycle #%d: peak DOA=%s deg | SPL=%.2f dB | "
        "bump=%.2f deg | ring time=%.4f s",
        counter,
        peak_angles,
        db_spl,
        bump_angle,
        ring_time,
    )

    # 7. Turn, preserving ROBAT's sign convention.
    if robot_move.check_stop_all_motion():
        logger.warning("Safety condition before turn")
        robot_move.stop()
        return counter

    if bump_angle < 0:
        logger.info(
            "Bump negative: rotate RIGHT %.2f deg",
            abs(bump_angle),
        )

        robot_move.smooth_rotate_right(
            abs(bump_angle)
        )

    elif bump_angle > 0:
        logger.info(
            "Bump positive: rotate LEFT %.2f deg",
            bump_angle,
        )

        robot_move.smooth_rotate_left(
            bump_angle
        )

    else:
        logger.info("Bump angle zero: no turn")

    robot_move.heading = 0

    # 8. Move forward for 0.5 seconds, checking safety throughout.
    if robot_move.move_forward():
        movement_start = time.time()

        while time.time() - movement_start < FORWARD_DURATION_S:
            if robot_move.check_stop_all_motion():
                logger.warning(
                    "Safety stop during forward movement"
                )
                break

            time.sleep(0.005)

    else:
        logger.warning("Forward movement was not started")

    # 9. Stop.
    robot_move.stop()
    logger.info("Forward movement complete; robot stopped")

    # 10. Preserve the actual five-second delay.
    logger.info(
        "Waiting %.1f seconds before next DOA",
        POST_MOVEMENT_WAIT_S,
    )

    time.sleep(POST_MOVEMENT_WAIT_S)

    return counter


# =============================================================================
# AUDIO DEVICE SELECTION + MAIN
# =============================================================================

def choose_audio_device(requested_device):
    devices = sd.query_devices()

    print("\nAvailable audio devices:")
    print(sd.query_devices())

    if requested_device is not None:
        info = sd.query_devices(
            requested_device,
            "input",
        )

        if int(info["max_input_channels"]) < CHANNELS:
            raise RuntimeError(
                f"Selected device has {info['max_input_channels']} input "
                f"channels; this script requires {CHANNELS}."
            )

        return requested_device

    # Match the circular-array test script's ReSpeaker device name if available.
    for index, info in enumerate(devices):
        if (
            info["max_input_channels"] >= CHANNELS
            and "respeaker" in info["name"].lower()
        ):
            return index

    raise RuntimeError(
        "Could not identify a ReSpeaker input device with at least five "
        "channels. Run with --list-devices, then pass its index using "
        "--device INDEX."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ROBAT Thymio controller using a circular microphone array"
        )
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (must provide 5 channels)",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio devices and exit",
    )

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    np.random.seed(300)
    random.seed(300)

    logger.info("============================================================")
    logger.info("Starting ROBAT circular-array controller")

    logger.info(
        "fs=%d | channels=%d | radius=%.4f m | shift=%.1f deg",
        FS,
        CHANNELS,
        ARRAY_RADIUS,
        ARRAY_SHIFT_DEG,
    )

    logger.info(
        "DOA band=%s Hz | SPL trigger=%.1f | critical=%.1f",
        DOA_FREQ_RANGE,
        TRIGGER_LEVEL,
        CRITICAL_LEVEL,
    )

    logger.info("============================================================")

    device_index = choose_audio_device(args.device)

    ring = RingAttractor()
    audio_processor = CircularAudioProcessor(device_index)
    robot_move = None

    try:
        # Initialize robot before starting movement decisions.
        robot_move = RobotMove(
            ring=ring,
            ground_sensors_bool=False,
        )

        # Start continuous passive audio acquisition.
        audio_processor.start()

        logger.info("Waiting for the first complete audio window")

        while audio_processor.get_latest_chunk() is None:
            time.sleep(0.02)

        # Sequential robot-control loop.
        counter = 0

        logger.info("Entering sequential ROBAT control loop")

        while robot_move.running:
            cycle_start = time.time()

            counter = process_one_doa_cycle(
                audio_processor=audio_processor,
                robot_move=robot_move,
                ring=ring,
                counter=counter,
            )

            logger.info(
                "End cycle #%d | elapsed=%.3f s",
                counter,
                time.time() - cycle_start,
            )

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")

    except Exception:
        logger.exception(
            "Fatal error in circular ROBAT controller"
        )
        raise

    finally:
        if robot_move is not None:
            robot_move.running = False
            robot_move.stop()

        audio_processor.stop()

        logger.info(
            "Controller terminated; robot stopped; audio stream closed"
        )


if __name__ == "__main__":
    main()