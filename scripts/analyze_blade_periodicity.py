#!/usr/bin/env python3
"""
Analyze periodicity in blade rotation using FFT and autocorrelation.
Extracts rotation frequency (RPM) from blade angle measurements.

Usage:
    python analyze_blade_periodicity.py --config config/tracking_config.yaml --preset drone_idle
"""

import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse
import yaml
from scipy import signal
from scipy.fft import fft, fftfreq
from evio.core.recording import open_dat


def load_config(config_path, preset=None):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply preset if specified
    if preset and "presets" in config and preset in config["presets"]:
        preset_config = config["presets"][preset]
        for key, value in preset_config.items():
            if isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

    return config


def load_dat_file(filepath, width=1280, height=720):
    """Load .dat file with event camera data using evio library."""
    rec = open_dat(filepath, width=width, height=height)

    # Decode event words
    x = rec.event_words & 0x3FFF
    y = (rec.event_words >> 14) & 0x3FFF
    p = ((rec.event_words >> 28) & 0xF).astype(np.int32)
    p = (p > 0).astype(np.int32) * 2 - 1

    # Apply time ordering
    x = x[rec.order]
    y = y[rec.order]
    p = p[rec.order]
    t = rec.timestamps

    events = np.column_stack([x, y, p, t])

    return events, rec.width, rec.height


def filter_hot_pixels(events, threshold=100):
    """Filter out hot pixels."""
    unique_coords, counts = np.unique(events[:, :2], axis=0, return_counts=True)
    hot_pixel_mask = counts > threshold
    hot_pixels = unique_coords[hot_pixel_mask]

    if len(hot_pixels) > 0:
        print(f"  Found {len(hot_pixels)} hot pixel locations")
        for hp in hot_pixels:
            hot_pixel_events = np.sum(
                (events[:, 0] == hp[0]) & (events[:, 1] == hp[1])
            )
        keep_mask = ~np.isin(events[:, :2], hot_pixels).all(axis=1)
        return events[keep_mask]
    return events


class BladeCluster:
    """Cluster representing a propeller blade with temporal tracking."""

    def __init__(self, cluster_id, initial_events, color=None):
        self.id = cluster_id
        self.events = initial_events.copy()
        self.color = color if color else plt.cm.tab10(cluster_id % 10)[:3]

        # Center tracking with history for stability
        self.center_history = []
        self.center_x = 0
        self.center_y = 0
        self.confidence = 0

        # Angle tracking
        self.angle_history = []
        self.confidence_history = []

        self.update_statistics()

    def update_statistics(self):
        """Update cluster statistics: center, slope, angle."""
        if len(self.events) < 2:
            self.slope = 0
            self.angle_deg = 0
            if len(self.events) == 1:
                self.center_history.append(
                    (self.events[0, 3], self.events[0, 0], self.events[0, 1])
                )
                self.center_x = self.events[:, 0].mean()
                self.center_y = self.events[:, 1].mean()
            return

        # Fit line to events
        X = self.events[:, 0].reshape(-1, 1)
        y = self.events[:, 1]
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression().fit(X, y)
        self.slope = reg.coef_[0]
        self.angle_deg = np.degrees(np.arctan(self.slope))

        # Calculate confidence based on line-ness
        # Project points onto fitted line and perpendicular
        points = np.column_stack([self.events[:, 0], self.events[:, 1]])
        center = np.array([points[:, 0].mean(), points[:, 1].mean()])

        # Direction vector along the line
        line_vec = np.array([1, self.slope])
        line_vec = line_vec / np.linalg.norm(line_vec)

        # Project points
        proj_along = np.dot(points - center, line_vec)
        var_along = np.var(proj_along) if len(proj_along) > 1 else 0

        # Perpendicular direction
        perp_vec = np.array([-self.slope, 1])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        proj_perp = np.dot(points - center, perp_vec)
        var_perp = np.var(proj_perp) if len(proj_perp) > 1 else 0

        # Confidence: log ratio of variance along vs perpendicular
        # Higher ratio = more line-like = higher confidence
        if var_perp > 0:
            ratio = var_along / var_perp
            # Use log scale for better sensitivity
            log_ratio = np.log10(max(ratio, 1))
            # Map to 0-1 range (log10(100) = 2 -> confidence 1.0)
            self.confidence = min(log_ratio / 2.0, 1.0)
        else:
            self.confidence = 1.0

        # Record angle history with timestamp
        if len(self.events) > 0:
            current_timestamp = self.events[:, 3].max()
            self.angle_history.append((current_timestamp, self.angle_deg))
            self.confidence_history.append((current_timestamp, self.confidence))

        # Update instantaneous center
        self.center_x = self.events[:, 0].mean()
        self.center_y = self.events[:, 1].mean()

        # Record in center history with timestamp
        if len(self.events) > 0:
            timestamp = self.events[:, 3].max()
            self.center_history.append(
                (timestamp, self.center_x, self.center_y)
            )
            self.slope = self.slope
        else:
            return

        # Use center history for stable center estimation
        if len(self.center_history) > 3:
            centers = np.array([(h[1], h[2]) for h in self.center_history])
            self.center_x = centers[:, 0].mean()
            self.center_y = centers[:, 1].mean()

    def add_event(self, event):
        """Add event to cluster."""
        self.events = np.vstack([self.events, event.reshape(1, -1)])

    def remove_old_events(self, cutoff_time_us):
        """Remove events older than cutoff time."""
        mask = self.events[:, 3] >= cutoff_time_us
        self.events = self.events[mask]
        if len(self.events) > 0:
            self.update_statistics()

    def remove_old_center_history(self, cutoff_time_us):
        """Remove old center history entries."""
        self.center_history = [
            h for h in self.center_history if h[0] >= cutoff_time_us
        ]

    def __len__(self):
        return len(self.events)

    def distance_to_point(self, x, y):
        """Distance from point to cluster center."""
        return np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)


class TemporalTracker:
    """Real-time blade tracker."""

    def __init__(
        self,
        assignment_distance=15,
        min_cluster_size=5,
        window_duration_us=750,
        center_history_window_us=3000,
        initial_eps=6,
        initial_min_samples=10,
    ):
        self.clusters = {}
        self.next_cluster_id = 0
        self.assignment_distance = assignment_distance
        self.min_cluster_size = min_cluster_size
        self.window_duration_us = window_duration_us
        self.center_history_window_us = center_history_window_us
        self.initial_eps = initial_eps
        self.initial_min_samples = initial_min_samples
        self.current_time_us = 0

    def initialize_from_events(self, events):
        """Initialize clusters from initial batch of events."""
        if len(events) < 10:
            return

        # Use DBSCAN for initial clustering
        coords = events[:, :2]
        db = DBSCAN(eps=self.initial_eps, min_samples=self.initial_min_samples)
        labels = db.fit_predict(coords)

        # Create clusters from DBSCAN results
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue

            cluster_mask = labels == label
            cluster_events = events[cluster_mask]

            if len(cluster_events) >= self.min_cluster_size:
                cluster = BladeCluster(self.next_cluster_id, cluster_events)
                self.clusters[self.next_cluster_id] = cluster
                self.next_cluster_id += 1

        if len(events) > 0:
            self.current_time_us = events[:, 3].max()

    def process_event(self, event):
        """Process single event: assign to cluster, update statistics."""
        x, y, polarity, timestamp = event
        self.current_time_us = timestamp

        # Try to assign to existing cluster
        min_distance = float("inf")
        closest_cluster_id = None

        for cluster_id, cluster in self.clusters.items():
            distance = cluster.distance_to_point(x, y)
            if distance < self.assignment_distance and distance < min_distance:
                min_distance = distance
                closest_cluster_id = cluster_id

        if closest_cluster_id is not None:
            self.clusters[closest_cluster_id].add_event(event)
        else:
            new_cluster = BladeCluster(
                self.next_cluster_id, event.reshape(1, -1)
            )
            self.clusters[self.next_cluster_id] = new_cluster
            self.next_cluster_id += 1

        # Remove old events from all clusters
        cutoff_time = self.current_time_us - self.window_duration_us
        center_cutoff_time = (
            self.current_time_us - self.center_history_window_us
        )

        clusters_to_remove = []

        for cluster_id, cluster in self.clusters.items():
            cluster.remove_old_events(cutoff_time)
            cluster.remove_old_center_history(center_cutoff_time)
            if len(cluster) < self.min_cluster_size:
                clusters_to_remove.append(cluster_id)

        for cluster_id in clusters_to_remove:
            del self.clusters[cluster_id]


def analyze_periodicity(times_ms, angles_deg, cluster_id):
    """Analyze periodicity in blade angle data using FFT and autocorrelation."""
    print(f"\n{'=' * 60}")
    print(f"Periodicity Analysis for Blade ID{cluster_id}")
    print(f"{'=' * 60}")

    # Remove duplicate timestamps
    unique_mask = np.concatenate([[True], np.diff(times_ms) > 0])
    times_ms = times_ms[unique_mask]
    angles_deg = angles_deg[unique_mask]

    if len(times_ms) < 10:
        print(f"  Not enough unique samples after filtering ({len(times_ms)})")
        return None

    # Basic stats
    duration_ms = times_ms[-1] - times_ms[0]
    duration_s = duration_ms / 1000.0
    n_samples = len(times_ms)

    print(f"\nData Overview:")
    print(f"  Duration: {duration_ms:.3f} ms ({duration_s:.6f} s)")
    print(f"  Samples: {n_samples}")
    print(f"  Sample rate: {n_samples / duration_s:.1f} Hz")

    # Resample to uniform time grid for FFT
    # Use median time step (but ensure it's not zero)
    dt_samples = np.diff(times_ms)
    median_dt = np.median(dt_samples)
    mean_dt = np.mean(dt_samples)
    print(f"  Time step - median: {median_dt:.6f} ms, mean: {mean_dt:.6f} ms")

    # Use a reasonable time step for resampling
    # Aim for ~1000-10000 samples
    target_samples = min(10000, n_samples)
    resample_dt = duration_ms / target_samples

    if resample_dt <= 0:
        resample_dt = mean_dt if mean_dt > 0 else 0.001

    print(f"  Resampling with dt = {resample_dt:.6f} ms")

    # Create uniform time grid
    t_uniform = np.arange(times_ms[0], times_ms[-1], resample_dt)

    # Interpolate angles onto uniform grid
    angles_uniform = np.interp(t_uniform, times_ms, angles_deg)

    n_uniform = len(t_uniform)
    sample_rate = 1000.0 / resample_dt  # Hz (convert from ms)

    print(f"  Resampled to {n_uniform} uniform samples at {sample_rate:.1f} Hz")

    # Remove DC component (mean)
    angles_centered = angles_uniform - np.mean(angles_uniform)

    # Apply window to reduce spectral leakage
    window = signal.windows.hann(n_uniform)
    angles_windowed = angles_centered * window

    # Compute FFT
    fft_vals = fft(angles_windowed)
    freqs = fftfreq(n_uniform, d=median_dt / 1000.0)  # Convert to seconds

    # Only positive frequencies
    positive_freq_mask = freqs > 0
    freqs_pos = freqs[positive_freq_mask]
    fft_mag = np.abs(fft_vals[positive_freq_mask])

    # Find dominant frequencies
    if len(fft_mag) > 0 and np.max(fft_mag) > 0:
        peak_indices = signal.find_peaks(fft_mag, height=np.max(fft_mag) * 0.1)[
            0
        ]

        print(f"\n  Top 5 Frequency Peaks:")
        sorted_peaks = sorted(
            peak_indices, key=lambda i: fft_mag[i], reverse=True
        )[:5]

        for i, peak_idx in enumerate(sorted_peaks):
            freq_hz = freqs_pos[peak_idx]
            rpm = freq_hz * 60
            magnitude = fft_mag[peak_idx]
            print(
                f"    {i + 1}. Frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM), Magnitude: {magnitude:.1f}"
            )
    else:
        print(f"\n  No significant frequency peaks detected")

    # Autocorrelation analysis
    print(f"\n  Autocorrelation Analysis:")
    autocorr = np.correlate(angles_centered, angles_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Only positive lags
    autocorr = autocorr / autocorr[0]  # Normalize

    # Find peaks in autocorrelation
    autocorr_peaks = signal.find_peaks(
        autocorr, height=0.3, distance=int(n_uniform * 0.05)
    )[0]

    if len(autocorr_peaks) > 0:
        first_peak = autocorr_peaks[0]
        period_samples = first_peak
        period_ms = period_samples * resample_dt
        period_s = period_ms / 1000.0
        if period_s > 0:
            freq_hz = 1.0 / period_s
            rpm = freq_hz * 60

            print(
                f"    First autocorrelation peak at lag {period_samples} samples"
            )
            print(f"    Period: {period_ms:.3f} ms ({period_s:.6f} s)")
            print(f"    Frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")
        else:
            print(f"    Invalid period detected")
    else:
        print(f"    No clear periodic pattern detected in autocorrelation")

    # Estimate RPM from angle jumps (alternative method)
    angle_diffs = np.diff(angles_deg)
    large_jumps = np.abs(angle_diffs) > 15  # Large angle changes
    jump_indices = np.where(large_jumps)[0]

    if len(jump_indices) > 1:
        jump_times = times_ms[jump_indices]
        jump_intervals = np.diff(jump_times)

        if len(jump_intervals) > 0:
            mean_interval_ms = np.mean(jump_intervals)
            mean_interval_s = mean_interval_ms / 1000.0
            freq_from_jumps = 1.0 / mean_interval_s
            rpm_from_jumps = freq_from_jumps * 60

            print(f"\n  Angle Jump Analysis:")
            print(f"    Found {len(jump_indices)} large angle jumps")
            print(f"    Mean interval: {mean_interval_ms:.3f} ms")
            print(
                f"    Estimated frequency: {freq_from_jumps:.2f} Hz ({rpm_from_jumps:.0f} RPM)"
            )

    return {
        "freqs": freqs_pos,
        "fft_mag": fft_mag,
        "times_uniform": t_uniform,
        "angles_uniform": angles_uniform,
        "autocorr": autocorr,
        "sample_rate": sample_rate,
        "duration_s": duration_s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze blade rotation periodicity from event camera data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tracking_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name to use from config file",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config, preset=args.preset)
    if args.preset:
        print(f"Using preset: {args.preset}")

    # Load data
    dat_file = Path(__file__).parent.parent / config["data"]["input_file"]
    print(f"Loading {dat_file}...")
    events, width, height = load_dat_file(
        dat_file, config["data"]["width"], config["data"]["height"]
    )
    print(f"Loaded {len(events)} events")

    # Parameters from config
    start_time_sec = config["data"]["start_time_sec"]
    duration_sec = config["data"]["duration_sec"]
    polarity = config["data"]["polarity"]
    min_time = events[:, 3].min()
    start_time = min_time + int(start_time_sec * 1_000_000)
    init_window_us = config["clustering"]["temporal"]["window_duration_us"]

    # Filter events
    end_time = start_time + int(duration_sec * 1_000_000)
    mask = (events[:, 3] >= start_time) & (events[:, 3] < end_time)
    if polarity is not None:
        mask = mask & (events[:, 2] == polarity)
    filtered_events = events[mask]

    print(f"Filtered {len(filtered_events)} events")

    # Filter hot pixels
    filtered_events = filter_hot_pixels(
        filtered_events, config["hot_pixel"]["threshold"]
    )
    print(f"After hot pixel filtering: {len(filtered_events)} events")

    if len(filtered_events) < 100:
        print("Not enough events!")
        return

    # Initialize tracker
    init_end_time = start_time + init_window_us
    init_mask = filtered_events[:, 3] < init_end_time
    init_events = filtered_events[init_mask]
    remaining_events = filtered_events[~init_mask]

    print(f"\nInitializing tracker with {len(init_events)} events...")

    center_history_window_us = (
        config["clustering"]["temporal"]["window_duration_us"] * 4
    )

    tracker = TemporalTracker(
        assignment_distance=config["clustering"]["temporal"][
            "assignment_distance"
        ],
        min_cluster_size=config["clustering"]["temporal"]["min_cluster_size"],
        window_duration_us=config["clustering"]["temporal"][
            "window_duration_us"
        ],
        center_history_window_us=center_history_window_us,
        initial_eps=config["clustering"]["initial"]["eps"],
        initial_min_samples=config["clustering"]["initial"]["min_samples"],
    )
    tracker.initialize_from_events(init_events)
    print(f"Initialized {len(tracker.clusters)} clusters")

    # Process all events
    print(f"Processing {len(remaining_events)} events...")
    for i, event in enumerate(remaining_events):
        tracker.process_event(event)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(remaining_events)} events...")

    print(f"Final clusters: {len(tracker.clusters)}")

    # Analyze periodicity for each cluster
    results = {}
    for cluster_id, cluster in tracker.clusters.items():
        if len(cluster.angle_history) > 100:
            times_ms = np.array([t / 1000.0 for t, _ in cluster.angle_history])
            angles_deg = np.array([a for _, a in cluster.angle_history])

            result = analyze_periodicity(times_ms, angles_deg, cluster_id)
            if result is not None:
                results[cluster_id] = result

    # Create visualization
    if len(results) > 0:
        n_clusters = len(results)
        fig, axes = plt.subplots(n_clusters, 3, figsize=(18, 5 * n_clusters))

        if n_clusters == 1:
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, result) in enumerate(results.items()):
            # Plot 1: Angle vs Time with periodic markers
            ax1 = axes[idx, 0]
            cluster = tracker.clusters[cluster_id]
            times_ms = np.array([t / 1000.0 for t, _ in cluster.angle_history])
            angles_deg = np.array([a for _, a in cluster.angle_history])

            ax1.plot(
                times_ms,
                angles_deg,
                "b-",
                alpha=0.7,
                linewidth=0.5,
                label="Angle",
            )

            # Mark angle jumps
            angle_diffs = np.diff(angles_deg)
            large_jumps = np.abs(angle_diffs) > 15
            jump_indices = np.where(large_jumps)[0] + 1
            if len(jump_indices) > 0:
                ax1.scatter(
                    times_ms[jump_indices],
                    angles_deg[jump_indices],
                    c="red",
                    s=50,
                    marker="o",
                    alpha=0.7,
                    zorder=5,
                    label=f"Large jumps ({len(jump_indices)})",
                )

            # Show FFT-derived period as vertical lines
            if len(result["fft_mag"]) > 0:
                # Get dominant frequency
                peak_idx = np.argmax(result["fft_mag"])
                dominant_freq = result["freqs"][peak_idx]
                period_ms = 1000.0 / dominant_freq if dominant_freq > 0 else 0

                if period_ms > 0:
                    # Draw vertical lines at expected period intervals
                    n_periods = int((times_ms[-1] - times_ms[0]) / period_ms)
                    for i in range(1, n_periods + 1):
                        t_mark = times_ms[0] + i * period_ms
                        if t_mark <= times_ms[-1]:
                            ax1.axvline(
                                t_mark,
                                color="green",
                                linestyle="--",
                                alpha=0.4,
                                linewidth=1.5,
                            )

                    # Add text annotation
                    ax1.text(
                        0.02,
                        0.98,
                        f"FFT Period: {period_ms:.3f} ms\n({dominant_freq:.1f} Hz, {dominant_freq * 60:.0f} RPM)",
                        transform=ax1.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round", facecolor="white", alpha=0.8
                        ),
                    )

            ax1.set_xlabel("Time (ms)", fontsize=10)
            ax1.set_ylabel("Angle (degrees)", fontsize=10)
            ax1.set_title(
                f"Blade ID{cluster_id}: Angle vs Time (green lines = FFT period, red dots = jumps)",
                fontsize=12,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right", fontsize=8)

            # Plot 2: FFT Spectrum with peak markers
            ax2 = axes[idx, 1]
            ax2.semilogy(result["freqs"], result["fft_mag"], "b-", linewidth=1)

            # Mark top 3 peaks
            if len(result["fft_mag"]) > 0:
                peak_indices = signal.find_peaks(
                    result["fft_mag"], height=np.max(result["fft_mag"]) * 0.1
                )[0]
                if len(peak_indices) > 0:
                    sorted_peaks = sorted(
                        peak_indices,
                        key=lambda i: result["fft_mag"][i],
                        reverse=True,
                    )[:3]

                    colors = ["red", "orange", "yellow"]
                    for i, peak_idx in enumerate(sorted_peaks):
                        freq_hz = result["freqs"][peak_idx]
                        rpm = freq_hz * 60
                        ax2.scatter(
                            freq_hz,
                            result["fft_mag"][peak_idx],
                            c=colors[i],
                            s=100,
                            marker="*",
                            zorder=5,
                            edgecolors="black",
                            linewidths=1,
                        )
                        ax2.annotate(
                            f"{freq_hz:.1f} Hz\n({rpm:.0f} RPM)",
                            xy=(freq_hz, result["fft_mag"][peak_idx]),
                            xytext=(10, 10),
                            textcoords="offset points",
                            fontsize=8,
                            bbox=dict(
                                boxstyle="round", facecolor=colors[i], alpha=0.7
                            ),
                        )

            ax2.set_xlabel("Frequency (Hz)", fontsize=10)
            ax2.set_ylabel("Magnitude", fontsize=10)
            ax2.set_title(
                f"Blade ID{cluster_id}: Frequency Spectrum (stars = top peaks)",
                fontsize=12,
                fontweight="bold",
            )
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, min(500, result["freqs"][-1]))

            # Plot 3: Autocorrelation with peak markers
            ax3 = axes[idx, 2]
            lag_ms = np.arange(len(result["autocorr"])) * np.median(
                np.diff(result["times_uniform"])
            )
            ax3.plot(
                lag_ms,
                result["autocorr"],
                "b-",
                linewidth=1,
                label="Autocorrelation",
            )

            # Find and mark peaks
            autocorr_peaks = signal.find_peaks(
                result["autocorr"],
                height=0.3,
                distance=int(len(result["autocorr"]) * 0.05),
            )[0]

            if len(autocorr_peaks) > 0:
                # Mark first 3 peaks
                for i, peak_idx in enumerate(autocorr_peaks[:3]):
                    peak_lag_ms = lag_ms[peak_idx]
                    peak_val = result["autocorr"][peak_idx]

                    color = ["red", "orange", "yellow"][i] if i < 3 else "green"
                    ax3.scatter(
                        peak_lag_ms,
                        peak_val,
                        c=color,
                        s=100,
                        marker="^",
                        zorder=5,
                        edgecolors="black",
                        linewidths=1,
                    )

                    if i == 0:  # Annotate first peak
                        period_s = peak_lag_ms / 1000.0
                        freq_hz = 1.0 / period_s if period_s > 0 else 0
                        rpm = freq_hz * 60
                        ax3.annotate(
                            f"Period: {peak_lag_ms:.2f} ms\n({freq_hz:.1f} Hz, {rpm:.0f} RPM)",
                            xy=(peak_lag_ms, peak_val),
                            xytext=(10, -20),
                            textcoords="offset points",
                            fontsize=8,
                            bbox=dict(
                                boxstyle="round", facecolor=color, alpha=0.7
                            ),
                            arrowprops=dict(
                                arrowstyle="->", connectionstyle="arc3,rad=0"
                            ),
                        )

            ax3.set_xlabel("Lag (ms)", fontsize=10)
            ax3.set_ylabel("Autocorrelation", fontsize=10)
            ax3.set_title(
                f"Blade ID{cluster_id}: Autocorrelation (triangles = peaks)",
                fontsize=12,
                fontweight="bold",
            )
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color="k", linestyle="--", linewidth=0.5)
            # Limit x-axis to first 20% of data for clarity
            ax3.set_xlim(0, lag_ms[-1] * 0.2)
            ax3.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        output_file = "blade_periodicity_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\n{'=' * 60}")
        print(f"Saved periodicity analysis plot: {output_file}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
