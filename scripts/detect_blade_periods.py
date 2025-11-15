#!/usr/bin/env python3
"""
Detect blade rotation periods using multiple methods suited for sawtooth/reset patterns.

Methods:
1. Angle reset detection (jumps back after decrease)
2. Cluster lifecycle analysis (blade disappearance/reappearance)
3. Peak-to-peak interval analysis
4. Derivative zero-crossing detection
5. Wavelet analysis

Usage:
    python detect_blade_periods.py --config config/tracking_config.yaml --preset drone_idle
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
from evio.core.recording import open_dat


def load_config(config_path, preset=None):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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

    x = rec.event_words & 0x3FFF
    y = (rec.event_words >> 14) & 0x3FFF
    p = ((rec.event_words >> 28) & 0xF).astype(np.int32)
    p = (p > 0).astype(np.int32) * 2 - 1

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
        keep_mask = ~np.isin(events[:, :2], hot_pixels).all(axis=1)
        return events[keep_mask]
    return events


class BladeCluster:
    """Cluster representing a propeller blade with temporal tracking."""

    def __init__(self, cluster_id, initial_events, color=None):
        self.id = cluster_id
        self.events = initial_events.copy()
        self.color = color if color else plt.cm.tab10(cluster_id % 10)[:3]
        self.center_history = []
        self.center_x = 0
        self.center_y = 0
        self.confidence = 0
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

        from sklearn.linear_model import LinearRegression

        X = self.events[:, 0].reshape(-1, 1)
        y = self.events[:, 1]
        reg = LinearRegression().fit(X, y)
        self.slope = reg.coef_[0]
        self.angle_deg = np.degrees(np.arctan(self.slope))

        points = np.column_stack([self.events[:, 0], self.events[:, 1]])
        center = np.array([points[:, 0].mean(), points[:, 1].mean()])

        line_vec = np.array([1, self.slope])
        line_vec = line_vec / np.linalg.norm(line_vec)
        proj_along = np.dot(points - center, line_vec)
        var_along = np.var(proj_along) if len(proj_along) > 1 else 0

        perp_vec = np.array([-self.slope, 1])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        proj_perp = np.dot(points - center, perp_vec)
        var_perp = np.var(proj_perp) if len(proj_perp) > 1 else 0

        if var_perp > 0:
            ratio = var_along / var_perp
            log_ratio = np.log10(max(ratio, 1))
            self.confidence = min(log_ratio / 2.0, 1.0)
        else:
            self.confidence = 1.0

        if len(self.events) > 0:
            current_timestamp = self.events[:, 3].max()
            self.angle_history.append((current_timestamp, self.angle_deg))
            self.confidence_history.append((current_timestamp, self.confidence))

        self.center_x = self.events[:, 0].mean()
        self.center_y = self.events[:, 1].mean()

        if len(self.events) > 0:
            timestamp = self.events[:, 3].max()
            self.center_history.append(
                (timestamp, self.center_x, self.center_y)
            )

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
        self.cluster_lifecycle = []  # Track (cluster_id, birth_time, death_time)

    def initialize_from_events(self, events):
        """Initialize clusters from initial batch of events."""
        if len(events) < 10:
            return

        coords = events[:, :2]
        db = DBSCAN(eps=self.initial_eps, min_samples=self.initial_min_samples)
        labels = db.fit_predict(coords)

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue

            cluster_mask = labels == label
            cluster_events = events[cluster_mask]

            if len(cluster_events) >= self.min_cluster_size:
                cluster = BladeCluster(self.next_cluster_id, cluster_events)
                self.clusters[self.next_cluster_id] = cluster
                birth_time = cluster_events[:, 3].min()
                self.cluster_lifecycle.append(
                    (self.next_cluster_id, birth_time, None)
                )
                self.next_cluster_id += 1

        if len(events) > 0:
            self.current_time_us = events[:, 3].max()

    def process_event(self, event):
        """Process single event: assign to cluster, update statistics."""
        x, y, polarity, timestamp = event
        self.current_time_us = timestamp

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
            self.cluster_lifecycle.append(
                (self.next_cluster_id, timestamp, None)
            )
            self.next_cluster_id += 1

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
            # Record death time
            for i, (cid, birth, death) in enumerate(self.cluster_lifecycle):
                if cid == cluster_id and death is None:
                    self.cluster_lifecycle[i] = (
                        cid,
                        birth,
                        self.current_time_us,
                    )
                    break
            del self.clusters[cluster_id]


def method1_angle_resets(times_ms, angles_deg, cluster_id):
    """Method 1: Detect angle resets (sawtooth pattern)."""
    print(f"\n--- Method 1: Angle Reset Detection ---")

    # Look for large positive jumps (angle resets)
    angle_diffs = np.diff(angles_deg)

    # A reset is a large positive jump after decreasing trend
    # We're looking for jumps > 20 degrees
    reset_threshold = 20
    resets = angle_diffs > reset_threshold
    reset_indices = np.where(resets)[0] + 1

    if len(reset_indices) < 2:
        print(
            f"  Found {len(reset_indices)} resets - not enough for period estimation"
        )
        return None

    reset_times = times_ms[reset_indices]
    periods = np.diff(reset_times)

    print(f"  Found {len(reset_indices)} angle resets")
    print(f"  Reset times: {reset_times[:10]} ms...")
    print(f"  Periods: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")
    print(f"  Std period: {np.std(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "reset_times": reset_times,
        "periods": periods,
        "mean_period": np.mean(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def method2_cluster_lifecycle(lifecycle_events, cluster_id):
    """Method 2: Analyze cluster birth/death cycles."""
    print(f"\n--- Method 2: Cluster Lifecycle Analysis ---")

    # Get all events for this cluster
    cluster_events = [
        (birth, death)
        for cid, birth, death in lifecycle_events
        if cid == cluster_id and death is not None
    ]

    if len(cluster_events) < 2:
        print(f"  Found {len(cluster_events)} complete lifecycles - not enough")
        return None

    birth_times = np.array([birth / 1000.0 for birth, death in cluster_events])
    death_times = np.array([death / 1000.0 for birth, death in cluster_events])
    lifespans = death_times - birth_times

    # Period is time between births (or deaths)
    periods = np.diff(birth_times)

    print(f"  Found {len(cluster_events)} complete lifecycles")
    print(f"  Mean lifespan: {np.mean(lifespans):.3f} ms")
    print(f"  Periods between births: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "birth_times": birth_times,
        "death_times": death_times,
        "periods": periods,
        "mean_period": np.mean(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def method3_peak_detection(times_ms, angles_deg, cluster_id):
    """Method 3: Detect peaks/valleys in angle signal."""
    print(f"\n--- Method 3: Peak/Valley Detection ---")

    # Find local maxima and minima
    peaks, _ = signal.find_peaks(angles_deg, distance=10, prominence=5)
    valleys, _ = signal.find_peaks(-angles_deg, distance=10, prominence=5)

    if len(peaks) < 2 and len(valleys) < 2:
        print(
            f"  Found {len(peaks)} peaks, {len(valleys)} valleys - not enough"
        )
        return None

    # Use whichever has more detections
    if len(peaks) >= len(valleys):
        feature_indices = peaks
        feature_name = "peaks"
    else:
        feature_indices = valleys
        feature_name = "valleys"

    feature_times = times_ms[feature_indices]
    periods = np.diff(feature_times)

    print(f"  Found {len(feature_indices)} {feature_name}")
    print(f"  {feature_name.capitalize()} at times: {feature_times[:10]} ms...")
    print(f"  Periods: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "feature_times": feature_times,
        "feature_type": feature_name,
        "periods": periods,
        "mean_period": np.mean(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def method4_derivative_analysis(times_ms, angles_deg, cluster_id):
    """Method 4: Analyze angle derivative for transitions."""
    print(f"\n--- Method 4: Derivative Analysis ---")

    # Compute derivative
    angle_derivative = np.diff(angles_deg) / np.diff(times_ms)
    derivative_times = times_ms[:-1]

    # Look for large negative derivatives (fast decreases)
    # and large positive derivatives (resets)
    negative_threshold = np.percentile(angle_derivative, 10)
    positive_threshold = np.percentile(angle_derivative, 90)

    # Find transitions
    negative_transitions = angle_derivative < negative_threshold
    positive_transitions = angle_derivative > positive_threshold

    # Use positive transitions (resets) as markers
    reset_indices = np.where(positive_transitions)[0]

    if len(reset_indices) < 2:
        print(
            f"  Found {len(reset_indices)} derivative-based resets - not enough"
        )
        return None

    reset_times = derivative_times[reset_indices]
    periods = np.diff(reset_times)

    print(f"  Found {len(reset_indices)} derivative resets")
    print(
        f"  Derivative range: [{np.min(angle_derivative):.2f}, {np.max(angle_derivative):.2f}] deg/ms"
    )
    print(f"  Periods: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "reset_times": reset_times,
        "angle_derivative": angle_derivative,
        "derivative_times": derivative_times,
        "periods": periods,
        "mean_period": np.mean(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def method5_threshold_crossing(times_ms, angles_deg, cluster_id):
    """Method 5: Detect threshold crossings (e.g., angle crossing 0)."""
    print(f"\n--- Method 5: Threshold Crossing Analysis ---")

    # Use median angle as threshold
    threshold = np.median(angles_deg)

    # Find upward crossings
    above_threshold = angles_deg > threshold
    crossings = np.diff(above_threshold.astype(int))
    upward_crossings = crossings > 0
    downward_crossings = crossings < 0

    up_indices = np.where(upward_crossings)[0] + 1
    down_indices = np.where(downward_crossings)[0] + 1

    # Use whichever has more crossings
    if len(up_indices) >= len(down_indices):
        crossing_indices = up_indices
        crossing_type = "upward"
    else:
        crossing_indices = down_indices
        crossing_type = "downward"

    if len(crossing_indices) < 2:
        print(
            f"  Found {len(crossing_indices)} {crossing_type} crossings - not enough"
        )
        return None

    crossing_times = times_ms[crossing_indices]
    periods = np.diff(crossing_times)

    print(f"  Threshold: {threshold:.2f} degrees")
    print(f"  Found {len(crossing_indices)} {crossing_type} crossings")
    print(f"  Crossing times: {crossing_times[:10]} ms...")
    print(f"  Periods: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "crossing_times": crossing_times,
        "crossing_type": crossing_type,
        "threshold": threshold,
        "periods": periods,
        "mean_period": np.mean(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def method6_large_angle_transition(times_ms, angles_deg, cluster_id):
    """Method 6: Detect large angle transitions (bidirectional: <-30 to >+30 OR >+30 to <-30)."""
    print(
        f"\n--- Method 6: Large Angle Transition Detection (Bidirectional) ---"
    )

    # Define thresholds
    low_threshold = -30
    high_threshold = 30
    max_transition_time = 2.0  # ms - max time for transition to be valid
    min_time_between_transitions = (
        3.0  # ms - minimum time between consecutive transitions
    )

    # Find transitions in both directions
    transitions = []
    last_transition_time = (
        -min_time_between_transitions
    )  # Allow first transition

    for i in range(len(angles_deg) - 1):
        # Skip if too close to last transition
        if times_ms[i] - last_transition_time < min_time_between_transitions:
            continue

        # Look ahead for a transition (both directions)
        if angles_deg[i] < low_threshold:
            # Search forward for angle > high_threshold (LOW to HIGH)
            for j in range(i + 1, len(angles_deg)):
                time_diff = times_ms[j] - times_ms[i]

                if time_diff > max_transition_time:
                    break  # Too far ahead

                if angles_deg[j] > high_threshold:
                    # Found a LOW to HIGH transition!
                    transitions.append(
                        {
                            "start_idx": i,
                            "end_idx": j,
                            "start_time": times_ms[i],
                            "end_time": times_ms[j],
                            "start_angle": angles_deg[i],
                            "end_angle": angles_deg[j],
                            "duration": time_diff,
                            "angle_change": angles_deg[j] - angles_deg[i],
                            "direction": "low_to_high",
                        }
                    )
                    last_transition_time = times_ms[i]
                    break  # Found transition, move to next search

        elif angles_deg[i] > high_threshold:
            # Search forward for angle < low_threshold (HIGH to LOW)
            for j in range(i + 1, len(angles_deg)):
                time_diff = times_ms[j] - times_ms[i]

                if time_diff > max_transition_time:
                    break  # Too far ahead

                if angles_deg[j] < low_threshold:
                    # Found a HIGH to LOW transition!
                    transitions.append(
                        {
                            "start_idx": i,
                            "end_idx": j,
                            "start_time": times_ms[i],
                            "end_time": times_ms[j],
                            "start_angle": angles_deg[i],
                            "end_angle": angles_deg[j],
                            "duration": time_diff,
                            "angle_change": angles_deg[j] - angles_deg[i],
                            "direction": "high_to_low",
                        }
                    )
                    last_transition_time = times_ms[i]
                    break  # Found transition, move to next search

    if len(transitions) < 2:
        print(
            f"  Found {len(transitions)} large angle transitions - not enough"
        )
        return None

    # Extract transition times and compute periods
    transition_times = np.array([t["start_time"] for t in transitions])
    periods = np.diff(transition_times)

    # Count direction types
    low_to_high = sum(1 for t in transitions if t["direction"] == "low_to_high")
    high_to_low = sum(1 for t in transitions if t["direction"] == "high_to_low")

    print(
        f"  Found {len(transitions)} transitions: {low_to_high} low→high, {high_to_low} high→low"
    )
    print(f"  Transition start times: {transition_times[:10]} ms...")

    if len(transitions) > 0:
        durations = [t["duration"] for t in transitions]
        angle_changes = [t["angle_change"] for t in transitions]
        directions = [t["direction"] for t in transitions]
        print(f"  Transition durations: {durations[:10]} ms...")
        print(
            f"  Angle changes: {[f'{a:.1f}' for a in angle_changes[:10]]} degrees..."
        )
        print(f"  Directions: {directions[:10]}...")

    print(f"  Periods between transitions: {periods[:10]} ms...")
    print(f"  Mean period: {np.mean(periods):.3f} ms")
    print(f"  Std period: {np.std(periods):.3f} ms")

    if np.mean(periods) > 0:
        freq_hz = 1000.0 / np.mean(periods)
        rpm = freq_hz * 60
        print(f"  Estimated frequency: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")

    return {
        "transitions": transitions,
        "transition_times": transition_times,
        "periods": periods,
        "mean_period": np.mean(periods),
        "std_period": np.std(periods),
        "freq_hz": 1000.0 / np.mean(periods) if np.mean(periods) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect blade rotation periods using multiple methods"
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

    # Analyze each cluster with all methods
    all_results = {}

    for cluster_id, cluster in tracker.clusters.items():
        if len(cluster.angle_history) < 50:
            continue

        print(f"\n{'=' * 70}")
        print(f"BLADE ID{cluster_id} - PERIOD DETECTION")
        print(f"{'=' * 70}")

        times_ms = np.array([t / 1000.0 for t, _ in cluster.angle_history])
        angles_deg = np.array([a for _, a in cluster.angle_history])

        results = {}

        # Apply all methods
        results["method1"] = method1_angle_resets(
            times_ms, angles_deg, cluster_id
        )
        results["method2"] = method2_cluster_lifecycle(
            tracker.cluster_lifecycle, cluster_id
        )
        results["method3"] = method3_peak_detection(
            times_ms, angles_deg, cluster_id
        )
        results["method4"] = method4_derivative_analysis(
            times_ms, angles_deg, cluster_id
        )
        results["method5"] = method5_threshold_crossing(
            times_ms, angles_deg, cluster_id
        )
        results["method6"] = method6_large_angle_transition(
            times_ms, angles_deg, cluster_id
        )

        # Summary
        print(f"\n--- SUMMARY for Blade ID{cluster_id} ---")
        valid_methods = []
        for method_name, result in results.items():
            if result is not None and "freq_hz" in result:
                freq_hz = result["freq_hz"]
                rpm = freq_hz * 60
                print(f"  {method_name}: {freq_hz:.2f} Hz ({rpm:.0f} RPM)")
                valid_methods.append((method_name, freq_hz, rpm))

        if len(valid_methods) > 0:
            # Compute consensus
            freqs = [f for _, f, _ in valid_methods]
            mean_freq = np.mean(freqs)
            std_freq = np.std(freqs)
            mean_rpm = mean_freq * 60

            print(f"\n  CONSENSUS:")
            print(
                f"    Mean frequency: {mean_freq:.2f} Hz ({mean_rpm:.0f} RPM)"
            )
            print(
                f"    Std deviation: {std_freq:.2f} Hz ({std_freq * 60:.0f} RPM)"
            )

        results["times_ms"] = times_ms
        results["angles_deg"] = angles_deg
        results["valid_methods"] = valid_methods
        all_results[cluster_id] = results

    # Create visualization
    if len(all_results) > 0:
        n_clusters = len(all_results)
        fig, axes = plt.subplots(n_clusters, 2, figsize=(16, 6 * n_clusters))

        if n_clusters == 1:
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, results) in enumerate(all_results.items()):
            times_ms = results["times_ms"]
            angles_deg = results["angles_deg"]

            # Plot 1: Angle vs Time with all detections
            ax1 = axes[idx, 0]
            ax1.plot(
                times_ms,
                angles_deg,
                "b-",
                alpha=0.5,
                linewidth=0.5,
                label="Angle",
            )

            # Overlay method 1 (resets)
            if results["method1"] is not None:
                reset_times = results["method1"]["reset_times"]
                reset_angles = np.interp(reset_times, times_ms, angles_deg)
                ax1.scatter(
                    reset_times,
                    reset_angles,
                    c="red",
                    s=100,
                    marker="v",
                    label="Method 1: Resets",
                    zorder=5,
                    edgecolors="black",
                    linewidths=1,
                )

            # Overlay method 3 (peaks)
            if results["method3"] is not None:
                feature_times = results["method3"]["feature_times"]
                feature_angles = np.interp(feature_times, times_ms, angles_deg)
                ax1.scatter(
                    feature_times,
                    feature_angles,
                    c="orange",
                    s=80,
                    marker="^",
                    label=f"Method 3: {results['method3']['feature_type']}",
                    zorder=4,
                )

            # Overlay method 5 (crossings)
            if results["method5"] is not None:
                crossing_times = results["method5"]["crossing_times"]
                crossing_angles = np.interp(
                    crossing_times, times_ms, angles_deg
                )
                threshold = results["method5"]["threshold"]
                ax1.axhline(
                    threshold,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    label="Threshold",
                )
                ax1.scatter(
                    crossing_times,
                    crossing_angles,
                    c="green",
                    s=60,
                    marker="x",
                    label="Method 5: Crossings",
                    zorder=3,
                )

            # Overlay method 6 (large transitions)
            if results["method6"] is not None:
                transitions = results["method6"]["transitions"]
                # Draw arrows from start to end of each transition
                for trans in transitions:
                    # Color based on direction
                    arrow_color = (
                        "magenta"
                        if trans["direction"] == "low_to_high"
                        else "cyan"
                    )
                    ax1.annotate(
                        "",
                        xy=(trans["end_time"], trans["end_angle"]),
                        xytext=(trans["start_time"], trans["start_angle"]),
                        arrowprops=dict(
                            arrowstyle="->", color=arrow_color, lw=2, alpha=0.7
                        ),
                        zorder=6,
                    )
                # Add legend points for both directions
                if len(transitions) > 0:
                    low_to_high_trans = [
                        t
                        for t in transitions
                        if t["direction"] == "low_to_high"
                    ]
                    high_to_low_trans = [
                        t
                        for t in transitions
                        if t["direction"] == "high_to_low"
                    ]

                    if len(low_to_high_trans) > 0:
                        ax1.scatter(
                            [low_to_high_trans[0]["start_time"]],
                            [low_to_high_trans[0]["start_angle"]],
                            c="magenta",
                            s=100,
                            marker="D",
                            label=f"Method 6: Low→High ({len(low_to_high_trans)})",
                            zorder=6,
                            edgecolors="black",
                            linewidths=1,
                        )
                    if len(high_to_low_trans) > 0:
                        ax1.scatter(
                            [high_to_low_trans[0]["start_time"]],
                            [high_to_low_trans[0]["start_angle"]],
                            c="cyan",
                            s=100,
                            marker="D",
                            label=f"Method 6: High→Low ({len(high_to_low_trans)})",
                            zorder=6,
                            edgecolors="black",
                            linewidths=1,
                        )

            ax1.set_xlabel("Time (ms)", fontsize=11)
            ax1.set_ylabel("Angle (degrees)", fontsize=11)
            ax1.set_title(
                f"Blade ID{cluster_id}: Angle with Period Markers",
                fontsize=12,
                fontweight="bold",
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=8)

            # Plot 2: Period distribution from all methods
            ax2 = axes[idx, 1]

            colors = ["red", "blue", "orange", "purple", "green", "magenta"]
            method_names = [
                "Resets",
                "Lifecycle",
                "Peaks",
                "Derivative",
                "Crossings",
                "Large Trans.",
            ]

            all_periods = []
            for i, method_key in enumerate(
                [
                    "method1",
                    "method2",
                    "method3",
                    "method4",
                    "method5",
                    "method6",
                ]
            ):
                if (
                    results[method_key] is not None
                    and "periods" in results[method_key]
                ):
                    periods = results[method_key]["periods"]
                    if len(periods) > 0:
                        ax2.hist(
                            periods,
                            bins=20,
                            alpha=0.5,
                            color=colors[i],
                            label=f"{method_names[i]} (n={len(periods)})",
                        )
                        all_periods.extend(periods)

            if len(all_periods) > 0:
                mean_period = np.mean(all_periods)
                ax2.axvline(
                    mean_period,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_period:.2f} ms",
                )

            ax2.set_xlabel("Period (ms)", fontsize=11)
            ax2.set_ylabel("Count", fontsize=11)
            ax2.set_title(
                f"Blade ID{cluster_id}: Period Distribution from All Methods",
                fontsize=12,
                fontweight="bold",
            )
            ax2.legend(loc="best", fontsize=8)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = "blade_period_detection.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"\n{'=' * 70}")
        print(f"Saved period detection plot: {output_file}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
