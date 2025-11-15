#!/usr/bin/env python3
"""
Test RPM calculation robustness across different time points and parameters.
Tests how consistent RPM estimates are across time without generating videos.

Usage:
    python test_rpm_robustness.py --config config/tracking_config.yaml --preset drone_idle
    python test_rpm_robustness.py --config config/tracking_config.yaml --preset drone_idle --num-time-points 20
"""

import numpy as np
from pathlib import Path
import argparse
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from evio.core.recording import open_dat
import csv
from itertools import product
from collections import defaultdict


def load_config(config_path, preset=None):
    """Load configuration from YAML file and optionally apply preset."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if preset and preset in config.get("presets", {}):
        preset_config = config["presets"][preset]
        for key, value in preset_config.items():
            if isinstance(value, dict) and key in config:
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
    hot_pixel_mask = counts >= threshold
    hot_pixels = unique_coords[hot_pixel_mask]

    if len(hot_pixels) > 0:
        keep_mask = np.ones(len(events), dtype=bool)
        for hp in hot_pixels:
            hot_pixel_events = (events[:, 0] == hp[0]) & (events[:, 1] == hp[1])
            keep_mask &= ~hot_pixel_events
        return events[keep_mask]
    return events


def initialize_clusters(
    events, eps, min_samples, min_cluster_events, max_y_spread
):
    """Initialize clusters from events using DBSCAN with quality filtering."""
    if len(events) < 10:
        return []

    coords = events[:, :2]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    clusters = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_events = events[cluster_mask]
        cluster_size = len(cluster_events)
        y_spread = cluster_events[:, 1].std()

        # Apply quality filters
        if cluster_size >= min_cluster_events and y_spread <= max_y_spread:
            clusters.append(
                {
                    "id": label,
                    "events": cluster_events,
                    "initial_region": cluster_events[:, :2].copy(),
                }
            )

    return clusters


def process_events_for_cluster(
    cluster, all_events, window_duration_us, assignment_distance
):
    """Process events and track width over time for a single cluster."""
    initial_region = cluster["initial_region"]
    width_history = []

    # Sort events by time
    sorted_events = all_events[all_events[:, 3].argsort()]

    if len(sorted_events) == 0:
        return width_history

    start_time = sorted_events[0, 3]
    end_time = sorted_events[-1, 3]
    current_time = start_time

    cluster_events = []

    while current_time <= end_time:
        # Get events in current window
        window_mask = (sorted_events[:, 3] >= current_time) & (
            sorted_events[:, 3] < current_time + window_duration_us
        )
        window_events = sorted_events[window_mask]

        # Assign events to cluster based on distance to initial region
        assigned_events = []
        for event in window_events:
            x, y = event[0], event[1]
            # Calculate distance to initial region
            distances = np.sqrt(
                (initial_region[:, 0] - x) ** 2
                + (initial_region[:, 1] - y) ** 2
            )
            min_distance = (
                np.min(distances) if len(distances) > 0 else float("inf")
            )

            if min_distance < assignment_distance:
                assigned_events.append(event)

        if len(assigned_events) > 0:
            cluster_events = np.array(assigned_events)

            # Calculate blade width (95th - 5th percentile)
            x_values = cluster_events[:, 0]
            if len(x_values) >= 2:
                x_5th = np.percentile(x_values, 5)
                x_95th = np.percentile(x_values, 95)
                blade_width = x_95th - x_5th

                timestamp = current_time
                width_history.append((timestamp, blade_width))

        # Move to next update
        current_time += window_duration_us

    return width_history


def calculate_rpm_from_width(width_history, smoothing_window):
    """Calculate RPM from blade width using peak detection."""
    if len(width_history) < smoothing_window:
        return None, 0, []

    times = np.array([t / 1000.0 for t, _ in width_history])  # Convert to ms
    widths = np.array([w for _, w in width_history])

    # Apply smoothing
    smoothed_width = np.convolve(
        widths, np.ones(smoothing_window) / smoothing_window, mode="valid"
    )
    smoothed_times = times[smoothing_window - 1 :]

    if len(smoothed_width) < 10:
        return None, 0, []

    # Find peaks
    prominence = np.std(smoothed_width) * 0.5
    min_distance = max(5, len(smoothed_width) // 20)

    peaks, properties = find_peaks(
        smoothed_width,
        prominence=prominence,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return None, len(peaks), []

    # Calculate periods between peaks
    peak_times = smoothed_times[peaks]
    periods_ms = np.diff(peak_times)

    if len(periods_ms) > 0:
        mean_period_ms = np.mean(periods_ms)
        mean_period_s = mean_period_ms / 1000.0
        freq_hz = 1.0 / mean_period_s if mean_period_s > 0 else 0
        rpm = (
            freq_hz * 60
        ) / 2  # Divide by 2: each width cycle is half rotation

        return rpm, len(peaks), periods_ms

    return None, len(peaks), []


def test_rpm_at_timepoint(
    events,
    start_time,
    test_duration_us,
    init_window_us,
    window_duration_us,
    assignment_distance,
    smoothing_window,
    eps,
    min_samples,
    min_cluster_events,
    max_y_spread,
):
    """Test RPM calculation at a specific time point."""
    # Get events for this test
    end_time = start_time + test_duration_us
    mask = (events[:, 3] >= start_time) & (events[:, 3] < end_time)
    test_events = events[mask]

    if len(test_events) < 100:
        return None

    # Initialize clusters from first window
    init_end_time = start_time + init_window_us
    init_mask = test_events[:, 3] < init_end_time
    init_events = test_events[init_mask]
    remaining_events = test_events[~init_mask]

    clusters = initialize_clusters(
        init_events, eps, min_samples, min_cluster_events, max_y_spread
    )

    if len(clusters) == 0:
        return None

    # Process each cluster to get width history and calculate RPM
    cluster_rpms = []
    for cluster in clusters:
        width_history = process_events_for_cluster(
            cluster, remaining_events, window_duration_us, assignment_distance
        )

        if len(width_history) > 0:
            rpm, num_peaks, periods = calculate_rpm_from_width(
                width_history, smoothing_window
            )

            if rpm is not None:
                cluster_rpms.append(
                    {
                        "cluster_id": cluster["id"],
                        "rpm": rpm,
                        "num_peaks": num_peaks,
                        "num_width_samples": len(width_history),
                        "mean_period_ms": np.mean(periods)
                        if len(periods) > 0
                        else 0,
                        "std_period_ms": np.std(periods)
                        if len(periods) > 0
                        else 0,
                    }
                )

    return {
        "num_clusters": len(clusters),
        "num_clusters_with_rpm": len(cluster_rpms),
        "cluster_rpms": cluster_rpms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test RPM calculation robustness across time points"
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
    parser.add_argument(
        "--output",
        type=str,
        default="rpm_robustness_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--num-time-points",
        type=int,
        default=10,
        help="Number of time points to test",
    )
    parser.add_argument(
        "--test-duration-ms",
        type=float,
        default=100.0,
        help="Duration to analyze at each time point (ms)",
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

    # Get parameters
    start_time_sec = config["data"]["start_time_sec"]
    duration_sec = config["data"]["duration_sec"]
    polarity = config["data"]["polarity"]
    min_time = events[:, 3].min()
    base_start_time = min_time + int(start_time_sec * 1_000_000)
    duration_us = int(duration_sec * 1_000_000)

    # Filter events
    end_time = base_start_time + duration_us
    mask = (events[:, 3] >= base_start_time) & (events[:, 3] < end_time)
    if polarity is not None:
        mask = mask & (events[:, 2] == polarity)
    filtered_events = events[mask]

    print(f"Filtered {len(filtered_events)} events in time range")

    # Filter hot pixels
    filtered_events = filter_hot_pixels(
        filtered_events, config["hot_pixel"]["threshold"]
    )
    print(f"After hot pixel filtering: {len(filtered_events)} events")

    # Clustering parameters (fixed from config)
    eps = config["clustering"]["initial"]["eps"]
    min_samples = config["clustering"]["initial"]["min_samples"]
    init_window_us = config["clustering"]["initial"]["window_us"]
    min_cluster_events = config["clustering"]["initial"]["min_cluster_events"]
    max_y_spread = config["clustering"]["initial"]["max_y_spread"]

    # Parameter ranges to test
    window_duration_us_values = [500, 750, 1000, 1500, 2000]  # Rolling window
    assignment_distance_values = [10, 15, 20, 25]
    smoothing_window_values = [40, 60, 80, 100, 120]

    # Generate time points
    test_duration_us = int(args.test_duration_ms * 1000)
    time_points = np.linspace(
        base_start_time,
        base_start_time + duration_us - test_duration_us,
        args.num_time_points,
    ).astype(int)

    print(f"\n{'=' * 60}")
    print(f"TESTING RPM CALCULATION ROBUSTNESS")
    print(f"{'=' * 60}")
    print(f"Time points: {args.num_time_points}")
    print(f"Test duration at each point: {args.test_duration_ms}ms")
    print(f"\nFixed clustering parameters:")
    print(f"  eps: {eps}")
    print(f"  min_samples: {min_samples}")
    print(f"  init_window_us: {init_window_us}")
    print(f"  min_cluster_events: {min_cluster_events}")
    print(f"  max_y_spread: {max_y_spread}")
    print(f"\nParameter combinations to test:")
    print(f"  window_duration_us: {window_duration_us_values}")
    print(f"  assignment_distance: {assignment_distance_values}")
    print(f"  smoothing_window: {smoothing_window_values}")

    total_tests = (
        len(time_points)
        * len(window_duration_us_values)
        * len(assignment_distance_values)
        * len(smoothing_window_values)
    )
    print(f"\nTotal tests: {total_tests}")
    print(f"{'=' * 60}\n")

    # Run tests
    results = []
    test_count = 0

    for time_idx, start_time in enumerate(time_points):
        time_offset_ms = (start_time - base_start_time) / 1000.0
        print(
            f"Testing time point {time_idx + 1}/{len(time_points)} (offset: {time_offset_ms:.1f}ms)"
        )

        for window_us, assign_dist, smooth_win in product(
            window_duration_us_values,
            assignment_distance_values,
            smoothing_window_values,
        ):
            test_count += 1
            if test_count % 50 == 0:
                print(
                    f"  Progress: {test_count}/{total_tests} ({100 * test_count / total_tests:.1f}%)"
                )

            result = test_rpm_at_timepoint(
                filtered_events,
                start_time,
                test_duration_us,
                init_window_us,
                window_us,
                assign_dist,
                smooth_win,
                eps,
                min_samples,
                min_cluster_events,
                max_y_spread,
            )

            if result is not None:
                # Store aggregate results
                base_result = {
                    "time_point_idx": time_idx,
                    "time_offset_ms": time_offset_ms,
                    "window_duration_us": window_us,
                    "assignment_distance": assign_dist,
                    "smoothing_window": smooth_win,
                    "num_clusters": result["num_clusters"],
                    "num_clusters_with_rpm": result["num_clusters_with_rpm"],
                }

                # Add per-cluster RPM data
                for i, cluster_rpm in enumerate(result["cluster_rpms"]):
                    cluster_result = base_result.copy()
                    cluster_result.update(
                        {
                            "cluster_id": cluster_rpm["cluster_id"],
                            "rpm": cluster_rpm["rpm"],
                            "num_peaks": cluster_rpm["num_peaks"],
                            "num_width_samples": cluster_rpm[
                                "num_width_samples"
                            ],
                            "mean_period_ms": cluster_rpm["mean_period_ms"],
                            "std_period_ms": cluster_rpm["std_period_ms"],
                        }
                    )
                    results.append(cluster_result)

    # Save to CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 60}\n")

    # Analyze results
    print("ANALYSIS:")
    print(f"{'=' * 60}\n")

    # Overall statistics
    if results:
        all_rpms = [r["rpm"] for r in results]
        print(f"Total RPM measurements: {len(all_rpms)}")
        print(f"Overall RPM statistics:")
        print(f"  Mean: {np.mean(all_rpms):.1f} RPM")
        print(f"  Std: {np.std(all_rpms):.1f} RPM")
        print(f"  Min: {np.min(all_rpms):.1f} RPM")
        print(f"  Max: {np.max(all_rpms):.1f} RPM")
        print(
            f"  Coefficient of Variation: {100 * np.std(all_rpms) / np.mean(all_rpms):.1f}%"
        )

        # RPM consistency across time points
        print(f"\nRPM consistency across time points:")
        for time_idx in range(args.num_time_points):
            time_rpms = [
                r["rpm"] for r in results if r["time_point_idx"] == time_idx
            ]
            if time_rpms:
                time_offset = results[0]["time_offset_ms"] if results else 0
                for r in results:
                    if r["time_point_idx"] == time_idx:
                        time_offset = r["time_offset_ms"]
                        break
                print(
                    f"  Time {time_idx} ({time_offset:.1f}ms): Mean={np.mean(time_rpms):.1f}, Std={np.std(time_rpms):.1f}, N={len(time_rpms)}"
                )

        # Parameter impact
        print(f"\nParameter impact on RPM variance:")

        for param_name, param_key in [
            ("window_duration_us", "window_duration_us"),
            ("assignment_distance", "assignment_distance"),
            ("smoothing_window", "smoothing_window"),
        ]:
            print(f"\n{param_name}:")
            param_stats = defaultdict(list)
            for r in results:
                param_stats[r[param_key]].append(r["rpm"])

            print(f"  {'Value':<10} {'Mean RPM':<12} {'Std RPM':<12} {'N':<6}")
            print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 6}")
            for val in sorted(param_stats.keys()):
                vals = param_stats[val]
                print(
                    f"  {val:<10} {np.mean(vals):<12.1f} {np.std(vals):<12.1f} {len(vals):<6}"
                )

        # Most stable configurations
        print(f"\n{'=' * 60}")
        print("MOST STABLE CONFIGURATIONS (lowest RPM variance):")
        print(f"{'=' * 60}\n")

        config_rpms = defaultdict(list)
        for r in results:
            key = (
                r["window_duration_us"],
                r["assignment_distance"],
                r["smoothing_window"],
            )
            config_rpms[key].append(r["rpm"])

        stable_configs = []
        for key, rpms in config_rpms.items():
            if (
                len(rpms) >= args.num_time_points
            ):  # Must have data from all time points
                std = np.std(rpms)
                mean = np.mean(rpms)
                cv = 100 * std / mean if mean > 0 else float("inf")
                stable_configs.append(
                    {
                        "params": key,
                        "mean_rpm": mean,
                        "std_rpm": std,
                        "cv_pct": cv,
                        "n": len(rpms),
                    }
                )

        stable_configs.sort(key=lambda x: x["cv_pct"])

        print(
            f"  {'window_us':<12} {'assign_dist':<13} {'smooth_win':<12} {'Mean RPM':<12} {'Std RPM':<12} {'CV %':<8}"
        )
        print(
            f"  {'-' * 12} {'-' * 13} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8}"
        )
        for config in stable_configs[:15]:
            win_us, assign_dist, smooth_win = config["params"]
            print(
                f"  {win_us:<12} {assign_dist:<13} {smooth_win:<12} {config['mean_rpm']:<12.1f} {config['std_rpm']:<12.1f} {config['cv_pct']:<8.1f}"
            )

        if stable_configs:
            print(f"\nBest configuration (lowest coefficient of variation):")
            best = stable_configs[0]
            win_us, assign_dist, smooth_win = best["params"]
            print(f"  window_duration_us: {win_us}")
            print(f"  assignment_distance: {assign_dist}")
            print(f"  smoothing_window: {smooth_win}")
            print(f"  Mean RPM: {best['mean_rpm']:.1f}")
            print(f"  Std RPM: {best['std_rpm']:.1f}")
            print(f"  Coefficient of Variation: {best['cv_pct']:.1f}%")

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
