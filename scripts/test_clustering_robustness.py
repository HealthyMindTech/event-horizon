#!/usr/bin/env python3
"""
Test clustering robustness across different time points and parameter combinations.
Helps optimize DBSCAN parameters and validate consistency.

Usage:
    python test_clustering_robustness.py --config config/tracking_config.yaml --preset drone_idle
    python test_clustering_robustness.py --config config/tracking_config.yaml --preset drone_idle --output results.csv
"""

import numpy as np
from pathlib import Path
import argparse
import yaml
from sklearn.cluster import DBSCAN
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


def test_clustering_at_time(
    events,
    start_time,
    window_us,
    eps,
    min_samples,
    min_cluster_events,
    max_y_spread,
):
    """Test clustering at a specific time point with given parameters."""
    # Get events in window
    end_time = start_time + window_us
    mask = (events[:, 3] >= start_time) & (events[:, 3] < end_time)
    window_events = events[mask]

    if len(window_events) < min_samples:
        return {
            "total_events": len(window_events),
            "total_clusters": 0,
            "valid_clusters": 0,
            "filtered_clusters": 0,
            "noise_points": 0,
            "noise_pct": 0,
            "valid_cluster_sizes": [],
            "valid_cluster_centers": [],
            "valid_cluster_y_spreads": [],
        }

    # Run DBSCAN
    coords = window_events[:, :2]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    # Analyze results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    valid_clusters = []
    filtered_clusters = []

    for label in unique_labels:
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_events = window_events[cluster_mask]
        cluster_size = len(cluster_events)

        # Calculate quality metrics
        y_spread = cluster_events[:, 1].std()
        center_x = cluster_events[:, 0].mean()
        center_y = cluster_events[:, 1].mean()

        # Apply filters
        passes_min_events = cluster_size >= min_cluster_events
        passes_y_spread = y_spread <= max_y_spread

        if passes_min_events and passes_y_spread:
            valid_clusters.append(
                {
                    "size": cluster_size,
                    "center": (center_x, center_y),
                    "y_spread": y_spread,
                    "x_spread": cluster_events[:, 0].std(),
                }
            )
        else:
            filtered_clusters.append(
                {
                    "size": cluster_size,
                    "y_spread": y_spread,
                    "reason": (
                        "too_small"
                        if not passes_min_events
                        else "high_y_spread"
                    ),
                }
            )

    return {
        "total_events": len(window_events),
        "total_clusters": n_clusters,
        "valid_clusters": len(valid_clusters),
        "filtered_clusters": len(filtered_clusters),
        "noise_points": n_noise,
        "noise_pct": 100 * n_noise / len(window_events)
        if len(window_events) > 0
        else 0,
        "valid_cluster_sizes": [c["size"] for c in valid_clusters],
        "valid_cluster_centers": [c["center"] for c in valid_clusters],
        "valid_cluster_y_spreads": [c["y_spread"] for c in valid_clusters],
        "valid_cluster_x_spreads": [c["x_spread"] for c in valid_clusters],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test clustering robustness across time points and parameters"
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
        default="clustering_robustness_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--num-time-points",
        type=int,
        default=10,
        help="Number of time points to test",
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

    # Get base parameters
    start_time_sec = config["data"]["start_time_sec"]
    duration_sec = config["data"]["duration_sec"]
    polarity = config["data"]["polarity"]
    min_time = events[:, 3].min()
    base_start_time = min_time + int(start_time_sec * 1_000_000)
    duration_us = int(duration_sec * 1_000_000)

    # Filter events: time range and polarity
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

    # Define parameter ranges to test (reduced for faster execution)
    eps_values = [2, 3, 4, 5]
    min_samples_values = [5, 10, 15]
    window_us_values = [5000, 10000, 15000, 20000]  # 5ms to 20ms
    min_cluster_events_values = [100, 200, 300]
    max_y_spread_values = [4.0, 5.0, 6.0]

    # Generate time points to test
    time_points = np.linspace(
        base_start_time,
        base_start_time + duration_us - max(window_us_values),
        args.num_time_points,
    ).astype(int)

    print(f"\n{'=' * 60}")
    print(f"TESTING CLUSTERING ROBUSTNESS")
    print(f"{'=' * 60}")
    print(f"Time points: {args.num_time_points}")
    print(f"Parameter combinations to test:")
    print(f"  eps: {eps_values}")
    print(f"  min_samples: {min_samples_values}")
    print(f"  window_us: {[w / 1000 for w in window_us_values]} ms")
    print(f"  min_cluster_events: {min_cluster_events_values}")
    print(f"  max_y_spread: {max_y_spread_values}")

    total_tests = (
        len(time_points)
        * len(eps_values)
        * len(min_samples_values)
        * len(window_us_values)
        * len(min_cluster_events_values)
        * len(max_y_spread_values)
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

        for (
            eps,
            min_samples,
            window_us,
            min_cluster_events,
            max_y_spread,
        ) in product(
            eps_values,
            min_samples_values,
            window_us_values,
            min_cluster_events_values,
            max_y_spread_values,
        ):
            test_count += 1
            if test_count % 100 == 0:
                print(
                    f"  Progress: {test_count}/{total_tests} ({100 * test_count / total_tests:.1f}%)"
                )

            result = test_clustering_at_time(
                filtered_events,
                start_time,
                window_us,
                eps,
                min_samples,
                min_cluster_events,
                max_y_spread,
            )

            # Store result
            results.append(
                {
                    "time_point_idx": time_idx,
                    "time_offset_ms": time_offset_ms,
                    "start_time_us": start_time,
                    "eps": eps,
                    "min_samples": min_samples,
                    "window_us": window_us,
                    "window_ms": window_us / 1000.0,
                    "min_cluster_events": min_cluster_events,
                    "max_y_spread": max_y_spread,
                    "total_events": result["total_events"],
                    "total_clusters": result["total_clusters"],
                    "valid_clusters": result["valid_clusters"],
                    "filtered_clusters": result["filtered_clusters"],
                    "noise_points": result["noise_points"],
                    "noise_pct": result["noise_pct"],
                    "min_cluster_size": (
                        min(result["valid_cluster_sizes"])
                        if result["valid_cluster_sizes"]
                        else 0
                    ),
                    "max_cluster_size": (
                        max(result["valid_cluster_sizes"])
                        if result["valid_cluster_sizes"]
                        else 0
                    ),
                    "mean_cluster_size": (
                        np.mean(result["valid_cluster_sizes"])
                        if result["valid_cluster_sizes"]
                        else 0
                    ),
                    "mean_y_spread": (
                        np.mean(result["valid_cluster_y_spreads"])
                        if result["valid_cluster_y_spreads"]
                        else 0
                    ),
                    "mean_x_spread": (
                        np.mean(result["valid_cluster_x_spreads"])
                        if result["valid_cluster_x_spreads"]
                        else 0
                    ),
                }
            )

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

    # Find configurations that consistently find 4 valid clusters
    ideal_results = [r for r in results if r["valid_clusters"] == 4]
    print(
        f"Configurations that found 4 valid clusters: {len(ideal_results)}/{len(results)} ({100 * len(ideal_results) / len(results):.1f}%)"
    )

    if len(ideal_results) > 0:
        print(
            "\nMost consistent parameter combinations (found 4 clusters most often):"
        )
        # Count parameter combinations
        param_counts = defaultdict(int)
        for r in ideal_results:
            key = (
                r["eps"],
                r["min_samples"],
                r["window_ms"],
                r["min_cluster_events"],
                r["max_y_spread"],
            )
            param_counts[key] += 1

        # Sort by count
        sorted_params = sorted(
            param_counts.items(), key=lambda x: x[1], reverse=True
        )

        print("\n  Top 10 parameter combinations:")
        print(
            f"  {'eps':<5} {'min_samp':<9} {'window_ms':<11} {'min_events':<11} {'max_y_sp':<9} {'count':<6}"
        )
        print(
            f"  {'-' * 5} {'-' * 9} {'-' * 11} {'-' * 11} {'-' * 9} {'-' * 6}"
        )
        for (eps, min_samp, win_ms, min_ev, max_y), count in sorted_params[:10]:
            print(
                f"  {eps:<5} {min_samp:<9} {win_ms:<11.1f} {min_ev:<11} {max_y:<9.1f} {count:<6}"
            )

        if sorted_params:
            print("\nBest parameters (most occurrences of 4 clusters):")
            best = sorted_params[0]
            eps, min_samp, win_ms, min_ev, max_y = best[0]
            count = best[1]
            print(f"  eps: {eps}")
            print(f"  min_samples: {min_samp}")
            print(f"  window_ms: {win_ms}")
            print(f"  min_cluster_events: {min_ev}")
            print(f"  max_y_spread: {max_y}")
            print(
                f"  Found 4 clusters in: {count}/{args.num_time_points} time points"
            )

    # Statistics by parameter
    print("\n" + "=" * 60)
    print("PARAMETER IMPACT ON VALID CLUSTER COUNT:")
    print("=" * 60 + "\n")

    for param in [
        "eps",
        "min_samples",
        "window_ms",
        "min_cluster_events",
        "max_y_spread",
    ]:
        print(f"\n{param}:")
        # Group by parameter value
        param_stats = defaultdict(list)
        for r in results:
            param_stats[r[param]].append(r["valid_clusters"])

        print(f"  {'Value':<10} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max':<6}")
        print(f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6}")
        for val in sorted(param_stats.keys()):
            vals = param_stats[val]
            mean = np.mean(vals)
            std = np.std(vals)
            min_val = min(vals)
            max_val = max(vals)
            print(
                f"  {val:<10.1f} {mean:<8.2f} {std:<8.2f} {min_val:<6} {max_val:<6}"
            )

    # Find robust configurations (low variance in cluster count)
    print("\n" + "=" * 60)
    print("MOST ROBUST CONFIGURATIONS (low variance across time):")
    print("=" * 60 + "\n")

    # Group by parameter combination
    param_groups = defaultdict(list)
    for r in results:
        key = (
            r["eps"],
            r["min_samples"],
            r["window_ms"],
            r["min_cluster_events"],
            r["max_y_spread"],
        )
        param_groups[key].append(r["valid_clusters"])

    # Calculate statistics
    robust_configs = []
    for key, vals in param_groups.items():
        mean = np.mean(vals)
        if mean >= 3.5:  # At least average of 3.5 clusters
            std = np.std(vals)
            robust_configs.append(
                {
                    "params": key,
                    "mean": mean,
                    "std": std,
                    "min": min(vals),
                    "max": max(vals),
                }
            )

    # Sort by std (lower is better)
    robust_configs.sort(key=lambda x: x["std"])

    print(
        f"  {'eps':<5} {'min_samp':<9} {'window_ms':<11} {'min_events':<11} {'max_y_sp':<9} {'mean':<8} {'std':<8}"
    )
    print(
        f"  {'-' * 5} {'-' * 9} {'-' * 11} {'-' * 11} {'-' * 9} {'-' * 8} {'-' * 8}"
    )
    for config in robust_configs[:10]:
        eps, min_samp, win_ms, min_ev, max_y = config["params"]
        print(
            f"  {eps:<5} {min_samp:<9} {win_ms:<11.1f} {min_ev:<11} {max_y:<9.1f} {config['mean']:<8.2f} {config['std']:<8.2f}"
        )

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
