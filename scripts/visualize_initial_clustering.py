#!/usr/bin/env python3
"""
Visualize the initial clustering window to troubleshoot DBSCAN clustering.
Creates a single image showing all events from the initial window with colors
indicating which cluster each event belongs to.

Usage:
    python visualize_initial_clustering.py --config config/tracking_config.yaml --preset fan_const_rpm
    python visualize_initial_clustering.py --config config/tracking_config.yaml --preset drone_idle
"""

import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import argparse
import yaml
from evio.core.recording import open_dat


def load_config(config_path, preset=None):
    """Load configuration from YAML file and optionally apply preset."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply preset if specified
    if preset and preset in config.get("presets", {}):
        preset_config = config["presets"][preset]
        # Deep merge preset into config
        for key, value in preset_config.items():
            if isinstance(value, dict) and key in config:
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
    hot_pixel_mask = counts >= threshold
    hot_pixels = unique_coords[hot_pixel_mask]

    if len(hot_pixels) > 0:
        print(f"  Found {len(hot_pixels)} hot pixel locations")
        keep_mask = np.ones(len(events), dtype=bool)
        for hp in hot_pixels:
            hot_pixel_events = (events[:, 0] == hp[0]) & (events[:, 1] == hp[1])
            keep_mask &= ~hot_pixel_events
        return events[keep_mask]
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Visualize initial clustering window for troubleshooting"
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
        help="Preset name to use from config file (e.g., 'drone_idle', 'fan_const_rpm')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: initial_clustering_debug.png)",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=None,
        help="Override initial window duration in milliseconds (e.g., 20 for 20ms)",
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=None,
        help="Minimum number of events per cluster (e.g., 200 to filter out small clusters)",
    )
    parser.add_argument(
        "--max-y-spread",
        type=float,
        default=None,
        help="Maximum y-spread (std dev) to filter out diffuse clusters (e.g., 5.0)",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=None,
        help="Minimum event density (events per pixel area, e.g., 2.0)",
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

    # Use command-line override if provided, otherwise use config
    if args.window_ms is not None:
        init_window_us = int(args.window_ms * 1000)
        print(
            f"Using command-line window duration: {args.window_ms}ms ({init_window_us}μs)"
        )
    else:
        init_window_us = config["clustering"]["temporal"]["window_duration_us"]

    # Filter events: time range and polarity
    end_time = start_time + int(duration_sec * 1_000_000)
    mask = (events[:, 3] >= start_time) & (events[:, 3] < end_time)
    if polarity is not None:
        mask = mask & (events[:, 2] == polarity)
    filtered_events = events[mask]

    polarity_str = (
        "negative"
        if polarity == -1
        else "positive"
        if polarity == 1
        else "both"
    )
    print(
        f"Filtered {len(filtered_events)} events ({polarity_str} polarity) "
        f"from {start_time_sec}s to {start_time_sec + duration_sec}s"
    )

    # Filter hot pixels
    filtered_events = filter_hot_pixels(
        filtered_events, config["hot_pixel"]["threshold"]
    )
    print(f"After hot pixel filtering: {len(filtered_events)} events")

    if len(filtered_events) < 100:
        print("Not enough events!")
        return

    # Get initial window of events
    init_end_time = start_time + init_window_us
    init_mask = filtered_events[:, 3] < init_end_time
    init_events = filtered_events[init_mask]

    print(f"\nInitial window: {len(init_events)} events")
    print(f"Window duration: {init_window_us} microseconds")

    # Run DBSCAN clustering
    eps = config["clustering"]["initial"]["eps"]
    min_samples = config["clustering"]["initial"]["min_samples"]
    min_cluster_size = config["clustering"]["temporal"]["min_cluster_size"]

    print(f"\nDBSCAN parameters:")
    print(f"  eps: {eps}")
    print(f"  min_samples: {min_samples}")
    print(f"  min_cluster_size: {min_cluster_size}")

    coords = init_events[:, :2]
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    # Analyze clustering results
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\nClustering results:")
    print(f"  Total clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise} ({100 * n_noise / len(labels):.1f}%)")

    # Apply quality filtering
    filtered_labels = {}  # Maps original label to filtered status
    quality_stats = {}  # Store quality metrics for each cluster

    for label in unique_labels:
        if label == -1:
            continue

        cluster_mask = labels == label
        cluster_events = init_events[cluster_mask]
        cluster_size = len(cluster_events)

        # Calculate quality metrics
        y_spread = cluster_events[:, 1].std()
        x_spread = cluster_events[:, 0].std()

        # Calculate density (events per pixel area)
        area = max(1, x_spread * y_spread * 4)  # 4 = 2*sigma in each direction
        density = cluster_size / area

        quality_stats[label] = {
            "size": cluster_size,
            "y_spread": y_spread,
            "x_spread": x_spread,
            "density": density,
        }

        # Apply filters
        passes_min_size = cluster_size >= min_cluster_size
        passes_min_events = (
            args.min_events is None or cluster_size >= args.min_events
        )
        passes_y_spread = (
            args.max_y_spread is None or y_spread <= args.max_y_spread
        )
        passes_density = args.min_density is None or density >= args.min_density

        filtered_labels[label] = (
            passes_min_size
            and passes_min_events
            and passes_y_spread
            and passes_density
        )

    # Count valid and filtered clusters
    valid_clusters = sum(1 for v in filtered_labels.values() if v)
    filtered_out = sum(1 for v in filtered_labels.values() if not v)

    print(f"\nFiltering criteria:")
    if args.min_events:
        print(f"  Minimum events: {args.min_events}")
    if args.max_y_spread:
        print(f"  Maximum y-spread: {args.max_y_spread}")
    if args.min_density:
        print(f"  Minimum density: {args.min_density}")

    print(f"\nAfter filtering:")
    print(f"  Valid clusters: {valid_clusters}")
    print(f"  Filtered out: {filtered_out}")

    # Print cluster details
    for label in sorted(unique_labels):
        if label == -1:
            continue
        stats = quality_stats[label]
        is_valid = filtered_labels[label]
        status = "VALID ✓" if is_valid else "FILTERED ✗"
        print(
            f"  Cluster {label}: {stats['size']} events, "
            f"σy={stats['y_spread']:.1f}, density={stats['density']:.2f} [{status}]"
        )

    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

    # Get colormap for clusters
    cmap = plt.cm.tab20

    # Plot each cluster with a different color
    for label in unique_labels:
        if label == -1:
            # Noise points in gray
            cluster_mask = labels == label
            cluster_events = init_events[cluster_mask]
            ax.scatter(
                cluster_events[:, 0],
                cluster_events[:, 1],
                c="gray",
                s=1,
                alpha=0.3,
                label=f"Noise ({n_noise} events)",
            )
        else:
            cluster_mask = labels == label
            cluster_events = init_events[cluster_mask]
            cluster_size = len(cluster_events)
            is_valid = filtered_labels.get(label, False)

            # Use different styles for valid vs filtered clusters
            if is_valid:
                marker_size = 4
                alpha = 0.9
                color = cmap(label % 20)
                label_text = f"Cluster {label} ({cluster_size} events) ✓"
            else:
                marker_size = 1
                alpha = 0.2
                color = "lightgray"
                label_text = f"Cluster {label} ({cluster_size} events) ✗"

            ax.scatter(
                cluster_events[:, 0],
                cluster_events[:, 1],
                c=[color],
                s=marker_size,
                alpha=alpha,
                label=label_text,
            )

            # Draw cluster center only for valid clusters
            if is_valid:
                center_x = cluster_events[:, 0].mean()
                center_y = cluster_events[:, 1].mean()
                ax.scatter(
                    center_x,
                    center_y,
                    c=[color],
                    s=100,
                    marker="x",
                    linewidths=3,
                )

    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)

    filter_text = ""
    if args.min_events or args.max_y_spread or args.min_density:
        filter_text = "\nFilters: "
        filters = []
        if args.min_events:
            filters.append(f"events>={args.min_events}")
        if args.max_y_spread:
            filters.append(f"σy<={args.max_y_spread}")
        if args.min_density:
            filters.append(f"density>={args.min_density}")
        filter_text += ", ".join(filters)

    ax.set_title(
        f"Initial Clustering Visualization\n"
        f"Window: {init_window_us}μs, DBSCAN(eps={eps}, min_samples={min_samples})\n"
        f"Valid Clusters: {valid_clusters}/{n_clusters}, Filtered out: {filtered_out}, Noise: {n_noise} events"
        f"{filter_text}",
        fontsize=14,
        fontweight="bold",
    )

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    # Add legend (but limit to reasonable number of entries)
    if len(unique_labels) <= 15:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        print("\nToo many clusters to show legend in plot")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Set aspect ratio to equal
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save figure
    if args.output:
        output_path = args.output
    else:
        preset_suffix = f"_{args.preset}" if args.preset else ""
        output_path = f"initial_clustering_debug{preset_suffix}.png"

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"\nVisualization saved to: {output_path}")

    # Also save a detailed text report
    report_path = output_path.replace(".png", "_report.txt")
    with open(report_path, "w") as f:
        f.write("Initial Clustering Debug Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration: {args.config}\n")
        if args.preset:
            f.write(f"Preset: {args.preset}\n")
        f.write(f"\nData:\n")
        f.write(f"  Input file: {config['data']['input_file']}\n")
        f.write(f"  Start time: {start_time_sec}s\n")
        f.write(f"  Polarity: {polarity_str}\n")
        f.write(f"  Total events in window: {len(init_events)}\n")
        f.write(f"  Window duration: {init_window_us}μs\n")
        f.write(f"\nDBSCAN Parameters:\n")
        f.write(f"  eps: {eps}\n")
        f.write(f"  min_samples: {min_samples}\n")
        f.write(f"  min_cluster_size: {min_cluster_size}\n")
        f.write(f"\nClustering Results:\n")
        f.write(f"  Total clusters found: {n_clusters}\n")
        f.write(f"  Valid clusters: {valid_clusters}\n")
        f.write(f"  Filtered out: {filtered_out}\n")
        f.write(
            f"  Noise points: {n_noise} ({100 * n_noise / len(labels):.1f}%)\n"
        )

        if args.min_events or args.max_y_spread or args.min_density:
            f.write(f"\nFiltering Criteria:\n")
            if args.min_events:
                f.write(f"  Minimum events: {args.min_events}\n")
            if args.max_y_spread:
                f.write(f"  Maximum y-spread: {args.max_y_spread}\n")
            if args.min_density:
                f.write(f"  Minimum density: {args.min_density}\n")

        f.write(f"\nCluster Details:\n")
        for label in sorted(unique_labels):
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_events = init_events[cluster_mask]
            stats = quality_stats[label]
            is_valid = filtered_labels[label]
            center_x = cluster_events[:, 0].mean()
            center_y = cluster_events[:, 1].mean()
            status = "VALID ✓" if is_valid else "FILTERED ✗"

            f.write(f"\n  Cluster {label} [{status}]:\n")
            f.write(f"    Events: {stats['size']}\n")
            f.write(f"    Center: ({center_x:.1f}, {center_y:.1f})\n")
            f.write(
                f"    Spread: σx={stats['x_spread']:.1f}, σy={stats['y_spread']:.1f}\n"
            )
            f.write(f"    Density: {stats['density']:.2f} events/px²\n")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
