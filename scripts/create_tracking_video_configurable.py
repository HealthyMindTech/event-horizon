#!/usr/bin/env python3
"""
Create a video showing real-time temporal blade tracking.
Reads configuration from YAML file to support different datasets.

Usage:
    python create_tracking_video_configurable.py --config config/tracking_config.yaml --preset fan_const_rpm
    python create_tracking_video_configurable.py --config config/tracking_config.yaml --preset drone_idle
"""

import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
import cv2
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


class BladeCluster:
    """Tracked blade cluster with rolling window."""

    def __init__(self, cluster_id, initial_events, color=None):
        self.id = cluster_id
        self.events = initial_events.copy()
        self.color = (
            color if color is not None else plt.cm.tab10(cluster_id % 10)
        )
        # Historical data for stable center calculation
        self.center_history = []  # List of (x, y, slope, timestamp) tuples
        self.center_x = 0
        self.center_y = 0
        self.confidence = 0.0  # Confidence score based on line fit
        # Tracking history for plotting
        self.angle_history = []  # List of (timestamp, angle_deg)
        self.confidence_history = []  # List of (timestamp, confidence)
        self.update_statistics()

    def update_statistics(self):
        """Recalculate slope and angle from current events."""
        if len(self.events) < 2:
            self.slope = 0
            self.angle_deg = 0
            if len(self.events) > 0:
                # If we have history, use historical center
                if len(self.center_history) == 0:
                    self.center_x = np.mean(self.events[:, 0])
                    self.center_y = np.mean(self.events[:, 1])
            return

        # Fit linear regression to get blade orientation
        X = self.events[:, 0].reshape(-1, 1)
        y = self.events[:, 1]
        lr = LinearRegression()
        lr.fit(X, y)
        self.slope = lr.coef_[0]
        self.angle_deg = np.degrees(np.arctan(self.slope))

        # Calculate confidence score based on "line-ness" vs "blob-ness"
        # Use ratio of variance along the line vs perpendicular to it
        # A true line has high variance along line, low variance perpendicular

        # Get the line direction (normalized)
        line_angle = np.arctan(self.slope)
        line_dir = np.array([np.cos(line_angle), np.sin(line_angle)])
        perp_dir = np.array([-np.sin(line_angle), np.cos(line_angle)])

        # Project all points onto line direction and perpendicular direction
        points = np.column_stack([self.events[:, 0], self.events[:, 1]])
        center_point = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
        centered_points = points - center_point

        # Variance along the line direction
        proj_along = np.dot(centered_points, line_dir)
        var_along = np.var(proj_along) if len(proj_along) > 1 else 0

        # Variance perpendicular to the line
        proj_perp = np.dot(centered_points, perp_dir)
        var_perp = np.var(proj_perp) if len(proj_perp) > 1 else 0

        # Confidence: ratio of along/perpendicular variance
        # High ratio = line-like, Low ratio = blob-like
        # Use logarithmic scale for better sensitivity
        if var_perp > 0.1:  # Avoid division by very small numbers
            ratio = var_along / var_perp
            # Use log scale: ratio 1 = 0%, ratio 3 = ~50%, ratio 10 = ~80%, ratio 30+ = 100%
            # log10(ratio) ranges: 0 (ratio=1) to 1.5 (ratio=30)
            log_ratio = np.log10(max(1.0, ratio))
            # Scale to 0-1: log10(3) ≈ 0.48 maps to 0.5
            self.confidence = min(1.0, max(0.0, log_ratio / 1.5))
        else:
            # Perfect line (no perpendicular variance) or too few points
            self.confidence = 1.0

        # Record angle and confidence history
        if len(self.events) > 0:
            current_timestamp = self.events[:, 3].max()
            self.angle_history.append((current_timestamp, self.angle_deg))
            self.confidence_history.append((current_timestamp, self.confidence))

        # Calculate instantaneous center (midpoint of events)
        instant_center_x = np.mean(self.events[:, 0])
        instant_center_y = np.mean(self.events[:, 1])

        # Add to history
        if len(self.events) > 0:
            timestamp = self.events[:, 3].max()
            self.center_history.append(
                (instant_center_x, instant_center_y, self.slope, timestamp)
            )

        # Calculate stable center from history
        if len(self.center_history) > 0:
            centers = np.array([(h[0], h[1]) for h in self.center_history])
            self.center_x = np.mean(centers[:, 0])
            self.center_y = np.mean(centers[:, 1])
        else:
            self.center_x = instant_center_x
            self.center_y = instant_center_y

    def add_event(self, event):
        """Add single event to cluster."""
        self.events = np.vstack([self.events, event.reshape(1, -1)])
        self.update_statistics()

    def remove_old_events(self, cutoff_time_us):
        """Remove events older than cutoff time."""
        mask = self.events[:, 3] >= cutoff_time_us
        self.events = self.events[mask]
        if len(self.events) > 0:
            self.update_statistics()

    def remove_old_center_history(self, cutoff_time_us):
        """Remove old center history entries (longer window than events)."""
        self.center_history = [
            h for h in self.center_history if h[3] >= cutoff_time_us
        ]

    def __len__(self):
        return len(self.events)

    def distance_to_point(self, x, y):
        """Calculate distance from point to cluster center."""
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
            if label == -1:  # Skip noise
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
            # Add to existing cluster
            self.clusters[closest_cluster_id].add_event(event)
        else:
            # Create new cluster
            new_cluster = BladeCluster(
                self.next_cluster_id, event.reshape(1, -1)
            )
            self.clusters[self.next_cluster_id] = new_cluster
            self.next_cluster_id += 1

        # Remove old events from all clusters (short window)
        cutoff_time = self.current_time_us - self.window_duration_us
        # Remove old center history (longer window for stable center)
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


def render_frame(tracker, frame_size, roi, config):
    """Render current tracking state to image."""
    x_min, x_max, y_min, y_max = roi
    vis_config = config["video"]["visualization"]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")
    ax.set_facecolor(vis_config["background_color"])
    fig.patch.set_facecolor(vis_config["background_color"])

    # Title with timing info
    ax.set_title(
        f"Temporal Blade Tracking - {config['data']['input_file']}\n"
        f"Time: {tracker.current_time_us / 1000:.3f}ms | Active Blades: {len(tracker.clusters)}",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("X [pixels]", color="white", fontsize=10)
    ax.set_ylabel("Y [pixels]", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(
        True, alpha=vis_config["grid_alpha"], color=vis_config["grid_color"]
    )

    # Draw each cluster
    for cluster in tracker.clusters.values():
        if len(cluster) < tracker.min_cluster_size:
            continue

        # Plot events
        ax.scatter(
            cluster.events[:, 0],
            cluster.events[:, 1],
            c=[cluster.color],
            s=vis_config["event_size"],
            alpha=vis_config["event_alpha"],
            edgecolors="white",
            linewidths=0.5,
        )

        # Draw cluster center
        ax.add_patch(
            Circle(
                (cluster.center_x, cluster.center_y),
                radius=vis_config["center_size"],
                color=vis_config["center_color"],
                fill=True,
                zorder=10,
            )
        )

        # Draw fitted line (blade orientation)
        if len(cluster.events) >= 2:
            x_min_cluster = cluster.events[:, 0].min()
            x_max_cluster = cluster.events[:, 0].max()
            x_line = np.array([x_min_cluster, x_max_cluster])
            y_line = cluster.slope * x_line + (
                cluster.center_y - cluster.slope * cluster.center_x
            )
            line_style = "--" if vis_config["line_style"] == "dashed" else "-"
            ax.plot(
                x_line,
                y_line,
                color=vis_config["line_color"],
                linestyle=line_style,
                linewidth=vis_config["line_width"],
                alpha=0.8,
                zorder=5,
            )

        # Label with ID, angle, and confidence
        confidence_pct = cluster.confidence * 100
        ax.text(
            cluster.center_x + 8,
            cluster.center_y - 5,
            f"ID{cluster.id}\n{cluster.angle_deg:.1f}°\n{len(cluster)}ev\nC:{confidence_pct:.0f}%",
            fontsize=vis_config["label_fontsize"],
            color=vis_config["label_color"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            zorder=11,
        )

    # Convert to image using savefig to buffer
    from io import BytesIO

    buf = BytesIO()
    plt.savefig(
        buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    buf.seek(0)

    # Read from buffer with opencv
    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    buf.close()
    plt.close(fig)

    # Resize to target size and convert to RGB
    img = cv2.resize(img, (frame_size[0], frame_size[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Create temporal blade tracking video from event camera data"
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

    # Initialize tracker with first window of events
    init_end_time = start_time + init_window_us
    init_mask = filtered_events[:, 3] < init_end_time
    init_events = filtered_events[init_mask]
    remaining_events = filtered_events[~init_mask]

    print(f"\nInitializing tracker with {len(init_events)} events...")

    # Center history window is 4x longer than event window for stability
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

    # Record initial cluster IDs for tracking
    initial_cluster_ids = list(tracker.clusters.keys())

    # Define ROI
    if config["roi"]["manual"] is not None:
        x_min, x_max, y_min, y_max = config["roi"]["manual"]
        print(f"Using manual ROI: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    elif len(tracker.clusters) > 0:
        all_events = np.vstack([c.events for c in tracker.clusters.values()])
        margin = config["roi"]["margin"]
        x_min = max(0, all_events[:, 0].min() - margin)
        x_max = min(width, all_events[:, 0].max() + margin)
        y_min = max(0, all_events[:, 1].min() - margin)
        y_max = min(height, all_events[:, 1].max() + margin)
        print(
            f"Auto-detected ROI: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]"
        )
    else:
        x_min, x_max = 0, width
        y_min, y_max = 0, height
        print(f"Using full frame ROI")

    roi = (x_min, x_max, y_min, y_max)

    # Video setup
    fps = config["video"]["fps"]
    frame_size = (config["video"]["width"], config["video"]["height"])
    output_video = config["video"]["output_file"]
    fourcc = cv2.VideoWriter_fourcc(*config["video"]["codec"])
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    print(f"\nCreating video: {output_video}")
    print(f"FPS: {fps}, Size: {frame_size}")
    print(f"Processing {len(remaining_events)} events...")

    # Process events and render frames
    frame_interval = config["video"]["frame_interval"]
    if frame_interval is None:
        # Auto-calculate to make ~10 seconds of video
        frame_interval = max(1, len(remaining_events) // (fps * 10))
    print(f"Frame interval: every {frame_interval} events")

    frame_count = 0

    for i, event in enumerate(remaining_events):
        # Process event
        tracker.process_event(event)

        # Render frame at intervals
        if i % frame_interval == 0 or i == len(remaining_events) - 1:
            frame = render_frame(tracker, frame_size, roi, config)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            frame_count += 1

            if frame_count % 30 == 0:
                print(
                    f"  Frame {frame_count}: Event {i}/{len(remaining_events)} "
                    f"({100 * i / len(remaining_events):.1f}%), "
                    f"Time: {tracker.current_time_us / 1000:.3f}ms, "
                    f"Clusters: {len(tracker.clusters)}"
                )

    video_writer.release()

    print(f"\n{'=' * 60}")
    print(f"Video created: {output_video}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count / fps:.2f} seconds")
    print(f"Events processed: {len(remaining_events)}")
    print(f"{'=' * 60}")

    # Generate plots of angle and confidence over time for tracked blades
    print(f"\nGenerating tracking plots...")

    # Find clusters that were tracked throughout (prefer initial clusters)
    clusters_to_plot = []
    for cluster_id in initial_cluster_ids:
        if cluster_id in tracker.clusters:
            cluster = tracker.clusters[cluster_id]
            if len(cluster.angle_history) > 10:  # Only plot if enough data
                clusters_to_plot.append(cluster)

    # Also add any other long-lived clusters
    for cluster in tracker.clusters.values():
        if cluster not in clusters_to_plot and len(cluster.angle_history) > 10:
            clusters_to_plot.append(cluster)

    if len(clusters_to_plot) > 0:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        # Plot raw angle over time
        ax1 = axes[0]
        ax1.set_title(
            "Blade Angle Over Time (raw)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Time (ms)", fontsize=12)
        ax1.set_ylabel("Angle (degrees)", fontsize=12)
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Plot angle over time (as sin)
        ax2 = axes[1]
        ax2.set_title(
            "Blade Orientation Over Time (sin(2×angle))",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_xlabel("Time (ms)", fontsize=12)
        ax2.set_ylabel("sin(2×angle)", fontsize=12)
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # Plot confidence over time
        ax3 = axes[2]
        ax3.set_title(
            "Blade Confidence Over Time", fontsize=14, fontweight="bold"
        )
        ax3.set_xlabel("Time (ms)", fontsize=12)
        ax3.set_ylabel("Confidence (0-1)", fontsize=12)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

        # Plot each cluster
        for cluster in clusters_to_plot[:6]:  # Limit to 6 for readability
            # Extract angle history and convert to sin/cos
            if len(cluster.angle_history) > 0:
                angle_times = np.array(
                    [t / 1000.0 for t, _ in cluster.angle_history]
                )
                angle_values = np.array([a for _, a in cluster.angle_history])

                # Debug: Print angle values
                print(f"\nBlade ID{cluster.id} angle analysis:")
                print(f"  Number of samples: {len(angle_values)}")
                print(
                    f"  Angle range: [{angle_values.min():.2f}°, {angle_values.max():.2f}°]"
                )
                print(f"  First 20 angles: {angle_values[:20]}")
                print(f"  Last 20 angles: {angle_values[-20:]}")
                print(f"  Mean angle: {np.mean(angle_values):.2f}°")
                print(f"  Std dev: {np.std(angle_values):.2f}°")

                # Check for angle progression
                angle_diffs = np.diff(angle_values)
                print(f"  Angle changes (first 20): {angle_diffs[:20]}")
                print(f"  Mean change per update: {np.mean(angle_diffs):.4f}°")
                print(f"  Max angle jump: {np.max(np.abs(angle_diffs)):.2f}°")

                # Find large jumps that indicate wrapping
                large_jumps = np.where(np.abs(angle_diffs) > 90)[0]
                print(f"  Number of large jumps (>90°): {len(large_jumps)}")
                if len(large_jumps) > 0:
                    print(
                        f"  Large jump locations (first 5): {large_jumps[:5]}"
                    )
                    for idx in large_jumps[:5]:
                        print(
                            f"    Jump at i={idx}: {angle_values[idx]:.2f}° -> {angle_values[idx + 1]:.2f}° (diff={angle_diffs[idx]:.2f}°)"
                        )

                # No unwrapping - just use raw angles
                unwrapped_angles = angle_values.copy()

                # Convert angles to sin(2×angle) to handle 180° period
                angle_radians = np.radians(unwrapped_angles * 2)
                sin_values = np.sin(angle_radians)
                cos_values = np.cos(angle_radians)

                print(
                    f"  Sin range: [{sin_values.min():.3f}, {sin_values.max():.3f}]"
                )
                print(
                    f"  Cos range: [{cos_values.min():.3f}, {cos_values.max():.3f}]"
                )

                # Show sample of time, angle, and sin values
                print(f"  Sample data (first 10):")
                for i in range(min(10, len(angle_times))):
                    print(
                        f"    t={angle_times[i]:.3f}ms, angle={angle_values[i]:.2f}°, sin={sin_values[i]:.3f}"
                    )

                # Plot raw angle (no unwrapping)
                ax1.plot(
                    angle_times,
                    angle_values,
                    marker="o",
                    markersize=1.5,
                    linewidth=1.5,
                    label=f"Blade ID{cluster.id}",
                    color=cluster.color,
                    linestyle="-",
                )

                # Highlight large angle changes
                large_changes = np.where(np.abs(angle_diffs) > 10)[0]
                print(f"  Large angle changes (>10°): {len(large_changes)}")
                if len(large_changes) > 0:
                    print(f"  First 10 large changes:")
                    for idx in large_changes[:10]:
                        print(
                            f"    t={angle_times[idx]:.3f}ms -> {angle_times[idx + 1]:.3f}ms: "
                            f"{angle_values[idx]:.2f}° -> {angle_values[idx + 1]:.2f}° "
                            f"(Δ={angle_diffs[idx]:.2f}°, sin: {sin_values[idx]:.3f} -> {sin_values[idx + 1]:.3f})"
                        )

                # Plot sin component only
                ax2.plot(
                    angle_times,
                    sin_values,
                    marker="o",
                    markersize=1.5,
                    linewidth=1.5,
                    label=f"Blade ID{cluster.id}",
                    color=cluster.color,
                    linestyle="-",
                )

                # Mark large jumps with red circles on sin plot
                if len(large_changes) > 0:
                    ax2.scatter(
                        angle_times[large_changes + 1],
                        sin_values[large_changes + 1],
                        c="red",
                        s=30,
                        marker="o",
                        edgecolors="black",
                        linewidths=1,
                        zorder=10,
                    )

            # Extract confidence history
            if len(cluster.confidence_history) > 0:
                conf_times = np.array(
                    [t / 1000.0 for t, _ in cluster.confidence_history]
                )
                conf_values = np.array(
                    [c for _, c in cluster.confidence_history]
                )
                ax3.plot(
                    conf_times,
                    conf_values,
                    marker="o",
                    markersize=2,
                    linewidth=1.5,
                    label=f"Blade ID{cluster.id}",
                    color=cluster.color,
                )

        ax1.legend(loc="best", fontsize=10)
        ax2.legend(loc="best", fontsize=10)
        ax3.legend(loc="best", fontsize=10)

        plt.tight_layout()
        plot_filename = "blade_tracking_plots.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved tracking plots to: {plot_filename}")
        print(f"Plotted {len(clusters_to_plot)} blade(s)")
    else:
        print("No clusters with sufficient tracking history to plot")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
