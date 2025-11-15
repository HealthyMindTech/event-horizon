#!/usr/bin/env python3
"""
Temporal blade tracking with rolling window.

Tracks propeller blades over time by maintaining clusters with a rolling window,
assigning new events to existing clusters, removing old events, and updating
cluster statistics (center, slope, angle) continuously.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from evio.core.recording import open_dat


def load_dat_file(filepath, width=1280, height=720):
    """Load .dat file with event camera data using evio library."""
    rec = open_dat(filepath, width=width, height=height)

    # Decode event words to get x, y, polarity
    x = rec.event_words & 0x3FFF  # 14 bits for x (bits 0-13)
    y = (rec.event_words >> 14) & 0x3FFF  # 14 bits for y (bits 14-27)
    p = ((rec.event_words >> 28) & 0xF).astype(
        np.int32
    )  # 4 bits for polarity (bits 28-31)
    p = (p > 0).astype(np.int32) * 2 - 1  # Convert to -1/1

    # Apply time ordering
    x = x[rec.order]
    y = y[rec.order]
    p = p[rec.order]
    t = rec.timestamps  # Already sorted

    # Stack into events array
    events = np.column_stack([x, y, p, t])

    print(f"Resolution: {rec.width}x{rec.height}")

    return events, rec.width, rec.height


def filter_events_by_time_and_polarity(
    events, start_time_us, duration_us, polarity=None
):
    """Filter events within a time window and optionally by polarity."""
    end_time = start_time_us + duration_us

    mask = (events[:, 3] >= start_time_us) & (events[:, 3] < end_time)

    if polarity is not None:
        mask = mask & (events[:, 2] == polarity)

    return events[mask]


def filter_hot_pixels(events, threshold=100):
    """Filter out hot pixels - locations with too many events at exact same coordinate."""
    unique_coords, counts = np.unique(events[:, :2], axis=0, return_counts=True)

    hot_pixel_mask = counts >= threshold
    hot_pixels = unique_coords[hot_pixel_mask]

    if len(hot_pixels) > 0:
        print(f"Found {len(hot_pixels)} hot pixel locations")
        keep_mask = np.ones(len(events), dtype=bool)
        for hp in hot_pixels:
            hot_pixel_events = (events[:, 0] == hp[0]) & (events[:, 1] == hp[1])
            keep_mask &= ~hot_pixel_events
        return events[keep_mask]

    return events


class BladeCluster:
    """Represents a tracked blade cluster with rolling window of events."""

    def __init__(self, cluster_id, initial_events):
        self.id = cluster_id
        self.events = initial_events.copy()  # (x, y, polarity, time)
        self.update_statistics()

    def update_statistics(self):
        """Recalculate center, slope, angle from current events."""
        if len(self.events) < 2:
            self.center_x = (
                np.mean(self.events[:, 0]) if len(self.events) > 0 else 0
            )
            self.center_y = (
                np.mean(self.events[:, 1]) if len(self.events) > 0 else 0
            )
            self.slope = 0
            self.angle_deg = 0
            self.std_x = 0
            self.std_y = 0
            return

        self.center_x = np.mean(self.events[:, 0])
        self.center_y = np.mean(self.events[:, 1])
        self.std_x = np.std(self.events[:, 0])
        self.std_y = np.std(self.events[:, 1])

        # Fit linear regression
        X = self.events[:, 0].reshape(-1, 1)
        y = self.events[:, 1]

        lr = LinearRegression()
        lr.fit(X, y)
        self.slope = lr.coef_[0]
        self.angle_deg = np.degrees(np.arctan(self.slope))

    def add_events(self, new_events):
        """Add new events to the cluster."""
        self.events = np.vstack([self.events, new_events])
        self.update_statistics()

    def remove_old_events(self, cutoff_time_us):
        """Remove events older than cutoff time."""
        mask = self.events[:, 3] >= cutoff_time_us
        self.events = self.events[mask]
        if len(self.events) > 0:
            self.update_statistics()

    def __len__(self):
        return len(self.events)

    def distance_to_point(self, x, y):
        """Calculate distance from point to cluster center."""
        return np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)


class TemporalBladeTracker:
    """Tracks blade clusters over time with rolling window."""

    def __init__(
        self,
        assignment_distance=15,
        min_cluster_size=5,
        window_duration_us=750,
    ):
        self.clusters = {}  # cluster_id -> BladeCluster
        self.next_cluster_id = 0
        self.assignment_distance = assignment_distance
        self.min_cluster_size = min_cluster_size
        self.window_duration_us = window_duration_us
        self.current_time_us = 0

    def process_events(self, new_events):
        """Process a batch of new events."""
        if len(new_events) == 0:
            return

        # Update current time
        self.current_time_us = new_events[:, 3].max()

        # Assign each event to nearest cluster or create new cluster
        for event in new_events:
            x, y, polarity, timestamp = event
            assigned = False

            # Try to assign to existing cluster
            min_distance = float("inf")
            closest_cluster_id = None

            for cluster_id, cluster in self.clusters.items():
                distance = cluster.distance_to_point(x, y)
                if (
                    distance < self.assignment_distance
                    and distance < min_distance
                ):
                    min_distance = distance
                    closest_cluster_id = cluster_id
                    assigned = True

            if assigned:
                # Add to existing cluster
                self.clusters[closest_cluster_id].add_events(
                    event.reshape(1, -1)
                )
            else:
                # Create new cluster
                new_cluster = BladeCluster(
                    self.next_cluster_id, event.reshape(1, -1)
                )
                self.clusters[self.next_cluster_id] = new_cluster
                self.next_cluster_id += 1

        # Remove old events from all clusters (rolling window)
        cutoff_time = self.current_time_us - self.window_duration_us
        clusters_to_remove = []

        for cluster_id, cluster in self.clusters.items():
            cluster.remove_old_events(cutoff_time)
            if len(cluster) < self.min_cluster_size:
                clusters_to_remove.append(cluster_id)

        # Remove clusters that are too small
        for cluster_id in clusters_to_remove:
            del self.clusters[cluster_id]

    def get_cluster_info(self):
        """Get current cluster information sorted by size."""
        cluster_list = []
        for cluster in self.clusters.values():
            if len(cluster) >= self.min_cluster_size:
                cluster_list.append(
                    {
                        "id": cluster.id,
                        "count": len(cluster),
                        "center_x": cluster.center_x,
                        "center_y": cluster.center_y,
                        "std_x": cluster.std_x,
                        "std_y": cluster.std_y,
                        "slope": cluster.slope,
                        "angle_deg": cluster.angle_deg,
                        "events": cluster.events,
                    }
                )

        # Sort by event count (descending)
        cluster_list.sort(key=lambda x: x["count"], reverse=True)
        return cluster_list


def visualize_temporal_tracking(tracker, width, height, time_ms, window_idx):
    """Visualize current state of temporal tracking."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    cluster_info = tracker.get_cluster_info()

    # Calculate region of interest
    if len(cluster_info) > 0:
        all_x = np.concatenate([info["events"][:, 0] for info in cluster_info])
        all_y = np.concatenate([info["events"][:, 1] for info in cluster_info])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        margin = 50
        x_min = max(0, x_min - margin)
        x_max = min(width, x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(height, y_max + margin)
    else:
        x_min, x_max = 0, width
        y_min, y_max = 0, height

    # Left plot: Tracked clusters
    ax1 = axes[0]
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_max, y_min)
    ax1.set_aspect("equal")
    ax1.set_title(
        f"Temporal Blade Tracking - Window {window_idx}\nTime: {time_ms:.3f}ms (rolling window: {tracker.window_duration_us / 1000:.2f}ms)"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, info in enumerate(cluster_info[:10]):
        cluster_events = info["events"]

        ax1.scatter(
            cluster_events[:, 0],
            cluster_events[:, 1],
            c=[colors[i % 10]],
            s=3,
            alpha=0.6,
            label=f"Blade {info['id']}: {info['count']} events",
        )

        # Mark cluster center
        ax1.plot(
            info["center_x"],
            info["center_y"],
            "x",
            markersize=15,
            markeredgewidth=3,
            color="red",
        )

        # Draw fitted line
        x_min_cluster = cluster_events[:, 0].min()
        x_max_cluster = cluster_events[:, 0].max()
        x_line = np.array([x_min_cluster, x_max_cluster])
        y_line = info["slope"] * x_line + (
            info["center_y"] - info["slope"] * info["center_x"]
        )
        ax1.plot(x_line, y_line, "k--", linewidth=2, alpha=0.7)

        # Label with angle
        ax1.text(
            info["center_x"] + 10,
            info["center_y"],
            f"ID{info['id']}\n{info['angle_deg']:.1f}°",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right plot: Statistics
    ax2 = axes[1]
    ax2.axis("off")

    stats_text = f"Temporal Tracking Statistics\n"
    stats_text += f"{'=' * 50}\n\n"
    stats_text += f"Current time: {tracker.current_time_us / 1000:.3f} ms\n"
    stats_text += (
        f"Rolling window: {tracker.window_duration_us / 1000:.2f} ms\n"
    )
    stats_text += f"Active clusters: {len(cluster_info)}\n"
    stats_text += f"Assignment distance: {tracker.assignment_distance} px\n"
    stats_text += f"Min cluster size: {tracker.min_cluster_size}\n\n"

    stats_text += f"Tracked Blades:\n"
    stats_text += f"{'-' * 50}\n"

    for i, info in enumerate(cluster_info[:10]):
        stats_text += f"\nBlade ID {info['id']}:\n"
        stats_text += f"  Events: {info['count']}\n"
        stats_text += (
            f"  Center: ({info['center_x']:.1f}, {info['center_y']:.1f})\n"
        )
        stats_text += (
            f"  Spread: ({info['std_x']:.1f}, {info['std_y']:.1f}) px\n"
        )
        stats_text += (
            f"  Angle: {info['angle_deg']:.1f}° (slope: {info['slope']:.3f})\n"
        )

    ax2.text(
        0.05,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    return fig


def main():
    # Load the data file
    dat_file = Path(__file__).parent.parent / "drone_idle.dat"
    print(f"Loading {dat_file}...")

    events, width, height = load_dat_file(dat_file)
    print(f"Loaded {len(events)} events")

    # Get time range
    min_time = events[:, 3].min()
    max_time = events[:, 3].max()
    duration = (max_time - min_time) / 1000.0  # ms

    print(f"Time range: {min_time} to {max_time} ({duration:.2f} ms)")

    # Parameters
    time_window_ms = 0.75  # 0.75ms = 1/8th revolution at 10,000 RPM
    time_window_us = int(time_window_ms * 1000)
    start_time_sec = 3.0  # Start at 3 seconds
    start_time = min_time + int(start_time_sec * 1_000_000)
    polarity = -1  # Negative polarity (clearer)

    # Initialize temporal tracker
    tracker = TemporalBladeTracker(
        assignment_distance=15,  # pixels
        min_cluster_size=8,  # minimum events per cluster
        window_duration_us=time_window_us,  # 0.75ms rolling window
    )

    print(f"\n{'=' * 60}")
    print(f"Temporal Blade Tracking")
    print(f"{'=' * 60}")
    print(f"Starting at: {start_time_sec}s")
    print(f"Window size: {time_window_ms}ms")
    print(f"Number of windows: 3")
    print(f"Assignment distance: {tracker.assignment_distance}px")
    print(f"Min cluster size: {tracker.min_cluster_size}")

    # Process 3 consecutive windows
    num_windows = 3
    for window_idx in range(num_windows):
        window_start = start_time + window_idx * time_window_us
        window_label = f"Window {window_idx + 1}"

        print(f"\n{'-' * 60}")
        print(f"{window_label}: {window_start / 1000:.3f}ms")
        print(f"{'-' * 60}")

        # Get events for this window
        window_events = filter_events_by_time_and_polarity(
            events, window_start, time_window_us, polarity
        )

        print(f"New events: {len(window_events)}")

        # Filter hot pixels
        window_events = filter_hot_pixels(window_events, threshold=100)
        print(f"After hot pixel filtering: {len(window_events)}")

        # Process events through tracker
        tracker.process_events(window_events)

        # Get current cluster info
        cluster_info = tracker.get_cluster_info()
        print(f"Active clusters: {len(cluster_info)}")

        # Print cluster details
        if len(cluster_info) > 0:
            print(f"\nCluster Details:")
            print(
                f"{'ID':<8} {'Events':<10} {'Center (X, Y)':<20} {'Angle':<15}"
            )
            print(f"{'-' * 55}")
            for info in cluster_info:
                print(
                    f"{info['id']:<8} {info['count']:<10} ({info['center_x']:6.1f}, {info['center_y']:6.1f})    {info['angle_deg']:6.1f}° ({info['slope']:6.3f})"
                )

        # Visualize
        fig = visualize_temporal_tracking(
            tracker,
            width,
            height,
            (window_start - min_time) / 1000,
            window_idx + 1,
        )

        output_filename = f"temporal_tracking_window{window_idx + 1}.png"
        plt.savefig(output_filename, dpi=150)
        print(f"Saved: {output_filename}")
        plt.close(fig)

    print("\n" + "=" * 60)
    print("Temporal tracking complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
