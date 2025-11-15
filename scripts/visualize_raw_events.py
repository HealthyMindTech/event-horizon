#!/usr/bin/env python3
"""
Simple raw event visualization to see propeller blade patterns.

Shows events as scatter plots without any clustering or processing,
just to see what the raw data looks like.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from evio.core.recording import open_dat


def load_dat_file(filepath, width=1280, height=720):
    """Load .dat file with event camera data using evio library."""
    rec = open_dat(filepath, width=width, height=height)

    # Decode event words to get x, y, polarity
    # Event word format: bits 0-10: x, bits 11-21: y, bit 22: polarity
    event_words = rec.event_words
    timestamps = rec.timestamps

    x = event_words & 0x3FFF  # 14 bits for x (bits 0-13)
    y = (event_words >> 14) & 0x3FFF  # 14 bits for y (bits 14-27)
    p = ((event_words >> 28) & 0xF).astype(
        np.int32
    )  # 4 bits for polarity (bits 28-31)
    p = (p > 0).astype(np.int32) * 2 - 1  # Convert to -1/1

    # Apply time ordering
    x = x[rec.order]
    y = y[rec.order]
    p = p[rec.order]
    t = timestamps  # Already sorted

    # Stack into events array
    events = np.column_stack([x, y, p, t])

    print(f"Resolution: {rec.width}x{rec.height}")

    return events, rec.width, rec.height


def filter_events_by_time_and_polarity(
    events, start_time_us, duration_us, polarity=None
):
    """
    Filter events within a time window and optionally by polarity.

    Args:
        events: array of (x, y, polarity, time)
        start_time_us: start time in microseconds
        duration_us: duration in microseconds
        polarity: 1 for positive, -1 for negative, None for both

    Returns:
        filtered events array
    """
    end_time = start_time_us + duration_us

    mask = (events[:, 3] >= start_time_us) & (events[:, 3] < end_time)

    if polarity is not None:
        mask = mask & (events[:, 2] == polarity)

    return events[mask]


def visualize_raw_events(
    events1, events2, width, height, time_window_ms, start_time_sec
):
    """Visualize raw events as scatter plots - positive polarity only, two consecutive time windows."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Only positive polarity
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"Negative Polarity Events - Two Consecutive Windows\nWindow 1 (red): {len(events1)} events | Window 2 (blue): {len(events2)} events\nt={start_time_sec}s to t={start_time_sec + 2 * time_window_ms / 1000}s, Δt={time_window_ms}ms each",
        fontsize=12,
    )
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)

    # First window in red
    if len(events1) > 0:
        ax.scatter(
            events1[:, 0],
            events1[:, 1],
            c="red",
            s=3,
            alpha=0.6,
            label=f"Window 1 (t={start_time_sec}s)",
        )

    # Second window in blue
    if len(events2) > 0:
        ax.scatter(
            events2[:, 0],
            events2[:, 1],
            c="blue",
            s=3,
            alpha=0.6,
            label=f"Window 2 (t={start_time_sec + time_window_ms / 1000}s)",
        )

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

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

    # Analyze polarity distribution
    pos_count = np.sum(events[:, 2] == 1)
    neg_count = np.sum(events[:, 2] == -1)
    print(f"\nPolarity distribution:")
    print(
        f"  Positive (+1): {pos_count} events ({100 * pos_count / len(events):.1f}%)"
    )
    print(
        f"  Negative (-1): {neg_count} events ({100 * neg_count / len(events):.1f}%)"
    )

    # Parameters
    time_window_ms = 0.75  # 0.75ms = 1/8th revolution at 10,000 RPM
    time_window_us = int(time_window_ms * 1000)
    start_time_sec = 3.0  # Start at 3 seconds

    # Start at 3 seconds as requested
    start_time = min_time + int(start_time_sec * 1_000_000)

    print(f"\n{'=' * 60}")
    print(
        f"Visualizing events at t={start_time_sec}s, window={time_window_ms}ms"
    )
    print(f"{'=' * 60}")

    # Filter events - only negative polarity - first window
    filtered_events_1 = filter_events_by_time_and_polarity(
        events, start_time, time_window_us, polarity=-1
    )

    # Filter events for second window (immediately after first)
    start_time_2 = start_time + time_window_us
    filtered_events_2 = filter_events_by_time_and_polarity(
        events, start_time_2, time_window_us, polarity=-1
    )

    print(f"Window 1 events: {len(filtered_events_1)}")
    print(f"Window 2 events: {len(filtered_events_2)}")

    if len(filtered_events_1) < 10 and len(filtered_events_2) < 10:
        print("Too few events to visualize!")
        return

    # Visualize
    fig = visualize_raw_events(
        filtered_events_1,
        filtered_events_2,
        width,
        height,
        time_window_ms,
        start_time_sec,
    )

    output_file = "raw_events_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {output_file}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
