"""Simulate an event camera viewing a 3D rotating propeller.

This script:
1. Creates a 3D propeller with configurable blades
2. Renders it frame-by-frame at high speed
3. Detects brightness changes between frames
4. Generates ON/OFF events based on thresholds
5. Writes events in Prophesee .dat format
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


class Propeller3D:
    """3D propeller mesh with rotation and projection capabilities."""

    def __init__(
        self,
        num_blades: int = 3,
        blade_length: float = 0.4,
        blade_width: float = 0.04,
        hub_radius: float = 0.05,
    ):
        self.num_blades = num_blades
        self.blade_length = blade_length
        self.blade_width = blade_width
        self.hub_radius = hub_radius
        self.vertices = self._create_geometry()

    def _create_geometry(self) -> np.ndarray:
        """Create 3D vertices for propeller (hub + blades)."""
        vertices = []

        # Hub (small cylinder approximated as octagon)
        hub_segments = 8
        for i in range(hub_segments):
            angle = 2 * np.pi * i / hub_segments
            x = self.hub_radius * np.cos(angle)
            y = self.hub_radius * np.sin(angle)
            vertices.append([x, y, -0.02])  # front
            vertices.append([x, y, 0.02])  # back

        # Blades
        for blade_idx in range(self.num_blades):
            blade_angle = 2 * np.pi * blade_idx / self.num_blades
            cos_a, sin_a = np.cos(blade_angle), np.sin(blade_angle)

            # Each blade: 4 corners (quad)
            # Root near hub
            x1 = self.hub_radius * cos_a
            y1 = self.hub_radius * sin_a
            # Tip at blade_length
            x2 = self.blade_length * cos_a
            y2 = self.blade_length * sin_a

            # Perpendicular offset for width
            perp_x = -sin_a * self.blade_width / 2
            perp_y = cos_a * self.blade_width / 2

            # 4 corners of blade quad
            vertices.extend(
                [
                    [x1 + perp_x, y1 + perp_y, 0],
                    [x1 - perp_x, y1 - perp_y, 0],
                    [x2 - perp_x, y2 - perp_y, 0],
                    [x2 + perp_x, y2 + perp_y, 0],
                ]
            )

        return np.array(vertices, dtype=np.float32)

    def rotate(
        self, angle_x: float, angle_y: float, angle_z: float
    ) -> np.ndarray:
        """Apply 3D rotation to vertices."""
        # Rotation matrices
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)

        rx = np.array(
            [
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx],
            ]
        )
        ry = np.array(
            [
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy],
            ]
        )
        rz = np.array(
            [
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1],
            ]
        )

        # Apply rotations: spin first (Z), then tilt for viewing (X, Y)
        # This makes blades spin in their natural plane, then tilts the whole thing
        rotation = rx @ ry @ rz
        return self.vertices @ rotation.T

    def project(
        self,
        vertices_3d: np.ndarray,
        distance: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Perspective projection to 2D screen coordinates."""
        # Simple perspective projection
        z_offset = vertices_3d[:, 2] + distance
        z_offset = np.maximum(z_offset, 0.1)  # avoid division by zero

        scale = 300  # scale factor for visibility
        x_proj = (vertices_3d[:, 0] * scale / z_offset) + width / 2
        y_proj = (-vertices_3d[:, 1] * scale / z_offset) + height / 2

        return np.stack([x_proj, y_proj], axis=1)


class EventCameraSimulator:
    """Simulate event camera by detecting brightness changes."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        pos_threshold: int = 15,
        neg_threshold: int = 15,
        noise_drop_rate: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.noise_drop_rate = noise_drop_rate
        self.prev_frame = np.full((height, width), 128, dtype=np.uint8)
        self.events = []

    def render_propeller_frame(
        self,
        propeller: Propeller3D,
        angle_x: float,
        angle_y: float,
        angle_z: float,
        distance: float = 2.0,
    ) -> np.ndarray:
        """Render propeller to grayscale frame."""
        frame = np.full((self.height, self.width), 0, dtype=np.uint8)

        rotated = propeller.rotate(angle_x, angle_y, angle_z)
        points_2d = propeller.project(
            rotated, distance, self.width, self.height
        )

        # Draw hub
        hub_points = points_2d[:16].astype(np.int32)
        if len(hub_points) > 0:
            cv2.fillPoly(frame, [hub_points], color=200)

        # Draw blades
        offset = 16
        for blade_idx in range(propeller.num_blades):
            blade_points = points_2d[offset : offset + 4].astype(np.int32)
            if len(blade_points) == 4:
                cv2.fillPoly(frame, [blade_points], color=220)
            offset += 4

        return frame

    def process_frame(self, frame: np.ndarray, timestamp_us: int) -> None:
        """Detect events by comparing with previous frame."""
        diff = frame.astype(np.int16) - self.prev_frame.astype(np.int16)

        # ON events (brightness increase)
        on_mask = diff > self.pos_threshold
        on_coords = np.argwhere(on_mask)
        for y, x in on_coords:
            # Drop event with probability noise_drop_rate
            if np.random.random() > self.noise_drop_rate:
                self.events.append((timestamp_us, x, y, 1))

        # OFF events (brightness decrease)
        off_mask = diff < -self.neg_threshold
        off_coords = np.argwhere(off_mask)
        for y, x in off_coords:
            # Drop event with probability noise_drop_rate
            if np.random.random() > self.noise_drop_rate:
                self.events.append((timestamp_us, x, y, 0))

        self.prev_frame = frame.copy()

    def get_events(self) -> list[tuple[int, int, int, int]]:
        """Return collected events as (timestamp_us, x, y, polarity)."""
        return self.events


def pack_event_word(x: int, y: int, polarity: int) -> np.uint32:
    """Pack x, y, polarity into a 32-bit word.

    Bit layout: [31:28]=polarity, [27:14]=y, [13:0]=x
    """
    polarity_bits = (1 if polarity else 0) << 28
    y_bits = (int(y) & 0x3FFF) << 14
    x_bits = int(x) & 0x3FFF
    return np.uint32(polarity_bits | y_bits | x_bits)


def write_dat_file(
    events: list[tuple[int, int, int, int]],
    output_path: Path,
    width: int,
    height: int,
) -> None:
    """Write events to a .dat file in Prophesee format."""
    with open(output_path, "wb") as f:
        # Write ASCII header
        header = f"% Date 2024-01-01 00:00:00\n"
        header += f"% Width {width}\n"
        header += f"% Height {height}\n"
        header += f"% Format EVT3\n"
        f.write(header.encode("ascii"))

        # Write format descriptor (2 bytes)
        f.write(bytes([0x00]))  # event_type
        f.write(bytes([0x08]))  # event_size (8 bytes)

        # Write events
        for timestamp_us, x, y, polarity in events:
            # Pack as little-endian uint32 + uint32
            t32 = np.uint32(timestamp_us)
            w32 = pack_event_word(x, y, polarity)

            f.write(t32.tobytes())
            f.write(w32.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate an event camera viewing a 3D rotating propeller"
    )
    parser.add_argument("output", type=Path, help="Output .dat file path")
    parser.add_argument(
        "--width", type=int, default=1280, help="Image width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Image height (default: 720)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Simulation duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1000,
        help="Rendering frame rate (default: 1000)",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=600.0,
        help="Propeller rotation speed in RPM (default: 600)",
    )
    parser.add_argument(
        "--tilt-x",
        type=float,
        default=65.0,
        help="Tilt around X axis in degrees (default: 65, drone viewed from ground)",
    )
    parser.add_argument(
        "--tilt-y",
        type=float,
        default=5.0,
        help="Tilt around Y axis in degrees (default: 5)",
    )
    parser.add_argument(
        "--blades",
        type=int,
        default=3,
        help="Number of propeller blades (default: 3)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Event threshold for brightness change detection (default: 15)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Event drop rate for noise simulation (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview during simulation",
    )

    args = parser.parse_args()

    print(f"Simulating {args.blades}-blade propeller at {args.rpm} RPM")
    print(f"Duration: {args.duration}s @ {args.fps} fps")
    print(f"Tilt: X={args.tilt_x}°, Y={args.tilt_y}°")
    print(f"Event threshold: ±{args.threshold}")
    if args.noise > 0:
        print(f"Noise: {args.noise * 100:.1f}% event drop rate")

    # Create propeller
    propeller = Propeller3D(num_blades=args.blades)

    # Create simulator
    simulator = EventCameraSimulator(
        width=args.width,
        height=args.height,
        pos_threshold=args.threshold,
        neg_threshold=args.threshold,
        noise_drop_rate=args.noise,
    )

    # Simulation parameters
    total_frames = int(args.duration * args.fps)
    dt_us = int(1_000_000 / args.fps)
    angular_velocity = (args.rpm / 60.0) * 2 * np.pi  # rad/s

    # Fixed tilt angles
    tilt_x = np.radians(args.tilt_x)
    tilt_y = np.radians(args.tilt_y)

    print(f"\nSimulating {total_frames} frames...")

    # Render first frame to initialize prev_frame (no events generated)
    first_frame = simulator.render_propeller_frame(
        propeller,
        angle_x=tilt_x,
        angle_y=tilt_y,
        angle_z=0.0,
        distance=2.0,
    )
    simulator.prev_frame = first_frame.copy()

    for frame_idx in range(total_frames):
        timestamp_us = frame_idx * dt_us
        t_seconds = timestamp_us / 1_000_000.0

        # Calculate rotation angle for propeller spin
        angle_z = angular_velocity * t_seconds

        # Render frame
        frame = simulator.render_propeller_frame(
            propeller,
            angle_x=tilt_x,
            angle_y=tilt_y,
            angle_z=angle_z,
            distance=2.0,
        )

        # Detect events
        simulator.process_frame(frame, timestamp_us)

        # Preview
        if args.preview and frame_idx % 10 == 0:
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                preview,
                f"Frame {frame_idx}/{total_frames}  Events: {len(simulator.events)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Propeller Simulation", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Progress indicator
        if (frame_idx + 1) % (total_frames // 20) == 0:
            pct = 100 * (frame_idx + 1) / total_frames
            print(
                f"  {pct:.0f}% complete ({len(simulator.events)} events so far)"
            )

    if args.preview:
        cv2.destroyAllWindows()

    # Get events and sort by timestamp
    events = simulator.get_events()
    events.sort(key=lambda e: e[0])

    print(f"\nGenerated {len(events)} total events")
    print(f"Event rate: {len(events) / args.duration:.0f} events/second")

    # Write .dat file
    write_dat_file(events, args.output, args.width, args.height)

    print(f"\n✓ Done! Play with:")
    print(f"  uv run scripts/play_dat.py {args.output}")


if __name__ == "__main__":
    main()
