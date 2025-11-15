"""Compare custom detection vs YOLO for drone detection in event camera data.

This script runs both detection methods side-by-side for comparison.
"""

import argparse  # noqa: INP001
import time
from collections import deque

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource

# Import detection functions from detect_drone.py
# Use relative import since both are in scripts/
import sys
from pathlib import Path

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from detect_drone import (
    accumulate_events,
    merge_nearby_detections,
    filter_drone_detections,
    filter_detections_by_temporal_consistency,
    normalize_frame,
)

# Try to import YOLO, but make it optional
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    print("YOLO detection will be disabled.")


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract event coordinates and polarities from a time window."""
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def events_to_rgb_frame(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarities: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert events to RGB frame for YOLO (similar to play_dat.py)."""
    frame = np.full((height, width, 3), (127, 127, 127), dtype=np.uint8)
    frame[y_coords[polarities], x_coords[polarities]] = (255, 255, 255)
    frame[y_coords[~polarities], x_coords[~polarities]] = (0, 0, 0)
    return frame


def custom_detection(
    frame: np.ndarray,
    threshold: int = 30,
    min_area: int = 200,
    max_area: int = 50000,
    detection_history: deque | None = None,
    merge_distance: float = 80.0,
    min_consistency: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    """Full custom detection method from detect_drone.py."""
    # Simple blob detection - find all motion regions
    _, binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find all contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Simple detection: find compact, roughly square regions
    all_detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Check compactness and spatial distribution
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        non_zero = np.count_nonzero(roi > threshold * 0.5)
        compactness = non_zero / area if area > 0 else 0

        # Filter very sparse regions (trees/background)
        if compactness < 0.12:
            continue

        # Filter very large sparse regions (trees)
        if area > 20000 and compactness < 0.25:
            continue

        # Check for irregular distribution (trees have irregular patterns)
        if area > 1000:
            grid_size = 4
            cell_w, cell_h = w // grid_size, h // grid_size
            if cell_w > 0 and cell_h > 0:
                cell_activities = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell_y1 = i * cell_h
                        cell_y2 = min((i + 1) * cell_h, h)
                        cell_x1 = j * cell_w
                        cell_x2 = min((j + 1) * cell_w, w)
                        cell_roi = roi[cell_y1:cell_y2, cell_x1:cell_x2]
                        cell_activity = np.count_nonzero(cell_roi > threshold * 0.5)
                        cell_activities.append(cell_activity)

                if len(cell_activities) > 0:
                    activity_std = np.std(cell_activities)
                    activity_mean = np.mean(cell_activities)
                    if activity_mean > 0:
                        cv = activity_std / activity_mean
                        if cv > 1.2:  # High coefficient of variation
                            continue

        all_detections.append((x, y, w, h))

    # Merge nearby detections (fixes issue where rotating drone is detected as multiple)
    merged_detections = merge_nearby_detections(
        all_detections,
        max_distance=merge_distance,
        min_overlap=0.15,
    )

    # Apply additional filtering to remove false positives
    filtered_detections = filter_drone_detections(
        merged_detections,
        frame,
        min_area=min_area,
        max_area=max_area,
    )

    # Apply temporal filtering to remove flickering detections (trees)
    if detection_history is not None and len(detection_history) >= 3:
        filtered_detections = filter_detections_by_temporal_consistency(
            filtered_detections,
            detection_history,
            max_distance=150.0,
            min_consistency=min_consistency,
        )

    return filtered_detections


def yolo_detection(
    frame_rgb: np.ndarray, model: "YOLO", conf_threshold: float = 0.25
) -> list[tuple[int, int, int, int]]:
    """Run YOLO detection on frame."""
    if not YOLO_AVAILABLE:
        return []

    # YOLO expects BGR for some models, but ultralytics handles RGB
    results = model(frame_rgb, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls = int(box.cls[0])
            class_name = result.names[cls]

            # Filter for objects that might be drones
            # YOLO classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat
            # We'll look for airplane, or any small flying object
            if class_name in ["airplane", "bird"] or box.conf[0] > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = int(x2 - x1)
                h = int(y2 - y1)
                detections.append((int(x1), int(y1), w, h))

    return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "",
) -> np.ndarray:
    """Draw bounding boxes on frame."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()
    for x, y, w, h in detections:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(
                frame_bgr,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    return frame_bgr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare custom detection vs YOLO for drone detection"
    )
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window",
        type=float,
        default=50.0,
        help="Window duration in ms (default: 50ms)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1 is real time)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Custom detection threshold (default: 30)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum detection area (default: 200)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=50000,
        help="Maximum detection area (default: 50000)",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=80.0,
        help="Maximum distance to merge nearby detections (default: 80)",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.4,
        help="Minimum consistency for temporal filtering (default: 0.4)",
    )
    args = parser.parse_args()

    # Initialize data source
    src = DatFileSource(
        args.dat,
        width=1280,
        height=720,
        window_length_us=args.window * 1000,
    )

    # Initialize YOLO model if available
    yolo_model = None
    yolo_ready = False
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO(args.yolo_model)
            print(f"Loaded YOLO model: {args.yolo_model}")
            yolo_ready = True
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Continuing with custom detection only.")
            yolo_ready = False

    # Initialize pacer
    pacer = Pacer(speed=args.speed, force_speed=False)

    # Detection history for temporal filtering
    detection_history: deque = deque(maxlen=5)

    cv2.namedWindow("Detection Comparison", cv2.WINDOW_NORMAL)

    print(f"Processing {args.dat}")
    print(f"Window: {args.window}ms, Speed: {args.speed}x")
    if yolo_ready:
        print("Running: Custom Detection | YOLO Detection")
    else:
        print("Running: Custom Detection only (YOLO not available)")
    print("Press 'q' or ESC to quit")

    for batch_range in pacer.pace(src.ranges()):
        # Extract events
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        x_coords, y_coords, polarities = window

        # Create frames for both methods
        event_frame = accumulate_events(
            x_coords, y_coords, polarities, src.width, src.height, use_polarity=False
        )
        frame_gray = normalize_frame(event_frame)
        frame_rgb = events_to_rgb_frame(
            x_coords, y_coords, polarities, src.width, src.height
        )

        # Run custom detection (full pipeline from detect_drone.py)
        custom_detections = custom_detection(
            frame_gray,
            threshold=args.threshold,
            min_area=args.min_area,
            max_area=args.max_area,
            detection_history=detection_history,
            merge_distance=args.merge_distance,
            min_consistency=args.min_consistency,
        )

        # Update history for temporal filtering
        detection_history.append(custom_detections)

        # Run YOLO detection
        yolo_detections = []
        if yolo_ready:
            yolo_detections = yolo_detection(frame_rgb, yolo_model, args.yolo_conf)

        # Create comparison display
        # Left side: Custom detection
        custom_display = draw_detections(
            frame_gray, custom_detections, color=(0, 255, 0), label="CUSTOM"
        )
        cv2.putText(
            custom_display,
            f"Custom: {len(custom_detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Right side: YOLO detection
        if yolo_ready:
            yolo_display = draw_detections(
                frame_rgb, yolo_detections, color=(255, 0, 0), label="YOLO"
            )
            cv2.putText(
                yolo_display,
                f"YOLO: {len(yolo_detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            # Combine side by side
            comparison = np.hstack([custom_display, yolo_display])
        else:
            comparison = custom_display

        # Display
        cv2.imshow("Detection Comparison", comparison)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

