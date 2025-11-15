"""Drone detection from event camera data.

This script processes .dat event files to detect drones with 4 propellers by:
1. Accumulating events over time windows
2. Detecting circular motion patterns (propellers) using HoughCircles
3. Finding 4-propeller cross patterns characteristic of drones
4. Filtering out false positives (planes, trees) by:
   - Aspect ratio filtering (planes are elongated)
   - Compactness filtering (trees are irregular/sparse)
   - Multiple activity center detection (drones have propellers + body)
5. Temporal consistency filtering for stability
6. Visualizing detections with bounding boxes
"""

import argparse  # noqa: INP001
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


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


def accumulate_events(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    polarities: np.ndarray,
    width: int,
    height: int,
    *,
    use_polarity: bool = True,
) -> np.ndarray:
    """Accumulate events into a frame with polarity encoding.

    Args:
        x_coords: X coordinates of events
        y_coords: Y coordinates of events
        polarities: Boolean array indicating ON (True) or OFF (False) events
        width: Frame width
        height: Frame height
        use_polarity: If True, ON events add +1, OFF events add -1.
                     If False, all events add +1.

    Returns:
        Accumulated frame as float32 array
    """
    frame = np.zeros((height, width), dtype=np.float32)
    if use_polarity:
        # Positive events add, negative events subtract
        frame[y_coords[polarities], x_coords[polarities]] += 1.0
        frame[y_coords[~polarities], x_coords[~polarities]] -= 1.0
    else:
        # All events add equally (motion intensity)
        frame[y_coords, x_coords] += 1.0
    return frame


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame to 0-255 range for visualization."""
    frame_abs = np.abs(frame)
    if frame_abs.max() > 0:
        frame_norm = (frame_abs / frame_abs.max() * 255).astype(np.uint8)
    else:
        frame_norm = np.zeros_like(frame, dtype=np.uint8)
    return frame_norm


def detect_circular_patterns(
    frame: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 50,
    threshold: int = 50,
) -> list[tuple[int, int, int]]:
    """Detect circular motion patterns (propellers) using HoughCircles.

    Args:
        frame: Normalized event accumulation frame
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius

    Returns:
        List of (x, y, radius) circles
    """
    # Use HoughCircles to detect circular patterns
    circles = cv2.HoughCircles(
        frame,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max_radius * 2,
        param1=threshold,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return [(x, y, r) for x, y, r in circles]
    return []


def detect_propellers_in_roi(
    roi: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 50,
    threshold: int = 40,
) -> list[tuple[int, int, int, int, float]]:
    """Detect propellers (elliptical/circular patterns) within a region of interest.
    
    Uses multiple methods:
    1. HoughCircles for circular patterns
    2. Ellipse fitting for elliptical patterns
    3. Contour analysis for irregular circular shapes

    Args:
        roi: Region of interest (part of frame containing a drone)
        min_radius: Minimum propeller radius
        max_radius: Maximum propeller radius
        threshold: Threshold for binary conversion

    Returns:
        List of (x, y, width, height, angle) ellipses/circles in ROI coordinates
    """
    propellers = []
    
    # Use adaptive thresholding for better detection in varying conditions
    binary_adaptive = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Also try simple thresholding
    _, binary_simple = cv2.threshold(roi, threshold * 0.7, 255, cv2.THRESH_BINARY)
    
    # Combine both
    binary = cv2.bitwise_or(binary_adaptive, binary_simple)
    
    # Apply morphological operations to enhance circular patterns
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Method 1: HoughCircles for circular patterns
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max_radius,
        param1=30,  # Lower for more sensitivity
        param2=15,  # Lower threshold for more detections
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            # Convert to ellipse format (x, y, width, height, angle=0 for circles)
            # Circles (w=h) are not horizontally elongated, but still acceptable
            # Give them a neutral elongation score
            propellers.append((x, y, r * 2, r * 2, 0.0, 1.0, False))
    
    # Method 2: Ellipse fitting for elliptical patterns
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        if len(contour) < 5:  # Need at least 5 points for ellipse fitting
            continue
        
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius ** 2 or area > np.pi * max_radius ** 2:
            continue
        
        # Fit ellipse
        try:
            ellipse = cv2.fitEllipse(contour)
            center, (width, height), angle = ellipse
            
            # Check if it's roughly circular/elliptical (not too elongated)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if aspect_ratio > 4.0:  # Too elongated, skip
                continue
            
            # Prefer horizontally elongated ellipses (propellers are typically wider than tall)
            # Give higher score to horizontally elongated shapes
            is_horizontal = width > height
            elongation_score = width / height if height > 0 else 1.0
            
            # Check size
            avg_radius = (width + height) / 4
            if min_radius <= avg_radius <= max_radius:
                x, y = int(center[0]), int(center[1])
                w, h = int(width), int(height)
                # Store with score for sorting (prefer horizontal)
                propellers.append((x, y, w, h, angle, elongation_score, is_horizontal))
        except cv2.error:
            continue
    
    # Remove duplicates (propellers detected by multiple methods)
    # Group by proximity
    if len(propellers) > 1:
        unique_propellers = []
        used = [False] * len(propellers)
        
        for i, prop1 in enumerate(propellers):
            if used[i]:
                continue
            
            x1, y1, w1, h1, a1 = prop1[:5]
            group = [prop1]
            used[i] = True
            
            for j, prop2 in enumerate(propellers):
                if used[j] or i == j:
                    continue
                
                x2, y2, w2, h2, a2 = prop2[:5]
                # Check if centers are close
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if distance < max_radius:
                    group.append(prop2)
                    used[j] = True
            
            # Use the best detection in the group (prefer horizontal, then largest)
            if len(group) > 1:
                # Sort by: horizontal first, then by size
                best = max(
                    group,
                    key=lambda p: (
                        p[6] if len(p) > 6 else False,  # is_horizontal
                        p[5] if len(p) > 5 else 1.0,  # elongation_score
                        p[2] * p[3],  # area
                    ),
                )
                unique_propellers.append(best)
            else:
                unique_propellers.append(group[0])
        
        propellers = unique_propellers
    
    # Sort by preference: horizontal ellipses first, then by size
    propellers.sort(
        key=lambda p: (
            not (p[6] if len(p) > 6 else False),  # horizontal first
            -(p[5] if len(p) > 5 else 1.0),  # higher elongation score
            -(p[2] * p[3]),  # larger area
        )
    )
    
    # Limit to maximum 4 propellers (drones have 4 propellers)
    propellers = propellers[:4]
    
    # Return without the scoring fields
    return [(x, y, w, h, angle) for x, y, w, h, angle, *_ in propellers]


@dataclass
class TrackedPropeller:
    """Represents a tracked propeller with identity across frames."""
    id: int  # Propeller ID (0-3 for P1-P4)
    x: float
    y: float
    w: float
    h: float
    angle: float
    confidence: float  # How confident we are this is the correct match
    age: int  # Number of frames this propeller has been tracked
    velocity_x: float = 0.0  # Estimated velocity
    velocity_y: float = 0.0


class PropellerTracker:
    """Tracks 4 propellers per drone, maintaining identity across frames."""
    
    def __init__(self, max_distance: float = 100.0, max_age: int = 5):
        """Initialize propeller tracker.
        
        Args:
            max_distance: Maximum pixel distance for matching propellers
            max_age: Maximum frames a propeller can be missing before being removed
        """
        self.max_distance = max_distance
        self.max_age = max_age
        # Dictionary mapping drone bbox to list of 4 TrackedPropeller objects
        self.tracked_propellers: dict[tuple[int, int, int, int], list[Optional[TrackedPropeller]]] = {}
    
    def update(
        self,
        drone_bbox: tuple[int, int, int, int],
        detected_propellers: list[tuple[int, int, int, int, float]],
    ) -> list[tuple[int, int, int, int, float, int]]:
        """Update tracked propellers for a drone.
        
        Args:
            drone_bbox: Drone bounding box (x, y, w, h)
            detected_propellers: List of detected propellers (x, y, w, h, angle)
        
        Returns:
            List of tracked propellers (x, y, w, h, angle, id) - always 4 propellers
        """
        # Initialize if this is a new drone
        if drone_bbox not in self.tracked_propellers:
            self.tracked_propellers[drone_bbox] = [None] * 4
        
        tracked = self.tracked_propellers[drone_bbox]
        
        # Convert detected propellers to centers for matching
        # Input format: (x, y, w, h, angle) where x,y is top-left corner
        detected_centers = [
            ((x + w // 2, y + h // 2), (x, y, w, h, angle))
            for x, y, w, h, angle in detected_propellers
        ]
        
        # Match detected propellers to tracked ones
        matched_indices = set()
        updated_tracked = [None] * 4
        
        # First pass: match existing tracked propellers to detections
        for track_idx, old_prop in enumerate(tracked):
            if old_prop is None:
                continue
            
            # Predict position based on velocity
            predicted_x = old_prop.x + old_prop.velocity_x
            predicted_y = old_prop.y + old_prop.velocity_y
            
            best_match = None
            best_distance = float('inf')
            best_det_idx = None
            
            for det_idx, ((cx, cy), (x, y, w, h, angle)) in enumerate(detected_centers):
                if det_idx in matched_indices:
                    continue
                
                # Calculate distance to predicted position
                distance = np.sqrt(
                    (cx - predicted_x) ** 2 + (cy - predicted_y) ** 2
                )
                
                # Also check size similarity
                old_area = old_prop.w * old_prop.h
                new_area = w * h
                area_ratio = min(old_area, new_area) / max(old_area, new_area) if max(old_area, new_area) > 0 else 0
                
                if distance < self.max_distance and distance < best_distance and area_ratio > 0.3:
                    best_match = (x, y, w, h, angle)
                    best_distance = distance
                    best_det_idx = det_idx
            
            if best_match is not None:
                # Update tracked propeller
                x, y, w, h, angle = best_match
                cx, cy = x + w // 2, y + h // 2
                
                # Update velocity (simple exponential moving average)
                alpha = 0.3
                new_velocity_x = alpha * (cx - old_prop.x) + (1 - alpha) * old_prop.velocity_x
                new_velocity_y = alpha * (cy - old_prop.y) + (1 - alpha) * old_prop.velocity_y
                
                updated_tracked[track_idx] = TrackedPropeller(
                    id=old_prop.id,
                    x=float(cx),
                    y=float(cy),
                    w=float(w),
                    h=float(h),
                    angle=angle,
                    confidence=1.0,
                    age=old_prop.age + 1,
                    velocity_x=new_velocity_x,
                    velocity_y=new_velocity_y,
                )
                matched_indices.add(best_det_idx)
            else:
                # Propeller not detected - predict position or mark as missing
                if old_prop.age < self.max_age:
                    # Predict position based on velocity
                    predicted_x = old_prop.x + old_prop.velocity_x
                    predicted_y = old_prop.y + old_prop.velocity_y
                    
                    # Keep tracking with predicted position, but lower confidence
                    updated_tracked[track_idx] = TrackedPropeller(
                        id=old_prop.id,
                        x=predicted_x,
                        y=predicted_y,
                        w=old_prop.w,
                        h=old_prop.h,
                        angle=old_prop.angle,
                        confidence=0.5,  # Lower confidence for predicted
                        age=old_prop.age + 1,
                        velocity_x=old_prop.velocity_x * 0.9,  # Decay velocity
                        velocity_y=old_prop.velocity_y * 0.9,
                    )
        
        # Second pass: assign unmatched detections to empty slots
        for det_idx, ((cx, cy), (x, y, w, h, angle)) in enumerate(detected_centers):
            if det_idx in matched_indices:
                continue
            
            # Find empty slot
            for track_idx in range(4):
                if updated_tracked[track_idx] is None:
                    updated_tracked[track_idx] = TrackedPropeller(
                        id=track_idx,
                        x=float(cx),
                        y=float(cy),
                        w=float(w),
                        h=float(h),
                        angle=angle,
                        confidence=0.8,  # New detection, moderate confidence
                        age=1,
                        velocity_x=0.0,
                        velocity_y=0.0,
                    )
                    break
        
        # Update tracked propellers
        self.tracked_propellers[drone_bbox] = updated_tracked
        
        # Return always 4 propellers (some may be predicted)
        result = []
        for prop in updated_tracked:
            if prop is not None:
                # Convert center back to top-left corner
                x = int(prop.x - prop.w // 2)
                y = int(prop.y - prop.h // 2)
                result.append((x, y, int(prop.w), int(prop.h), prop.angle, prop.id))
            else:
                # No propeller in this slot - use placeholder (will be filtered)
                result.append((0, 0, 0, 0, 0.0, -1))
        
        return result
    
    def remove_drone(self, drone_bbox: tuple[int, int, int, int]) -> None:
        """Remove tracking for a drone that's no longer detected."""
        if drone_bbox in self.tracked_propellers:
            del self.tracked_propellers[drone_bbox]


def filter_propellers_by_temporal_consistency(
    current_propellers: list[tuple[int, int, int, int, float]],
    history: deque,
    max_distance: float = 100.0,
    min_consistency: float = 0.2,
) -> list[tuple[int, int, int, int, float]]:
    """Filter propellers that appear consistently over time.
    
    Propellers spin very fast but their position relative to the drone
    should be relatively stable. Filters out flickering false detections.

    Args:
        current_propellers: Current frame propellers (x, y, w, h, angle)
        history: Deque of previous frame propellers
        max_distance: Maximum pixel distance for matching (larger for fast movement)
        min_consistency: Minimum fraction of frames propeller must appear (lower threshold)

    Returns:
        Filtered list of propellers
    """
    if len(history) == 0:
        return current_propellers

    # Calculate centers and sizes of current propellers
    current_info = [
        ((x + w // 2, y + h // 2), w * h, (x, y, w, h, angle))
        for x, y, w, h, angle in current_propellers
    ]

    # Track consistency for each propeller
    propeller_consistency = []

    for (cx, cy), area, propeller in current_info:
        matches = 0
        total_checks = 0

        # Check against all history frames
        for past_propellers in history:
            total_checks += 1
            found_match = False

            for px, py, pw, ph, pangle in past_propellers:
                past_center = (px + pw // 2, py + ph // 2)
                past_area = pw * ph

                # Calculate distance between centers
                distance = np.sqrt(
                    (cx - past_center[0]) ** 2 + (cy - past_center[1]) ** 2
                )

                # Calculate size similarity
                area_ratio = min(area, past_area) / max(area, past_area) if max(area, past_area) > 0 else 0

                # Match if close enough and similar size
                # Larger max_distance accounts for fast drone/propeller movement
                if distance <= max_distance and area_ratio > 0.3:
                    matches += 1
                    found_match = True
                    break

        # Calculate consistency ratio
        consistency = matches / total_checks if total_checks > 0 else 0
        propeller_consistency.append((consistency, propeller))

    # Return propellers that appear consistently (not flickering)
    # Lower threshold (0.2) since propellers can be occluded or move fast
    filtered = [
        prop
        for consistency, prop in propeller_consistency
        if consistency >= min_consistency
    ]

    return filtered


def detect_propellers_for_drones(
    frame: np.ndarray,
    drone_detections: list[tuple[int, int, int, int]],
    min_radius: int = 5,
    max_radius: int = 50,
    threshold: int = 40,
    propeller_tracker: PropellerTracker | None = None,
) -> dict[tuple[int, int, int, int], list[tuple[int, int, int, int, float, int]]]:
    """Detect and track propellers for each detected drone.
    
    Always returns exactly 4 propellers per drone, maintaining identity across frames.
    Propellers are labeled P1-P4 and tracked using historical positions.

    Args:
        frame: Full normalized event frame
        drone_detections: List of drone bounding boxes (x, y, w, h)
        min_radius: Minimum propeller radius
        max_radius: Maximum propeller radius
        threshold: Threshold for detection
        propeller_tracker: PropellerTracker instance for maintaining identity

    Returns:
        Dictionary mapping drone bbox to list of propellers (x, y, w, h, angle, id) in frame coordinates
        Always 4 propellers per drone (some may be predicted if not detected)
    """
    drone_propellers = {}
    
    if propeller_tracker is None:
        propeller_tracker = PropellerTracker()

    for drone_bbox in drone_detections:
        x, y, w, h = drone_bbox
        
        # Extract ROI with some padding
        padding = 10
        roi_x = max(0, x - padding)
        roi_y = max(0, y - padding)
        roi_w = min(frame.shape[1] - roi_x, w + 2 * padding)
        roi_h = min(frame.shape[0] - roi_y, h + 2 * padding)
        
        roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        
        if roi.size == 0:
            continue
        
        # Detect propellers in ROI
        propellers_roi = detect_propellers_in_roi(
            roi, min_radius, max_radius, threshold
        )
        
        # Convert ROI coordinates to frame coordinates and filter to within drone bbox
        propellers_frame = []
        for px, py, pw, ph, angle in propellers_roi:
            # Convert to frame coordinates
            frame_x = px + roi_x
            frame_y = py + roi_y
            
            # Propeller center must be strictly within the drone bounding box
            # (propellers cannot be outside the detected drone zone)
            propeller_center_x = frame_x
            propeller_center_y = frame_y
            
            # Check if propeller center is within drone bbox (strict check)
            if (x <= propeller_center_x <= x + w and
                y <= propeller_center_y <= y + h):
                propellers_frame.append((frame_x, frame_y, pw, ph, angle))
        
        # Use tracker to maintain identity and ensure exactly 4 propellers
        tracked_propellers = propeller_tracker.update(drone_bbox, propellers_frame)
        
        # Filter out invalid propellers (id == -1) and ensure they're within bbox
        valid_propellers = []
        for prop_x, prop_y, prop_w, prop_h, prop_angle, prop_id in tracked_propellers:
            if prop_id == -1:
                continue
            
            # Double-check propeller is within drone bbox
            prop_center_x = prop_x + prop_w // 2
            prop_center_y = prop_y + prop_h // 2
            if (x <= prop_center_x <= x + w and
                y <= prop_center_y <= y + h):
                valid_propellers.append((prop_x, prop_y, prop_w, prop_h, prop_angle, prop_id))
        
        # Always return 4 propellers (fill with predicted if needed)
        # If we have fewer than 4 valid, keep the tracked ones (some may be predicted)
        if len(valid_propellers) < 4:
            # Use tracked propellers even if predicted (they maintain identity)
            valid_propellers = tracked_propellers[:4]
            # Filter out invalid ones
            valid_propellers = [p for p in valid_propellers if p[5] != -1]
        
        drone_propellers[drone_bbox] = valid_propellers[:4]  # Ensure max 4
    
    return drone_propellers


def find_four_propeller_pattern(
    circles: list[tuple[int, int, int]],
    frame_shape: tuple[int, int],
    max_propeller_distance: float = 150.0,
    min_propeller_distance: float = 30.0,
) -> list[tuple[int, int, int, int]]:
    """Find groups of circles arranged in a 4-fold symmetric pattern (drone propellers).
    Works at any rotation angle.

    Args:
        circles: List of detected circles (x, y, radius)
        frame_shape: (height, width) of the frame
        max_propeller_distance: Maximum distance from center to propeller
        min_propeller_distance: Minimum distance between propellers

    Returns:
        List of bounding boxes (x, y, w, h) for detected drones
    """
    if len(circles) < 2:
        return []

    detections = []
    circles_array = np.array([(x, y) for x, y, _ in circles])

    # Try to find groups of circles that form a symmetric pattern
    # Check for 4-fold symmetry at any angle (rotation-invariant)
    for i in range(len(circles)):
        center_x, center_y, center_r = circles[i]

        # Find circles that could be propellers around this center
        distances = np.sqrt(
            (circles_array[:, 0] - center_x) ** 2
            + (circles_array[:, 1] - center_y) ** 2
        )

        # Find circles at roughly equal distances (propellers)
        valid_indices = np.where(
            (distances >= min_propeller_distance)
            & (distances <= max_propeller_distance)
        )[0]

        if len(valid_indices) >= 2:  # Need at least 2 propellers visible
            # Get angles of propellers relative to center
            angles = []
            dists = []
            for idx in valid_indices:
                dx = circles_array[idx, 0] - center_x
                dy = circles_array[idx, 1] - center_y
                angle = np.arctan2(dy, dx)
                angles.append(angle)
                dists.append(distances[idx])

            # Check for 4-fold symmetry (rotation-invariant)
            # Look for angles that are roughly 90 degrees apart (at any rotation)
            angles_sorted = sorted(angles)
            n = len(angles_sorted)

            # Calculate all pairwise angle differences
            angle_diffs = []
            for j in range(n):
                for k in range(j + 1, n):
                    diff = abs(angles_sorted[j] - angles_sorted[k])
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    angle_diffs.append(diff)

            # Check for 90-degree spacing (4-fold symmetry)
            # Allow tolerance: 70-110 degrees
            ninety_deg_matches = sum(
                1
                for diff in angle_diffs
                if np.pi * 0.39 <= diff <= np.pi * 0.61  # ~90 degrees
            )

            # Check for 180-degree spacing (opposite pairs)
            opposite_matches = sum(
                1
                for diff in angle_diffs
                if np.pi * 0.83 <= diff <= np.pi * 1.17  # ~180 degrees
            )

            # Check for roughly equal distances (propellers should be equidistant)
            if len(dists) >= 2:
                dist_std = np.std(dists)
                dist_mean = np.mean(dists)
                dist_cv = dist_std / dist_mean if dist_mean > 0 else 1.0
                # Coefficient of variation should be low (< 0.3) for equidistant propellers
                is_equidistant = dist_cv < 0.3
            else:
                is_equidistant = True

            # Accept if we have:
            # - At least 2 ninety-degree pairs (suggesting 4-fold symmetry), OR
            # - At least 1 opposite pair AND good distance consistency, OR
            # - 3+ propellers with reasonable spacing
            if (
                ninety_deg_matches >= 2
                or (opposite_matches >= 1 and is_equidistant)
                or (n >= 3 and is_equidistant)
            ):
                # Create bounding box around the drone
                all_x = [center_x] + [
                    circles_array[idx, 0] for idx in valid_indices
                ]
                all_y = [center_y] + [
                    circles_array[idx, 1] for idx in valid_indices
                ]
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)

                # Add padding
                padding = int(max_propeller_distance * 0.5)
                x = max(0, int(x_min - padding))
                y = max(0, int(y_min - padding))
                w = min(
                    frame_shape[1] - x, int(x_max - x_min + 2 * padding)
                )
                h = min(
                    frame_shape[0] - y, int(y_max - y_min + 2 * padding)
                )

                if w > 0 and h > 0:
                    detections.append((x, y, w, h))

    return detections


def detect_drone_by_motion_pattern(
    frame: np.ndarray,
    min_area: int = 500,
    max_area: int = 50000,
    threshold: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Alternative detection method: look for compact motion regions with
    multiple activity centers (propellers). This is the primary method for
    event data where circular patterns may not be clearly visible.

    Args:
        frame: Normalized event accumulation frame
        min_area: Minimum detection area
        max_area: Maximum detection area
        threshold: Threshold for binary conversion

    Returns:
        List of bounding boxes
    """
    # Use adaptive thresholding to handle varying intensities
    binary = cv2.adaptiveThreshold(
        frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Also try simple thresholding as fallback
    _, binary_simple = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_or(binary, binary_simple)

    # Apply morphological operations to connect nearby regions
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        # Filter out very elongated shapes (planes) - be more lenient
        if aspect_ratio > 3.0 or aspect_ratio < 1 / 3.0:
            continue

        # Check for multiple activity centers (propellers)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        # Use a lower threshold to find activity regions
        _, thresh_roi = cv2.threshold(roi, threshold * 0.6, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh_roi, connectivity=8
        )

        # Count significant components (propellers + body)
        significant_components = sum(
            1
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > 15  # Lower threshold
        )

        # Check if components are roughly symmetrically distributed (4 propellers)
        if significant_components >= 2:
            # Calculate center of mass of all components
            if len(centroids) > 1:
                component_centers = centroids[1:]  # Skip background
                center_x, center_y = w / 2, h / 2

                # Check if components are distributed around center
                # (characteristic of drone with propellers)
                distances_from_center = [
                    np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                    for cx, cy in component_centers
                ]

                # If we have multiple components at similar distances from center,
                # it suggests a symmetric pattern (drone)
                if len(distances_from_center) >= 2:
                    dist_std = np.std(distances_from_center)
                    dist_mean = np.mean(distances_from_center)
                    if dist_mean > 0:
                        dist_cv = dist_std / dist_mean
                        # Accept if components are roughly equidistant (symmetric)
                        if dist_cv < 0.5 or significant_components >= 3:
                            detections.append((x, y, w, h))
                    else:
                        detections.append((x, y, w, h))
                else:
                    detections.append((x, y, w, h))

    return detections


def merge_nearby_detections(
    detections: list[tuple[int, int, int, int]],
    max_distance: float = 100.0,
    min_overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """Merge nearby detections that are likely the same object.
    
    When a drone rotates, it may be detected as multiple separate blobs.
    This function merges overlapping or nearby detections.

    Args:
        detections: List of bounding boxes (x, y, w, h)
        max_distance: Maximum distance between centers to merge (pixels)
        min_overlap: Minimum overlap ratio to merge (0-1)

    Returns:
        Merged list of detections
    """
    if len(detections) <= 1:
        return detections

    # Calculate centers and areas
    detections_info = [
        ((x + w // 2, y + h // 2), (x, y, w, h), w * h)
        for x, y, w, h in detections
    ]

    merged = []
    used = [False] * len(detections_info)

    for i, ((cx1, cy1), (x1, y1, w1, h1), area1) in enumerate(detections_info):
        if used[i]:
            continue

        # Start with current detection
        group = [(x1, y1, w1, h1)]
        used[i] = True

        # Find nearby detections to merge
        for j, ((cx2, cy2), (x2, y2, w2, h2), area2) in enumerate(detections_info):
            if used[j] or i == j:
                continue

            # Calculate distance between centers
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            min_area = min(area1, area2)
            overlap_ratio = overlap_area / min_area if min_area > 0 else 0

            # Merge if close enough or overlapping significantly
            if distance <= max_distance or overlap_ratio >= min_overlap:
                group.append((x2, y2, w2, h2))
                used[j] = True

        # Merge the group into a single bounding box
        if len(group) > 1:
            all_x = [x for x, _, _, _ in group]
            all_y = [y for _, y, _, _ in group]
            all_x2 = [x + w for x, _, w, _ in group]
            all_y2 = [y + h for _, y, _, h in group]

            x_min = min(all_x)
            y_min = min(all_y)
            x_max = max(all_x2)
            y_max = max(all_y2)

            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        else:
            merged.append(group[0])

    return merged


def filter_drone_detections(
    detections: list[tuple[int, int, int, int]],
    frame: np.ndarray,
    min_area: int = 500,
    max_area: int = 50000,
    max_aspect_ratio: float = 2.5,
    min_compactness: float = 0.15,
) -> list[tuple[int, int, int, int]]:
    """Filter detections to remove planes, trees, and other false positives.
    More lenient to avoid filtering out actual drones.

    Args:
        detections: List of bounding boxes
        frame: Event accumulation frame
        min_area: Minimum detection area
        max_area: Maximum detection area
        max_aspect_ratio: Maximum width/height ratio (filters elongated planes)
        min_compactness: Minimum compactness to filter very sparse regions

    Returns:
        Filtered list of detections
    """
    filtered = []

    for x, y, w, h in detections:
        area = w * h
        if not (min_area <= area <= max_area):
            continue

        aspect_ratio = w / h if h > 0 else 0
        # No plane filtering - removed per user request
        # Keep aspect ratio check only for very extreme cases to avoid errors
        if aspect_ratio > 10.0 or aspect_ratio < 1 / 10.0:
            continue  # Only filter extremely distorted shapes

        # Check compactness (how filled the bounding box is)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        # Count non-zero pixels in ROI (use lower threshold)
        non_zero = np.count_nonzero(roi > 20)  # Lower threshold
        compactness = non_zero / area if area > 0 else 0

        # Be more lenient with compactness - event data can be sparse
        if compactness < min_compactness:
            continue  # Too sparse (likely trees/background)

        # Additional check: filter out very large, sparse regions (likely trees)
        if area > 15000 and compactness < 0.3:
            continue
        
        # Simplified tree filtering - only filter very large sparse regions
        # Removed strict aspect ratio checks per user request

        # Accept the detection
        filtered.append((x, y, w, h))

    return filtered


def filter_detections_by_temporal_consistency(
    detections: list[tuple[int, int, int, int]],
    history: deque,
    max_distance: float = 150.0,
    min_consistency: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    """Filter detections that appear consistently over time.
    
    Drones move smoothly and consistently, while tree leaves flicker in/out.
    This filters out flickering detections (trees) and keeps stable ones (drones).
    
    Key insight: Drones appear in most frames (high consistency), while tree
    leaves appear sporadically (low consistency).

    Args:
        detections: Current frame detections
        history: Deque of previous frame detections (most recent first)
        max_distance: Maximum pixel distance for matching (allows for movement)
        min_consistency: Minimum fraction of recent frames detection must appear (0-1)

    Returns:
        Filtered list of detections (only stable ones)
    """
    if len(history) == 0:
        return detections

    # Calculate centers and areas of current detections
    current_info = [
        ((x + w // 2, y + h // 2), (x, y, w, h), w * h) for x, y, w, h in detections
    ]

    # Track consistency for each detection
    detection_consistency = []
    
    for (cx, cy), (x, y, w, h), area in current_info:
        matches = 0
        total_checks = 0
        
        # Check against all history frames (most recent first)
        for past_detections in history:
            total_checks += 1
            found_match = False
            
            for px, py, pw, ph in past_detections:
                past_center = (px + pw // 2, py + ph // 2)
                past_area = pw * ph
                
                # Calculate distance between centers
                distance = np.sqrt(
                    (cx - past_center[0]) ** 2 + (cy - past_center[1]) ** 2
                )
                
                # Calculate size similarity
                area_ratio = min(area, past_area) / max(area, past_area) if max(area, past_area) > 0 else 0
                
                # Match if close enough and similar size
                # Allow some movement and size variation
                if distance <= max_distance and area_ratio > 0.4:
                    matches += 1
                    found_match = True
                    break
        
        # Calculate consistency ratio (how often it appears)
        consistency = matches / total_checks if total_checks > 0 else 0
        detection_consistency.append(consistency)

    # Return detections that appear consistently (not flickering)
    # Lower threshold (0.4) means detection must appear in at least 40% of frames
    filtered = [
        detections[i]
        for i, consistency in enumerate(detection_consistency)
        if consistency >= min_consistency
    ]

    return filtered


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int]],
    circles: list[tuple[int, int, int]] | None = None,
    drone_propellers: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    circle_color: tuple[int, int, int] = (255, 0, 0),
    propeller_color: tuple[int, int, int] = (0, 165, 255),  # Orange
) -> np.ndarray:
    """Draw bounding boxes, propeller circles, and detected propellers on frame."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw detected propeller circles (for debugging)
    if circles is not None:
        for x, y, r in circles:
            cv2.circle(frame_bgr, (x, y), r, circle_color, 1)

    # Draw drone bounding boxes and propellers
    for drone_bbox in detections:
        x, y, w, h = drone_bbox
        
        # Draw drone bounding box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(frame_bgr, (center_x, center_y), 5, color, -1)
        
        # Label as drone
        cv2.putText(
            frame_bgr,
            "DRONE",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        
        # Draw detected propellers for this drone
        if drone_propellers is not None and drone_bbox in drone_propellers:
            propellers = drone_propellers[drone_bbox]
            for propeller in propellers:
                # Handle both old format (5 elements) and new format (6 elements with id)
                if len(propeller) >= 6:
                    px, py, pw, ph, angle, prop_id = propeller[:6]
                    # Label as P1, P2, P3, P4
                    label = f"P{prop_id + 1}"
                elif len(propeller) == 5:
                    px, py, pw, ph, angle = propeller
                    label = "P"
                else:
                    # Backward compatibility
                    px, py, pw, ph = propeller[:4]
                    angle = 0.0
                    label = "P"
                
                # Skip if invalid (w or h is 0)
                if pw == 0 or ph == 0:
                    continue
                
                # Draw ellipse (propellers are typically elliptical)
                center = (int(px + pw // 2), int(py + ph // 2))
                axes = (int(pw // 2), int(ph // 2))
                cv2.ellipse(
                    frame_bgr,
                    center,
                    axes,
                    angle,  # Use the angle from ellipse fitting
                    0,  # start angle
                    360,  # end angle
                    propeller_color,
                    2,  # thickness
                )
                # Draw center point
                cv2.circle(frame_bgr, center, 3, propeller_color, -1)
                
                # Label propeller with ID (P1, P2, P3, P4)
                cv2.putText(
                    frame_bgr,
                    label,
                    (int(px + pw // 2) - 8, int(py + ph // 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    propeller_color,
                    2,
                    cv2.LINE_AA,
                )

    return frame_bgr


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    num_detections: int,
    num_propellers: int = 0,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Overlay timing info, detection count, and propeller count."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    first_row_str = f"Detections: {num_detections} | Propellers: {num_propellers} | speed={pacer.speed:.2f}x"
    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect drones in event camera .dat files"
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
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum detection area in pixels (default: 200)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=50000,
        help="Maximum detection area in pixels (default: 50000)",
    )
    parser.add_argument(
        "--show-circles",
        action="store_true",
        help="Show detected propeller circles (for debugging)",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=5,
        help="Number of frames to use for temporal filtering (default: 5)",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.4,
        help="Minimum consistency ratio for temporal filtering (0-1, default: 0.4). "
        "Higher values filter more aggressively (removes flickering trees).",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=80.0,
        help="Maximum distance to merge nearby detections (pixels, default: 80)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Event accumulation threshold (default: 30)",
    )
    parser.add_argument(
        "--propeller-min-radius",
        type=int,
        default=5,
        help="Minimum propeller radius in pixels (default: 5)",
    )
    parser.add_argument(
        "--propeller-max-radius",
        type=int,
        default=50,
        help="Maximum propeller radius in pixels (default: 50)",
    )
    parser.add_argument(
        "--show-propeller-debug",
        action="store_true",
        help="Show propeller detection debug visualization",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    parser.add_argument(
        "--min-propeller-distance",
        type=float,
        default=30.0,
        help="Minimum distance between propellers in pixels (default: 30)",
    )
    parser.add_argument(
        "--max-propeller-distance",
        type=float,
        default=150.0,
        help="Maximum distance from center to propeller in pixels (default: 150)",
    )
    parser.add_argument(
        "--detect-propellers",
        action="store_true",
        help="Enable propeller detection and tracking (disabled by default)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video file path (e.g., output.mp4). If specified, saves detection video instead of displaying.",
    )
    args = parser.parse_args()

    # Initialize data source
    src = DatFileSource(
        args.dat,
        width=1280,
        height=720,
        window_length_us=args.window * 1000,
    )

    # Initialize pacer for playback
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Detection history for temporal filtering
    detection_history: deque = deque(maxlen=args.history_frames)
    
    # Propeller tracker for maintaining identity across frames (only if enabled)
    propeller_tracker = PropellerTracker(max_distance=100.0, max_age=5) if args.detect_propellers else None

    # Initialize video writer if output path is specified
    video_writer = None
    if args.output_video:
        # Calculate frame rate based on window duration
        fps = 1000.0 / args.window  # frames per second
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.output_video,
            fourcc,
            fps,
            (src.width, src.height),
        )
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {args.output_video}")
            return
        print(f"Writing video to {args.output_video} at {fps:.2f} fps")
    else:
        cv2.namedWindow("Drone Detection", cv2.WINDOW_NORMAL)

    print(f"Processing {args.dat}")
    print(f"Window: {args.window}ms, Speed: {args.speed}x")
    if args.output_video:
        print("Writing video file...")
    else:
        print("Press 'q' or ESC to quit")

    for batch_range in pacer.pace(src.ranges()):
        # Extract events in current window
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        x_coords, y_coords, polarities = window

        # Accumulate events into frame (using absolute motion for detection)
        event_frame = accumulate_events(
            x_coords, y_coords, polarities, src.width, src.height, use_polarity=False
        )

        # Normalize for visualization
        frame = normalize_frame(event_frame)

        # Simple blob detection - find all motion regions
        _, binary = cv2.threshold(frame, args.threshold, 255, cv2.THRESH_BINARY)
        
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
            if not (args.min_area <= area <= args.max_area):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # No plane filtering - removed per user request
            
            # Check compactness and spatial distribution
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            
            non_zero = np.count_nonzero(roi > args.threshold * 0.5)
            compactness = non_zero / area if area > 0 else 0
            
            # Filter very sparse regions (trees/background)
            if compactness < 0.12:
                continue
            
            # Filter very large sparse regions (trees)
            if area > 20000 and compactness < 0.25:
                continue
            
            # Check for irregular distribution (trees have irregular patterns)
            # Drones have more regular, symmetric patterns
            if area > 1000:
                # Divide ROI into grid and check distribution
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
                            cell_activity = np.count_nonzero(cell_roi > args.threshold * 0.5)
                            cell_activities.append(cell_activity)
                    
                    # Check variance - trees have high variance (irregular)
                    # Drones have lower variance (more uniform)
                    if len(cell_activities) > 0:
                        activity_std = np.std(cell_activities)
                        activity_mean = np.mean(cell_activities)
                        if activity_mean > 0:
                            cv = activity_std / activity_mean
                            # Filter very irregular patterns (likely trees)
                            if cv > 1.2:  # High coefficient of variation
                                continue
            
            all_detections.append((x, y, w, h))
        
        # Merge nearby detections (fixes issue where rotating drone is detected as multiple)
        merged_detections = merge_nearby_detections(
            all_detections,
            max_distance=args.merge_distance,
            min_overlap=0.15,  # Minimum overlap to merge
        )

        # Apply additional filtering to remove false positives
        filtered_detections = filter_drone_detections(
            merged_detections,
            frame,
            min_area=args.min_area,
            max_area=args.max_area,
        )
        
        # Debug output
        if args.debug and len(all_detections) > 0:
            print(
                f"Frame: {len(all_detections)} candidates, "
                f"{len(merged_detections)} after merging, "
                f"{len(filtered_detections)} after filtering"
            )
            for i, (x, y, w, h) in enumerate(all_detections):
                area = w * h
                aspect = w / h if h > 0 else 0
                roi = frame[y : y + h, x : x + w]
                compact = (
                    np.count_nonzero(roi > args.threshold * 0.5) / area
                    if area > 0
                    else 0
                )
                print(
                    f"  Candidate {i}: area={area}, aspect={aspect:.2f}, "
                    f"compact={compact:.3f}"
                )

        # Apply temporal filtering to remove flickering detections (trees)
        # Drones appear consistently, tree leaves flicker in/out
        if len(detection_history) >= 3:  # Need at least 3 frames of history
            filtered_detections = filter_detections_by_temporal_consistency(
                filtered_detections,
                detection_history,
                max_distance=150.0,  # Allow some movement
                min_consistency=args.min_consistency,  # Must appear in X% of frames
            )

        # Update history
        detection_history.append(filtered_detections)

        # Detect and track propellers for each drone (only if enabled)
        drone_propellers = {}
        if args.detect_propellers:
            drone_propellers = detect_propellers_for_drones(
                frame,
                filtered_detections,
                min_radius=args.propeller_min_radius,
                max_radius=args.propeller_max_radius,
                threshold=args.threshold,
                propeller_tracker=propeller_tracker,
            )
            
            # Clean up tracker for drones that are no longer detected
            if propeller_tracker is not None:
                active_drones = set(filtered_detections)
                tracked_drones = set(propeller_tracker.tracked_propellers.keys())
                drones_to_remove = tracked_drones - active_drones
                for bbox in drones_to_remove:
                    propeller_tracker.remove_drone(bbox)

        # Debug: print propeller detection info
        if args.debug:
            for drone_bbox, props in drone_propellers.items():
                print(f"Drone at {drone_bbox}: {len(props)} propellers tracked")
                for prop in props:
                    if len(prop) >= 6:
                        px, py, pw, ph, angle, prop_id = prop[:6]
                        print(f"  P{prop_id + 1}: center=({px + pw // 2}, {py + ph // 2}), size=({pw}, {ph}), angle={angle:.1f}°")
                    else:
                        px, py, pw, ph, angle = prop[:5]
                        print(f"  Propeller: center=({px + pw // 2}, {py + ph // 2}), size=({pw}, {ph}), angle={angle:.1f}°")

        # Draw detections with propellers
        display_frame = draw_detections(
            frame, filtered_detections, None, drone_propellers
        )
        
        # Debug visualization: show ROI and binary for propeller detection (only if enabled)
        if args.detect_propellers and args.show_propeller_debug and len(filtered_detections) > 0:
            # Show the first drone's ROI for debugging
            drone_bbox = filtered_detections[0]
            x, y, w, h = drone_bbox
            padding = 10
            roi_x = max(0, x - padding)
            roi_y = max(0, y - padding)
            roi_w = min(frame.shape[1] - roi_x, w + 2 * padding)
            roi_h = min(frame.shape[0] - roi_y, h + 2 * padding)
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            
            if roi.size > 0:
                # Create binary for visualization
                _, binary_vis = cv2.threshold(roi, args.threshold * 0.7, 255, cv2.THRESH_BINARY)
                binary_vis = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
                
                # Draw detected propellers on debug view
                if drone_bbox in drone_propellers:
                    for prop in drone_propellers[drone_bbox]:
                        # Handle both old and new format
                        if len(prop) >= 6:
                            px, py, pw, ph, angle, prop_id = prop[:6]
                        else:
                            px, py, pw, ph, angle = prop[:5]
                        
                        # Skip if invalid
                        if pw == 0 or ph == 0:
                            continue
                        
                        # Convert to ROI coordinates
                        center_x = px + pw // 2
                        center_y = py + ph // 2
                        px_roi = center_x - roi_x
                        py_roi = center_y - roi_y
                        center = (px_roi, py_roi)
                        axes = (pw // 2, ph // 2)
                        cv2.ellipse(
                            binary_vis,
                            center,
                            axes,
                            angle,
                            0,
                            360,
                            (0, 255, 255),  # Yellow for debug
                            2,
                        )
                
                # Resize for display
                debug_h = 200
                debug_w = int(roi_w * debug_h / roi_h)
                binary_vis_resized = cv2.resize(binary_vis, (debug_w, debug_h))
                
                # Place in top-right corner
                corner_y = 60
                corner_x = display_frame.shape[1] - debug_w - 10
                if corner_x > 0 and corner_y + debug_h < display_frame.shape[0]:
                    display_frame[corner_y:corner_y + debug_h, corner_x:corner_x + debug_w] = binary_vis_resized
                    cv2.putText(
                        display_frame,
                        "Propeller ROI",
                        (corner_x, corner_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

        # Count total propellers (should be 4 per drone, but count actual valid ones)
        total_propellers = sum(
            len([p for p in props if len(p) >= 6 and p[5] != -1]) 
            for props in drone_propellers.values()
        )

        # Draw HUD
        draw_hud(
            display_frame,
            pacer,
            batch_range,
            len(filtered_detections),
            total_propellers,
        )

        # Write to video or display
        if video_writer is not None:
            video_writer.write(display_frame)
        else:
            cv2.imshow("Drone Detection", display_frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {args.output_video}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

