"""Drone detection from event camera data.

This script processes .dat event files to detect drones by:
1. Accumulating events over time windows
2. Detecting high-density regions (density lumps) - regions with density >= 0.4
   are automatically considered drones
3. Detecting motion regions using blob detection and filtering by:
   - Density-based filtering (intensity >= 0.4 of max)
   - Spatial regularity (drones have uniform activity distribution)
   - Area constraints
4. Filtering out false positives (trees, background) by:
   - Density threshold (sparse regions filtered out)
   - Spatial regularity (irregular patterns like trees filtered out)
   - Temporal consistency (flickering objects filtered out)
5. Temporal consistency filtering for stability (drones appear consistently)
6. Optional propeller detection and tracking (--detect-propellers flag)
7. Visualizing detections with bounding boxes, density lumps, and trajectories
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

# Constants
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_MIN_COMPACTNESS = 0.4
DEFAULT_LARGE_AREA_THRESHOLD = 20000
DEFAULT_LARGE_AREA_COMPACTNESS = 0.25
DEFAULT_GRID_SIZE = 4
DEFAULT_CV_THRESHOLD = 1.2
DEFAULT_MIN_OVERLAP_MERGE = 0.15
DEFAULT_MIN_TEMPORAL_HISTORY = 2
DEFAULT_MAX_TRAIL_DISTANCE = 200.0
DEFAULT_PADDING = 10


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


class DroneTrajectoryTracker:
    """Tracks drone trajectories to draw movement trails.
    
    Maintains identity of drones across frames and stores their movement history
    for visualization purposes. Handles occlusion and disappearing/reappearing drones
    by maintaining trajectories for a grace period (max_age frames).
    """
    
    def __init__(
        self,
        max_distance: float = 150.0,
        max_trail_length: int = 50,
        max_age: int = 10,
    ) -> None:
        """Initialize trajectory tracker.
        
        Args:
            max_distance: Maximum pixel distance for matching drones between frames
            max_trail_length: Maximum number of positions to store in trail
            max_age: Maximum frames a drone can be missing before removing trajectory
        """
        self.max_distance = max_distance
        self.max_trail_length = max_trail_length
        self.max_age = max_age
        # Dictionary mapping drone ID to deque of (center_x, center_y) positions
        self.trajectories: dict[int, deque] = {}
        # Dictionary mapping drone ID to number of frames since last seen
        self.trajectory_age: dict[int, int] = {}
        # Dictionary mapping drone ID to estimated velocity (vx, vy)
        self.trajectory_velocity: dict[int, tuple[float, float]] = {}
        self.next_id = 0
    
    def update(
        self,
        detections: list[tuple[int, int, int, int]],
    ) -> dict[tuple[int, int, int, int], int]:
        """Update trajectories with new detections.
        
        Handles disappearing/reappearing drones by maintaining trajectories for
        max_age frames even when not detected, using velocity prediction.
        
        Args:
            detections: List of drone bounding boxes (x, y, w, h)
        
        Returns:
            Dictionary mapping drone bbox to trajectory ID
        """
        # Calculate centers for current detections
        current_centers = [
            ((x + w // 2, y + h // 2), (x, y, w, h))
            for x, y, w, h in detections
        ]
        
        # Match current detections to existing trajectories
        matched_ids = set()
        bbox_to_id = {}
        
        # First pass: match existing trajectories (including predicted positions)
        for traj_id, trajectory in self.trajectories.items():
            if len(trajectory) == 0:
                continue
            
            # Get last known position
            last_x, last_y = trajectory[-1]
            
            # Predict current position based on velocity if drone was missing
            if traj_id in self.trajectory_velocity:
                vx, vy = self.trajectory_velocity[traj_id]
                age = self.trajectory_age.get(traj_id, 0)
                # Predict position: last position + velocity * frames_missing
                predicted_x = last_x + vx * age
                predicted_y = last_y + vy * age
            else:
                predicted_x, predicted_y = last_x, last_y
            
            best_match = None
            best_distance = float('inf')
            best_bbox = None
            
            for (cx, cy), bbox in current_centers:
                if bbox in bbox_to_id:  # Already matched
                    continue
                
                # Check distance to predicted position (allows for movement during occlusion)
                distance = np.sqrt((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2)
                # Increase max_distance for older trajectories (drone may have moved more)
                effective_max_distance = self.max_distance * (1 + self.trajectory_age.get(traj_id, 0) * 0.2)
                
                if distance < effective_max_distance and distance < best_distance:
                    best_match = (cx, cy)
                    best_distance = distance
                    best_bbox = bbox
            
            if best_match is not None:
                # Update trajectory
                trajectory.append(best_match)
                if len(trajectory) > self.max_trail_length:
                    trajectory.popleft()
                bbox_to_id[best_bbox] = traj_id
                matched_ids.add(traj_id)
                
                # Update velocity estimate (exponential moving average)
                if len(trajectory) >= 2:
                    prev_x, prev_y = trajectory[-2]
                    new_vx = (best_match[0] - prev_x) * 0.3 + self.trajectory_velocity.get(traj_id, (0.0, 0.0))[0] * 0.7
                    new_vy = (best_match[1] - prev_y) * 0.3 + self.trajectory_velocity.get(traj_id, (0.0, 0.0))[1] * 0.7
                    self.trajectory_velocity[traj_id] = (new_vx, new_vy)
                
                # Reset age (drone was found)
                self.trajectory_age[traj_id] = 0
            else:
                # Drone not detected - increment age
                self.trajectory_age[traj_id] = self.trajectory_age.get(traj_id, 0) + 1
        
        # Second pass: create new trajectories for unmatched detections
        # (only if they're not near any existing trajectory's predicted position)
        for (cx, cy), bbox in current_centers:
            if bbox not in bbox_to_id:
                # Check if this detection is near any existing trajectory's predicted position
                is_near_existing = False
                for traj_id, trajectory in self.trajectories.items():
                    if traj_id in matched_ids:  # Already matched
                        continue
                    if len(trajectory) == 0:
                        continue
                    
                    last_x, last_y = trajectory[-1]
                    if traj_id in self.trajectory_velocity:
                        vx, vy = self.trajectory_velocity[traj_id]
                        age = self.trajectory_age.get(traj_id, 0)
                        predicted_x = last_x + vx * age
                        predicted_y = last_y + vy * age
                    else:
                        predicted_x, predicted_y = last_x, last_y
                    
                    distance = np.sqrt((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2)
                    effective_max_distance = self.max_distance * (1 + self.trajectory_age.get(traj_id, 0) * 0.2)
                    
                    if distance < effective_max_distance:
                        # Re-associate with existing trajectory
                        trajectory.append((cx, cy))
                        if len(trajectory) > self.max_trail_length:
                            trajectory.popleft()
                        bbox_to_id[bbox] = traj_id
                        matched_ids.add(traj_id)
                        self.trajectory_age[traj_id] = 0
                        is_near_existing = True
                        break
                
                if not is_near_existing:
                    # Create new trajectory
                    traj_id = self.next_id
                    self.next_id += 1
                    self.trajectories[traj_id] = deque([(cx, cy)], maxlen=self.max_trail_length)
                    self.trajectory_age[traj_id] = 0
                    self.trajectory_velocity[traj_id] = (0.0, 0.0)
                    bbox_to_id[bbox] = traj_id
        
        # Remove trajectories that are too old (drones disappeared permanently)
        ids_to_remove = [
            tid for tid, age in self.trajectory_age.items()
            if age > self.max_age
        ]
        for tid in ids_to_remove:
            del self.trajectories[tid]
            del self.trajectory_age[tid]
            if tid in self.trajectory_velocity:
                del self.trajectory_velocity[tid]
        
        return bbox_to_id
    
    def get_trajectory(self, traj_id: int) -> list[tuple[int, int]]:
        """Get trajectory points for a given ID."""
        if traj_id in self.trajectories:
            return list(self.trajectories[traj_id])
        return []


class PropellerTracker:
    """Tracks 4 propellers per drone, maintaining identity across frames.
    
    Each drone has 4 propellers labeled P1-P4. The tracker maintains propeller
    identity even when propellers are temporarily occluded or not detected.
    """
    
    def __init__(self, max_distance: float = 100.0, max_age: int = 5) -> None:
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
        padding = DEFAULT_PADDING
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
    
    More lenient to avoid filtering out actual drones. Filters based on:
    - Area constraints
    - Aspect ratio (very extreme shapes only)
    - Compactness (sparse regions like trees)

    Args:
        detections: List of bounding boxes (x, y, w, h)
        frame: Event accumulation frame
        min_area: Minimum detection area in pixels
        max_area: Maximum detection area in pixels
        max_aspect_ratio: Maximum width/height ratio (unused, kept for compatibility)
        min_compactness: Minimum compactness to filter very sparse regions

    Returns:
        Filtered list of detections (x, y, w, h)
    """
    filtered = []

    for x, y, w, h in detections:
        area = w * h
        if not (min_area <= area <= max_area):
            continue

        aspect_ratio = w / h if h > 0 else 0
        # Keep aspect ratio check only for very extreme cases to avoid errors
        # (filters out invalid or extremely distorted detections)
        if aspect_ratio > 10.0 or aspect_ratio < 1 / 10.0:
            continue

        # Check compactness (spatial coverage)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        # Calculate compactness (spatial coverage) - fraction of pixels with activity
        non_zero = np.count_nonzero(roi > 20)  # Lower threshold
        compactness = non_zero / area if area > 0 else 0

        # Filter using compactness (spatial coverage) - threshold of 0.4 (40% of pixels active)
        min_compactness_threshold = DEFAULT_MIN_COMPACTNESS  # 0.4
        if compactness < min_compactness_threshold:
            continue  # Too sparse (likely trees/background)

        # Additional check: filter out very large, sparse regions (likely trees)
        if area > 15000 and compactness < DEFAULT_LARGE_AREA_COMPACTNESS:
            continue

        # Accept the detection
        filtered.append((x, y, w, h))

    return filtered


def filter_detections_by_temporal_consistency(
    detections: list[tuple[int, int, int, int]],
    history: deque[list[tuple[int, int, int, int]]],
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


def detect_density_lumps(
    frame: np.ndarray,
    compactness_threshold: float = 0.4,
    min_lump_area: int = 600,
    max_lump_area: int = 10000,
    activity_threshold: int = 20,
) -> list[tuple[int, int, int, int, float]]:
    """Detect regions of high spatial coverage (density lumps).
    
    These are areas with good pixel coverage that may represent:
    - Propellers (circular/elliptical regions with good coverage)
    - Drone body (compact region with good coverage)
    - Other high-activity objects
    
    Args:
        frame: Normalized event accumulation frame (0-255)
        compactness_threshold: Minimum fraction of pixels that must be active (0-1)
        min_lump_area: Minimum area for a density lump
        max_lump_area: Maximum area for a density lump
        activity_threshold: Pixel intensity threshold for considering a pixel active
        
    Returns:
        List of density lumps as (x, y, w, h, compactness_score) where compactness_score
        is the fraction of active pixels in that region (0-1)
    """
    if frame.max() == 0:
        return []
    
    # Create binary mask of active pixels
    _, binary = cv2.threshold(frame, activity_threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to connect nearby active pixels
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours of active regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    density_lumps = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_lump_area <= area <= max_lump_area):
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate compactness (spatial coverage) in this region
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        
        # Calculate compactness score (fraction of active pixels)
        non_zero = np.count_nonzero(roi > activity_threshold)
        compactness_score = float(non_zero / (w * h)) if (w * h) > 0 else 0.0
        
        # Exclude lumps with compactness less than threshold
        if compactness_score < compactness_threshold:
            continue
        
        density_lumps.append((x, y, w, h, compactness_score))
    
    return density_lumps


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int]],
    drone_propellers: dict[tuple[int, int, int, int], list[tuple[int, int, int, int, float, int]]] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    propeller_color: tuple[int, int, int] = (0, 165, 255),  # Orange
    trajectories: dict[tuple[int, int, int, int], list[tuple[int, int]]] | None = None,
    trail_color: tuple[int, int, int] = (0, 255, 0),
        density_lumps: list[tuple[int, int, int, int, float]] | None = None,
        density_color: tuple[int, int, int] = (255, 0, 255),  # Magenta
) -> np.ndarray:
    """Draw bounding boxes, detected propellers, trajectory trails, and density lumps on frame.
    
    Args:
        frame: Grayscale frame to draw on
        detections: List of drone bounding boxes (x, y, w, h)
        drone_propellers: Dictionary mapping drone bbox to list of propellers
            (x, y, w, h, angle, id)
        color: Color for drone bounding boxes
        propeller_color: Color for propeller ellipses
        trajectories: Dictionary mapping drone bbox to list of trail points
        trail_color: Color for trajectory trails
        density_lumps: List of density lumps (x, y, w, h, compactness_score)
        density_color: Color for density lump visualization
        
    Returns:
        BGR frame with detections drawn
    """
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Draw density lumps first (so they appear behind other detections)
    if density_lumps is not None:
        for x, y, w, h, compactness_score in density_lumps:
            # Draw semi-transparent rectangle with intensity based on compactness
            # Higher compactness = more opaque
            alpha = 0.3 + (compactness_score * 0.4)  # 0.3 to 0.7 opacity
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), density_color, 2)
            cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)
            
            # Draw center point
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(frame_bgr, (center_x, center_y), 3, density_color, -1)
            
            # Label with compactness score
            label = f"C:{compactness_score:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                density_color,
                1,
                cv2.LINE_AA,
            )

    # Draw trajectories (trails) before bounding boxes so they appear behind
    if trajectories is not None:
        for drone_bbox, trail_points in trajectories.items():
            if len(trail_points) < 2:
                continue
            
            # Draw trail as connected lines
            points = np.array(trail_points, dtype=np.int32)
            cv2.polylines(
                frame_bgr,
                [points],
                isClosed=False,
                color=trail_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            
            # Optionally draw points along the trail (fade out)
            for i, (px, py) in enumerate(trail_points):
                # Fade intensity based on age (older points are dimmer)
                alpha = 1.0 - (i / len(trail_points)) * 0.7
                point_color = tuple(int(c * alpha) for c in trail_color)
                cv2.circle(frame_bgr, (int(px), int(py)), 2, point_color, -1)

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
        "--history-frames",
        type=int,
        default=10,
        help="Number of frames to use for temporal filtering (default: 10). "
        "Higher values provide more stable tracking but slower reaction to new drones.",
    )
    parser.add_argument(
        "--min-consistency",
        type=float,
        default=0.25,
        help="Minimum consistency ratio for temporal filtering (0-1, default: 0.25). "
        "Lower values allow temporary occlusion. Higher values filter more aggressively.",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=120.0,
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
    parser.add_argument(
        "--show-trail",
        action="store_true",
        help="Show trajectory trail for detected drones",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=50,
        help="Maximum number of positions to show in trail (default: 50)",
    )
    parser.add_argument(
        "--show-density-lumps",
        action="store_true",
        help="Show detected density lumps (high activity regions) in magenta",
    )
    parser.add_argument(
        "--compactness-threshold",
        type=float,
        default=0.4,
        help="Compactness threshold for detecting lumps (0-1, default: 0.4). "
        "Fraction of pixels that must be active. Higher values detect only regions with better coverage.",
    )
    parser.add_argument(
        "--min-lump-area",
        type=int,
        default=600,
        help="Minimum area for density lumps in pixels (default: 600)",
    )
    parser.add_argument(
        "--max-lump-area",
        type=int,
        default=10000,
        help="Maximum area for density lumps in pixels (default: 10000)",
    )
    args = parser.parse_args()

    # Initialize data source
    src = DatFileSource(
        args.dat,
        width=DEFAULT_FRAME_WIDTH,
        height=DEFAULT_FRAME_HEIGHT,
        window_length_us=args.window * 1000,
    )

    # Initialize pacer for playback
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # Detection history for temporal filtering
    detection_history: deque = deque(maxlen=args.history_frames)
    
    # Propeller tracker for maintaining identity across frames (only if enabled)
    propeller_tracker = PropellerTracker(max_distance=100.0, max_age=5) if args.detect_propellers else None
    
    # Trajectory tracker for drawing trails (only if enabled)
    # max_age=15 allows drones to disappear for up to 15 frames before losing identity
    trajectory_tracker = DroneTrajectoryTracker(
        max_distance=250.0,  # Increased for faster drone movement
        max_trail_length=args.trail_length,
        max_age=15,  # Frames a drone can be missing before losing identity (increased for occlusion)
    ) if args.show_trail else None

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
                        
            # Check compactness (spatial coverage) and spatial distribution
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            
            # Calculate compactness (spatial coverage) - fraction of pixels with activity
            non_zero = np.count_nonzero(roi > args.threshold * 0.5)
            compactness = non_zero / area if area > 0 else 0
            
            # Filter very sparse regions (trees/background) using compactness
            # Use threshold of 0.4 (40% of pixels must be active)
            min_compactness_threshold = DEFAULT_MIN_COMPACTNESS  # 0.4
            if compactness < min_compactness_threshold:
                continue
            
            # Filter very large sparse regions (trees) using compactness
            if area > DEFAULT_LARGE_AREA_THRESHOLD and compactness < DEFAULT_LARGE_AREA_COMPACTNESS:
                continue
            
            # Check for irregular distribution (trees have irregular patterns)
            # Drones have more regular, symmetric patterns
            if area > 1000:
                # Divide ROI into grid and check distribution
                grid_size = DEFAULT_GRID_SIZE
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
                            if cv > DEFAULT_CV_THRESHOLD:  # High coefficient of variation
                                continue
            
            all_detections.append((x, y, w, h))
        
        # Merge nearby detections (fixes issue where rotating drone is detected as multiple)
        merged_detections = merge_nearby_detections(
            all_detections,
            max_distance=args.merge_distance,
            min_overlap=DEFAULT_MIN_OVERLAP_MERGE,
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
        if len(detection_history) >= DEFAULT_MIN_TEMPORAL_HISTORY:
            filtered_detections = filter_detections_by_temporal_consistency(
                filtered_detections,
                detection_history,
                max_distance=DEFAULT_MAX_TRAIL_DISTANCE,  # Allow some movement
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

        # Update trajectory tracker and get trails
        trajectories = {}
        if trajectory_tracker is not None:
            bbox_to_id = trajectory_tracker.update(filtered_detections)
            # Build dictionary mapping bbox to trail points
            for bbox, traj_id in bbox_to_id.items():
                trail_points = trajectory_tracker.get_trajectory(traj_id)
                if len(trail_points) > 1:  # Need at least 2 points for a trail
                    trajectories[bbox] = trail_points
        
        # Detect density lumps (always detect, but only show if flag is set)
        density_lumps = detect_density_lumps(
            frame,
            compactness_threshold=args.compactness_threshold,
            min_lump_area=args.min_lump_area,
            max_lump_area=args.max_lump_area,
        )
        
        # Add high-compactness lumps (>= threshold) as drone detections
        high_compactness_drones = []
        for x, y, w, h, compactness_score in density_lumps:
            if compactness_score >= args.compactness_threshold:
                # Check if this lump overlaps with existing detections
                lump_center = (x + w // 2, y + h // 2)
                is_duplicate = False
                
                for det_x, det_y, det_w, det_h in filtered_detections:
                    det_center = (det_x + det_w // 2, det_y + det_h // 2)
                    distance = np.sqrt(
                        (lump_center[0] - det_center[0]) ** 2 +
                        (lump_center[1] - det_center[1]) ** 2
                    )
                    # If within 50 pixels, consider it a duplicate
                    if distance < 50:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    high_compactness_drones.append((x, y, w, h))
        
        # Add high-compactness lumps to filtered detections
        if high_compactness_drones:
            filtered_detections.extend(high_compactness_drones)
        
        # Only show density lumps in visualization if flag is set
        if not args.show_density_lumps:
            density_lumps = None
        
        # Draw detections with propellers, trails, and density lumps
        display_frame = draw_detections(
            frame,
            filtered_detections,
            drone_propellers,
            trajectories=trajectories if args.show_trail else None,
            density_lumps=density_lumps,
        )
        
        # Debug visualization: show ROI and binary for propeller detection (only if enabled)
        if args.detect_propellers and args.show_propeller_debug and len(filtered_detections) > 0:
            # Show the first drone's ROI for debugging
            drone_bbox = filtered_detections[0]
            x, y, w, h = drone_bbox
            padding = DEFAULT_PADDING
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

