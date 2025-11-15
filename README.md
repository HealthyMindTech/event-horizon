# evio

Minimal Python library for standardized handling of event camera data.

**evio** provides a single abstraction for event streams. Each source yields standardized event packets containing `x_coords, y_coords, timestamps, polarities` arrays. This makes algorithms and filters source-agnostic.

---

## Features
- Unified async interface for event streams
- Read `.dat` recordings with optional real-time pacing
- Extensible to live cameras via adapter classes (requires Metavision SDK)

---

## Repository Structure

```
.
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ scripts/
│  └─ play_dat.py    
└─ src/
   └─ evio/
      ├─ __init__.py
      ├── core/
      │   ├── __init__.py
      │   ├── index_scheduler.py
      │   ├── mmap.py
      │   ├── pacer.py
      │   └── recording.py
      └─── source/
          ├── __init__.py
          └── dat_file.py
       
```

---

## Quick start using UV
If not already installed, install UV (instructions [here](https://docs.astral.sh/uv/getting-started/installation/)) \
Clone the repo and in the repo root run

```bash
# create venv and install dependencies.
uv sync

# play a .dat file in real time
uv run scripts/play_dat.py path/to/dat/file.dat
```

Adjust window duration in ms using `--window` argument and playback speed factor with `--speed` argument. When event data is constructed to frames we take all events between t and t + window and display them in the frame. With very short windows the rendering of the frames can take longer than the actual window duration and the player falls behind (depends on the playback speed), you can see this by comparing the wall clock to the recording clock in the GUI. In such cases you can force the playback speed with a `--force-speed` argument. This drops enough frames to make the recording play according to the set speed.

---


## `.dat` File Encoding

`evio` reads Prophesee Metavision-style DAT files, which store events as fixed-width binary records following a short ASCII header.

### Header
The file starts with text lines beginning with `%`, for example:

```
% Width 1280
% Height 720
% Format EVT3
```

After the header, two bytes appear:

- **event_type** — currently only stored in metadata (not interpreted by `evio`)
- **event_size** — must be `8`, meaning each event occupies 8 bytes

### Event Record Format (8 bytes)
The binary payload is interpreted as an array of structured records with dtype:

```python
_DTYPE_CD8 = np.dtype([("t32", "<u4"), ("w32", "<u4")])
```

Each event record is 8 bytes (64 bits):

- `t32` (upper 32 bits) is a little-endian `uint32` timestamp in microseconds.
- `w32` (lower 32 bits) packs polarity and coordinates as:

| Bits  | Meaning                                   |
|-------|-------------------------------------------|
| 31–28 | polarity (4 bits; > 0 → ON, 0 → OFF)      |
| 27–14 | y coordinate (14 bits)                    |
| 13–0  | x coordinate (14 bits)                    |

This matches the decoder:

```python
packed_w32 = raw_events["w32"].astype(np.uint32, copy=False)

decoded_x = (packed_w32 & 0x3FFF).astype(np.uint16, copy=False)
decoded_y = ((packed_w32 >> 14) & 0x3FFF).astype(np.uint16, copy=False)
raw_polarity = ((packed_w32 >> 28) & 0xF).astype(np.uint8, copy=False)
decoded_polarity = (raw_polarity > 0).astype(np.int8, copy=False)
```

### Decoded Arrays in `evio`
`evio` exposes the following decoded NumPy arrays:

- `x_coords` — uint16 (from bits 0–13)
- `y_coords` — uint16 (from bits 14–27)
- `timestamps` — int64 (from `t32` promoted from uint32)
- `polarities` — int8 (0 for OFF, 1 for ON)

### Memory-Mapped Reading
`evio` uses a `numpy.memmap` view of the event region with `_DTYPE_CD8` and performs zero-copy decoding of the packed fields. This allows:

- fast slicing of large recordings
- stable real-time playback
- minimal memory use even with millions of events




---

## Drone Detection

The repository includes a drone detection system that processes event camera data (`.dat` files) to detect drones and optionally track their propellers.

### Algorithm Overview

The drone detection pipeline consists of the following steps:

1. **Event Accumulation**: Events from a time window (default 50ms) are accumulated into a frame, converting sparse event data into a dense image representation.

2. **Motion Blob Detection**: 
   - Binary thresholding is applied to identify regions with significant event activity
   - Morphological operations (closing and opening) clean up the binary image
   - Contours are extracted to identify potential objects

3. **Drone Filtering**:
   - **Area filtering**: Objects must be within a reasonable size range (default: 200-50000 pixels)
   - **Compactness filtering**: Filters out sparse regions (like trees) by checking the density of events within bounding boxes
   - **Spatial regularity check**: Analyzes the distribution of events using a grid-based approach to identify irregular patterns (trees) versus uniform patterns (drones)
   - **Aspect ratio filtering**: Can filter out elongated objects (like planes) if needed

4. **Detection Merging**: Nearby or overlapping detections are merged to handle cases where a rotating drone is detected as multiple objects.

5. **Temporal Filtering**: 
   - Maintains a history of detections across frames
   - Filters out flickering false positives (like moving tree leaves) by requiring detections to appear consistently over time
   - Only detections that appear in at least 40% of recent frames are kept

6. **Propeller Detection** (optional, enabled with `--detect-propellers`):
   - For each detected drone, analyzes the region of interest (ROI) to detect propellers
   - Uses multiple detection methods:
     - **HoughCircles**: Detects circular patterns
     - **Ellipse fitting**: Detects elliptical patterns (propellers are typically horizontally elongated)
   - **Propeller tracking**: Maintains identity of 4 propellers per drone across frames:
     - Tracks each propeller's position and velocity
     - Matches detections to tracked propellers using predicted positions
     - Predicts positions for temporarily occluded propellers
     - Labels propellers as P1, P2, P3, P4
   - Ensures propellers are within the drone's bounding box
   - Limits to maximum 4 propellers per drone

### Running the Detection Script

#### Basic Usage

```bash
# Display detection in real-time window
uv run scripts/detection/detect_drone.py drone_moving.dat
```

#### Command-Line Options

**Core Options:**
- `--window WINDOW`: Window duration in milliseconds (default: 50ms)
- `--speed SPEED`: Playback speed factor (default: 1.0, 1.0 = real-time)
- `--force-speed`: Force playback speed by dropping frames if needed

**Detection Parameters:**
- `--min-area AREA`: Minimum detection area in pixels (default: 200)
- `--max-area AREA`: Maximum detection area in pixels (default: 50000)
- `--threshold THRESHOLD`: Event accumulation threshold (default: 30)
- `--merge-distance DISTANCE`: Maximum distance to merge nearby detections in pixels (default: 80)
- `--history-frames N`: Number of frames for temporal filtering (default: 5)
- `--min-consistency RATIO`: Minimum consistency ratio for temporal filtering, 0-1 (default: 0.4)

**Propeller Detection:**
- `--detect-propellers`: Enable propeller detection and tracking (disabled by default)
- `--propeller-min-radius RADIUS`: Minimum propeller radius in pixels (default: 5)
- `--propeller-max-radius RADIUS`: Maximum propeller radius in pixels (default: 50)
- `--show-propeller-debug`: Show debug visualization of propeller detection ROI

**Output:**
- `--output-video PATH`: Save detection video to file instead of displaying (e.g., `output.mp4`)
- `--debug`: Print debug information to console

#### Examples

```bash
# Basic detection with default settings
uv run scripts/detection/detect_drone.py drone_moving.dat

# Detection with propeller tracking enabled
uv run scripts/detection/detect_drone.py drone_moving.dat --detect-propellers

# Save detection video to file
uv run scripts/detection/detect_drone.py drone_moving.dat --output-video detection.mp4

# Detection with custom parameters and propeller tracking
uv run scripts/detection/detect_drone.py drone_moving.dat \
    --window 100 \
    --min-area 500 \
    --detect-propellers \
    --output-video output.mp4

# Faster playback with debug information
uv run scripts/detection/detect_drone.py drone_moving.dat \
    --speed 2.0 \
    --debug
```

#### Output

When displaying in a window:
- **Green bounding boxes**: Detected drones
- **Orange ellipses**: Detected propellers (if `--detect-propellers` is enabled)
- **Labels**: "DRONE" for drones, "P1", "P2", "P3", "P4" for propellers
- **HUD**: Shows detection count, propeller count, playback speed, and timing information

When saving to video:
- All visualizations are included in the output video
- Frame rate is automatically calculated from window duration
- No window is displayed (faster processing)

---

## License
MIT

