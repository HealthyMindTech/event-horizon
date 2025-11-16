# Event Horizon 🎯

**Microsecond motion radar for rotating objects**

Event Horizon transforms event cameras into microsecond-precision motion radars for ultra-fast rotating objects. Using only asynchronous event data, we achieve real-time propeller blade tracking and RPM measurement with performance optimized for ARM devices.

---

## 🚀 What We Built

### Core Capabilities

- **Propeller blade tracking** with microsecond temporal resolution
- **RPM estimation** using multiple methods (FFT, autocorrelation, zero-crossing analysis)
- **Real-time clustering** with DBSCAN-based temporal tracking
- **ARM SIMD optimization** achieving **1.71x speedup** using Neon intrinsics
- **Near real-time performance**: 10 seconds of event data processes in ~10 seconds

### Why This Matters

Traditional frame-based cameras are fundamentally limited by exposure time and frame rate. Event cameras capture every brightness change with microsecond precision, enabling us to:

- **Track individual propeller blades** rotating at thousands of RPM
- **Measure rotation frequency** to predict drone thrust and future positions
- **Detect periodic patterns** from various angles using FFT analysis
- **Deploy on mobile ARM devices** for field use (not just lab PCs)

This is critical for autonomous drone interception, no-fly zone enforcement, collision avoidance, and high-speed tracking applications.

---

## 📊 Performance

### ARM SIMD Optimization (@arm 📱)

Implemented SIMD parallelization using ARM Neon intrinsics in Rust:

- **1.71x speedup** on ARM processors
- Optimized operations: mean, variance, distance calculations, batch processing
- Enables **mobile deployment** for real-time tracking in the field
- Near real-time: 10-second event file processes in ~10 seconds

**Implementation**: Following [ARM's SIMD on Rust guide](https://learn.arm.com/learning-paths/cross-platform/simd-on-rust/simd-on-rust-part1/), we parallelized statistical calculations and clustering operations critical for blade tracking.

### Cloud Infrastructure (@Vultr ☁️)

Deployed hyperparameter optimization on **Vultr cloud infrastructure**:

- Memory-optimized compute instances for processing large event datasets
- Distributed parameter search for optimal DBSCAN clustering thresholds
- Fast iteration on detection algorithms

---

## 🏗️ System Architecture

### Pipeline Overview

```
Event Stream (.dat) 
  → Hot Pixel Filtering 
  → Temporal Windowing 
  → DBSCAN Clustering 
  → Blade Tracking 
  → Angle/Width Analysis 
  → RPM Estimation (FFT/Autocorrelation/Zero-Crossing)
```

### Core Components

1. **Event Stream Processing** (`evio` library)
   - Memory-mapped `.dat` file reading (Prophesee Metavision format)
   - Zero-copy decoding of packed event data
   - Real-time playback pacing

2. **Hot Pixel Filtering**
   - Identifies and removes noisy pixels with excessive events
   - Improves clustering quality

3. **Temporal Blade Clustering**
   - Initial DBSCAN clustering on time windows (10-20ms)
   - Rolling window tracking (0.5-0.75ms) with grace periods
   - Event-to-cluster assignment with distance thresholds
   - Periodic re-initialization to detect new blades

4. **Blade Statistics & Analysis**
   - Center position tracking with history
   - Angle calculation via linear regression
   - Width measurement (blade extent)
   - Confidence scoring

5. **RPM Estimation Methods**
   - **FFT Analysis**: Frequency domain periodicity detection
   - **Autocorrelation**: Time-domain period detection
   - **Angle Zero-Crossing**: Track sin(angle) zero crossings
   - **Width Maxima**: Detect peaks in blade width cycles

### Quality Filtering

Multi-stage filtering ensures robust blade detection:

- **Size constraints**: Min/max cluster events (100-10000)
- **Spatial filtering**: Max Y-spread to reject diffuse patterns
- **Temporal consistency**: Grace periods for intermittent blade visibility
- **Statistical validation**: Confidence scores based on linear fit quality

---

## 📦 Repository Structure

```
evio/
├── pyproject.toml                          # Python package configuration
├── config/
│   └── tracking_config.yaml                # Hyperparameters and presets
├── scripts/
│   ├── play_dat.py                         # Event stream visualizer
│   ├── simulate_propeller.py               # Synthetic event generation
│   ├── temporal_blade_tracking.py          # Core tracking implementation
│   ├── create_tracking_video_configurable.py  # Video generation with RPM
│   ├── analyze_blade_periodicity.py        # FFT & autocorrelation analysis
│   ├── detect_blade_periods.py             # Period detection methods
│   ├── test_clustering_robustness.py       # Parameter optimization (Python)
│   ├── test_rpm_robustness.py              # RPM consistency testing
│   ├── visualize_raw_events.py             # Event data exploration
│   └── detection/
│       ├── detect_drone.py                 # Drone detection & propeller tracking
│       ├── detect_drone_v2.py              # Enhanced detection pipeline
│       └── compare_detection_methods.py    # Method comparison
├── rust_tools/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs                         # Clustering robustness (Rust)
│   │   ├── simd_ops.rs                     # ARM Neon SIMD optimizations
│   │   ├── benchmark.rs                    # SIMD vs scalar benchmarks
│   │   └── bin/
│   │       ├── benchmark.rs                # Benchmark executable
│   │       └── create_tracking_video.rs    # Video generation (Rust)
└── src/evio/
    ├── core/
    │   ├── recording.py                    # .dat file abstraction
    │   ├── mmap.py                         # Memory-mapped I/O
    │   ├── pacer.py                        # Real-time playback
    │   └── index_scheduler.py              # Event indexing
    └── source/
        └── dat_file.py                     # .dat decoder
```

---

## 🏃 Quick Start

### Prerequisites

Install [UV](https://docs.astral.sh/uv/getting-started/installation/) for Python package management.

For Rust tools (optional):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd evio

# Install Python dependencies
uv sync

# Build Rust tools (optional, for SIMD benchmarks)
cd rust_tools
cargo build --release
cd ..
```

### Quick Examples

```bash
# Visualize event stream
uv run scripts/play_dat.py drone_idle.dat

# Track propeller blades and estimate RPM
uv run scripts/create_tracking_video_configurable.py \
    --config config/tracking_config.yaml \
    --preset drone_idle

# Analyze blade periodicity with FFT
uv run scripts/analyze_blade_periodicity.py \
    --config config/tracking_config.yaml \
    --preset drone_idle

# Detect drones with propeller tracking
uv run scripts/detection/detect_drone.py drone_moving.dat --detect-propellers

# Test clustering parameter robustness
uv run scripts/test_clustering_robustness.py \
    --config config/tracking_config.yaml \
    --preset drone_idle
```

---

## 🎮 Usage Guide

### Blade Tracking & RPM Estimation

Generate tracking video with comprehensive RPM analysis:

```bash
uv run scripts/create_tracking_video_configurable.py \
    --config config/tracking_config.yaml \
    --preset drone_idle
```

**Output:**
- Tracking video (`blade_tracking_video.mp4`) with annotated blades
- Analysis plots showing:
  - Blade angles over time
  - Confidence scores
  - Cluster widths
  - FFT frequency spectrum
  - Autocorrelation periodicity
  - RPM calculations

### Periodicity Analysis

Deep dive into rotation frequency detection:

```bash
uv run scripts/analyze_blade_periodicity.py \
    --config config/tracking_config.yaml \
    --preset drone_idle
```

**Methods:**
- **FFT**: Identifies dominant frequencies in angle/position signals
- **Autocorrelation**: Detects repeating patterns in time series
- **Angle Jump Analysis**: Measures intervals between large angular changes

**Output:** Comprehensive plots showing all three methods with RPM estimates.

### Hyperparameter Optimization

Test clustering robustness across parameters:

**Python version:**
```bash
uv run scripts/test_clustering_robustness.py \
    --config config/tracking_config.yaml \
    --preset drone_idle \
    --output results.csv
```

**Rust version (with SIMD optimization):**
```bash
cd rust_tools
cargo run --release -- \
    --config ../config/tracking_config.yaml \
    --preset drone_idle \
    --output ../clustering_robustness_results.csv
cd ..
```

Exports CSV with metrics across parameter combinations:
- eps (DBSCAN distance threshold)
- min_samples (minimum cluster size)
- window_us (temporal window duration)
- Cluster counts, sizes, spreads, noise percentages

### Drone Detection

Full drone detection with optional propeller tracking:

```bash
# Basic detection
uv run scripts/detection/detect_drone.py drone_moving.dat

# With propeller tracking and video output
uv run scripts/detection/detect_drone.py drone_moving.dat \
    --detect-propellers \
    --output-video detection.mp4 \
    --window 50 \
    --speed 1.0
```

**Visualization:**
- 🟩 Green boxes: Detected drones
- 🟧 Orange ellipses: Tracked propellers (P1-P4)
- HUD: Detection counts, timing info

### SIMD Benchmark

Compare SIMD vs scalar performance:

```bash
cd rust_tools
cargo run --release --bin benchmark
cd ..
```

Tests mean/variance calculations, distance computations, and batch operations across data sizes.

---

## ⚙️ Configuration

All scripts use `config/tracking_config.yaml` with preset support:

### Available Presets

- **`drone_idle`**: Stationary drone with sparse propeller events
- **`drone_moving`**: Moving drone with denser event patterns

### Key Parameters

**Data:**
- `input_file`: Path to .dat file
- `start_time_sec`: Start time offset
- `duration_sec`: Analysis window duration
- `polarity`: Event polarity filter (-1, 1, or null)

**Clustering:**
- `eps`: DBSCAN distance threshold (pixels)
- `min_samples`: Minimum events per cluster
- `window_us`: Initial clustering window (microseconds)
- `assignment_distance`: Max distance for event-to-cluster assignment
- `grace_period_us`: Time to keep sparse clusters alive

**Temporal Tracking:**
- `window_duration_us`: Rolling window size
- `reinit_interval_us`: Period for re-running DBSCAN
- `min_cluster_size`: Minimum events to keep cluster active

Edit `config/tracking_config.yaml` or add new presets for your datasets.

---

## 🔬 Technical Deep Dive

### Event Camera Data

Event cameras output asynchronous events when pixel brightness changes:

```
Event = (x, y, timestamp, polarity)
```

- **Microsecond timestamps**: Far exceeding traditional frame rates
- **Sparse representation**: Only changing pixels generate events
- **Polarity**: ON (+1) for brightness increase, OFF (-1) for decrease

### .dat File Format

Prophesee Metavision binary format:

**Header:**
```
% Width 1280
% Height 720
% Format EVT3
```

**Binary Events (8 bytes each):**
- Bits 0-13: X coordinate (14 bits)
- Bits 14-27: Y coordinate (14 bits)
- Bits 28-31: Polarity (4 bits)
- Bits 32-63: Timestamp in microseconds (32 bits)

**Memory-mapped I/O** enables zero-copy processing of millions of events.

### Blade Tracking Algorithm

**1. Initialization:**
- Load events in initial window (10-20ms)
- Apply DBSCAN clustering on (x, y) coordinates
- Filter clusters by size, Y-spread (reject diffuse patterns)

**2. Temporal Tracking:**
- Process events in rolling window (0.5-0.75ms)
- Assign new events to nearest cluster within threshold
- Update cluster statistics: center, angle (via linear regression), width
- Remove old events from rolling window
- Re-run DBSCAN periodically to detect new blades

**3. Grace Periods:**
- Keep clusters alive during sparse event periods (5-20ms)
- Critical for tracking fast-rotating blades that intermittently generate events

**4. Statistics Calculation:**
```python
# Linear regression for blade angle
slope, intercept = fit_line(x_coords, y_coords)
angle = atan(slope)

# Width = extent along blade axis
width = max(coords_along_axis) - min(coords_along_axis)

# Confidence from R² of linear fit
```

### RPM Estimation Methods

**Method 1: FFT (Frequency Domain)**
```python
angles_over_time = [...]
fft_magnitudes = fft(angles_over_time)
dominant_freq = frequency_of_max_magnitude
rpm = dominant_freq * 60
```

**Method 2: Autocorrelation (Time Domain)**
```python
autocorr = correlate(signal, signal, mode='full')
period = time_lag_of_first_peak
rpm = (1 / period) * 60
```

**Method 3: Zero-Crossing (Phase Tracking)**
```python
sin_angles = sin(angles)
zero_crossings = find_upward_crossings(sin_angles)
periods = diff(zero_crossings)
rpm = (1 / mean(periods)) * 60
```

**Method 4: Width Maxima**
```python
# Width cycles twice per rotation (blade appears/disappears)
peaks = find_peaks(widths)
period = mean(diff(peak_times))
rpm = (1 / period) * 60 / 2  # Divide by 2 for half-rotation cycles
```

### SIMD Optimization Details

Key operations parallelized with ARM Neon:

**Mean Calculation:**
```rust
// Process 2 f64 values per iteration (128-bit Neon registers)
let mut sum = vdupq_n_f64(0.0);
for chunk in values.chunks(2) {
    let v = vld1q_f64(chunk);
    sum = vaddq_f64(sum, v);
}
```

**Variance Calculation:**
```rust
let vmean = vdupq_n_f64(mean);
let diff = vsubq_f64(values, vmean);
let squared = vmulq_f64(diff, diff);
// Horizontal reduction for final sum
```

**Distance Computation:**
```rust
// Batch calculate distances from point to multiple targets
// Vectorized subtraction, multiplication, addition
```

**Speedup**: 1.71x on representative blade tracking workloads.

---

## 🎯 Key Insights & Challenges

### What Worked Well

✅ **Rolling window clustering** with grace periods handles intermittent blade visibility  
✅ **Multiple RPM methods** provide cross-validation (FFT most robust)  
✅ **SIMD optimization** enables mobile deployment on ARM devices  
✅ **YAML configuration** with presets enables easy parameter tuning  
✅ **Linear regression** for blade angle works well for thin, straight blades

### Challenges Overcome

⚠️ **Sparse events from distant/fast-rotating propellers**
- Solution: Long grace periods (20ms), periodic re-initialization

⚠️ **Parameter sensitivity** (eps, min_samples, window sizes)
- Solution: Hyperparameter search tools (Python + Rust)

⚠️ **Distinguishing blades from background**
- Solution: Y-spread filtering, temporal consistency checks

### Future Improvements

- **Real camera integration**: Test with live Prophesee devices (currently validated on recordings)
- **3D trajectory estimation**: Combine RPM with spatial tracking
- **Multi-drone scenarios**: Handle multiple quadcopters simultaneously
- **Machine learning**: Neural networks for event-based detection
- **Improved blade models**: Handle curved or flexible blades
- **Better tree/foliage rejection** for outdoor scenarios

---

## 🛠️ Development

### Adding New Event Sources

Extend `evio` library to support additional cameras:

1. Implement async stream in `src/evio/source/`
2. Yield standardized packets: `x_coords, y_coords, timestamps, polarities`
3. All algorithms work with any source automatically

### Creating New Datasets

Generate synthetic event data:

```bash
uv run scripts/simulate_propeller.py \
    --output output_quadcopter.dat \
    --rpm 5500 \
    --num-blades 4 \
    --duration 10.0 \
    --fps 1000
```

Parameters: rotation angle, position, noise level, etc.

### Testing & Validation

```bash
# Test RPM consistency across time points
uv run scripts/test_rpm_robustness.py \
    --config config/tracking_config.yaml \
    --preset drone_idle \
    --num-time-points 20

# Compare detection methods
uv run scripts/detection/compare_detection_methods.py
```

---

## 📚 Papers & References

- **Magrini et al. (2025)** - [Drone Detection with Event Cameras](https://arxiv.org/abs/2508.04564) - Comprehensive survey of event-based vision for drone detection, tracking, trajectory forecasting, and propeller signature analysis. arXiv:2508.04564 [cs.CV]
- [ARM SIMD on Rust](https://learn.arm.com/learning-paths/cross-platform/simd-on-rust/simd-on-rust-part1/)
- [Prophesee Metavision SDK](https://docs.prophesee.ai/)
- [Event Cameras: Principles and Applications](https://www.prophesee.ai/event-based-vision/)
- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

---

## 🏆 Acknowledgments

**@arm 📱** - SIMD optimization resources enabling 1.71x speedup on ARM processors  
**@Vultr ☁️** - Cloud infrastructure for hyperparameter optimization  

Special thanks to the event camera community and Prophesee for the Metavision SDK.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🤝 Contributing

We welcome contributions! Priority areas:

- Real-time camera integration (Prophesee, Inivation)
- Additional RPM estimation methods
- Performance optimization (GPU, more SIMD)
- Documentation improvements
- Dataset contributions

---

**Event Horizon** - Tracking the future at microsecond precision 🎯⚡

---

## 📖 Citation

If you use this work, please cite:

```
Event Horizon: Microsecond Motion Radar for Rotating Objects
https://github.com/<your-repo>/evio
```
