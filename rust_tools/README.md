# Rust Clustering Robustness Test

This is a Rust port of the Python script `test_clustering_robustness.py`, which tests clustering robustness across different time points and parameter combinations to help optimize DBSCAN parameters and validate consistency.

## Features

- Load and parse event camera data from `.dat` files
- Filter hot pixels based on configurable thresholds
- Run DBSCAN clustering with various parameter combinations
- Test clustering across multiple time points
- Export results to CSV for analysis
- Analyze parameter effectiveness and robustness

## Building

Make sure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

```bash
cd rust_tools
cargo build --release
```

## Usage

Basic usage with default configuration:

```bash
cargo run --release -- --config ../config/tracking_config.yaml --preset drone_idle
```

Specify custom output file:

```bash
cargo run --release -- --config ../config/tracking_config.yaml --preset drone_idle --output my_results.csv
```

Test with a custom number of time points:

```bash
cargo run --release -- --config ../config/tracking_config.yaml --preset drone_idle --num-time-points 20
```

## Command-line Arguments

- `--config`: Path to configuration YAML file (default: `../config/tracking_config.yaml`)
- `--preset`: Preset name to use from config file (optional)
- `--output`: Output CSV file for results (default: `clustering_robustness_results.csv`)
- `--num-time-points`: Number of time points to test (default: `10`)

## Configuration

The tool expects a YAML configuration file with the following structure:

```yaml
data:
  input_file: "path/to/data.dat"
  width: 1280
  height: 720
  start_time_sec: 0.0
  duration_sec: 1.0
  polarity: 1  # Optional: 1, -1, or null for both

hot_pixel:
  threshold: 100

presets:
  drone_idle:
    data:
      start_time_sec: 0.5
      duration_sec: 0.2
```

## Parameter Ranges Tested

The tool tests various combinations of the following parameters:

- **eps**: DBSCAN epsilon (neighborhood radius) - [2.0, 3.0, 4.0, 5.0]
- **min_samples**: Minimum samples for core points - [5, 10, 15]
- **window_us**: Time window in microseconds - [5000, 10000, 15000, 20000]
- **min_cluster_events**: Minimum events per cluster - [100, 200, 300]
- **max_y_spread**: Maximum Y-axis spread for valid clusters - [4.0, 5.0, 6.0]

## Output

The tool generates:

1. **CSV file**: Contains detailed results for each parameter combination and time point
2. **Console analysis**: Displays statistics about:
   - Configurations that found the target number of clusters
   - Most consistent parameter combinations
   - Parameter impact on clustering quality

## Performance

The Rust implementation is significantly faster than the Python version due to:

- Compiled code vs interpreted Python
- Efficient memory management
- Native DBSCAN implementation via `linfa-clustering`

## Dependencies

Key Rust crates used:

- `clap`: Command-line argument parsing
- `serde` & `serde_yaml`: Configuration file parsing
- `csv`: CSV output generation
- `ndarray`: Multi-dimensional array support
- `linfa-clustering`: DBSCAN implementation
- `anyhow`: Error handling

## Differences from Python Version

The Rust version maintains functional parity with the Python version but with some implementation differences:

1. **Type safety**: Strong typing prevents many runtime errors
2. **Memory efficiency**: More explicit memory management
3. **Error handling**: Uses Result types instead of exceptions
4. **Performance**: Generally faster execution
5. **Preset merging**: Simplified compared to Python (basic implementation)

## Future Improvements

Potential enhancements:

- [ ] Parallel processing of time points using `rayon`
- [ ] More sophisticated preset merging
- [ ] Progress bar using `indicatif`
- [ ] Interactive visualization of results
- [ ] Support for additional clustering algorithms
- [ ] Memory-mapped file I/O for very large datasets

## License

MIT (same as parent project)