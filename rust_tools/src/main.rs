//! Test clustering robustness across different time points and parameter combinations.
//! Helps optimize DBSCAN parameters and validate consistency.
//!
//! Usage:
//!     cargo run --release -- --config ../config/tracking_config.yaml --preset drone_idle
//!     cargo run --release -- --config ../config/tracking_config.yaml --preset drone_idle --output results.csv

mod simd_ops;

use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use linfa::prelude::*;
use linfa_clustering::Dbscan;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

/// Event structure matching the Python implementation
#[derive(Debug, Clone, Copy)]
struct Event {
    x: u16,
    y: u16,
    p: i32,
    t: u64,
}

/// Configuration structure for YAML parsing
#[derive(Debug, Deserialize)]
struct Config {
    data: DataConfig,
    hot_pixel: HotPixelConfig,
    #[serde(default)]
    presets: HashMap<String, serde_yaml::Value>,
}

#[derive(Debug, Deserialize)]
struct DataConfig {
    input_file: String,
    width: usize,
    height: usize,
    start_time_sec: f64,
    duration_sec: f64,
    polarity: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct HotPixelConfig {
    threshold: usize,
}

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(name = "test_clustering_robustness")]
#[command(about = "Test clustering robustness across time points and parameters")]
struct Args {
    /// Path to configuration YAML file
    #[arg(long, default_value = "../config/tracking_config.yaml")]
    config: PathBuf,

    /// Preset name to use from config file
    #[arg(long)]
    preset: Option<String>,

    /// Output CSV file for results
    #[arg(long, default_value = "clustering_robustness_results.csv")]
    output: PathBuf,

    /// Number of time points to test
    #[arg(long, default_value = "10")]
    num_time_points: usize,

    /// Use expanded hyperparameter search space (more comprehensive but slower)
    #[arg(long)]
    expanded_search: bool,

    /// Use quick search (fewer parameters, faster execution)
    #[arg(long)]
    quick_search: bool,
}

/// Result of clustering test at a specific time point
#[derive(Debug)]
struct ClusteringResult {
    total_events: usize,
    total_clusters: usize,
    valid_clusters: usize,
    filtered_clusters: usize,
    noise_points: usize,
    noise_pct: f64,
    valid_cluster_sizes: Vec<usize>,
    valid_cluster_y_spreads: Vec<f64>,
    valid_cluster_x_spreads: Vec<f64>,
}

/// CSV output row
#[derive(Debug, Serialize)]
struct ResultRow {
    time_point_idx: usize,
    time_offset_ms: f64,
    start_time_us: u64,
    eps: f64,
    min_samples: usize,
    window_us: u64,
    window_ms: f64,
    min_cluster_events: usize,
    max_y_spread: f64,
    total_events: usize,
    total_clusters: usize,
    valid_clusters: usize,
    filtered_clusters: usize,
    noise_points: usize,
    noise_pct: f64,
    min_cluster_size: usize,
    max_cluster_size: usize,
    mean_cluster_size: f64,
    mean_y_spread: f64,
    mean_x_spread: f64,
}

fn load_config(config_path: &Path, preset: Option<&str>) -> Result<Config> {
    let file = File::open(config_path)
        .with_context(|| format!("Failed to open config file: {:?}", config_path))?;
    let reader = BufReader::new(file);
    let mut config: Config =
        serde_yaml::from_reader(reader).with_context(|| "Failed to parse YAML config")?;

    // Apply preset if specified
    if let Some(preset_name) = preset {
        if let Some(preset_config) = config.presets.get(preset_name) {
            if let serde_yaml::Value::Mapping(preset_map) = preset_config {
                // Merge data config
                if let Some(serde_yaml::Value::Mapping(data_map)) =
                    preset_map.get(&serde_yaml::Value::String("data".to_string()))
                {
                    if let Some(serde_yaml::Value::String(input_file)) =
                        data_map.get(&serde_yaml::Value::String("input_file".to_string()))
                    {
                        config.data.input_file = input_file.clone();
                    }
                    if let Some(serde_yaml::Value::Number(start_time)) =
                        data_map.get(&serde_yaml::Value::String("start_time_sec".to_string()))
                    {
                        if let Some(val) = start_time.as_f64() {
                            config.data.start_time_sec = val;
                        }
                    }
                    if let Some(serde_yaml::Value::Number(duration)) =
                        data_map.get(&serde_yaml::Value::String("duration_sec".to_string()))
                    {
                        if let Some(val) = duration.as_f64() {
                            config.data.duration_sec = val;
                        }
                    }
                    if let Some(polarity_val) =
                        data_map.get(&serde_yaml::Value::String("polarity".to_string()))
                    {
                        match polarity_val {
                            serde_yaml::Value::Number(n) => {
                                if let Some(val) = n.as_i64() {
                                    config.data.polarity = Some(val as i32);
                                }
                            }
                            serde_yaml::Value::Null => {
                                config.data.polarity = None;
                            }
                            _ => {}
                        }
                    }
                }

                // Merge hot_pixel config
                if let Some(serde_yaml::Value::Mapping(hp_map)) =
                    preset_map.get(&serde_yaml::Value::String("hot_pixel".to_string()))
                {
                    if let Some(serde_yaml::Value::Number(threshold)) =
                        hp_map.get(&serde_yaml::Value::String("threshold".to_string()))
                    {
                        if let Some(val) = threshold.as_u64() {
                            config.hot_pixel.threshold = val as usize;
                        }
                    }
                }

                println!("Applied preset: {}", preset_name);
            }
        }
    }

    Ok(config)
}

fn load_dat_file(filepath: &Path, _width: usize, _height: usize) -> Result<Vec<Event>> {
    let mut file =
        File::open(filepath).with_context(|| format!("Failed to open dat file: {:?}", filepath))?;

    // Skip header lines starting with '%'
    let mut offset = 0u64;
    let mut buf = [0u8; 1];
    loop {
        file.read_exact(&mut buf)?;
        if buf[0] != b'%' {
            // Found non-header byte, read type and size bytes
            let mut type_byte = [0u8; 1];
            let mut size_byte = [0u8; 1];
            type_byte[0] = buf[0];
            file.read_exact(&mut size_byte)?;

            if size_byte[0] != 8 {
                anyhow::bail!("Only 8-byte CD events supported");
            }

            offset = file.stream_position()?;
            break;
        }
        // Read rest of header line
        loop {
            file.read_exact(&mut buf)?;
            if buf[0] == b'\n' {
                break;
            }
        }
    }

    // Read all events from offset
    file.seek(std::io::SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);

    let mut event_data = Vec::new();
    loop {
        let mut ts_buf = [0u8; 4];
        let mut word_buf = [0u8; 4];

        match reader.read_exact(&mut ts_buf) {
            Ok(_) => {
                reader.read_exact(&mut word_buf)?;
                let timestamp = u32::from_le_bytes(ts_buf) as u64;
                let word = u32::from_le_bytes(word_buf);
                event_data.push((timestamp, word));
            }
            Err(_) => break,
        }
    }

    // Sort by timestamp (stable sort to preserve order for equal timestamps)
    event_data.sort_by_key(|&(t, _)| t);

    // Parse events
    let mut events = Vec::with_capacity(event_data.len());
    for (ts, word) in event_data {
        let x = (word & 0x3FFF) as u16;
        let y = ((word >> 14) & 0x3FFF) as u16;
        let p_raw = ((word >> 28) & 0xF) as i32;
        let p = if p_raw > 0 { 1 } else { -1 };

        events.push(Event { x, y, p, t: ts });
    }

    println!("Loaded {} events", events.len());
    Ok(events)
}

fn filter_hot_pixels(events: Vec<Event>, threshold: usize) -> Vec<Event> {
    // Count occurrences of each (x, y) coordinate
    let mut coord_counts: HashMap<(u16, u16), usize> = HashMap::new();
    for event in &events {
        *coord_counts.entry((event.x, event.y)).or_insert(0) += 1;
    }

    // Identify hot pixels
    let hot_pixels: HashSet<(u16, u16)> = coord_counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold)
        .map(|(coord, _)| coord)
        .collect();

    if hot_pixels.is_empty() {
        return events;
    }

    // Filter out hot pixels
    let filtered: Vec<Event> = events
        .into_iter()
        .filter(|e| !hot_pixels.contains(&(e.x, e.y)))
        .collect();

    println!("After hot pixel filtering: {} events", filtered.len());
    filtered
}

fn test_clustering_at_time(
    events: &[Event],
    start_time: u64,
    window_us: u64,
    eps: f64,
    min_samples: usize,
    min_cluster_events: usize,
    max_y_spread: f64,
) -> ClusteringResult {
    // Get events in window
    let end_time = start_time + window_us;
    let window_events: Vec<&Event> = events
        .iter()
        .filter(|e| e.t >= start_time && e.t < end_time)
        .collect();

    if window_events.len() < min_samples {
        return ClusteringResult {
            total_events: window_events.len(),
            total_clusters: 0,
            valid_clusters: 0,
            filtered_clusters: 0,
            noise_points: 0,
            noise_pct: 0.0,
            valid_cluster_sizes: Vec::new(),
            valid_cluster_y_spreads: Vec::new(),
            valid_cluster_x_spreads: Vec::new(),
        };
    }

    // Prepare coordinates for DBSCAN
    let n_events = window_events.len();
    let mut coords = Array2::<f64>::zeros((n_events, 2));
    for (i, event) in window_events.iter().enumerate() {
        coords[[i, 0]] = event.x as f64;
        coords[[i, 1]] = event.y as f64;
    }

    // Run DBSCAN using linfa's API
    let dataset = DatasetBase::from(coords);

    let clustering = match Dbscan::params(min_samples)
        .tolerance(eps)
        .transform(dataset)
    {
        Ok(c) => c,
        Err(_) => {
            return ClusteringResult {
                total_events: window_events.len(),
                total_clusters: 0,
                valid_clusters: 0,
                filtered_clusters: 0,
                noise_points: 0,
                noise_pct: 0.0,
                valid_cluster_sizes: Vec::new(),
                valid_cluster_y_spreads: Vec::new(),
                valid_cluster_x_spreads: Vec::new(),
            };
        }
    };

    let labels = clustering.targets();

    // Analyze results
    let mut cluster_map: HashMap<Option<usize>, Vec<usize>> = HashMap::new();
    for (idx, label) in labels.iter().enumerate() {
        cluster_map.entry(*label).or_insert_with(Vec::new).push(idx);
    }

    let noise_points = cluster_map.get(&None).map(|v| v.len()).unwrap_or(0);
    let total_clusters = cluster_map.len()
        - if cluster_map.contains_key(&None) {
            1
        } else {
            0
        };

    let mut valid_clusters = Vec::new();
    let mut filtered_clusters = 0;

    for (label, indices) in cluster_map.iter() {
        if label.is_none() {
            continue;
        }

        let cluster_size = indices.len();

        // Calculate cluster statistics
        let y_values: Vec<f64> = indices.iter().map(|&i| window_events[i].y as f64).collect();
        let x_values: Vec<f64> = indices.iter().map(|&i| window_events[i].x as f64).collect();

        // Use SIMD-optimized statistics calculations
        let (_y_mean, y_spread) = simd_ops::calculate_mean_and_std(&y_values);
        let (_x_mean, x_spread) = simd_ops::calculate_mean_and_std(&x_values);

        // Apply filters
        let passes_min_events = cluster_size >= min_cluster_events;
        let passes_y_spread = y_spread <= max_y_spread;

        if passes_min_events && passes_y_spread {
            valid_clusters.push((cluster_size, y_spread, x_spread));
        } else {
            filtered_clusters += 1;
        }
    }

    let noise_pct = if n_events > 0 {
        100.0 * noise_points as f64 / n_events as f64
    } else {
        0.0
    };

    ClusteringResult {
        total_events: n_events,
        total_clusters,
        valid_clusters: valid_clusters.len(),
        filtered_clusters,
        noise_points,
        noise_pct,
        valid_cluster_sizes: valid_clusters.iter().map(|c| c.0).collect(),
        valid_cluster_y_spreads: valid_clusters.iter().map(|c| c.1).collect(),
        valid_cluster_x_spreads: valid_clusters.iter().map(|c| c.2).collect(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    println!("Loading configuration from: {:?}", args.config);
    let config = load_config(&args.config, args.preset.as_deref())?;
    if let Some(ref preset) = args.preset {
        println!("Using preset: {}", preset);
    }

    // Load data
    let dat_file = args
        .config
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(&config.data.input_file);
    println!("Loading {:?}...", dat_file);
    let mut events = load_dat_file(&dat_file, config.data.width, config.data.height)?;

    // Get base parameters
    let min_time = events.iter().map(|e| e.t).min().unwrap_or(0);
    let base_start_time = min_time + (config.data.start_time_sec * 1_000_000.0) as u64;
    let duration_us = (config.data.duration_sec * 1_000_000.0) as u64;

    println!("\nTimestamp filtering:");
    println!("  Min time in file: {} µs", min_time);
    println!(
        "  Start time offset: {} µs ({} sec)",
        (config.data.start_time_sec * 1_000_000.0) as u64,
        config.data.start_time_sec
    );
    println!("  Base start time: {} µs", base_start_time);
    println!(
        "  Duration: {} µs ({} sec)",
        duration_us, config.data.duration_sec
    );
    println!("  End time: {} µs", base_start_time + duration_us);
    println!("  Polarity filter: {:?}", config.data.polarity);

    // Filter events: time range and polarity
    let end_time = base_start_time + duration_us;
    let before_filter = events.len();
    events.retain(|e| {
        let in_time_range = e.t >= base_start_time && e.t < end_time;
        let matches_polarity = config.data.polarity.map_or(true, |p| e.p == p);
        in_time_range && matches_polarity
    });

    println!("  Events before filter: {}", before_filter);
    println!("  Events after filter: {}", events.len());

    // Filter hot pixels
    events = filter_hot_pixels(events, config.hot_pixel.threshold);

    // Define parameter ranges to test based on search mode
    let (
        eps_values,
        min_samples_values,
        window_us_values,
        min_cluster_events_values,
        max_y_spread_values,
    ) = if args.quick_search {
        // Quick search: minimal parameter space for fast validation
        (
            vec![2.0, 3.0, 4.0],
            vec![5, 10, 15],
            vec![5000, 10000, 20000],
            vec![100, 200],
            vec![4.0, 5.0, 6.0],
        )
    } else if args.expanded_search {
        // Expanded search: comprehensive parameter space
        (
            vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0],
            vec![3, 5, 8, 10, 12, 15, 20, 25],
            vec![3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000],
            vec![50, 100, 150, 200, 250, 300, 400, 500],
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        )
    } else {
        // Default search: balanced parameter space
        (
            vec![2.0, 3.0, 4.0, 5.0],
            vec![5, 10, 15],
            vec![5000, 10000, 15000, 20000],
            vec![100, 200, 300],
            vec![4.0, 5.0, 6.0],
        )
    };

    // Generate time points to test
    let max_window = *window_us_values.iter().max().unwrap();
    let time_step = (duration_us - max_window) / (args.num_time_points - 1) as u64;
    let time_points: Vec<u64> = (0..args.num_time_points)
        .map(|i| base_start_time + i as u64 * time_step)
        .collect();

    println!("\n{}", "=".repeat(60));
    println!("TESTING CLUSTERING ROBUSTNESS");
    println!("{}", "=".repeat(60));
    println!("Time points: {}", args.num_time_points);
    println!("Parameter combinations to test:");
    println!("  eps: {:?}", eps_values);
    println!("  min_samples: {:?}", min_samples_values);
    println!(
        "  window_us: {:?} ms",
        window_us_values
            .iter()
            .map(|w| w / 1000)
            .collect::<Vec<_>>()
    );
    println!("  min_cluster_events: {:?}", min_cluster_events_values);
    println!("  max_y_spread: {:?}", max_y_spread_values);

    let total_tests = time_points.len()
        * eps_values.len()
        * min_samples_values.len()
        * window_us_values.len()
        * min_cluster_events_values.len()
        * max_y_spread_values.len();
    println!("\nTotal tests: {}", total_tests);
    println!("{}\n", "=".repeat(60));

    // Run tests
    let mut results = Vec::new();
    let mut test_count = 0;

    for (time_idx, &start_time) in time_points.iter().enumerate() {
        let time_offset_ms = (start_time - base_start_time) as f64 / 1000.0;
        println!(
            "Testing time point {}/{} (offset: {:.1}ms)",
            time_idx + 1,
            time_points.len(),
            time_offset_ms
        );

        for &eps in &eps_values {
            for &min_samples in &min_samples_values {
                for &window_us in &window_us_values {
                    for &min_cluster_events in &min_cluster_events_values {
                        for &max_y_spread in &max_y_spread_values {
                            test_count += 1;
                            if test_count % 100 == 0 {
                                println!(
                                    "  Progress: {}/{} ({:.1}%)",
                                    test_count,
                                    total_tests,
                                    100.0 * test_count as f64 / total_tests as f64
                                );
                            }

                            let result = test_clustering_at_time(
                                &events,
                                start_time,
                                window_us,
                                eps,
                                min_samples,
                                min_cluster_events,
                                max_y_spread,
                            );

                            let mean_cluster_size = if !result.valid_cluster_sizes.is_empty() {
                                result.valid_cluster_sizes.iter().sum::<usize>() as f64
                                    / result.valid_cluster_sizes.len() as f64
                            } else {
                                0.0
                            };

                            let mean_y_spread = if !result.valid_cluster_y_spreads.is_empty() {
                                result.valid_cluster_y_spreads.iter().sum::<f64>()
                                    / result.valid_cluster_y_spreads.len() as f64
                            } else {
                                0.0
                            };

                            let mean_x_spread = if !result.valid_cluster_x_spreads.is_empty() {
                                result.valid_cluster_x_spreads.iter().sum::<f64>()
                                    / result.valid_cluster_x_spreads.len() as f64
                            } else {
                                0.0
                            };

                            results.push(ResultRow {
                                time_point_idx: time_idx,
                                time_offset_ms,
                                start_time_us: start_time,
                                eps,
                                min_samples,
                                window_us,
                                window_ms: window_us as f64 / 1000.0,
                                min_cluster_events,
                                max_y_spread,
                                total_events: result.total_events,
                                total_clusters: result.total_clusters,
                                valid_clusters: result.valid_clusters,
                                filtered_clusters: result.filtered_clusters,
                                noise_points: result.noise_points,
                                noise_pct: result.noise_pct,
                                min_cluster_size: result
                                    .valid_cluster_sizes
                                    .iter()
                                    .min()
                                    .copied()
                                    .unwrap_or(0),
                                max_cluster_size: result
                                    .valid_cluster_sizes
                                    .iter()
                                    .max()
                                    .copied()
                                    .unwrap_or(0),
                                mean_cluster_size,
                                mean_y_spread,
                                mean_x_spread,
                            });
                        }
                    }
                }
            }
        }
    }

    // Save to CSV
    if !results.is_empty() {
        let mut wtr = Writer::from_path(&args.output)?;
        for result in &results {
            wtr.serialize(result)?;
        }
        wtr.flush()?;
    }

    println!("\n{}", "=".repeat(60));
    println!("Results saved to: {:?}", args.output);
    println!("{}", "=".repeat(60));

    // Analyze results
    println!("\nANALYSIS:");
    println!("{}\n", "=".repeat(60));

    // Find configurations that consistently find 4 valid clusters
    let ideal_results: Vec<&ResultRow> = results.iter().filter(|r| r.valid_clusters == 4).collect();
    println!(
        "Configurations that found 4 valid clusters: {}/{} ({:.1}%)",
        ideal_results.len(),
        results.len(),
        100.0 * ideal_results.len() as f64 / results.len() as f64
    );

    if !ideal_results.is_empty() {
        println!("\nMost consistent parameter combinations (found 4 clusters most often):");

        // Count parameter combinations
        let mut param_counts: HashMap<(i32, usize, i32, usize, i32), usize> = HashMap::new();
        for r in &ideal_results {
            let key = (
                (r.eps * 10.0) as i32,
                r.min_samples,
                (r.window_ms * 10.0) as i32,
                r.min_cluster_events,
                (r.max_y_spread * 10.0) as i32,
            );
            *param_counts.entry(key).or_insert(0) += 1;
        }

        // Sort by count
        let mut sorted_params: Vec<_> = param_counts.into_iter().collect();
        sorted_params.sort_by(|a, b| b.1.cmp(&a.1));

        println!("\n  Top 10 parameter combinations:");
        println!(
            "  {:<5} {:<9} {:<11} {:<11} {:<9} {:<6}",
            "eps", "min_samp", "window_ms", "min_events", "max_y_sp", "count"
        );
        println!(
            "  {} {} {} {} {} {}",
            "-".repeat(5),
            "-".repeat(9),
            "-".repeat(11),
            "-".repeat(11),
            "-".repeat(9),
            "-".repeat(6)
        );

        for ((eps, min_samp, win_ms, min_ev, max_y), count) in sorted_params.iter().take(10) {
            println!(
                "  {:<5.1} {:<9} {:<11.1} {:<11} {:<9.1} {:<6}",
                *eps as f64 / 10.0,
                min_samp,
                *win_ms as f64 / 10.0,
                min_ev,
                *max_y as f64 / 10.0,
                count
            );
        }

        if let Some(((eps, min_samp, win_ms, min_ev, max_y), count)) = sorted_params.first() {
            println!("\nBest parameters (most occurrences of 4 clusters):");
            println!("  eps: {:.1}", *eps as f64 / 10.0);
            println!("  min_samples: {}", min_samp);
            println!("  window_ms: {:.1}", *win_ms as f64 / 10.0);
            println!("  min_cluster_events: {}", min_ev);
            println!("  max_y_spread: {:.1}", *max_y as f64 / 10.0);
            println!(
                "  Found 4 clusters in: {}/{} time points",
                count, args.num_time_points
            );
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Analysis complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
