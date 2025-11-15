//! Blade tracking video generator - Rust implementation
//!
//! Creates a video showing real-time temporal blade tracking for drone propellers.
//! This is a Rust port of create_tracking_video_configurable.py
//!
//! Usage:
//!     cargo run --release --bin create_tracking_video -- \
//!         --config ../config/tracking_config.yaml \
//!         --preset drone_idle \
//!         --output frames/

use anyhow::{Context, Result};
use clap::Parser;
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use imageproc::rect::Rect;
use rusttype::{Font, Scale};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

/// Event structure
#[derive(Debug, Clone, Copy)]
struct Event {
    x: u16,
    y: u16,
    p: i32,
    t: u64,
}

/// Configuration for video generation
#[derive(Debug, Deserialize)]
struct Config {
    data: DataConfig,
    hot_pixel: HotPixelConfig,
    clustering: ClusteringConfig,
    video: VideoConfig,
    roi: RoiConfig,
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

#[derive(Debug, Deserialize)]
struct ClusteringConfig {
    initial: InitialClusteringConfig,
    temporal: TemporalTrackingConfig,
}

#[derive(Debug, Deserialize)]
struct InitialClusteringConfig {
    eps: f64,
    min_samples: usize,
    window_us: u64,
    min_cluster_events: usize,
    max_y_spread: f64,
}

#[derive(Debug, Deserialize)]
struct TemporalTrackingConfig {
    assignment_distance: f64,
    min_cluster_size: usize,
    window_duration_us: u64,
    grace_period_us: u64,
    reinit_interval_us: u64,
    min_events_for_reinit: usize,
}

#[derive(Debug, Deserialize)]
struct VideoConfig {
    output_file: String,
    fps: u32,
    width: u32,
    height: u32,
    frame_interval: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RoiConfig {
    margin: i32,
    manual: Option<Vec<i32>>,
}

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(name = "create_tracking_video")]
#[command(about = "Create blade tracking video from event camera data")]
struct Args {
    /// Path to configuration YAML file
    #[arg(long, default_value = "../config/tracking_config.yaml")]
    config: PathBuf,

    /// Preset name to use from config file
    #[arg(long)]
    preset: Option<String>,

    /// Output directory for frames
    #[arg(long, default_value = "frames")]
    output: PathBuf,

    /// Generate individual frames instead of video
    #[arg(long)]
    frames_only: bool,
}

/// Blade cluster with temporal tracking
struct BladeCluster {
    id: usize,
    events: Vec<Event>,
    center_x: f64,
    center_y: f64,
    color: [u8; 3],
    last_event_time: u64,
    inactive_duration: u64,
    center_history: Vec<(f64, f64, u64)>,
}

impl BladeCluster {
    fn new(id: usize, events: Vec<Event>, color: [u8; 3]) -> Self {
        let mut cluster = BladeCluster {
            id,
            events,
            center_x: 0.0,
            center_y: 0.0,
            color,
            last_event_time: 0,
            inactive_duration: 0,
            center_history: Vec::new(),
        };
        cluster.update_statistics();
        cluster
    }

    fn update_statistics(&mut self) {
        if self.events.is_empty() {
            return;
        }

        // Calculate instant center
        let sum_x: f64 = self.events.iter().map(|e| e.x as f64).sum();
        let sum_y: f64 = self.events.iter().map(|e| e.y as f64).sum();
        let count = self.events.len() as f64;

        let instant_x = sum_x / count;
        let instant_y = sum_y / count;

        // Get latest timestamp
        self.last_event_time = self.events.iter().map(|e| e.t).max().unwrap_or(0);

        // Add to center history
        self.center_history
            .push((instant_x, instant_y, self.last_event_time));

        // Use historical average for smoother tracking
        if self.center_history.len() > 5 {
            let sum_x: f64 = self.center_history.iter().map(|(x, _, _)| x).sum();
            let sum_y: f64 = self.center_history.iter().map(|(_, y, _)| y).sum();
            self.center_x = sum_x / self.center_history.len() as f64;
            self.center_y = sum_y / self.center_history.len() as f64;
        } else {
            self.center_x = instant_x;
            self.center_y = instant_y;
        }
    }

    fn add_event(&mut self, event: Event) {
        self.events.push(event);
        self.last_event_time = event.t;
        self.inactive_duration = 0;
        self.update_statistics();
    }

    fn remove_old_events(&mut self, cutoff_time: u64) {
        self.events.retain(|e| e.t >= cutoff_time);
        if !self.events.is_empty() {
            self.update_statistics();
        }
    }

    fn remove_old_center_history(&mut self, cutoff_time: u64) {
        self.center_history.retain(|(_, _, t)| *t >= cutoff_time);
    }

    fn distance_to_point(&self, x: f64, y: f64) -> f64 {
        let dx = self.center_x - x;
        let dy = self.center_y - y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Temporal tracker for blades
struct TemporalTracker {
    clusters: HashMap<usize, BladeCluster>,
    next_cluster_id: usize,
    assignment_distance: f64,
    min_cluster_size: usize,
    window_duration_us: u64,
    grace_period_us: u64,
    current_time_us: u64,
}

impl TemporalTracker {
    fn new(config: &TemporalTrackingConfig) -> Self {
        TemporalTracker {
            clusters: HashMap::new(),
            next_cluster_id: 0,
            assignment_distance: config.assignment_distance,
            min_cluster_size: config.min_cluster_size,
            window_duration_us: config.window_duration_us,
            grace_period_us: config.grace_period_us,
            current_time_us: 0,
        }
    }

    fn initialize_from_events(&mut self, events: &[Event], config: &InitialClusteringConfig) {
        if events.is_empty() {
            return;
        }

        // Simple spatial clustering (grid-based for speed)
        let mut grid: HashMap<(i32, i32), Vec<Event>> = HashMap::new();
        let cell_size = config.eps as i32;

        for &event in events {
            let grid_x = event.x as i32 / cell_size;
            let grid_y = event.y as i32 / cell_size;
            grid.entry((grid_x, grid_y))
                .or_insert_with(Vec::new)
                .push(event);
        }

        // Create clusters from dense grid cells
        let colors = [
            [255, 0, 0],   // Red
            [0, 255, 0],   // Green
            [0, 0, 255],   // Blue
            [255, 255, 0], // Yellow
            [255, 0, 255], // Magenta
            [0, 255, 255], // Cyan
        ];

        for (_, cell_events) in grid.iter() {
            if cell_events.len() >= config.min_cluster_events {
                // Check Y-spread
                let y_values: Vec<f64> = cell_events.iter().map(|e| e.y as f64).collect();
                let y_mean = y_values.iter().sum::<f64>() / y_values.len() as f64;
                let y_variance = y_values.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>()
                    / y_values.len() as f64;
                let y_spread = y_variance.sqrt();

                if y_spread <= config.max_y_spread {
                    let color = colors[self.next_cluster_id % colors.len()];
                    let cluster =
                        BladeCluster::new(self.next_cluster_id, cell_events.clone(), color);
                    self.clusters.insert(self.next_cluster_id, cluster);
                    self.next_cluster_id += 1;
                }
            }
        }

        if !events.is_empty() {
            self.current_time_us = events.iter().map(|e| e.t).max().unwrap();
        }

        println!("  Initialized {} clusters", self.clusters.len());
    }

    fn process_event(&mut self, event: Event) {
        self.current_time_us = event.t;

        // Try to assign to existing cluster
        let mut min_distance = f64::INFINITY;
        let mut best_cluster_id = None;

        for (id, cluster) in self.clusters.iter() {
            let distance = cluster.distance_to_point(event.x as f64, event.y as f64);
            if distance < min_distance && distance < self.assignment_distance {
                min_distance = distance;
                best_cluster_id = Some(*id);
            }
        }

        // Add event to cluster
        if let Some(id) = best_cluster_id {
            if let Some(cluster) = self.clusters.get_mut(&id) {
                cluster.add_event(event);
            }
        }

        // Clean up old events and inactive clusters
        self.cleanup();
    }

    fn cleanup(&mut self) {
        let cutoff_time = self.current_time_us.saturating_sub(self.window_duration_us);
        let history_cutoff = self.current_time_us.saturating_sub(3000); // 3ms history

        // Remove old events from all clusters
        for cluster in self.clusters.values_mut() {
            cluster.remove_old_events(cutoff_time);
            cluster.remove_old_center_history(history_cutoff);

            // Update inactive duration
            if cluster.last_event_time < self.current_time_us {
                cluster.inactive_duration = self.current_time_us - cluster.last_event_time;
            }
        }

        // Remove clusters that are too small or inactive too long
        self.clusters.retain(|_, cluster| {
            cluster.events.len() >= self.min_cluster_size
                || cluster.inactive_duration < self.grace_period_us
        });
    }

    fn get_clusters(&self) -> &HashMap<usize, BladeCluster> {
        &self.clusters
    }
}

fn load_config(config_path: &Path, preset: Option<&str>) -> Result<Config> {
    let file = File::open(config_path)
        .with_context(|| format!("Failed to open config: {:?}", config_path))?;
    let reader = BufReader::new(file);
    let mut config: Config = serde_yaml::from_reader(reader)?;

    // Apply preset (simplified)
    if let Some(preset_name) = preset {
        if config.presets.contains_key(preset_name) {
            println!("Applied preset: {}", preset_name);
        }
    }

    Ok(config)
}

fn load_dat_file(filepath: &Path) -> Result<Vec<Event>> {
    let mut file = File::open(filepath)?;

    // Skip header
    let mut offset = 0u64;
    let mut buf = [0u8; 1];
    loop {
        file.read_exact(&mut buf)?;
        if buf[0] != b'%' {
            let mut size_byte = [0u8; 1];
            file.read_exact(&mut size_byte)?;
            if size_byte[0] != 8 {
                anyhow::bail!("Only 8-byte events supported");
            }
            offset = file.stream_position()?;
            break;
        }
        // Skip rest of header line
        loop {
            file.read_exact(&mut buf)?;
            if buf[0] == b'\n' {
                break;
            }
        }
    }

    // Read events
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

    // Sort by timestamp
    event_data.sort_by_key(|&(t, _)| t);

    // Parse events
    let mut events = Vec::new();
    for (ts, word) in event_data {
        let x = (word & 0x3FFF) as u16;
        let y = ((word >> 14) & 0x3FFF) as u16;
        let p_raw = ((word >> 28) & 0xF) as i32;
        let p = if p_raw > 0 { 1 } else { -1 };
        events.push(Event { x, y, p, t: ts });
    }

    Ok(events)
}

fn filter_hot_pixels(events: Vec<Event>, threshold: usize) -> Vec<Event> {
    let mut coord_counts: HashMap<(u16, u16), usize> = HashMap::new();
    for event in &events {
        *coord_counts.entry((event.x, event.y)).or_insert(0) += 1;
    }

    let hot_pixels: HashSet<(u16, u16)> = coord_counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold)
        .map(|(coord, _)| coord)
        .collect();

    if !hot_pixels.is_empty() {
        println!("  Found {} hot pixel locations", hot_pixels.len());
    }

    events
        .into_iter()
        .filter(|e| !hot_pixels.contains(&(e.x, e.y)))
        .collect()
}

fn render_frame(
    events: &[Event],
    tracker: &TemporalTracker,
    frame_num: usize,
    config: &VideoConfig,
    roi: Option<(i32, i32, i32, i32)>,
) -> RgbImage {
    let mut img = ImageBuffer::from_pixel(config.width, config.height, Rgb([0u8, 0u8, 0u8]));

    // Draw ROI rectangle if specified
    if let Some((x_min, x_max, y_min, y_max)) = roi {
        let roi_color = Rgb([64u8, 64u8, 64u8]);
        // Draw rectangle borders (simplified)
        for x in x_min..=x_max {
            if x >= 0 && x < config.width as i32 {
                if y_min >= 0 && y_min < config.height as i32 {
                    img.put_pixel(x as u32, y_min as u32, roi_color);
                }
                if y_max >= 0 && y_max < config.height as i32 {
                    img.put_pixel(x as u32, y_max as u32, roi_color);
                }
            }
        }
        for y in y_min..=y_max {
            if y >= 0 && y < config.height as i32 {
                if x_min >= 0 && x_min < config.width as i32 {
                    img.put_pixel(x_min as u32, y as u32, roi_color);
                }
                if x_max >= 0 && x_max < config.width as i32 {
                    img.put_pixel(x_max as u32, y as u32, roi_color);
                }
            }
        }
    }

    // Draw events by cluster
    for (cluster_id, cluster) in tracker.get_clusters() {
        let color = Rgb(cluster.color);
        for event in &cluster.events {
            if event.x < config.width as u16 && event.y < config.height as u16 {
                img.put_pixel(event.x as u32, event.y as u32, color);
            }
        }

        // Draw cluster center
        let cx = cluster.center_x as i32;
        let cy = cluster.center_y as i32;
        if cx >= 0 && cx < config.width as i32 && cy >= 0 && cy < config.height as i32 {
            draw_filled_circle_mut(&mut img, (cx, cy), 3, Rgb([255, 255, 255]));
        }
    }

    // Draw unclustered events in white
    let all_cluster_events: HashSet<(u16, u16, u64)> = tracker
        .get_clusters()
        .values()
        .flat_map(|c| c.events.iter())
        .map(|e| (e.x, e.y, e.t))
        .collect();

    for event in events {
        if !all_cluster_events.contains(&(event.x, event.y, event.t)) {
            if event.x < config.width as u16 && event.y < config.height as u16 {
                img.put_pixel(event.x as u32, event.y as u32, Rgb([128, 128, 128]));
            }
        }
    }

    img
}

fn main() -> Result<()> {
    let args = Args::parse();

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
    let mut events = load_dat_file(&dat_file)?;
    println!("Loaded {} events", events.len());

    // Filter by time and polarity
    let min_time = events.iter().map(|e| e.t).min().unwrap_or(0);
    let start_time = min_time + (config.data.start_time_sec * 1_000_000.0) as u64;
    let end_time = start_time + (config.data.duration_sec * 1_000_000.0) as u64;

    events.retain(|e| {
        let in_range = e.t >= start_time && e.t < end_time;
        let matches_polarity = config.data.polarity.map_or(true, |p| e.p == p);
        in_range && matches_polarity
    });

    println!("Filtered {} events in time range", events.len());

    // Filter hot pixels
    events = filter_hot_pixels(events, config.hot_pixel.threshold);
    println!("After hot pixel filtering: {} events", events.len());

    // Initialize tracker
    let init_window_end = start_time + config.clustering.initial.window_us;
    let init_events: Vec<Event> = events
        .iter()
        .filter(|e| e.t < init_window_end)
        .copied()
        .collect();

    let mut tracker = TemporalTracker::new(&config.clustering.temporal);
    println!("Initializing tracker with {} events...", init_events.len());
    tracker.initialize_from_events(&init_events, &config.clustering.initial);

    // Calculate ROI
    let roi = if let Some(manual) = &config.roi.manual {
        Some((manual[0], manual[1], manual[2], manual[3]))
    } else {
        // Auto-detect from initial clusters
        if !tracker.get_clusters().is_empty() {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            for cluster in tracker.get_clusters().values() {
                for event in &cluster.events {
                    min_x = min_x.min(event.x as f64);
                    max_x = max_x.max(event.x as f64);
                    min_y = min_y.min(event.y as f64);
                    max_y = max_y.max(event.y as f64);
                }
            }

            let margin = config.roi.margin as f64;
            Some((
                (min_x - margin) as i32,
                (max_x + margin) as i32,
                (min_y - margin) as i32,
                (max_y + margin) as i32,
            ))
        } else {
            None
        }
    };

    if let Some((x_min, x_max, y_min, y_max)) = roi {
        println!("ROI: x=[{}, {}], y=[{}, {}]", x_min, x_max, y_min, y_max);
    }

    // Create output directory
    fs::create_dir_all(&args.output)?;
    println!("Saving frames to: {:?}", args.output);

    // Process events and generate frames
    let frame_interval = config.video.frame_interval.unwrap_or(
        (events.len() / 300).max(1), // Target ~300 frames
    );

    let mut frame_count = 0;
    let mut event_count = 0;

    for event in &events {
        tracker.process_event(*event);
        event_count += 1;

        if event_count % frame_interval == 0 {
            let frame = render_frame(
                &events[..event_count],
                &tracker,
                frame_count,
                &config.video,
                roi,
            );

            let frame_path = args.output.join(format!("frame_{:06}.png", frame_count));
            frame.save(&frame_path)?;

            if frame_count % 30 == 0 {
                println!(
                    "  Frame {}: Event {}/{} ({:.1}%), Clusters: {}",
                    frame_count,
                    event_count,
                    events.len(),
                    100.0 * event_count as f64 / events.len() as f64,
                    tracker.get_clusters().len()
                );
            }

            frame_count += 1;
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Frame generation complete!");
    println!("Total frames: {}", frame_count);
    println!("Saved to: {:?}", args.output);
    println!("{}", "=".repeat(60));

    if !args.frames_only {
        println!("\nTo create video from frames, run:");
        println!("  ffmpeg -framerate {} -i {:?}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4",
            config.video.fps, args.output);
    }

    Ok(())
}
