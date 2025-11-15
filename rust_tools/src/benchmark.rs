//! Benchmark module to compare SIMD-optimized vs scalar implementations
//!
//! This module provides benchmarking utilities to measure the performance
//! difference between SIMD-optimized and scalar versions of statistics
//! calculations used in the clustering code.

use std::hint::black_box;
use std::time::Instant;

/// Scalar implementation of mean calculation (non-SIMD)
pub fn calculate_mean_scalar(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Scalar implementation of variance calculation (non-SIMD)
pub fn calculate_variance_scalar(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Scalar implementation of mean and std dev (non-SIMD)
pub fn calculate_mean_and_std_scalar(values: &[f64]) -> (f64, f64) {
    let mean = calculate_mean_scalar(values);
    let variance = calculate_variance_scalar(values, mean);
    (mean, variance.sqrt())
}

/// Benchmark result structure
#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration_ns: u128,
    pub iterations: usize,
    pub ns_per_iteration: f64,
}

impl BenchmarkResult {
    pub fn print(&self) {
        println!(
            "{}: {:.2}µs total, {:.2}ns per iteration ({} iterations)",
            self.name,
            self.duration_ns as f64 / 1000.0,
            self.ns_per_iteration,
            self.iterations
        );
    }
}

/// Run a benchmark comparing SIMD vs scalar implementation
pub fn benchmark_statistics(data_sizes: &[usize], iterations: usize) -> Vec<(usize, f64)> {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: SIMD vs Scalar Statistics Calculation");
    println!("{}", "=".repeat(60));

    let mut speedups = Vec::new();

    for &size in data_sizes {
        // Generate test data
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 1.5 + 10.0).collect();

        // Benchmark scalar version
        let start = Instant::now();
        for _ in 0..iterations {
            let result = calculate_mean_and_std_scalar(black_box(&data));
            black_box(result);
        }
        let scalar_duration = start.elapsed();

        // Benchmark SIMD version
        let start = Instant::now();
        for _ in 0..iterations {
            let result = crate::simd_ops::calculate_mean_and_std(black_box(&data));
            black_box(result);
        }
        let simd_duration = start.elapsed();

        let scalar_ns = scalar_duration.as_nanos();
        let simd_ns = simd_duration.as_nanos();
        let speedup = scalar_ns as f64 / simd_ns as f64;

        println!("\nData size: {} elements", size);
        println!(
            "  Scalar: {:.2}µs ({:.2}ns per iteration)",
            scalar_ns as f64 / 1000.0,
            scalar_ns as f64 / iterations as f64
        );
        println!(
            "  SIMD:   {:.2}µs ({:.2}ns per iteration)",
            simd_ns as f64 / 1000.0,
            simd_ns as f64 / iterations as f64
        );
        println!("  Speedup: {:.2}x", speedup);

        speedups.push((size, speedup));
    }

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY:");
    println!("{}", "=".repeat(60));
    println!("{:<15} | {:<15}", "Data Size", "Speedup");
    println!("{}", "-".repeat(32));
    for (size, speedup) in &speedups {
        println!("{:<15} | {:.2}x", size, speedup);
    }
    println!();

    speedups
}

/// Benchmark clustering statistics on realistic cluster sizes
pub fn benchmark_cluster_statistics() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Realistic Cluster Statistics");
    println!("{}", "=".repeat(60));

    // Typical cluster sizes in the event clustering application
    let cluster_sizes = vec![100, 200, 300, 500, 1000];
    let iterations = 10000;

    for &size in &cluster_sizes {
        // Simulate cluster coordinate data
        let y_coords: Vec<f64> = (0..size)
            .map(|i| 360.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let x_coords: Vec<f64> = (0..size)
            .map(|i| 640.0 + (i as f64 * 0.1).cos() * 50.0)
            .collect();

        println!("\nCluster size: {} events", size);

        // Benchmark scalar version for both coordinates
        let start = Instant::now();
        for _ in 0..iterations {
            let result1 = calculate_mean_and_std_scalar(black_box(&y_coords));
            black_box(result1);
            let result2 = calculate_mean_and_std_scalar(black_box(&x_coords));
            black_box(result2);
        }
        let scalar_duration = start.elapsed();

        // Benchmark SIMD version for both coordinates
        let start = Instant::now();
        for _ in 0..iterations {
            let result1 = crate::simd_ops::calculate_mean_and_std(black_box(&y_coords));
            black_box(result1);
            let result2 = crate::simd_ops::calculate_mean_and_std(black_box(&x_coords));
            black_box(result2);
        }
        let simd_duration = start.elapsed();

        let scalar_ns = scalar_duration.as_nanos();
        let simd_ns = simd_duration.as_nanos();
        let speedup = scalar_ns as f64 / simd_ns as f64;
        let time_saved_per_cluster_ns = (scalar_ns as f64 - simd_ns as f64) / iterations as f64;

        println!(
            "  Scalar: {:.2}µs total ({:.2}ns per cluster)",
            scalar_ns as f64 / 1000.0,
            scalar_ns as f64 / iterations as f64
        );
        println!(
            "  SIMD:   {:.2}µs total ({:.2}ns per cluster)",
            simd_ns as f64 / 1000.0,
            simd_ns as f64 / iterations as f64
        );
        println!("  Speedup: {:.2}x", speedup);
        println!(
            "  Time saved per cluster: {:.2}ns",
            time_saved_per_cluster_ns
        );
    }

    println!("\n{}", "=".repeat(60));
}

/// Estimate time savings in full clustering pipeline
pub fn estimate_pipeline_savings() {
    println!("\n{}", "=".repeat(60));
    println!("ESTIMATED TIME SAVINGS IN FULL PIPELINE");
    println!("{}", "=".repeat(60));

    // Typical parameters from the test
    let time_points = 10;
    let parameter_combinations = 1080; // 4*3*4*3*3
    let avg_clusters_per_test = 10;
    let avg_cluster_size = 200;

    println!("\nTest parameters:");
    println!("  Time points: {}", time_points);
    println!("  Parameter combinations: {}", parameter_combinations);
    println!("  Average clusters per test: {}", avg_clusters_per_test);
    println!("  Average cluster size: {} events", avg_cluster_size);

    // Benchmark a single cluster
    let coords: Vec<f64> = (0..avg_cluster_size)
        .map(|i| 360.0 + (i as f64 * 0.1).sin() * 5.0)
        .collect();

    let iterations = 1000;

    let start = Instant::now();
    for _ in 0..iterations {
        let result = calculate_mean_and_std_scalar(black_box(&coords));
        black_box(result);
    }
    let scalar_time_per_calc = start.elapsed().as_nanos() as f64 / iterations as f64;

    let start = Instant::now();
    for _ in 0..iterations {
        let result = crate::simd_ops::calculate_mean_and_std(black_box(&coords));
        black_box(result);
    }
    let simd_time_per_calc = start.elapsed().as_nanos() as f64 / iterations as f64;

    let time_saved_per_calc = scalar_time_per_calc - simd_time_per_calc;

    // Each cluster calculates stats for both x and y coordinates
    let calcs_per_cluster = 2;
    let total_clusters = time_points * parameter_combinations * avg_clusters_per_test;
    let total_calculations = total_clusters * calcs_per_cluster;

    let total_time_saved_ns = time_saved_per_calc * total_calculations as f64;
    let total_time_saved_ms = total_time_saved_ns / 1_000_000.0;
    let total_time_saved_sec = total_time_saved_ms / 1000.0;

    println!("\nBenchmark results:");
    println!(
        "  Scalar time per calculation: {:.2}ns",
        scalar_time_per_calc
    );
    println!("  SIMD time per calculation: {:.2}ns", simd_time_per_calc);
    println!("  Time saved per calculation: {:.2}ns", time_saved_per_calc);

    println!("\nEstimated savings for full test:");
    println!("  Total clusters analyzed: {}", total_clusters);
    println!("  Total calculations: {}", total_calculations);
    println!("  Total time saved: {:.2}ms", total_time_saved_ms);
    println!("  Total time saved: {:.3}s", total_time_saved_sec);

    let speedup = scalar_time_per_calc / simd_time_per_calc;
    println!("\n  Overall speedup: {:.2}x", speedup);
    println!(
        "  Percentage improvement: {:.1}%",
        ((speedup - 1.0) * 100.0)
    );

    println!("\n{}", "=".repeat(60));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_vs_simd_correctness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (scalar_mean, scalar_std) = calculate_mean_and_std_scalar(&data);
        let (simd_mean, simd_std) = crate::simd_ops::calculate_mean_and_std(&data);

        // Results should be very close (within floating point error)
        assert!((scalar_mean - simd_mean).abs() < 1e-10);
        assert!((scalar_std - simd_std).abs() < 1e-10);
    }

    #[test]
    fn test_benchmark_runs() {
        // Just verify the benchmark runs without crashing
        let sizes = vec![10, 100];
        let iterations = 10;
        let speedups = benchmark_statistics(&sizes, iterations);
        assert_eq!(speedups.len(), sizes.len());
    }
}
