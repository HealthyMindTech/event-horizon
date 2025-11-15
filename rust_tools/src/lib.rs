//! Test Clustering Robustness Library
//!
//! This library provides utilities for testing DBSCAN clustering robustness
//! on event camera data, including SIMD-optimized statistics calculations.

pub mod benchmark;
pub mod simd_ops;

// Re-export commonly used types
pub use benchmark::{
    benchmark_cluster_statistics, benchmark_statistics, estimate_pipeline_savings,
};
pub use simd_ops::{calculate_mean, calculate_mean_and_std, calculate_variance};
